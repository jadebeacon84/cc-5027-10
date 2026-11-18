/**
 * Streaming engine: the piece that turns a camera pose into a drawn, correct
 * point set and reports exactly what happened.
 *
 * Pipeline per frame:
 *   1. build a CameraView from the orbit camera
 *   2. select an ancestor-closed cut under point/byte/screen-error budgets
 *   3. reconcile the residency cache with that cut (cancel stale, evict safely)
 *   4. upload newly resident payloads, drop retired ones
 *   5. draw resident nodes belonging to the current cut
 *
 * The engine never removes a drawn node in the same frame it requests its
 * replacement. Coverage therefore only ever changes discontinuously upward
 * (more detail). When a node cannot be refined (budget, failure), its ancestor
 * stays and the region is reported as "coarse" rather than blank.
 */
import * as THREE from "three";
import type { Manifest, DatasetMeta } from "../shared/format";
import {
  aabbCenter, aabbRadius, perspectiveMatrix,
  projectionFactor, type AABB3, type CameraView, type Vec3,
} from "../shared/geo";
import { buildHierarchy, selectNodes, type Hierarchy, type NodeIndex, type SelectionResult } from "./hierarchy";
import { ResidencyCache, type CacheEvent, type CacheCounters, type RequestSpec } from "./cache";
import { createLoader, type NodeLoader } from "./loader";
import { PointRenderer, type PointCloudDisplay } from "../render/point-renderer";
import { OrbitCamera } from "../render/orbit-camera";

export type EngineOptions = {
  baseUrl: string;
  manifest: Manifest;
  meta: DatasetMeta;
  renderer: THREE.WebGLRenderer;
  canvas: HTMLCanvasElement;
  /** Injected latency for deterministic cancellation demos/tests. */
  latencyMs?: number;
  onEvent?: (event: CacheEvent) => void;
  onFrame?: (stats: FrameStats) => void;
};

export type FrameStats = {
  frame: number;
  time: number;
  dt: number;
  /** Points drawn on screen this frame (resident nodes in the cut). */
  drawnPoints: number;
  /** Nodes in the cut whose payload is resident and rendered. */
  drawnNodes: number;
  selectedNodes: number;
  selectedPointsExpected: number;
  residentBytes: number;
  inflightBytes: number;
  residentPoints: number;
  inflightPoints: number;
  residentNodes: number;
  inflightRequests: number;
  loadedBytesTotal: number;
  requestsIssued: number;
  requestsCompleted: number;
  requestsAborted: number;
  requestsFailed: number;
  evictions: number;
  /** Budget evictions deferred because the replacement ancestor was not resident. */
  evictionsDeferred: number;
  culledNodes: number;
  visitedNodes: number;
  budgetLimited: boolean;
  sseTolerance: number;
  pointBudget: number;
  /** Points selected for drawing but not yet resident (progress indicator). */
  pendingPoints: number;
  /** Nodes in the cut that are still in flight. */
  pendingNodes: number;
  /** Fraction of selected points actually on screen. */
  residency: number;
  /** Region coverage diagnostics. */
  unrefinedNodes: number;
  leafDepthHistogram: number[];
  cameraDistance: number;
  gpuBytes: number;
  fps: number;
  drawMs: number;
  selectMs: number;
  uploadMs: number;
  deepNodes: number;
};

export type SelectionSnapshot = {
  selected: NodeIndex[];
  selectedSet: Set<NodeIndex>;
  deferred: NodeIndex[];
  result: SelectionResult;
};

export class StreamingEngine {
  readonly hierarchy: Hierarchy;
  readonly cache: ResidencyCache;
  readonly renderer: PointRenderer;
  readonly orbit: OrbitCamera;
  readonly loader: NodeLoader;
  readonly meta: DatasetMeta;
  readonly manifest: Manifest;
  readonly view: CameraView;

  pointBudget: number;
  byteBudget: number;
  residentByteBudget: number;
  inflightByteBudget: number;
  sseTolerance: number;
  maxDepth: number;
  pointSize = 1.4;
  brightness = 1.0;

  private gl: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private frameCount = 0;
  private lastTime = 0;
  private fpsEma = 0;
  private selection: SelectionSnapshot;
  private epoch = 0;
  private focusBox: AABB3 | null = null;
  private display: PointCloudDisplay = "colour";
  private lastStats: FrameStats | null = null;
  private onFrame?: (stats: FrameStats) => void;
  private onEvent?: (event: CacheEvent) => void;
  private pendingUploads: Set<NodeIndex> = new Set();
  private forcedDepth = -1;
  private history: FrameStats[] = [];
  private traceLimit = 3600;

  constructor(opts: EngineOptions) {
    this.manifest = opts.manifest;
    this.meta = opts.meta;
    this.gl = opts.renderer;
    this.gl.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.hierarchy = buildHierarchy(opts.manifest);
    this.orbit = new OrbitCamera(1);
    this.renderer = new PointRenderer(this.hierarchy);
    this.scene = this.renderer.scene;
    this.scene.background = new THREE.Color(0x07090d);
    this.loader = createLoader({
      baseUrl: opts.baseUrl,
      manifest: opts.manifest,
      latencyMs: opts.latencyMs,
    });
    this.scene.add(new THREE.AmbientLight(0xffffff, 1));
    this.cache = new ResidencyCache({
      residentByteBudget: 0,
      inflightByteBudget: 0,
      maxConcurrent: 8,
      retryBackoffMs: 800,
      fetch: this.loader.fetch,
      onChange: (e) => this.handleCacheEvent(e),
      hooks: {
        parent: (n) => this.hierarchy.parent[n],
        depth: (n) => this.hierarchy.nodes[n].depth,
      },
    });
    this.onFrame = opts.onFrame;
    this.onEvent = opts.onEvent;

    const bounds = opts.manifest.bounds;
    const center = aabbCenter(bounds);
    const radius = aabbRadius(bounds);
    this.pointBudget = opts.meta.defaultBudget ?? 1_000_000;
    this.byteBudget = Number.POSITIVE_INFINITY;
    this.residentByteBudget = Math.max(64e6, opts.manifest.maxNodeBytes * 24);
    this.inflightByteBudget = Math.max(16e6, opts.manifest.maxNodeBytes * 6);
    this.sseTolerance = 2.5;
    this.maxDepth = this.hierarchy.depthMax;
    this.cache.setBudgets(this.residentByteBudget, this.inflightByteBudget);
    this.cache.setConcurrency(8);
    this.orbit.setPose({
      eye: opts.meta.home.eye,
      target: opts.meta.home.target,
      fov: opts.meta.home.fov,
    });
    this.orbit.minDistance = Math.max(0.02, radius * 0.002);
    this.orbit.maxDistance = radius * 40;
    this.orbit.setTargetImmediate([
      opts.meta.home.target[0],
      opts.meta.home.target[1],
      opts.meta.home.target[2],
    ]);
    this.orbit.camera.near = Math.max(0.02, radius * 0.0005);
    this.orbit.camera.far = radius * 80;
    this.renderer.setScanCount(opts.meta.scanCount);
    this.view = emptyCameraView();
    void center;
    this.selection = {
      selected: [],
      selectedSet: new Set(),
      deferred: [],
      result: emptySelection(),
    };
    this.resize();
  }

  setDisplay(mode: PointCloudDisplay) {
    this.display = mode;
    this.renderer.setDisplay(mode);
  }
  getDisplay() { return this.display; }
  setPointSize(px: number) {
    this.pointSize = px;
    this.renderer.setPointSize(px);
  }
  setBrightness(v: number) {
    this.brightness = v;
    this.renderer.setBrightness(v);
  }

  /**
   * Convert the split point/byte budgets into cache capacities. The cache keeps
   * more than the draw budget (about 2x) so small camera moves hit cache, but
   * the in-flight meter stays small so a fast zoom can cancel and refocus.
   */
  applyBudgets(pointBudget: number, byteBudget: number) {
    this.pointBudget = pointBudget;
    this.byteBudget = byteBudget;
    const bytesPerPoint = 36;
    const targetBytes = Math.min(this.byteBudget, pointBudget * bytesPerPoint * 2.5);
    this.residentByteBudget = Math.min(4e9, Math.max(48e6, targetBytes));
    this.inflightByteBudget = Math.min(this.residentByteBudget, Math.max(12e6, targetBytes * 0.18));
    this.cache.setBudgets(this.residentByteBudget, this.inflightByteBudget);
  }

  setBudgets(pointBudget: number, byteBudget: number, sseTolerance: number) {
    this.applyBudgets(pointBudget, byteBudget);
    this.sseTolerance = sseTolerance;
  }

  /** Limit refinement to a world box (region focus in the UI). */
  setFocusBox(box: AABB3 | null) { this.focusBox = box; }
  getFocusBox() { return this.focusBox; }

  /** Diagnostics: force a specific depth (or -1 for normal error-driven). */
  setForcedDepth(depth: number) { this.forcedDepth = depth; }

  resize() {
    const canvas = this.gl.domElement;
    const w = canvas.clientWidth || canvas.width;
    const h = canvas.clientHeight || canvas.height;
    this.gl.setSize(w, h, false);
    this.orbit.setViewport(w, h);
    this.renderer.setPixelRatio(this.gl.getPixelRatio());
  }

  private handleCacheEvent(e: CacheEvent) {
    if (e.kind === "resident") {
      this.pendingUploads.add(e.node);
    } else if (e.kind === "evicted") {
      this.renderer.removeNode(e.node);
    }
    this.onEvent?.(e);
  }

  /**
   * Run one frame. `now` is performance.now(); deterministic in tests when the
   * harness passes a synthetic clock.
   */
  frame(now: number): FrameStats {
    const t0 = performance.now();
    const dt = this.lastTime > 0 ? now - this.lastTime : 16;
    this.lastTime = now;
    this.frameCount++;
    this.orbit.update(now);
    this.orbit.setViewport(
      this.gl.domElement.clientWidth || this.gl.domElement.width,
      this.gl.domElement.clientHeight || this.gl.domElement.height,
    );

    // --- selection ---
    const selectStart = performance.now();
    const camera = this.orbit.camera;
    camera.updateMatrixWorld();
    const vp = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    const viewProjection = new Float64Array(vp.elements);
    const heightPx = this.gl.domElement.clientHeight || this.gl.domElement.height;
    const eye: Vec3 = [camera.position.x, camera.position.y, camera.position.z];
    const target: Vec3 = [this.orbit.target.x, this.orbit.target.y, this.orbit.target.z];
    this.view.eye = eye;
    this.view.target = target;
    this.view.viewProjection = viewProjection;
    this.view.projectionFactor = projectionFactor(camera.fov * Math.PI / 180, heightPx);
    this.view.near = camera.near;
    this.view.fov = camera.fov * Math.PI / 180;

    let result = selectNodes(this.hierarchy, this.view, {
      pointBudget: this.pointBudget,
      byteBudget: Number.isFinite(this.byteBudget) ? this.byteBudget : undefined,
      sseTolerance: this.sseTolerance,
      maxDepth: this.forcedDepth >= 0 ? Math.min(this.forcedDepth, this.hierarchy.depthMax) : this.maxDepth,
      focusBox: this.focusBox ?? undefined,
      viewportHeight: heightPx,
      near: camera.near,
    });
    const selectMs = performance.now() - selectStart;

    // The selected set already includes ancestors (selection enqueues only from
    // selected parents), so it is ancestor-closed by construction.
    const selectedSet = new Set(result.selected);
    this.selection = { selected: result.selected, selectedSet, deferred: result.deferred, result };

    // --- reconciliation ---
    this.epoch++;
    this.cache.beginFrame();
    const requests: RequestSpec[] = [];
    for (const node of result.selected) {
      const mapNode = this.hierarchy.nodes[node];
      if (!mapNode.hasPayload || mapNode.pointCount === 0) continue;
      this.cache.touch(node);
      if (this.cache.has(node) || this.cache.isInflight(node)) continue;
      const depth = mapNode.depth;
      requests.push({
        node,
        depth,
        bytes: mapNode.bytes,
        points: mapNode.pointCount,
        // Ancestors first; among siblings prefer nodes nearer the camera.
        priority: 1e6 - depth * 1000 + (node % 1000),
      });
    }
    this.cache.reconcile(selectedSet, this.epoch, requests);

    // --- GPU uploads ---
    // Upload every node in the cut that is resident but not yet on the GPU.
    // Doing this by inspecting the cut (rather than only this frame's arrival
    // events) means a payload that arrived for a previous cut is still uploaded
    // if the camera returns to it, and a failed upload cannot linger.
    const uploadStart = performance.now();
    for (const node of result.selected) {
      if (this.renderer.hasNode(node)) continue;
      const entry = this.cache.get(node);
      if (entry) this.renderer.addNode(node, entry.views);
    }
    this.pendingUploads.clear();
    // Remove any rendered node no longer in the cut. Its replacement (the
    // ancestor) is resident by construction, so this cannot create a hole.
    for (const node of this.rendererNodes()) {
      if (!selectedSet.has(node)) this.renderer.removeNode(node);
    }
    const uploadMs = performance.now() - uploadStart;

    // --- draw ---
    this.gl.render(this.scene, camera);

    const counters = this.cache.snapshot();
    const stats = this.buildStats(now, dt, result, counters, selectMs, uploadMs);
    this.lastStats = stats;
    this.fpsEma = this.fpsEma === 0 ? 1000 / Math.max(1, dt) : this.fpsEma * 0.9 + (1000 / Math.max(1, dt)) * 0.1;
    stats.fps = Math.round(this.fpsEma);
    stats.drawMs = performance.now() - t0;
    if (this.history.length >= this.traceLimit) this.history.shift();
    this.history.push(stats);
    this.onFrame?.(stats);
    return stats;
  }

  private buildStats(
    now: number, dt: number, result: SelectionResult,
    counters: CacheCounters & { failures: number }, selectMs: number, uploadMs: number,
  ): FrameStats {
    let pendingPoints = 0;
    let pendingNodes = 0;
    let unrefined = 0;
    const deferredSet = new Set(result.deferred);
    const depthHistogram = new Array(this.hierarchy.depthMax + 1).fill(0);
    for (const node of result.selected) {
      const mapNode = this.hierarchy.nodes[node];
      const entry = this.cache.get(node);
      if (entry) {
        depthHistogram[mapNode.depth] += entry.points;
      } else {
        pendingPoints += mapNode.pointCount;
        pendingNodes++;
      }
      // "Unrefined" = selected but not allowed to refine despite having
      // children whose error is above tolerance: budget-bound coarse regions.
      if (mapNode.children.length > 0 && deferredSet.has(node)) unrefined++;
    }
    // What the GPU actually draws, from the renderer's own bookkeeping.
    const render = this.renderer.getStats();
    const expected = result.drawnPoints;
    const cameraDistance = this.orbit.radius;
    return {
      frame: this.frameCount,
      time: now,
      dt,
      drawnNodes: render.drawnNodes,
      drawnPoints: render.drawnPoints,
      selectedNodes: result.selected.length,
      selectedPointsExpected: expected,
      residentBytes: counters.residentBytes,
      inflightBytes: counters.inflightBytes,
      residentPoints: counters.residentPoints,
      inflightPoints: counters.inflightPoints,
      residentNodes: counters.residentNodes,
      inflightRequests: counters.inflightRequests,
      loadedBytesTotal: counters.loadedBytesTotal,
      requestsIssued: counters.requestsIssued,
      requestsCompleted: counters.requestsCompleted,
      requestsAborted: counters.requestsAborted,
      requestsFailed: counters.requestsFailed,
      evictions: counters.evictions,
      evictionsDeferred: counters.evictionsDeferred,
      culledNodes: result.culled,
      visitedNodes: result.visited,
      budgetLimited: result.budgetLimited,
      sseTolerance: this.sseTolerance,
      pointBudget: this.pointBudget,
      pendingPoints,
      pendingNodes,
      residency: expected > 0 ? Math.min(1, render.drawnPoints / expected) : 1,
      unrefinedNodes: unrefined,
      leafDepthHistogram: depthHistogram,
      cameraDistance,
      gpuBytes: render.gpuBytes,
      fps: Math.round(this.fpsEma),
      drawMs: 0,
      selectMs,
      uploadMs,
      deepNodes: this.hierarchy.depthMax,
      };
  }

  private rendererNodes(): NodeIndex[] {
    return this.renderer.residentNodeList();
  }

  lastFrame(): FrameStats | null { return this.lastStats; }
  getHistory(): FrameStats[] { return this.history; }
  clearHistory() { this.history = []; }

  /** Abort all in-flight payload requests, leaving residents in place. */
  cancelInflight() { this.cache.cancelAllInflight(); }

  /**
   * Failure diagnostic used by the UI. Pick the deepest resident leaf in the
   * current cut, forget it, and force the re-fetch of it to fail. Its ancestors
   * remain resident, so the region stays covered at coarse density; the failure
   * and backoff are reported through the normal cache path.
   */
  injectLoadFailure(): { node: NodeIndex; path: string } | null {
    let pick = -1;
    let bestDepth = -1;
    for (const node of this.selection.selected) {
      if (node === this.hierarchy.root) continue;
      if (!this.cache.has(node)) continue;
      const depth = this.hierarchy.nodes[node].depth;
      if (depth > bestDepth) { bestDepth = depth; pick = node; }
    }
    if (pick < 0) return null;
    this.loader.failNext(1);
    this.cache.invalidate(pick);
    return { node: pick, path: this.hierarchy.nodes[pick].path };
  }

  /** True if the node's region is drawn at a coarser level than its detail. */
  isNodeUnrefined(node: NodeIndex): boolean {
    return this.selection.deferred.includes(node);
  }
  selectionSnapshot(): SelectionSnapshot { return this.selection; }

  /**
   * Coverage report in world space: sample a grid and report, for each sample,
   * the deepest selected node covering it. Used by the "unrefined regions"
   * overlay and by the trajectory report.
   */
  coverageGrid(resolution = 32): {
    cells: Array<{ x: number; y: number; z: number; depth: number; refined: boolean; drawn: boolean }>;
    minDepth: number;
    maxDepth: number;
    unrefined: number;
  } {
    const bounds = this.manifest.bounds;
    const cells: Array<{ x: number; y: number; z: number; depth: number; refined: boolean; drawn: boolean }> = [];
    let minDepth = 99, maxDepth = 0, unrefined = 0;
    const deferredSet = new Set(this.selection.deferred);
    for (let iz = 0; iz < resolution; iz++) {
      for (let iy = 0; iy < resolution; iy++) {
        for (let ix = 0; ix < resolution; ix++) {
          const fx = (ix + 0.5) / resolution;
          const fy = (iy + 0.5) / resolution;
          const fz = (iz + 0.5) / resolution;
          const x = bounds.min[0] + fx * (bounds.max[0] - bounds.min[0]);
          const y = bounds.min[1] + fy * (bounds.max[1] - bounds.min[1]);
          const z = bounds.min[2] + fz * (bounds.max[2] - bounds.min[2]);
          const node = deepestNodeContaining(this.hierarchy, x, y, z, this.selection.selectedSet);
          if (!node) continue;
          const depth = this.hierarchy.nodes[node].depth;
          minDepth = Math.min(minDepth, depth);
          maxDepth = Math.max(maxDepth, depth);
          const refined = !deferredSet.has(node);
          if (!refined) unrefined++;
          cells.push({ x, y, z, depth, refined, drawn: this.cache.has(node) });
        }
      }
    }
    return { cells, minDepth: minDepth === 99 ? 0 : minDepth, maxDepth, unrefined };
  }

  /** Load and draw the coarsest cut immediately; used by tests before timing. */
  warmUp(frames = 12) {
    for (let i = 0; i < frames; i++) this.frame(performance.now() + i * 16);
  }

  /** Release this dataset's GPU resources. The WebGL context itself is owned
   *  by the caller and survives dataset switches. */
  dispose() {
    this.cache.clear();
    this.renderer.dispose();
  }
}

function deepestNodeContaining(
  h: Hierarchy, x: number, y: number, z: number, selected: ReadonlySet<NodeIndex>,
): NodeIndex | null {
  let best: NodeIndex | null = null;
  // Selected is ancestor-closed; a linear scan is fine for diagnostics.
  for (const node of selected) {
    const b = h.nodes[node].bounds;
    if (x >= b.min[0] && x <= b.max[0] && y >= b.min[1] && y <= b.max[1] && z >= b.min[2] && z <= b.max[2]) {
      if (best === null || h.nodes[node].depth > h.nodes[best].depth) best = node;
    }
  }
  return best;
}

function emptySelection(): SelectionResult {
  return {
    selected: [],
    deferred: [],
    drawnPoints: 0,
    drawnBytes: 0,
    culled: 0,
    visited: 0,
    budgetLimited: false,
  };
}

function emptyCameraView(): CameraView {
  return {
    eye: [0, 0, 0],
    target: [0, 0, 0],
    viewProjection: perspectiveMatrix(Math.PI / 3, 1, 0.1, 1000),
    projectionFactor: 1,
    near: 0.1,
    fov: Math.PI / 3,
  };
}
