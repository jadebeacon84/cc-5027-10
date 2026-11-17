/**
 * Out-of-core octree builder.
 *
 * ============================ Coverage model ============================
 *
 * Every source point has a stable global id and a stable 32-bit rank
 * (hash of the id). The tree is built recursively:
 *
 *   build(cell, incoming):
 *     own = the `capacity` lowest-ranked points of incoming   (or all)
 *     rest = the other points, split into 8 child octants
 *
 * This gives the following invariants, which runtime selection relies on:
 *
 *   I1. Payloads are disjoint. A point sits in exactly one node payload, so
 *       drawing every visible node can never draw a point twice. Drawing a
 *       parent together with its children therefore does not double density.
 *
 *   I2. Rank selection is spatially independent, so a node's payload is a
 *       uniform random subsample of its whole subtree. The union of the
 *       ancestor chain's own points, ending at a node, is exactly "the lowest
 *       ranks of the subtree" — again a uniform sample, just denser. That is
 *       why ancestor points staying visible is not a density error.
 *
 *   I3. Coverage is ancestor-closed and additive. The renderer selects nodes
 *       only as a cut plus all ancestors. Refinement adds the child payloads;
 *       it never removes the parent payload. A late or failed child request
 *       therefore cannot open a hole where the parent used to be.
 *
 *   I4. A node is retired only when zoom/error makes its screen-space error
 *       below tolerance. Retiring happens bottom-up: descendants are dropped
 *       first, and the ancestor that replaces them is already resident, so the
 *       coverage never dips below the coarse level during the transition.
 *
 *   I5. Identity and attributes are coupled in one fixed-stride record. Payload
 *       headers carry checksums over exactly the bytes the GPU receives, so a
 *       shifted stride or a misaligned batch is detected rather than silently
 *       colouring the wrong vertex.
 *
 * =======================================================================
 */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import {
  HEADER_BYTES, MAGIC, VERSION, writeHeader, nodeBoundsAt, nodeFileName,
  stableRank, hashString, payloadLayout, fnv1a, fnv1aU32, LAYOUT_SOA,
  type AABB3, type Manifest, type NodeManifest,
} from "../src/shared/format";
import type { PointStore, GeneratedScan } from "./generators";

export type BuildOptions = {
  /** Maximum points kept in one node payload. */
  nodeCapacity?: number;
  maxDepth?: number;
  /** Refuse to build if a leaf payload exceeds this (sanity guard). */
  hardLeafLimit?: number;
  log?: (msg: string) => void;
};

export type BuiltNode = {
  manifest: NodeManifest;
  payload: Uint8Array;
};

export type BuildResult = {
  manifest: Manifest;
  nodes: Map<string, BuiltNode>;
  stats: {
    storePoints: number;
    payloadPoints: number;
    depthHistogram: number[];
    nodeCountByDepth: number[];
    maxPayload: number;
    avgPayload: number;
    maxDepthReached: number;
    buildMs: number;
  };
};

function octantOf(x: number, y: number, z: number, bounds: AABB3): number {
  const cx = (bounds.min[0] + bounds.max[0]) * 0.5;
  const cy = (bounds.min[1] + bounds.max[1]) * 0.5;
  const cz = (bounds.min[2] + bounds.max[2]) * 0.5;
  return (x >= cx ? 1 : 0) | (y >= cy ? 2 : 0) | (z >= cz ? 4 : 0);
}

/**
 * Iterative quickselect over `indices[left..right]` keyed by `key`.
 * Returns after the element at `k` is the k-th smallest; order is otherwise
 * unspecified. O(n) expected, no allocation.
 */
function selectKth(indices: Int32Array, key: Float64Array, left: number, right: number, k: number) {
  let l = left, r = right;
  while (l < r) {
    // Deterministic median-of-three pivot for repeatable builds.
    const mid = (l + r) >> 1;
    const a = key[indices[l]], b = key[indices[mid]], c = key[indices[r]];
    const pivot = a < b ? (b < c ? b : a < c ? c : a) : (a < c ? a : b < c ? c : b);
    let i = l, j = r;
    while (i <= j) {
      while (key[indices[i]] < pivot) i++;
      while (key[indices[j]] > pivot) j--;
      if (i <= j) {
        const t = indices[i]; indices[i] = indices[j]; indices[j] = t;
        i++; j--;
      }
    }
    if (k <= j) r = j;
    else if (k >= i) l = i;
    else return;
  }
}

/** Median nearest-neighbour distance estimated from a sparse uniform grid. */
export function estimateSpacing(
  store: PointStore,
  indices: Int32Array,
  bounds: AABB3,
  maxSamples = 600,
): number {
  const m = indices.length;
  const size = Math.max(
    bounds.max[0] - bounds.min[0],
    bounds.max[1] - bounds.min[1],
    bounds.max[2] - bounds.min[2],
  ) || 1;
  if (m < 8) return Math.max(1e-4, size / 12);
  const cellCount = Math.max(4, Math.min(96, Math.round(Math.cbrt(m) * 1.4)));
  const inv = cellCount / size;
  const cells = new Map<number, number[]>();
  const cellCoords = new Int32Array(m * 3);
  for (let k = 0; k < m; k++) {
    const i = indices[k];
    const cx = Math.min(cellCount - 1, Math.max(0, ((store.px[i] - bounds.min[0]) * inv) | 0));
    const cy = Math.min(cellCount - 1, Math.max(0, ((store.py[i] - bounds.min[1]) * inv) | 0));
    const cz = Math.min(cellCount - 1, Math.max(0, ((store.pz[i] - bounds.min[2]) * inv) | 0));
    const key = (cz * cellCount + cy) * cellCount + cx;
    cellCoords[k * 3] = cx; cellCoords[k * 3 + 1] = cy; cellCoords[k * 3 + 2] = cz;
    let list = cells.get(key);
    if (!list) { list = []; cells.set(key, list); }
    list.push(k);
  }
  const dists: number[] = [];
  const stride = Math.max(1, Math.floor(m / maxSamples));
  for (let k = 0; k < m && dists.length < maxSamples; k += stride) {
    const i = indices[k];
    const cx = cellCoords[k * 3], cy = cellCoords[k * 3 + 1], cz = cellCoords[k * 3 + 2];
    let best = Infinity;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx, ny = cy + dy, nz = cz + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= cellCount || ny >= cellCount || nz >= cellCount) continue;
          const list = cells.get((nz * cellCount + ny) * cellCount + nx);
          if (!list) continue;
          for (const j of list) {
            if (j <= k) continue;
            const i2 = indices[j];
            const d = Math.hypot(
              store.px[i2] - store.px[i], store.py[i2] - store.py[i], store.pz[i2] - store.pz[i],
            );
            if (d > 1e-7 && d < best) best = d;
          }
        }
      }
    }
    if (Number.isFinite(best)) dists.push(best);
  }
  if (dists.length === 0) return size / cellCount;
  dists.sort((a, b) => a - b);
  // 25th percentile of nearest-neighbour distance within this payload: the
  // spacing the viewer actually sees when this node is the finest level drawn.
  return dists[Math.floor(dists.length * 0.25)];
}

type PendingNode = {
  path: string;
  depth: number;
  indices: Int32Array;
  bounds: AABB3;
};

export function buildOctree(scan: GeneratedScan, opts: BuildOptions = {}): BuildResult {
  const t0 = Date.now();
  const log = opts.log ?? (() => {});
  const store = scan.store;
  const world = scan.bounds;
  const capacity = opts.nodeCapacity ?? 4096;
  const maxDepth = opts.maxDepth ?? 11;
  const hardLeafLimit = opts.hardLeafLimit ?? capacity * 8;

  const ranks = new Float64Array(store.n);
  for (let i = 0; i < store.n; i++) ranks[i] = stableRank(store.id[i]) / 4294967296;

  const rootBounds = nodeBoundsAt("0", 0, world);
  const rootIndices = new Int32Array(store.n);
  for (let i = 0; i < store.n; i++) rootIndices[i] = i;

  const byPath = new Map<string, NodeManifest>();
  const payloadIndex = new Map<string, Int32Array>();
  const stack: PendingNode[] = [{ path: "0", depth: 0, indices: rootIndices, bounds: rootBounds }];
  const seen = new Set<number>();
  let maxDepthReached = 0;

  while (stack.length) {
    const node = stack.pop()!;
    const { path, depth, indices, bounds } = node;
    maxDepthReached = Math.max(maxDepthReached, depth);

    let ownCount = indices.length;
    let restCount = 0;
    if (indices.length > capacity) {
      selectKth(indices, ranks, 0, indices.length - 1, capacity - 1);
      ownCount = capacity;
      restCount = indices.length - capacity;
    }
    const hitDepthLimit = depth >= maxDepth && restCount > 0;
    const isLeaf = restCount === 0 || hitDepthLimit;
    if (hitDepthLimit) {
      // Never drop source points. If the depth cap is reached with a crowded
      // leaf, that leaf takes the whole remainder; C5 then reports it as over
      // capacity so the depth/capacity settings get fixed rather than silently
      // losing detail.
      ownCount = indices.length;
      restCount = 0;
      if (ownCount > hardLeafLimit) {
        throw new Error(
          `leaf ${path} hit maxDepth with ${ownCount} points > hard limit ${hardLeafLimit}; raise --depth`,
        );
      }
    }

    const octantBuckets: number[][] = Array.from({ length: 8 }, () => []);
    const children: string[] = [];
    if (!isLeaf) {
      for (let k = 0; k < restCount; k++) {
        const i = indices[ownCount + k];
        octantBuckets[octantOf(store.px[i], store.py[i], store.pz[i], bounds)].push(i);
      }
      const base = path === "0" ? "0" : path;
      for (let oct = 0; oct < 8; oct++) {
        const bucket = octantBuckets[oct];
        if (bucket.length === 0) continue;
        const childPath = `${base}/${oct}`;
        children.push(childPath);
      }
    }

    const ownIdx = indices.slice(0, ownCount);
    for (let k = 0; k < ownIdx.length; k++) {
      const id = store.id[ownIdx[k]];
      if (seen.has(id)) throw new Error(`duplicate point id ${id} at ${path}`);
      seen.add(id);
    }

    const spacing = estimateSpacing(store, ownIdx, bounds);
    const manifest: NodeManifest = {
      id: path,
      path,
      depth,
      children,
      pointCount: ownCount,
      subtreePointCount: indices.length,
      bounds,
      spacing,
      geometricError: spacing,
      subtreeMinSpacing: spacing,
      bytes: 0,
      hash: "",
      payloadOffset: 0,
      payloadLength: 0,
      hasPayload: ownCount > 0,
      samplingRate: Math.min(1, capacity / Math.max(1, indices.length)),
    };
    byPath.set(path, manifest);
    payloadIndex.set(path, ownIdx);

    if (!isLeaf) {
      const base = path === "0" ? "0" : path;
      let oct = 0;
      for (const bucket of octantBuckets) {
        if (bucket.length === 0) { oct++; continue; }
        const childPath = `${base}/${oct}`;
        stack.push({
          path: childPath,
          depth: depth + 1,
          indices: Int32Array.from(bucket),
          bounds: nodeBoundsAt(childPath, depth + 1, world),
        });
        oct++;
      }
    }
  }

  // Bottom-up: `subtreeMinSpacing` is the finest spacing reachable below this
  // node (a diagnostic, and C7's density signal). `geometricError` is this
  // node's *own* sampling gap: the error a viewer sees while this payload is
  // the finest representation of the region. If it were the subtree minimum,
  // every coarse ancestor would look already-refined at any distance and the
  // traversal would stop at the root.
  const order = [...byPath.values()].sort((a, b) => b.depth - a.depth);
  for (const node of order) {
    let minSpacing = node.spacing;
    for (const c of node.children) {
      const cn = byPath.get(c)!;
      if (cn.subtreeMinSpacing > 0) minSpacing = Math.min(minSpacing, cn.subtreeMinSpacing);
    }
    node.subtreeMinSpacing = minSpacing;
    node.geometricError = node.spacing;
  }

  // Serialize in a stable order so byte layout and hashes are reproducible.
  const ordered = [...byPath.values()].sort((a, b) => comparePaths(a.path, b.path));
  const payloads = new Map<string, BuiltNode>();
  let payloadPoints = 0;
  let maxPayload = 0;
  const depthHistogram = new Array(maxDepth + 1).fill(0);
  const nodeCountByDepth = new Array(maxDepth + 1).fill(0);

  for (const node of ordered) {
    const own = payloadIndex.get(node.path)!;
    const count = own.length;
    const layout = payloadLayout(count);
    const payload = new Uint8Array(layout.end);
    const positions = new Float32Array(payload.buffer, layout.posOff, count * 3);
    const colours = new Uint8Array(payload.buffer, layout.colorOff, count * 4);
    const normals = new Int8Array(payload.buffer, layout.normalOff, count * 4);
    const intensity = new Float32Array(payload.buffer, layout.intensityOff, count);
    const ranges = new Float32Array(payload.buffer, layout.rangeOff, count);
    const ids = new Uint32Array(payload.buffer, layout.idOff, count);
    const scans = new Uint32Array(payload.buffer, layout.scanOff, count);
    const mn: [number, number, number] = [Infinity, Infinity, Infinity];
    const mx: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let k = 0; k < count; k++) {
      const i = own[k];
      const x = store.px[i], y = store.py[i], z = store.pz[i];
      positions[k * 3] = x; positions[k * 3 + 1] = y; positions[k * 3 + 2] = z;
      colours[k * 4] = store.r[i];
      colours[k * 4 + 1] = store.g[i];
      colours[k * 4 + 2] = store.b[i];
      colours[k * 4 + 3] = 255;
      normals[k * 4] = quantNormal(store.nx[i]);
      normals[k * 4 + 1] = quantNormal(store.ny[i]);
      normals[k * 4 + 2] = quantNormal(store.nz[i]);
      normals[k * 4 + 3] = 0;
      intensity[k] = store.intensity[i];
      ranges[k] = store.distance[i];
      ids[k] = store.id[i];
      scans[k] = store.scan[i];
      if (x < mn[0]) mn[0] = x; if (y < mn[1]) mn[1] = y; if (z < mn[2]) mn[2] = z;
      if (x > mx[0]) mx[0] = x; if (y > mx[1]) mx[1] = y; if (z > mx[2]) mx[2] = z;
    }
    const posBytes = new Uint8Array(payload.buffer, layout.posOff, count * 12);
    const colorBytes = new Uint8Array(payload.buffer, layout.colorOff, count * 4);
    const normalBytes = new Uint8Array(payload.buffer, layout.normalOff, count * 4);
    const checkPos = fnv1a(posBytes);
    const checkColor = fnv1a(colorBytes);
    const checkNormal = fnv1a(normalBytes);
    const checkId = fnv1aU32(ids);
    let checkAttr = fnv1a(posBytes);
    checkAttr = fnv1a(colorBytes, checkAttr);
    checkAttr = fnv1a(normalBytes, checkAttr);
    writeHeader(payload, {
      magic: MAGIC,
      version: VERSION,
      headerBytes: HEADER_BYTES,
      count,
      layout: LAYOUT_SOA,
      posOff: layout.posOff,
      colorOff: layout.colorOff,
      normalOff: layout.normalOff,
      intensityOff: layout.intensityOff,
      rangeOff: layout.rangeOff,
      idOff: layout.idOff,
      scanOff: layout.scanOff,
      checkPos,
      checkColor,
      checkNormal,
      checkId,
      checkAttr,
      nodeHash: hashString(node.path),
      min: mn,
      max: mx,
      payloadBytes: payload.byteLength,
    });
    node.bytes = payload.byteLength;
    node.payloadOffset = 0;
    node.payloadLength = payload.byteLength;
    node.hasPayload = count > 0;
    node.hash = createHash("sha1")
      .update(payload.subarray(0, Math.min(payload.length, 16384)))
      .update(node.path)
      .digest("hex")
      .slice(0, 16);
    payloads.set(node.path, { manifest: node, payload });
    payloadPoints += count;
    maxPayload = Math.max(maxPayload, count);
    depthHistogram[node.depth] += count;
    nodeCountByDepth[node.depth]++;
  }

  if (payloadPoints !== store.n) {
    throw new Error(`internal error: stored ${payloadPoints} of ${store.n} points`);
  }
  for (const node of byPath.values()) {
    for (const c of node.children) {
      if (!byPath.has(c)) throw new Error(`dangling child reference ${c}`);
    }
  }

  const spacings = ordered.map((n) => n.spacing).filter((s) => s > 0);
  const manifest: Manifest = {
    format: "pc-octree",
    version: VERSION,
    dataset: scan.generator,
    title: titleFor(scan.generator),
    generator: "scripts/generators.ts",
    bounds: world,
    totalPoints: store.n,
    nodeCount: byPath.size,
    payloadNodeCount: payloads.size,
    maxDepth: maxDepthReached,
    finestSpacing: spacings.length ? Math.min(...spacings) : 0.01,
    root: "0",
    childOrder: "octant bits x=1,y=2,z=4",
    maxNodeBytes: Math.max(0, ...ordered.map((n) => n.bytes)),
    nodeCapacity: capacity,
    nodes: ordered,
    createdAt: new Date().toISOString(),
    seed: scan.seed,
    notes: scan.notes,
  };

  const buildMs = Date.now() - t0;
  log(`built ${manifest.nodeCount} nodes, depth ${maxDepthReached}, ${payloadPoints} points, max payload ${maxPayload}, ${buildMs}ms`);

  return {
    manifest,
    nodes: payloads,
    stats: {
      storePoints: store.n,
      payloadPoints,
      depthHistogram,
      nodeCountByDepth,
      maxPayload,
      avgPayload: payloads.size ? payloadPoints / payloads.size : 0,
      maxDepthReached,
      buildMs,
    },
  };
}

function quantNormal(v: number): number {
  const q = Math.round(v * 127);
  return q < -127 ? -127 : q > 127 ? 127 : q;
}

function comparePaths(a: string, b: string): number {
  const pa = a.split("/").map(Number);
  const pb = b.split("/").map(Number);
  for (let i = 0; i < Math.min(pa.length, pb.length); i++) {
    if (pa[i] !== pb[i]) return pa[i] - pb[i];
  }
  return pa.length - pb.length;
}

function titleFor(generator: string): string {
  switch (generator) {
    case "assembly": return "Assembly hall — analytic surfaces, rods, density cliff";
    case "cliff": return "Cliff terrace — density cliff, suspended lattice, cables";
    case "small": return "Small mixed scan — fast validation dataset";
    default: return generator;
  }
}

export function writeDataset(outDir: string, scan: GeneratedScan, built: BuildResult) {
  fs.mkdirSync(path.join(outDir, "nodes"), { recursive: true });
  for (const [nodePath, entry] of built.nodes) {
    fs.writeFileSync(path.join(outDir, "nodes", nodeFileName(nodePath)), entry.payload);
  }
  const meta = {
    dataset: scan.generator,
    title: titleFor(scan.generator),
    generator: "scripts/generators.ts",
    description: scan.notes,
    seed: scan.seed,
    worldBounds: scan.bounds,
    totalPoints: scan.store.n,
    scanCount: scan.stations.length,
    stations: scan.stations,
    features: scan.features,
    surfaces: scan.surfaces,
    home: scan.home,
    defaultBudget: scan.defaultBudget,
    notes: "Synthetic deterministic scan; analytic surfaces allow attribute validation.",
  };
  fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta, null, 2));
  fs.writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(built.manifest, null, 2));
  fs.writeFileSync(path.join(outDir, "build-stats.json"), JSON.stringify(built.stats, null, 2));
}
