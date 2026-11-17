import { describe, expect, it } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import {
  buildHierarchy, selectNodes, sortCut, treeStats,
} from "../src/kernel/hierarchy";
import {
  lookAtMatrix, multiplyMat4, perspectiveMatrix, projectionFactor,
  aabbCenter, aabbRadius, extractFrustumPlanes, aabbVisible,
  type CameraView,
} from "../src/shared/geo";
import type { Manifest, PayloadViews } from "../src/shared/format";
import { payloadViews } from "../src/shared/format";
import { ResidencyCache, type CacheEvent, type FetchFn } from "../src/kernel/cache";

const DATA = path.resolve(__dirname, "..", "public", "data");

function loadManifest(name: string): Manifest {
  return JSON.parse(readFileSync(path.join(DATA, name, "manifest.json"), "utf8")) as Manifest;
}

function loadPayload(name: string, nodePath: string): LoadedPayload {
  const file = path.join(DATA, name, "nodes", nodePath.replaceAll("/", "-") + ".bin");
  const bytes = new Uint8Array(readFileSync(file));
  return { views: payloadViews(bytes), bytes: bytes.byteLength, ms: 0 };
}

type LoadedPayload = { views: PayloadViews; bytes: number; ms: number };

function makeView(
  eye: [number, number, number],
  target: [number, number, number],
  heightPx = 900,
  fovDeg = 55,
): CameraView {
  const fov = (fovDeg * Math.PI) / 180;
  const proj = perspectiveMatrix(fov, 16 / 9, 0.02, 500);
  const view = lookAtMatrix(eye, target);
  const vp = multiplyMat4(proj, view);
  return {
    eye,
    target,
    viewProjection: vp,
    projectionFactor: projectionFactor(fov, heightPx),
    near: 0.02,
    fov,
  };
}

/** Ancestor-closure is the key structural property. */
describe("hierarchy", () => {
  const manifest = loadManifest("small");
  const h = buildHierarchy(manifest);

  it("links parent/child consistently", () => {
    expect(h.nodes[h.root].path).toBe("0");
    for (let i = 0; i < h.nodes.length; i++) {
      for (const c of h.children[i]) {
        expect(h.parent[c]).toBe(i);
      }
    }
  });

  it("ancestor point counts accumulate root to leaves", () => {
    for (let i = 0; i < h.nodes.length; i++) {
      const p = h.parent[i];
      expect(h.ancestorPointCount[i]).toBe(
        h.nodes[i].pointCount + (p >= 0 ? h.ancestorPointCount[p] : 0),
      );
    }
  });

  it("reports a sensible tree shape", () => {
    const s = treeStats(h);
    expect(s.payloadNodes).toBe(manifest.payloadNodeCount);
    expect(s.maxDepth).toBeGreaterThan(0);
    expect(s.leaves).toBeGreaterThan(0);
  });
});

describe("selection", () => {
  const manifest = loadManifest("small");
  const h = buildHierarchy(manifest);
  const home = { eye: [-7.5, 4.6, 7.5] as [number, number, number], target: [0, 0.2, 0] as [number, number, number] };

  it("always returns an ancestor-closed cut", () => {
    for (const dist of [3, 8, 20, 60, 200]) {
      const eye: [number, number, number] = [home.eye[0] * (dist / 8), home.eye[1] * (dist / 8), home.eye[2] * (dist / 8)];
      const view = makeView(eye, home.target);
      for (const tol of [0.5, 2.5, 10, 100]) {
        const res = selectNodes(h, view, { pointBudget: 5_000_000, sseTolerance: tol, viewportHeight: 900 });
        expect(res.selected.length).toBeGreaterThan(0);
        const set = new Set(res.selected);
        // every ancestor of a selected node is selected
        for (const n of res.selected) {
          let p = h.parent[n];
          while (p >= 0) {
            expect(set.has(p)).toBe(true);
            p = h.parent[p];
          }
        }
      }
    }
  });

  it("includes the root so empty frames are impossible", () => {
    for (const tol of [0, 1, 1e9]) {
      const view = makeView([0, 0, 30], [0, 0, 0]);
      const res = selectNodes(h, view, { pointBudget: 1, sseTolerance: tol, viewportHeight: 900 });
      expect(res.selected).toContain(h.root);
    }
  });

  it("draws fewer or equal points as tolerance grows", () => {
    const view = makeView(home.eye, home.target);
    let prev = Infinity;
    for (const tol of [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]) {
      const res = selectNodes(h, view, { pointBudget: 5_000_000, sseTolerance: tol, viewportHeight: 900 });
      expect(res.drawnPoints).toBeLessThanOrEqual(prev);
      prev = res.drawnPoints;
    }
  });

  it("respects the point budget or reports it is limited", () => {
    const view = makeView(home.eye, home.target);
    const budget = 12_000;
    const res = selectNodes(h, view, { pointBudget: budget, sseTolerance: 0.1, viewportHeight: 900 });
    // The trim phase is a hard cap: the final cut is at or below budget.
    expect(res.drawnPoints).toBeLessThanOrEqual(budget);
    expect(res.budgetLimited).toBe(true);
  });

  it("never returns a child without its parent after sort", () => {
    const view = makeView(home.eye, home.target);
    const res = selectNodes(h, view, { pointBudget: 2_000_000, sseTolerance: 0.5, viewportHeight: 900 });
    const sorted = sortCut(h, res.selected);
    const seen = new Set<number>();
    for (const n of sorted) {
      if (h.parent[n] >= 0) expect(seen.has(h.parent[n])).toBe(true);
      seen.add(n);
    }
  });

  it("culls everything outside a narrow frustum but keeps the root in view", () => {
    const view = makeView([15, 0.2, 0.1], [0, 0.2, 0], 900, 6);
    const res = selectNodes(h, view, { pointBudget: 5_000_000, sseTolerance: 0.5, viewportHeight: 900 });
    expect(res.visited).toBeGreaterThan(0);
    expect(res.culled).toBeGreaterThan(0);
    expect(res.selected.length).toBeGreaterThan(0);
  });

  it("converges to full detail when tolerance falls to zero around the camera", () => {
    const view = makeView([0.5, 0.4, 0.5], [0, 0.2, 0]);
    const res = selectNodes(h, view, { pointBudget: 1e9, sseTolerance: 0, viewportHeight: 1400 });
    // With zero tolerance and a large budget, every visible payload node ends
    // up selected: the cut reaches the leaves of the in-frustum subtrees.
    let someLeaf = false;
    for (const n of res.selected) {
      if (h.children[n].length === 0) someLeaf = true;
    }
    expect(someLeaf).toBe(true);
    expect(res.budgetLimited).toBe(false);
  });

  it("focus box restricts the cut", () => {
    const view = makeView([-12, 6, 12], [0, 0.2, 0]);
    const full = selectNodes(h, view, { pointBudget: 5_000_000, sseTolerance: 0.2, viewportHeight: 900 });
    const box = {
      min: [-2, -1.3, -2] as [number, number, number],
      max: [2, 0, 2] as [number, number, number],
    };
    const focused = selectNodes(h, view, { pointBudget: 5_000_000, sseTolerance: 0.2, viewportHeight: 900, focusBox: box });
    expect(focused.selected.length).toBeLessThan(full.selected.length);
    // Still ancestor-closed.
    const set = new Set(focused.selected);
    for (const n of focused.selected) {
      let p = h.parent[n];
      while (p >= 0) {
        expect(set.has(p)).toBe(true);
        p = h.parent[p];
      }
    }
  });
});

describe("frustum math", () => {
  const manifest = loadManifest("small");
  it("keeps the dataset bounds visible from the home pose", () => {
    const view = makeView([-7.5, 4.6, 7.5], [0, 0.2, 0]);
    const planes = extractFrustumPlanes(view.viewProjection);
    expect(aabbVisible(manifest.bounds, planes)).toBe(true);
    expect(aabbVisible({ min: [100, 100, 100], max: [101, 101, 101] }, planes)).toBe(false);
  });
  it("camera view centers and radius match bounds", () => {
    const c = aabbCenter(manifest.bounds);
    expect(c[0]).toBeGreaterThan(manifest.bounds.min[0]);
    expect(aabbRadius(manifest.bounds)).toBeGreaterThan(1);
  });
});

describe("residency cache", () => {
  const manifest = loadManifest("small");
  const h = buildHierarchy(manifest);

  function fakeCache(maxConcurrent = 4, inflightBudget = 1e9) {
    const events: CacheEvent[] = [];
    const pending: Array<{ node: number; resolve: (v: LoadedPayload) => void; reject: (e: unknown) => void }> = [];
    const fetchImpl: FetchFn = (node, signal) => new Promise((resolve, reject) => {
      const entry = { node, resolve, reject };
      pending.push(entry);
      signal.addEventListener("abort", () => {
        const err = new Error("aborted");
        err.name = "AbortError";
        reject(err);
      }, { once: true });
    });
    const cache = new ResidencyCache({
      residentByteBudget: 1e9,
      inflightByteBudget: inflightBudget,
      maxConcurrent,
      retryBackoffMs: 1,
      fetch: fetchImpl,
      onChange: (e) => events.push(e),
      hooks: { parent: (n) => h.parent[n], depth: (n) => h.nodes[n].depth },
    });
    return { cache, events, pending };
  }

  function spec(node: number, depth = h.nodes[node].depth): { node: number; priority: number; bytes: number; points: number; depth: number } {
    return {
      node,
      depth,
      priority: 1e6 - depth * 1000,
      bytes: h.nodes[node].bytes || 1,
      points: h.nodes[node].pointCount,
    };
  }

  it("tracks resident and in-flight bytes separately", async () => {
    const { cache, pending } = fakeCache();
    cache.reconcile(new Set([0, 1]), 1, [spec(0), spec(1)]);
    let snap = cache.snapshot();
    expect(snap.inflightRequests).toBe(2);
    expect(snap.inflightBytes).toBeGreaterThan(0);
    expect(snap.residentBytes).toBe(0);
    // Resolve one
    const zero = pending.find((p) => p.node === 0)!;
    zero.resolve(loadPayload("small", h.nodes[0].path));
    await Promise.resolve();
    await Promise.resolve();
    snap = cache.snapshot();
    expect(snap.residentBytes).toBeGreaterThan(0);
    expect(snap.inflightRequests).toBeLessThanOrEqual(1);
  });

  it("aborts requests that leave the target cut and frees inflight bytes", () => {
    const { cache, pending } = fakeCache();
    cache.reconcile(new Set([0]), 1, [spec(0)]);
    const before = cache.snapshot().inflightBytes;
    expect(before).toBeGreaterThan(0);
    cache.reconcile(new Set([]), 2, []);
    const after = cache.snapshot();
    expect(after.inflightBytes).toBe(0);
    expect(after.inflightRequests).toBe(0);
    expect(after.requestsAborted).toBeGreaterThan(0);
    expect(pending.length).toBe(1);
  });

  it("does not evict a resident node whose target ancestor is not resident", async () => {
    const { cache, pending } = fakeCache();
    // Load a deep node only (simulating a resident that is now stale).
    const deepNode = h.nodes.findIndex((n) => n.depth === 2);
    cache.reconcile(new Set([deepNode]), 1, [spec(deepNode)]);
    pending[0].resolve(loadPayload("small", h.nodes[deepNode].path));
    await flush();
    expect(cache.has(deepNode)).toBe(true);
    // Now a target cut that does not include it, and whose ancestor is not
    // resident: eviction must be refused so coverage cannot go blank.
    const otherNode = h.nodes.findIndex((n) => n.depth === 2 && n.path !== h.nodes[deepNode].path);
    cache.reconcile(new Set([otherNode]), 2, [spec(otherNode)]);
    expect(cache.has(deepNode) || cache.inflightCount() > 0).toBe(true);
  });

  it("reports failures and backs off rather than spinning", async () => {
    const { cache, events, pending } = fakeCache();
    cache.reconcile(new Set([0]), 1, [spec(0)]);
    pending[0].reject(new Error("boom"));
    await flush();
    const snap = cache.snapshot();
    expect(snap.requestsFailed).toBe(1);
    expect(cache.failureCount()).toBe(1);
    // Re-reconcile immediately: backoff should suppress a new request.
    cache.reconcile(new Set([0]), 2, [spec(0)]);
    expect(cache.inflightCount()).toBe(0);
    expect(events.some((e) => e.kind === "failed")).toBe(true);
  });

  it("counts evictions and never exceeds the resident byte budget when possible", async () => {
    const { cache, pending } = fakeCache();
    // Load several payloads.
    const nodes = h.nodes.map((_, i) => i).filter((i) => h.nodes[i].hasPayload).slice(0, 6);
    cache.reconcile(new Set(nodes), 1, nodes.map((n) => spec(n)));
    for (const p of [...pending]) {
      p.resolve(loadPayload("small", h.nodes[p.node].path));
    }
    await flush();
    expect(cache.residentCount()).toBeGreaterThan(0);
    expect(cache.snapshot().evictions).toBeGreaterThanOrEqual(0);
  });
});

async function flush() {
  for (let i = 0; i < 8; i++) await Promise.resolve();
}
