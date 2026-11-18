/**
 * Point-block recycling regression.
 *
 * The residency cache may only recycle a resident block outside the target
 * cut once the block's nearest ancestor *in the current cut* is itself
 * resident: the replacement must exist before the detail goes away. The
 * budget-enforcement path (reconcile's budget trim and the arrival-triggered
 * evictToBudget) used to skip that check, so a budget shrink or a late
 * arrival could recycle a block whose replacement was still in flight or
 * backing off after a failure. The region then dropped to a much coarser
 * ancestor — or a hole — for a few frames: the flicker this regression pins.
 *
 * Every test uses explicit tiny budgets and an explicit synthetic tree, so no
 * default value can mask the failure. The real-dataset test at the end ties
 * the mechanism back to the scanned point cloud payloads on disk.
 */
import { describe, expect, it } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import { ResidencyCache, type CacheEvent, type FetchFn } from "../src/kernel/cache";
import { buildHierarchy } from "../src/kernel/hierarchy";
import { PointRenderer } from "../src/render/point-renderer";
import { payloadViews, type Manifest, type PayloadViews } from "../src/shared/format";

const DATA = path.resolve(__dirname, "..", "public", "data");

function loadManifest(name: string): Manifest {
  return JSON.parse(readFileSync(path.join(DATA, name, "manifest.json"), "utf8")) as Manifest;
}

function loadPayload(dataset: string, nodePath: string): Loaded {
  const file = path.join(DATA, dataset, "nodes", nodePath.replaceAll("/", "-") + ".bin");
  const bytes = new Uint8Array(readFileSync(file));
  return { views: payloadViews(bytes), bytes: bytes.byteLength, ms: 0 };
}

type Loaded = { views: PayloadViews; bytes: number; ms: number };

type FakeNode = { parent: number; depth: number; bytes: number; points: number };

function fakePayload(node: number, points: number, bytes: number): Loaded {
  // Distinct id stream per node so a slot mix-up between nodes is visible.
  const ids = new Uint32Array(points);
  for (let i = 0; i < points; i++) ids[i] = node * 1000 + i;
  return {
    views: { header: { count: points }, ids } as unknown as PayloadViews,
    bytes,
    ms: 0,
  };
}

function makeCache(tree: FakeNode[], residentByteBudget: number, retryBackoffMs = 25) {
  const events: CacheEvent[] = [];
  const pending = new Map<number, { resolve: (v: Loaded) => void; reject: (e: unknown) => void }>();
  const fetchImpl: FetchFn = (node, signal) => new Promise((resolve, reject) => {
    pending.set(node, { resolve, reject });
    signal.addEventListener("abort", () => {
      const err = new Error("aborted");
      err.name = "AbortError";
      reject(err);
    }, { once: true });
  });
  const cache = new ResidencyCache({
    residentByteBudget,
    inflightByteBudget: 1e9,
    maxConcurrent: 16,
    retryBackoffMs,
    fetch: fetchImpl,
    onChange: (e) => events.push(e),
    hooks: {
      parent: (n) => tree[n].parent,
      depth: (n) => tree[n].depth,
    },
  });
  const spec = (node: number) => ({
    node,
    depth: tree[node].depth,
    priority: 1e6 - tree[node].depth * 1000,
    bytes: tree[node].bytes,
    points: tree[node].points,
  });
  return { cache, events, pending, spec };
}

async function flush() {
  for (let i = 0; i < 10; i++) await Promise.resolve();
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe("point-block recycling under budget pressure", () => {
  it("defers budget eviction of a retiring block until its cut anchor is resident", async () => {
    // chain 0(root) -> 1(A) -> 2(B). A fails and backs off; the budget shrinks
    // while B retires from the cut. Bytes: 60 + 100 = 160 resident, budget 150.
    const tree: FakeNode[] = [
      { parent: -1, depth: 0, bytes: 60, points: 6 },
      { parent: 0, depth: 1, bytes: 60, points: 6 },
      { parent: 1, depth: 2, bytes: 100, points: 10 },
    ];
    const { cache, pending, spec } = makeCache(tree, 1000, 30);

    cache.beginFrame();
    cache.reconcile(new Set([0, 1, 2]), 1, [spec(0), spec(1), spec(2)]);
    pending.get(0)!.resolve(fakePayload(0, 6, 60));
    pending.get(1)!.reject(new Error("injected failure"));
    pending.get(2)!.resolve(fakePayload(2, 10, 100));
    await flush();
    expect(cache.has(1)).toBe(false);
    expect(cache.snapshot().residentBytes).toBe(160);

    // Budget shrinks below the resident bytes; B retires from the cut while
    // its anchor A is still missing. Recycling B now would leave the region
    // at root-only coverage, so B must be kept despite the overrun.
    cache.setBudgets(150, 1e9);
    cache.beginFrame();
    cache.reconcile(new Set([0, 1]), 2, [spec(0), spec(1)]);
    expect(cache.has(2)).toBe(true);
    expect(cache.snapshot().evictionsDeferred).toBeGreaterThan(0);

    // Once A's backoff expires it is re-issued; its arrival makes B safe to
    // recycle and the byte budget is restored. Eviction is deferred, not
    // disabled: the block does not stay resident forever.
    await sleep(40);
    cache.beginFrame();
    cache.reconcile(new Set([0, 1]), 3, [spec(0), spec(1)]);
    expect(pending.has(1)).toBe(true);
    pending.get(1)!.resolve(fakePayload(1, 6, 60));
    await flush();
    expect(cache.has(2)).toBe(false);
    expect(cache.snapshot().residentBytes).toBeLessThanOrEqual(150);
  });

  it("does not let a late arrival push out a block whose anchor is missing", async () => {
    // 0 root; 1 (P) child of 0; 2 (Y) child of 1; 3 (Z) child of 0.
    // Y retires while P is backing off; Z's late arrival then triggers the
    // arrival-side budget trim, which must not recycle Y either.
    const tree: FakeNode[] = [
      { parent: -1, depth: 0, bytes: 60, points: 6 },
      { parent: 0, depth: 1, bytes: 60, points: 6 },
      { parent: 1, depth: 2, bytes: 80, points: 8 },
      { parent: 0, depth: 1, bytes: 60, points: 6 },
    ];
    const { cache, pending, spec } = makeCache(tree, 1000, 10_000);

    cache.beginFrame();
    cache.reconcile(new Set([0, 1, 2, 3]), 1, [spec(0), spec(1), spec(2), spec(3)]);
    pending.get(0)!.resolve(fakePayload(0, 6, 60));
    pending.get(1)!.reject(new Error("injected failure"));
    pending.get(2)!.resolve(fakePayload(2, 8, 80));
    await flush();
    expect(cache.snapshot().residentBytes).toBe(140);

    // Y retires from the cut; its anchor P is missing, so Y is kept.
    // Z is still in flight and the budget shrinks before Z lands.
    cache.setBudgets(100, 1e9);
    cache.beginFrame();
    cache.reconcile(new Set([0, 1, 3]), 2, [spec(0), spec(1), spec(3)]);
    expect(cache.has(2)).toBe(true);

    // Z arrives late. The arrival-side trim is now 200 over a budget of 100,
    // but Y is the only non-target candidate and its anchor is still missing.
    pending.get(3)!.resolve(fakePayload(3, 6, 60));
    await flush();
    expect(cache.has(2)).toBe(true);
    expect(cache.snapshot().evictionsDeferred).toBeGreaterThan(0);
    // The cache may exceed its byte budget rather than tear a hole.
    expect(cache.snapshot().residentBytes).toBe(200);
  });

  it("recycles deepest-first among equally stale retirees", async () => {
    // 0 root; 1 (A) child of 0; 2 (B) child of 1; 3 (C) child of 2.
    // B and C retire together while A backs off. When A arrives and the
    // budget frees exactly one block, the deeper one (C) goes first so the
    // coarser base stays available longer.
    const tree: FakeNode[] = [
      { parent: -1, depth: 0, bytes: 60, points: 6 },
      { parent: 0, depth: 1, bytes: 60, points: 6 },
      { parent: 1, depth: 2, bytes: 50, points: 5 },
      { parent: 2, depth: 3, bytes: 50, points: 5 },
    ];
    const { cache, pending, spec } = makeCache(tree, 1000, 30);

    cache.beginFrame();
    cache.reconcile(new Set([0, 1, 2, 3]), 1, [spec(0), spec(1), spec(2), spec(3)]);
    pending.get(0)!.resolve(fakePayload(0, 6, 60));
    pending.get(1)!.reject(new Error("injected failure"));
    pending.get(2)!.resolve(fakePayload(2, 5, 50));
    pending.get(3)!.resolve(fakePayload(3, 5, 50));
    await flush();

    // B and C retire while A is missing; both are blocked from recycling.
    cache.setBudgets(170, 1e9);
    cache.beginFrame();
    cache.reconcile(new Set([0, 1]), 2, [spec(0), spec(1)]);
    expect(cache.has(2)).toBe(true);
    expect(cache.has(3)).toBe(true);

    // A recovers; its arrival frees exactly one block (220 -> 170). The
    // deeper retiree C is recycled first.
    await sleep(40);
    cache.beginFrame();
    cache.reconcile(new Set([0, 1]), 3, [spec(0), spec(1)]);
    pending.get(1)!.resolve(fakePayload(1, 6, 60));
    await flush();
    expect(cache.has(3)).toBe(false);
    expect(cache.has(2)).toBe(true);
    expect(cache.snapshot().residentBytes).toBeLessThanOrEqual(170);
  });

  it("aborts still supersede recycling: a cancelled block is never stored", async () => {
    // Fast view switch: node 1 leaves the cut while in flight; its late
    // response must be dropped, not stored as a fresh resident.
    const tree: FakeNode[] = [
      { parent: -1, depth: 0, bytes: 60, points: 6 },
      { parent: 0, depth: 1, bytes: 60, points: 6 },
    ];
    const { cache, pending, spec } = makeCache(tree, 1000);

    cache.beginFrame();
    cache.reconcile(new Set([0, 1]), 1, [spec(0), spec(1)]);
    cache.beginFrame();
    cache.reconcile(new Set([0]), 2, [spec(0)]);
    expect(cache.isInflight(1)).toBe(false);
    // The aborted request's late resolution is dropped.
    pending.get(1)!.resolve(fakePayload(1, 6, 60));
    await flush();
    expect(cache.has(1)).toBe(false);
    expect(cache.snapshot().residentBytes).toBe(0);
  });
});

describe("recycling preserves point identity and accounting (scanned point cloud)", () => {
  const manifest = loadManifest("small");
  const h = buildHierarchy(manifest);

  it("re-fetches an identical payload after recycling; points accounting stays exact", async () => {
    // Two real payload nodes with disjoint id streams: a slot mix-up between
    // them would surface as an id mismatch.
    const nodes = h.nodes
      .map((n, i) => ({ n, i }))
      .filter(({ n }) => n.depth === 2 && n.hasPayload)
      .slice(0, 2)
      .map(({ i }) => i);
    expect(nodes.length).toBe(2);
    const [a, b] = nodes;

    const events: CacheEvent[] = [];
    const cache = new ResidencyCache({
      residentByteBudget: 1e9,
      inflightByteBudget: 1e9,
      maxConcurrent: 4,
      retryBackoffMs: 1,
      fetch: (node) => Promise.resolve(loadPayload("small", h.nodes[node].path)),
      onChange: (e) => events.push(e),
      hooks: { parent: (n) => h.parent[n], depth: (n) => h.nodes[n].depth },
    });
    const spec = (node: number) => ({
      node,
      depth: h.nodes[node].depth,
      priority: 1e6 - h.nodes[node].depth * 1000,
      bytes: h.nodes[node].bytes,
      points: h.nodes[node].pointCount,
    });

    cache.beginFrame();
    cache.reconcile(new Set([a, b]), 1, [spec(a), spec(b)]);
    await flush();
    const idsA = Array.from(cache.get(a)!.views.ids);
    const idsB = Array.from(cache.get(b)!.views.ids);
    expect(idsA.length).toBeGreaterThan(0);
    expect(idsB.length).toBeGreaterThan(0);
    expect(idsA).not.toEqual(idsB);
    const pointsBefore = cache.snapshot().residentPoints;
    expect(pointsBefore).toBe(h.nodes[a].pointCount + h.nodes[b].pointCount);

    // Recycle A out of the cache, then let the next reconcile re-fetch it.
    expect(cache.invalidate(a)).toBe(true);
    expect(cache.has(a)).toBe(false);
    cache.beginFrame();
    cache.reconcile(new Set([a, b]), 2, [spec(a), spec(b)]);
    await flush();

    // Identity: the re-fetched payload is bit-identical in id space, and it
    // is A's own stream, not B's.
    expect(Array.from(cache.get(a)!.views.ids)).toEqual(idsA);
    expect(Array.from(cache.get(b)!.views.ids)).toEqual(idsB);
    // Accounting: resident points match the manifest exactly after recycling.
    expect(cache.snapshot().residentPoints).toBe(pointsBefore);
    expect(cache.snapshot().residentBytes).toBe(h.nodes[a].bytes + h.nodes[b].bytes);
  });
});

describe("renderer slots", () => {
  const manifest = loadManifest("small");
  const h = buildHierarchy(manifest);

  it("frees the slot on remove and rebuilds it on re-add (no stale slot)", () => {
    const node = h.nodes.findIndex((n) => n.depth === 2 && n.hasPayload);
    const views = loadPayload("small", h.nodes[node].path).views;
    const renderer = new PointRenderer(h);

    renderer.addNode(node, views);
    expect(renderer.hasNode(node)).toBe(true);
    const drawn = renderer.getStats().drawnPoints;
    expect(drawn).toBe(h.nodes[node].pointCount);

    // A duplicate add keeps the existing slot; it must not double-upload.
    renderer.addNode(node, views);
    expect(renderer.getStats().geometriesCreated).toBe(1);

    // Recycling removes the slot entirely: nothing stale stays drawable.
    renderer.removeNode(node);
    expect(renderer.hasNode(node)).toBe(false);
    expect(renderer.getStats().drawnPoints).toBe(0);
    renderer.removeNode(node);
    expect(renderer.getStats().geometriesDisposed).toBe(1);

    // Re-adding after recycling builds a fresh slot with the same points.
    renderer.addNode(node, views);
    expect(renderer.getStats().geometriesCreated).toBe(2);
    expect(renderer.getStats().drawnPoints).toBe(drawn);
    renderer.dispose();
    expect(renderer.getStats().drawnPoints).toBe(0);
  });
});
