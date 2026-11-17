#!/usr/bin/env node
/**
 * Budget-reclamation check (auxiliary to the scan viewer UI).
 *
 * The primary entry point for the scan viewer is the graphical viewport
 * (`npm run dev` / `npm start`); this script is a headless CI check that
 * reproduces the two conditions which used to make the draw list flash:
 *
 *   1. late replacement — a payload retired from the cut while its coarse
 *      replacement was still uploading, and the resident-byte budget was
 *      already exceeded by the retiring block;
 *   2. budget reclamation — stale blocks outside the target must still be
 *      reclaimed down to the byte cap; the fix must not "work" by pinning all
 *      blocks forever.
 *
 * For every step it records the point identities reachable through the draw
 * slots (union + FNV fingerprint + duplicate count) and the resident-byte peak,
 * so identity and memory behaviour are execution evidence rather than claims.
 *
 * Usage:
 *   npx tsx scripts/reclaim-check.ts [--dataset small] [--json out.json]
 * Exit code is non-zero if any invariant fails.
 */
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { payloadViews, type Manifest, type PayloadViews } from "../src/shared/format";
import { buildHierarchy, type Hierarchy } from "../src/kernel/hierarchy";
import { ResidencyCache, type FetchFn } from "../src/kernel/cache";

const here = path.dirname(fileURLToPath(import.meta.url));
const rootDir = path.resolve(here, "..");
const DATA = path.join(rootDir, "public", "data");

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 ? fallback : process.argv[i + 1] ?? fallback;
}
const dataset = arg("dataset", "small");
const jsonOut = arg("json", "");

const manifest = JSON.parse(
  readFileSync(path.join(DATA, dataset, "manifest.json"), "utf8"),
) as Manifest;
const h: Hierarchy = buildHierarchy(manifest);
const viewCache = new Map<number, PayloadViews>();
function viewsOf(node: number): PayloadViews {
  const cached = viewCache.get(node);
  if (cached) return cached;
  const file = path.join(DATA, dataset, "nodes", h.nodes[node].path.replaceAll("/", "-") + ".bin");
  const views = payloadViews(new Uint8Array(readFileSync(file)));
  viewCache.set(node, views);
  return views;
}

type Chain = { root: number; ancestor: number; deep: number };
function pickChain(h: Hierarchy): Chain {
  const root = h.root;
  const ancestor = h.nodes.findIndex((_, i) => h.parent[i] === root);
  let deep = -1;
  for (let i = 0; i < h.nodes.length; i++) {
    if (h.nodes[i].depth < 2) continue;
    let p = h.parent[i];
    while (p >= 0 && p !== ancestor) p = h.parent[p];
    if (p === ancestor && (deep < 0 || h.nodes[i].depth > h.nodes[deep].depth)) deep = i;
  }
  if (ancestor < 0 || deep < 0) throw new Error("could not pick root/ancestor/deep chain");
  return { root, ancestor, deep };
}
function pickUnrelated(h: Hierarchy, exclude: Set<number>): number {
  for (let i = 0; i < h.nodes.length; i++) {
    if (exclude.has(i)) continue;
    if (h.nodes[i].depth === 1 && h.nodes[i].hasPayload) return i;
  }
  throw new Error("no unrelated depth-1 node");
}

type Row = {
  scenario: string;
  step: string;
  drawnNodes: number;
  residentNodes: number;
  residentBytes: number;
  retainedBytes: number;
  evictions: number;
  uniqueIds: number;
  idFingerprint: string;
  peakResidentBytes: number;
  ok: boolean;
  note: string;
};

const rows: Row[] = [];
let failures = 0;

type Harness = {
  cache: ResidencyCache;
  pending: Map<number, (v: { views: PayloadViews; bytes: number; ms: number }) => void>;
  deliver: (node: number) => Promise<void>;
  peak: number;
};

function makeHarness(budget: number): Harness {
  const pending = new Map<number, (v: { views: PayloadViews; bytes: number; ms: number }) => void>();
  const fetchFn: FetchFn = (node) => new Promise((resolve) => { pending.set(node, resolve); });
  const cache = new ResidencyCache({
    residentByteBudget: budget,
    inflightByteBudget: Number.MAX_SAFE_INTEGER,
    maxConcurrent: 8,
    retryBackoffMs: 1,
    retentionMs: 10_000,
    fetch: fetchFn,
    onChange: () => {},
    hooks: { parent: (n) => h.parent[n], depth: (n) => h.nodes[n].depth },
  });
  const deliver = async (node: number) => {
    const resolve = pending.get(node);
    if (!resolve) throw new Error(`node ${node} was not requested`);
    pending.delete(node);
    resolve({ views: viewsOf(node), bytes: h.nodes[node].bytes, ms: 0 });
    for (let i = 0; i < 8; i++) await Promise.resolve();
  };
  return { cache, pending, deliver, peak: 0 };
}

type Identity = { total: number; unique: number; fingerprint: string; missingGenerations: number };

/** Union of payload ids behind the draw slots, with duplicates and generation gaps. */
function identityOf(cache: ResidencyCache, drawn: number[]): Identity {
  const ids: number[] = [];
  let missingGenerations = 0;
  for (const node of drawn) {
    const e = cache.get(node);
    if (!e || e.generation <= 0) { missingGenerations++; continue; }
    for (let i = 0; i < e.views.header.count; i++) ids.push(e.views.ids[i]);
  }
  let fingerprint = 0x811c9dc5;
  for (const id of ids) {
    for (let shift = 0; shift < 32; shift += 8) {
      fingerprint ^= (id >>> shift) & 0xff;
      fingerprint = Math.imul(fingerprint, 0x01000193) >>> 0;
    }
  }
  return {
    total: ids.length,
    unique: new Set(ids).size,
    fingerprint: fingerprint.toString(16).padStart(8, "0"),
    missingGenerations,
  };
}

function record(
  scenario: string, step: string, harness: Harness, drawn: number[], invariant: boolean, note: string,
): void {
  const { cache } = harness;
  const snap = cache.snapshot();
  harness.peak = Math.max(harness.peak, snap.residentBytes);
  const id = identityOf(cache, drawn);
  const duplicates = id.total !== id.unique;
  const ok = invariant && !duplicates && id.missingGenerations === 0;
  if (!ok) failures++;
  rows.push({
    scenario,
    step,
    drawnNodes: drawn.length,
    residentNodes: snap.residentNodes,
    residentBytes: snap.residentBytes,
    retainedBytes: snap.retainedBytes,
    evictions: snap.evictions,
    uniqueIds: id.unique,
    idFingerprint: id.fingerprint,
    peakResidentBytes: harness.peak,
    ok,
    note: duplicates ? `${note}; DUPLICATE IDS`
      : id.missingGenerations > 0 ? `${note}; DRAW ENTRY WITHOUT RESIDENCY`
      : note,
  });
}

const spec = (node: number) => ({
  node, depth: h.nodes[node].depth, priority: 0,
  bytes: h.nodes[node].bytes, points: h.nodes[node].pointCount,
});

/** Scenario 1: late replacement must be retained until it is resident. */
async function lateReplacement(chain: Chain): Promise<void> {
  const { root, ancestor, deep } = chain;
  // Budget exceeded by root+deep alone: this is what made the old budget pass
  // trim the retiring deep block the moment the cut shrank.
  const budget = h.nodes[root].bytes + h.nodes[deep].bytes - 1;
  const harness = makeHarness(budget);
  const { cache, deliver } = harness;
  const drawn: number[] = [];

  cache.reconcile(new Set([root, ancestor, deep]), 1, [spec(root), spec(ancestor), spec(deep)]);
  await deliver(root); drawn.push(root);
  await deliver(deep); drawn.push(deep);
  record("late-replacement", "deep delivered, ancestor in flight", harness, drawn,
    cache.has(deep), `budget ${budget}B < root+deep`);

  cache.reconcile(new Set([root, ancestor]), 2, []);
  record("late-replacement", "cut shrinks, replacement not resident", harness, drawn,
    cache.has(deep) && cache.snapshot().evictions === 0,
    cache.has(deep) ? "retained, no flicker" : "BUG: deep block reclaimed early");

  await deliver(ancestor);
  drawn.splice(0, drawn.length, root, ancestor);
  record("late-replacement", "replacement arrives, retiree released", harness, drawn,
    !cache.has(deep) && cache.has(ancestor),
    `evictions ${cache.snapshot().evictions}`);
}

/** Scenario 2: stale non-target blocks are still reclaimed to the byte cap. */
async function budgetReclaim(chain: Chain): Promise<void> {
  const { root, ancestor } = chain;
  const exclude = new Set<number>([root, ancestor]);
  for (let p = h.parent[ancestor]; p >= 0; p = h.parent[p]) exclude.add(p);
  const stale = pickUnrelated(h, exclude);
  // Over budget by one byte while both are target. The budget pass cannot trim
  // a target node, so this also proves the cap is not silently ignored.
  const budget = h.nodes[root].bytes + h.nodes[stale].bytes - 1;
  const harness = makeHarness(budget);
  const { cache, deliver } = harness;
  const drawn: number[] = [];

  cache.reconcile(new Set([root, stale]), 1, [spec(root), spec(stale)]);
  await deliver(root); await deliver(stale);
  drawn.push(root, stale);
  record("budget-reclaim", "target blocks loaded", harness, drawn,
    cache.has(root) && cache.has(stale), `budget ${budget}B; both are target`);

  // Target drops `stale`; root is its nearest target ancestor and is resident,
  // so the budget pass must reclaim it (the fix must not pin everything).
  cache.reconcile(new Set([root]), 2, []);
  drawn.splice(drawn.indexOf(stale), 1);
  const snap = cache.snapshot();
  record("budget-reclaim", "stale block reclaimed to cap", harness, drawn,
    !cache.has(stale) && snap.residentBytes <= budget && cache.has(root),
    `resident ${snap.residentBytes}B <= cap ${budget}B`);
}

async function main(): Promise<number> {
  const chain = pickChain(h);
  await lateReplacement(chain);
  await budgetReclaim(chain);

  const pad = (s: string, n: number) => (s.length >= n ? s : s + " ".repeat(n - s.length));
  console.log(`\nreclamation check — dataset ${dataset}`);
  let current = "";
  for (const r of rows) {
    if (r.scenario !== current) {
      current = r.scenario;
      console.log(`\n[${r.scenario}]`);
      console.log(
        pad("step", 38) + pad("drawn", 7) + pad("resident", 17) + pad("retained", 9)
        + pad("evict", 7) + pad("ids", 8) + pad("idHash", 10) + pad("peak", 11) + "status",
      );
    }
    console.log(
      pad(r.step, 38)
      + pad(String(r.drawnNodes), 7)
      + pad(`${r.residentNodes}n/${r.residentBytes}B`, 16)
      + pad(String(r.retainedBytes), 8)
      + pad(String(r.evictions), 7)
      + pad(String(r.uniqueIds), 8)
      + pad(r.idFingerprint, 10)
      + pad(`${r.peakResidentBytes}B`, 11)
      + (r.ok ? "ok" : "FAIL") + "  " + r.note,
    );
  }

  const report = {
    dataset,
    budgetBytes: {
      "late-replacement": h.nodes[chain.root].bytes + h.nodes[chain.deep].bytes - 1,
    },
    rows,
    ok: failures === 0,
  };
  if (jsonOut) writeFileSync(path.resolve(jsonOut), JSON.stringify(report, null, 2));
  console.log(`\n${failures === 0 ? "PASS" : "FAIL"}: ${failures} invariant(s) violated`);
  return failures === 0 ? 0 : 1;
}

main().then((code) => process.exit(code)).catch((err) => {
  console.error(err);
  process.exit(1);
});
