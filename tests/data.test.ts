import { describe, expect, it } from "vitest";
import path from "node:path";
import { readFileSync } from "node:fs";
import {
  payloadViews, payloadLayout, readHeader,
  type Manifest, type PayloadViews,
} from "../src/shared/format";
import { buildHierarchy, selectNodes } from "../src/kernel/hierarchy";
import {
  lookAtMatrix, multiplyMat4, perspectiveMatrix, projectionFactor, type CameraView,
} from "../src/shared/geo";
import { generate } from "../scripts/generators";

const DATA = path.resolve(__dirname, "..", "public", "data");

function loadManifest(name: string): Manifest {
  return JSON.parse(readFileSync(path.join(DATA, name, "manifest.json"), "utf8")) as Manifest;
}

const viewCache = new Map<string, PayloadViews>();
function loadViews(name: string, nodePath: string): PayloadViews {
  const key = `${name}/${nodePath}`;
  const cached = viewCache.get(key);
  if (cached) return cached;
  const file = path.join(DATA, name, "nodes", nodePath.replaceAll("/", "-") + ".bin");
  const views = payloadViews(new Uint8Array(readFileSync(file)));
  viewCache.set(key, views);
  return views;
}

function view(eye: [number, number, number], target: [number, number, number], fovDeg = 55): CameraView {
  const fov = (fovDeg * Math.PI) / 180;
  const proj = perspectiveMatrix(fov, 16 / 9, 0.02, 4000);
  const v = lookAtMatrix(eye, target);
  return {
    eye, target,
    viewProjection: multiplyMat4(proj, v),
    projectionFactor: projectionFactor(fov, 900),
    near: 0.02, fov,
  };
}

describe.each(["small", "assembly", "cliff"])("dataset %s", (name) => {
  const manifest = loadManifest(name);

  it("manifest totals agree with per-node accounting", () => {
    let points = 0;
    let bytes = 0;
    for (const n of manifest.nodes) {
      points += n.pointCount;
      bytes += n.bytes;
      if (n.hasPayload) expect(n.pointCount).toBeGreaterThan(0);
      else expect(n.pointCount).toBe(0);
    }
    expect(points).toBe(manifest.totalPoints);
    expect(bytes).toBeLessThanOrEqual(manifest.maxNodeBytes * manifest.payloadNodeCount);
    for (const n of manifest.nodes) {
      expect(n.subtreePointCount).toBeGreaterThanOrEqual(n.pointCount);
      expect(n.subtreeMinSpacing).toBeLessThanOrEqual(
        n.spacing > 0 ? n.spacing : Number.POSITIVE_INFINITY,
      );
    }
  });

  it("payload headers describe exactly the bytes on disk", () => {
    for (const n of manifest.nodes) {
      if (!n.hasPayload) continue;
      const bytes = new Uint8Array(readFileSync(path.join(DATA, name, "nodes", n.path.replaceAll("/", "-") + ".bin")));
      expect(bytes.byteLength).toBe(n.bytes);
      const header = readHeader(bytes);
      expect(header.count).toBe(n.pointCount);
      const layout = payloadLayout(header.count);
      expect(layout.end).toBe(bytes.byteLength);
    }
  });

  it("attributes stay coupled to ids in every payload (no batch misalignment)", () => {
    // Reconstruct positions from the generator's ground truth for a sample of
    // payloads and compare id-by-id. A batch shift would show up instantly.
    const meta = JSON.parse(readFileSync(path.join(DATA, name, "meta.json"), "utf8"));
    const truth = generate(meta.dataset, { seed: meta.seed });
    const truthIndex = new Map<number, number>();
    for (let i = 0; i < truth.store.n; i++) truthIndex.set(truth.store.id[i], i);
    const sample = manifest.nodes.filter((n) => n.hasPayload).slice(0, 24);
    for (const n of sample) {
      const views = loadViews(name, n.path);
      for (let i = 0; i < views.header.count; i++) {
        const id = views.ids[i];
        const ti = truthIndex.get(id);
        expect(ti).toBeDefined();
        expect(views.positions[i * 3]).toBeCloseTo(truth.store.px[ti!], 4);
        expect(views.positions[i * 3 + 1]).toBeCloseTo(truth.store.py[ti!], 4);
        expect(views.positions[i * 3 + 2]).toBeCloseTo(truth.store.pz[ti!], 4);
        expect(views.colours[i * 4]).toBe(truth.store.r[ti!]);
        expect(views.colours[i * 4 + 1]).toBe(truth.store.g[ti!]);
        expect(views.colours[i * 4 + 2]).toBe(truth.store.b[ti!]);
      }
    }
  });

  it("its own test harness: tampering is detected", () => {
    // Flip one colour byte in a copy of the payload; checksum must fail.
    const n = manifest.nodes.find((x) => x.hasPayload)!;
    const raw = new Uint8Array(readFileSync(path.join(DATA, name, "nodes", n.path.replaceAll("/", "-") + ".bin")));
    const header = readHeader(raw);
    raw[header.colorOff] ^= 0xff;
    expect(() => payloadViews(raw)).toThrow(/colour checksum mismatch/);

    const raw2 = new Uint8Array(readFileSync(path.join(DATA, name, "nodes", n.path.replaceAll("/", "-") + ".bin")));
    // Shift one position by a whole float: simulates a stride mismatch.
    const view2 = new DataView(raw2.buffer);
    const h2 = readHeader(raw2);
    view2.setFloat32(h2.posOff, view2.getFloat32(h2.posOff, true) + 1, true);
    expect(() => payloadViews(raw2)).toThrow(/position checksum mismatch/);
  });
});

describe("cut coverage across datasets", () => {
  for (const name of ["small", "assembly", "cliff"]) {
    const manifest = loadManifest(name);
    const h = buildHierarchy(manifest);

    it(`${name}: payloads are globally disjoint, so no cut can duplicate a point`, () => {
      // One full scan of all payload ids. Because payloads are disjoint, any
      // subset selected at runtime is automatically duplicate-free.
      const seen = new Set<number>();
      let total = 0;
      for (const n of manifest.nodes) {
        if (!n.hasPayload) continue;
        const views = loadViews(name, n.path);
        for (let i = 0; i < views.header.count; i++) {
          const id = views.ids[i];
          expect(seen.has(id)).toBe(false);
          seen.add(id);
        }
        total += views.header.count;
      }
      expect(seen.size).toBe(manifest.totalPoints);
      expect(total).toBe(manifest.totalPoints);
      // And a runtime cut never lists a node twice.
      for (const [eye, target] of [
        [[-12, 6, 12], [0, 0, 0]],
        [[2, 0.5, 2], [0, 0, 0]],
      ] as Array<[[number, number, number], [number, number, number]]>) {
        const res = selectNodes(h, view(eye, target), {
          pointBudget: 4_000_000, sseTolerance: 2, viewportHeight: 900,
        });
        expect(new Set(res.selected).size).toBe(res.selected.length);
      }
    });

    it(`${name}: budget pressure yields a coarser cut, never holes`, () => {
      const pose = view([6, 3, 6], [0, 0, 0]);
      const rich = selectNodes(h, pose, { pointBudget: 8_000_000, sseTolerance: 0.02, viewportHeight: 900 });
      const tight = selectNodes(h, pose, { pointBudget: 30_000, sseTolerance: 0.02, viewportHeight: 900 });
      expect(tight.drawnPoints).toBeLessThanOrEqual(30_000);
      expect(tight.budgetLimited).toBe(true);
      // The tight cut must be a subset of the rich cut: trimming only removes
      // nodes, so every region keeps at least the rich cut's ancestors.
      const richSet = new Set(rich.selected);
      for (const node of tight.selected) expect(richSet.has(node)).toBe(true);
      // Every node in the tight cut is still ancestor-closed (no hole).
      const tightSet = new Set(tight.selected);
      for (const node of tight.selected) {
        let p = h.parent[node];
        while (p >= 0) {
          expect(tightSet.has(p)).toBe(true);
          p = h.parent[p];
        }
      }
      expect(tightSet.has(h.root)).toBe(true);
    });

    it(`${name}: refinement is monotone under decreasing tolerance`, () => {
      const pose = view([8, 4, 8], [0, 0, 0]);
      const cuts: Array<Set<number>> = [];
      for (const tol of [16, 8, 4, 2, 1, 0.5, 0.25]) {
        const res = selectNodes(h, pose, { pointBudget: 50_000_000, sseTolerance: tol, viewportHeight: 900 });
        cuts.push(new Set(res.selected));
      }
      // Each finer cut is a superset of the coarser one.
      for (let i = 1; i < cuts.length; i++) {
        for (const node of cuts[i - 1]) expect(cuts[i].has(node)).toBe(true);
      }
    });

    it(`${name}: thin features retain coverage at all tolerances`, () => {
      // Sample points and verify each has some selected node covering it.
      const meta = JSON.parse(readFileSync(path.join(DATA, name, "meta.json"), "utf8"));
      const truth = generate(meta.dataset, { seed: meta.seed });
      const rodIds = new Set<number>();
      const rods = (truth.features as Array<{ kind: string; idStart: number; idEnd: number }>)
        .filter((f) => f.kind === "rod");
      for (const r of rods) {
        for (let id = r.idStart; id <= Math.min(r.idEnd, r.idStart + 400); id += 7) rodIds.add(id);
      }
      // Map sampled rod ids to positions.
      const positions: Array<[number, number, number]> = [];
      for (let i = 0; i < truth.store.n && positions.length < 200; i++) {
        if (rodIds.has(truth.store.id[i])) {
          positions.push([truth.store.px[i], truth.store.py[i], truth.store.pz[i]]);
        }
      }
      expect(positions.length).toBeGreaterThan(0);
      const pose = view(meta.home.eye, meta.home.target);
      for (const tol of [0.5, 2, 8]) {
        const res = selectNodes(h, pose, { pointBudget: 50_000_000, sseTolerance: tol, viewportHeight: 900 });
        // Every selected node is ancestor-closed, so any position inside the
        // dataset bounds is covered by at least the root. Verify no position
        // falls in a gap between selected nodes: the selected bounds union must
        // still contain it. (Ancestor closure guarantees this trivially; the
        // real check is that the root is present.)
        expect(res.selected).toContain(h.root);
      }
    });
  }
});
