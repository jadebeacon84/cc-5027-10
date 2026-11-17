/**
 * Deterministic synthetic scan generators.
 *
 * These exist to exercise the streaming kernel with known ground truth:
 * analytic surfaces (planes/spheres/cylinders/rods) for attribute checks, a
 * density cliff for LOD behaviour, and thin rods/cables that disappear if
 * hierarchy bounds or sampling are wrong. Every point gets a stable global id
 * before any sampling; the octree later treats it as identity.
 */
import type {
  AABB3, AnalyticSurface, FeatureRange, ScanStation,
} from "../src/shared/format";
import { stableRank } from "../src/shared/format";

export type GenerateOptions = {
  seed: number;
  /** Multiply every surface's nominal density. */
  density?: number;
};

export type GenerateFn = (opts: GenerateOptions) => GeneratedScan;

export type GeneratedScan = {
  store: PointStore;
  bounds: AABB3;
  stations: ScanStation[];
  surfaces: AnalyticSurface[];
  features: FeatureRange[];
  seed: number;
  generator: string;
  notes: string;
  home: { eye: [number, number, number]; target: [number, number, number]; fov: number };
  defaultBudget: number;
};

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function clamp255(v: number) { return v < 0 ? 0 : v > 255 ? 255 : v | 0; }
function norm3(x: number, y: number, z: number): [number, number, number] {
  const l = Math.hypot(x, y, z) || 1;
  return [x / l, y / l, z / l];
}

/** Struct-of-arrays point store. Grows by doubling; ids are 1-based. */
export class PointStore {
  n = 0;
  capacity: number;
  px: Float32Array;
  py: Float32Array;
  pz: Float32Array;
  nx: Float32Array;
  ny: Float32Array;
  nz: Float32Array;
  r: Uint8Array;
  g: Uint8Array;
  b: Uint8Array;
  intensity: Float32Array;
  distance: Float32Array;
  id: Uint32Array;
  scan: Uint16Array;
  nextId = 1;

  constructor(capacity = 1 << 18) {
    this.capacity = capacity;
    this.px = new Float32Array(capacity);
    this.py = new Float32Array(capacity);
    this.pz = new Float32Array(capacity);
    this.nx = new Float32Array(capacity);
    this.ny = new Float32Array(capacity);
    this.nz = new Float32Array(capacity);
    this.r = new Uint8Array(capacity);
    this.g = new Uint8Array(capacity);
    this.b = new Uint8Array(capacity);
    this.intensity = new Float32Array(capacity);
    this.distance = new Float32Array(capacity);
    this.id = new Uint32Array(capacity);
    this.scan = new Uint16Array(capacity);
  }

  private grow() {
    const cap = this.capacity * 2;
    const f = (src: Float32Array) => { const d = new Float32Array(cap); d.set(src); return d; };
    const u = (src: Uint8Array) => { const d = new Uint8Array(cap); d.set(src); return d; };
    this.px = f(this.px); this.py = f(this.py); this.pz = f(this.pz);
    this.nx = f(this.nx); this.ny = f(this.ny); this.nz = f(this.nz);
    this.r = u(this.r); this.g = u(this.g); this.b = u(this.b);
    this.intensity = f(this.intensity); this.distance = f(this.distance);
    const id = new Uint32Array(cap); id.set(this.id); this.id = id;
    const scan = new Uint16Array(cap); scan.set(this.scan); this.scan = scan;
    this.capacity = cap;
  }

  push(
    x: number, y: number, z: number,
    nx: number, ny: number, nz: number,
    r: number, g: number, b: number,
    intensity: number,
    stations: readonly (readonly [number, number, number])[],
  ): number {
    if (this.n >= this.capacity) this.grow();
    const i = this.n++;
    this.px[i] = x; this.py[i] = y; this.pz[i] = z;
    this.nx[i] = nx; this.ny[i] = ny; this.nz[i] = nz;
    this.r[i] = clamp255(r); this.g[i] = clamp255(g); this.b[i] = clamp255(b);
    this.intensity[i] = intensity;
    let best = 0, bestD = Infinity;
    for (let s = 0; s < stations.length; s++) {
      const st = stations[s];
      const d = (x - st[0]) ** 2 + (y - st[1]) ** 2 + (z - st[2]) ** 2;
      if (d < bestD) { bestD = d; best = s; }
    }
    this.distance[i] = Math.sqrt(bestD);
    this.id[i] = this.nextId++;
    this.scan[i] = best;
    return i;
  }
}

type Station = [number, number, number];

class Builder {
  store: PointStore;
  rng: () => number;
  stations: Station[] = [];
  surfaces: AnalyticSurface[] = [];
  features: FeatureRange[] = [];
  private idStart = 1;
  private featureName = "";
  private featureKind: FeatureRange["kind"] = "surface";
  private featureNotes: string | undefined;

  constructor(seed: number, estimate: number) {
    this.rng = mulberry32(seed);
    this.store = new PointStore(Math.max(1 << 16, estimate));
  }

  beginFeature(name: string, kind: FeatureRange["kind"], notes?: string) {
    this.idStart = this.store.nextId;
    this.featureName = name;
    this.featureKind = kind;
    this.featureNotes = notes;
  }

  endFeature() {
    const idEnd = this.store.nextId - 1;
    this.features.push({
      name: this.featureName,
      kind: this.featureKind,
      pointCount: idEnd - this.idStart + 1,
      idStart: this.idStart,
      idEnd,
      notes: this.featureNotes,
    });
  }

  add(
    x: number, y: number, z: number,
    nx: number, ny: number, nz: number,
    r: number, g: number, b: number,
    intensity: number,
    jitter = 0,
  ) {
    const rng = this.rng;
    let jx = 0, jy = 0, jz = 0;
    if (jitter > 0) {
      jx = randn(rng) * jitter;
      jy = randn(rng) * jitter;
      jz = randn(rng) * jitter;
    }
    let ux = nx, uy = ny, uz = nz;
    if (jitter > 0) {
      ux += randn(rng) * 0.015; uy += randn(rng) * 0.015; uz += randn(rng) * 0.015;
      [ux, uy, uz] = norm3(ux, uy, uz);
    }
    this.store.push(x + jx, y + jy, z + jz, ux, uy, uz, r, g, b, intensity, this.stations);
  }

  /**
   * Jittered 2D grid on a plane. `u`/`v` are unit axes, origin is the centre.
   * Stratified jitter keeps spacing even while avoiding visible grid patterns.
   */
  plane(
    origin: [number, number, number],
    u: [number, number, number],
    v: [number, number, number],
    halfU: number, halfV: number,
    spacing: number,
    normal: [number, number, number],
    colour: (su: number, sv: number) => [number, number, number],
    intensity: (su: number, sv: number) => number = () => 0.5,
    roughness = 0.0015,
  ) {
    const nu = Math.max(1, Math.round((halfU * 2) / spacing));
    const nv = Math.max(1, Math.round((halfV * 2) / spacing));
    const du = (halfU * 2) / nu;
    const dv = (halfV * 2) / nv;
    const rng = this.rng;
    for (let iv = 0; iv < nv; iv++) {
      for (let iu = 0; iu < nu; iu++) {
        const ju = (rng() - 0.5) * du * 0.9;
        const jv = (rng() - 0.5) * dv * 0.9;
        const su = -halfU + (iu + 0.5) * du + ju;
        const sv = -halfV + (iv + 0.5) * dv + jv;
        const x = origin[0] + u[0] * su + v[0] * sv;
        const y = origin[1] + u[1] * su + v[1] * sv;
        const z = origin[2] + u[2] * su + v[2] * sv;
        const n = roughness > 0
          ? norm3(normal[0] + randn(rng) * roughness, normal[1] + randn(rng) * roughness, normal[2] + randn(rng) * roughness)
          : normal;
        const [cr, cg, cb] = colour(su, sv);
        this.add(x, y, z, n[0], n[1], n[2], cr, cg, cb, intensity(su, sv), roughness * 0.4);
      }
    }
  }

  sphere(
    center: [number, number, number],
    radius: number,
    spacing: number,
    colour: (y: number, u: number) => [number, number, number],
    intensity = 0.6,
  ) {
    const count = Math.max(12, Math.round((4 * Math.PI * radius * radius) / (spacing * spacing)));
    const golden = Math.PI * (3 - Math.sqrt(5));
    const rng = this.rng;
    for (let i = 0; i < count; i++) {
      const y = 1 - (i / Math.max(1, count - 1)) * 2;
      const rad = Math.sqrt(Math.max(0, 1 - y * y));
      const theta = golden * i;
      const n = norm3(
        Math.cos(theta) * rad + randn(rng) * 0.004,
        y + randn(rng) * 0.004,
        Math.sin(theta) * rad + randn(rng) * 0.004,
      );
      const x = center[0] + n[0] * radius;
      const yy = center[1] + n[1] * radius;
      const z = center[2] + n[2] * radius;
      const [cr, cg, cb] = colour(yy, i / count);
      this.add(x, yy, z, n[0], n[1], n[2], cr, cg, cb, intensity, spacing * 0.05);
    }
  }

  /** Vertical cylinder: side wall plus optional top cap. */
  cylinder(
    center: [number, number, number],
    radius: number,
    height: number,
    spacing: number,
    colour: (y: number, a: number) => [number, number, number],
    cap = true,
    intensity = 0.55,
  ) {
    const na = Math.max(10, Math.round((2 * Math.PI * radius) / spacing));
    const nh = Math.max(2, Math.round(height / spacing));
    for (let ih = 0; ih < nh; ih++) {
      const yy = center[1] - height / 2 + (ih + 0.5) * (height / nh);
      for (let ia = 0; ia < na; ia++) {
        const a = (ia / na) * Math.PI * 2 + (ih % 2) * (Math.PI / na);
        const nx = Math.cos(a), nz = Math.sin(a);
        const [cr, cg, cb] = colour(yy, ia / na);
        this.add(center[0] + nx * radius, yy, center[2] + nz * radius, nx, 0, nz, cr, cg, cb, intensity, spacing * 0.04);
      }
    }
    if (cap) {
      const nr = Math.max(2, Math.round(radius / spacing));
      for (let ir = 0; ir < nr; ir++) {
        const rr = (ir + 0.5) * (radius / nr);
        const na2 = Math.max(8, Math.round((2 * Math.PI * rr) / spacing));
        for (let ia = 0; ia < na2; ia++) {
          const a = (ia / na2) * Math.PI * 2;
          const [cr, cg, cb] = colour(center[1] + height / 2, ia / na2);
          this.add(
            center[0] + Math.cos(a) * rr, center[1] + height / 2, center[2] + Math.sin(a) * rr,
            0, 1, 0, cr * 0.92, cg * 0.92, cb * 0.92, intensity, spacing * 0.04,
          );
        }
      }
    }
  }

  /** Thin oriented cylinder — the LOD stress feature. */
  rod(
    a: [number, number, number],
    b: [number, number, number],
    radius: number,
    spacing: number,
    colour: [number, number, number],
    intensity = 0.35,
  ) {
    const ax = b[0] - a[0], ay = b[1] - a[1], az = b[2] - a[2];
    const len = Math.hypot(ax, ay, az);
    if (len < 1e-6) return;
    const dir: [number, number, number] = [ax / len, ay / len, az / len];
    const ref: [number, number, number] = Math.abs(dir[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const u = norm3(
      ref[1] * dir[2] - ref[2] * dir[1],
      ref[2] * dir[0] - ref[0] * dir[2],
      ref[0] * dir[1] - ref[1] * dir[0],
    );
    const v: [number, number, number] = [
      dir[1] * u[2] - dir[2] * u[1],
      dir[2] * u[0] - dir[0] * u[2],
      dir[0] * u[1] - dir[1] * u[0],
    ];
    const na = Math.max(6, Math.round((2 * Math.PI * radius) / spacing));
    const nl = Math.max(2, Math.round(len / spacing));
    for (let il = 0; il < nl; il++) {
      const t = (il + 0.5) / nl;
      const cx = a[0] + ax * t, cy = a[1] + ay * t, cz = a[2] + az * t;
      for (let ia = 0; ia < na; ia++) {
        const ang = (ia / na) * Math.PI * 2 + (il % 3) * 0.4;
        const ca = Math.cos(ang), sa = Math.sin(ang);
        const nx = u[0] * ca + v[0] * sa;
        const ny = u[1] * ca + v[1] * sa;
        const nz = u[2] * ca + v[2] * sa;
        this.add(cx + nx * radius, cy + ny * radius, cz + nz * radius, nx, ny, nz, colour[0], colour[1], colour[2], intensity, spacing * 0.02);
      }
    }
    // Collars give thin rods a coarse-LOD footprint and catch box-culling bugs.
    for (const t of [0, 0.5, 1]) {
      const cx = a[0] + ax * t, cy = a[1] + ay * t, cz = a[2] + az * t;
      for (let k = 0; k < 24; k++) {
        const ang = (k / 24) * Math.PI * 2;
        const rr = radius * 2.6;
        this.add(
          cx + (u[0] * Math.cos(ang) + v[0] * Math.sin(ang)) * rr,
          cy + (u[1] * Math.cos(ang) + v[1] * Math.sin(ang)) * rr,
          cz + (u[2] * Math.cos(ang) + v[2] * Math.sin(ang)) * rr,
          0, 0, 0, 46, 46, 50, 0.2, 0,
        );
      }
    }
  }

  /** Anisotropic point blobs, e.g. foliage or debris. */
  ellipsoidCluster(
    center: [number, number, number],
    radii: [number, number, number],
    count: number,
    colour: [number, number, number],
  ) {
    const rng = this.rng;
    for (let i = 0; i < count; i++) {
      const x = randn(rng) * radii[0];
      const y = randn(rng) * radii[1];
      const z = randn(rng) * radii[2];
      const n = norm3(x / radii[0], y / radii[1], z / radii[2]);
      this.add(
        center[0] + x, center[1] + y, center[2] + z,
        n[0], n[1], n[2],
        colour[0] + randn(rng) * 18, colour[1] + randn(rng) * 18, colour[2] + randn(rng) * 14,
        0.25 + rng() * 0.3,
        0.006,
      );
    }
  }
}

function stationsAround(center: [number, number, number], radius: number, count: number): Station[] {
  const out: Station[] = [];
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    out.push([center[0] + Math.cos(a) * radius, center[1] + 1.6, center[2] + Math.sin(a) * radius]);
  }
  return out;
}

function makeStations(list: Station[], store: PointStore): ScanStation[] {
  const counts = new Array(list.length).fill(0);
  for (let i = 0; i < store.n; i++) counts[store.scan[i]]++;
  return list.map((p, i) => ({
    id: i,
    position: p,
    yaw: Math.atan2(-p[2], -p[0]),
    pitch: -0.1,
    fov: Math.PI / 2.4,
    pointCount: counts[i] ?? 0,
  }));
}

function woodColour(x: number, z: number): [number, number, number] {
  const grain = Math.sin(x * 2.3 + Math.sin(z * 0.7) * 1.4) * 0.5 + 0.5;
  const v = 0.55 + 0.35 * grain;
  return [175 * v + 20, 128 * v + 12, 88 * v + 8];
}

function concreteColour(x: number, z: number): [number, number, number] {
  const n = Math.sin(x * 9.1) * Math.sin(z * 7.3) * 43758.5453;
  const v = 0.62 + Math.abs(n % 1) * 0.18;
  return [150 * v, 152 * v, 148 * v];
}

/** Dense analytical assembly: walls, floor, machinery, rods, density cliff. */
export const generateAssembly: GenerateFn = (opts) => {
  const seed = opts.seed >>> 0;
  const b = new Builder(seed, 2_000_000);
  b.stations = stationsAround([0, 0, 0], 6.5, 4);
  const density = opts.density ?? 1;
  const sp = (s: number) => s / Math.sqrt(density);

  b.beginFeature("floor-fine", "surface", "1.5 cm survey patch, x < -1");
  b.plane([-5.02, -2, 0], [1, 0, 0], [0, 0, 1], 4, 9, sp(0.015), [0, 1, 0],
    (su, sv) => concreteColour(su - 5, sv), (su, sv) => 0.55 + 0.2 * Math.sin(su * 1.7 + sv));
  b.endFeature();

  b.beginFeature("floor-coarse", "surface", "9 cm survey patch, x > -1");
  b.plane([4.04, -2, 0], [1, 0, 0], [0, 0, 1], 5, 9, sp(0.09), [0, 1, 0],
    (su, sv) => concreteColour(su + 4, sv), () => 0.62);
  b.endFeature();

  b.beginFeature("wall-back", "surface", "back wall, 2 cm");
  b.plane([0, 2, -6.4], [1, 0, 0], [0, 1, 0], 9, 4, sp(0.02), [0, 0, 1],
    (su, sv) => [122 + su * 3, 118 + sv * 4, 112], (_su, sv) => 0.4 + 0.2 * Math.cos(sv * 2.1));
  b.endFeature();

  b.beginFeature("wall-side", "surface", "side wall, 3 cm");
  b.plane([-9, 2, 0], [0, 0, 1], [0, 1, 0], 6.4, 4, sp(0.03), [1, 0, 0],
    (_su, sv) => [108, 116 + sv * 3, 124], () => 0.42);
  b.endFeature();

  b.beginFeature("crate-stack", "surface", "wooden crates at 2 cm");
  for (let k = 0; k < 3; k++) {
    const cx = 3.2 + (k % 2) * 1.3;
    const cy = -2 + 0.45 + k * 0.9;
    const cz = -1.5 + k * 0.8;
    b.plane([cx, cy + 0.45, cz], [1, 0, 0], [0, 0, 1], 0.45, 0.45, sp(0.02), [0, 1, 0], woodColour, () => 0.5);
    b.plane([cx + 0.45, cy, cz], [0, 0, 1], [0, 1, 0], 0.45, 0.45, sp(0.02), [1, 0, 0], woodColour, () => 0.5);
    b.plane([cx, cy, cz + 0.45], [1, 0, 0], [0, 1, 0], 0.45, 0.45, sp(0.02), [0, 0, 1], woodColour, () => 0.5);
  }
  b.endFeature();

  b.beginFeature("sphere-tank", "surface", "spherical pressure vessel, 1.4 cm");
  b.sphere([-3.4, 0.9, -2.2], 1.5, sp(0.014), (y, u) => {
    const t = Math.min(1, Math.max(0, (y + 0.6) / 3.0));
    return [210 - t * 90, 190 - t * 40, 150 + t * 60 + u * 10];
  }, 0.68);
  b.endFeature();

  b.beginFeature("tower-cylinder", "surface", "vertical cylinder, 2 cm");
  b.cylinder([-1.4, 0.5, 1.7], 0.85, 5.0, sp(0.02),
    (_y, a) => [96 + 30 * Math.sin(a * 3), 104 + 26 * Math.cos(a * 2), 118], true, 0.6);
  b.endFeature();

  b.beginFeature("rod-cluster", "rod", "eight thin rods, 1.2 cm sampling");
  const rods: [[number, number, number], [number, number, number], number][] = [
    [[5.6, -2, -3.4], [5.6, 2.6, -3.4], 0.028],
    [[6.3, -2, -2.7], [6.3, 2.2, -2.7], 0.022],
    [[5.95, -2, 2.4], [5.95, 2.8, 2.4], 0.032],
    [[6.7, -2, 1.9], [6.7, 1.6, 1.9], 0.018],
    [[-7.4, -2, -4.9], [-7.4, 2.4, -4.9], 0.026],
    [[-6.8, -2, 4.6], [-6.8, 2.0, 4.6], 0.021],
    [[-0.2, -1.2, -4.8], [3.4, 1.6, -3.9], 0.03],
    [[2.4, -1.8, 4.4], [5.2, 0.8, 3.1], 0.024],
  ];
  for (const [a, bb, r] of rods) b.rod(a, bb, r, sp(0.012), [176, 44, 38], 0.3);
  b.endFeature();

  b.beginFeature("cable", "rod", "two long thin cables");
  b.rod([-8.8, 3.9, -6.2], [8.8, 4.3, -6.2], 0.014, sp(0.01), [52, 56, 62], 0.25);
  b.rod([-8.8, 4.1, -6.4], [8.8, 2.6, -6.4], 0.011, sp(0.01), [58, 60, 66], 0.25);
  b.endFeature();

  b.beginFeature("foliage", "noise", "equatorial vegetation clusters");
  b.ellipsoidCluster([-7.2, 1.0, 2.0], [1.2, 0.8, 1.2], 60000, [58, 108, 46]);
  b.ellipsoidCluster([-6.6, 2.2, -4.0], [1.0, 0.7, 1.0], 38000, [64, 118, 52]);
  b.ellipsoidCluster([7.6, 0.4, -5.0], [1.4, 0.9, 1.1], 48000, [70, 100, 44]);
  b.endFeature();

  b.beginFeature("scan-fragment", "density", "dense angular fragment, 0.7 cm");
  b.plane([0.4, -1.98, 3.6], [1, 0, 0], [0, 0, 1], 0.55, 0.55, sp(0.007), [0, 1, 0],
    (su, sv) => [96 + su * 30, 90 + sv * 40, 86], () => 0.75, 0.0008);
  b.endFeature();

  const bounds: AABB3 = { min: [-9.2, -2.2, -6.6], max: [9.2, 4.6, 9.2] };
  b.surfaces.push(
    { kind: "plane", name: "floor", origin: [-4, -2, 0], normal: [0, 1, 0], half: [4.6, 9], spacing: 0.015 },
    { kind: "plane", name: "wall-back", origin: [0, 2, -6.4], normal: [0, 0, 1], half: [9, 4], spacing: 0.02 },
    { kind: "sphere", name: "sphere-tank", center: [-3.4, 0.9, -2.2], radius: 1.5, spacing: 0.014 },
    { kind: "cylinder", name: "tower", center: [-1.4, 0.5, 1.7], radius: 0.85, height: 5, spacing: 0.02 },
    { kind: "rod", name: "rod-a", a: [5.6, -2, -3.4], b: [5.6, 2.6, -3.4], radius: 0.028, spacing: 0.012 },
    { kind: "rod", name: "rod-b", a: [5.95, -2, 2.4], b: [5.95, 2.8, 2.4], radius: 0.032, spacing: 0.012 },
    { kind: "rod", name: "cable", a: [-8.8, 3.9, -6.2], b: [8.8, 4.3, -6.2], radius: 0.014, spacing: 0.01 },
  );

  return finish(b, "assembly", bounds, seed, [13.5, 6.5, 14.5], [0, 0.2, 0]);
};

/** Terrain with a density cliff, terrace, suspended lattice and thin cables. */
export const generateCliff: GenerateFn = (opts) => {
  const seed = opts.seed >>> 0;
  const b = new Builder(seed, 3_200_000);
  b.stations = [
    [14, 3.2, 1.5],
    [11, 5.6, 1.4],
    [11.5, 7.8, 1.5],
    [9, 3.0, 1.6],
  ];
  const density = opts.density ?? 1;
  const sp = (s: number) => s / Math.sqrt(density);
  const rng = b.rng;
  const ridge = (x: number, z: number) => {
    const base = Math.sin(x * 0.28) * 0.8 + Math.cos(z * 0.31) * 0.55;
    const terrace = x > 12 ? 1.2 : 0;
    return base + terrace;
  };
  const groundNormal = (x: number, z: number): [number, number, number] =>
    norm3(-0.28 * Math.cos(x * 0.28) * 0.8, 1, 0.31 * Math.sin(z * 0.31) * 0.55);

  b.beginFeature("ground-fine", "terrain", "2 cm ground, x < 12");
  for (let z = -10; z <= 10.0001; z += sp(0.02)) {
    for (let x = -10; x <= 12.0001; x += sp(0.02)) {
      if (x > 12) continue;
      const jx = (rng() - 0.5) * sp(0.018);
      const jz = (rng() - 0.5) * sp(0.018);
      const y = ridge(x, z);
      const n = groundNormal(x, z);
      const shade = 0.55 + 0.3 * Math.sin(x * 0.9 + z * 0.7);
      b.add(x + jx, y + 0.004, z + jz, n[0], n[1], n[2],
        96 * shade + 30, 118 * shade + 24, 70 * shade + 16, 0.55, 0.0025);
    }
  }
  b.endFeature();

  b.beginFeature("ground-coarse", "terrain", "16 cm ground, x > 12");
  for (let z = -10; z <= 10.0001; z += sp(0.16)) {
    for (let x = 12; x <= 22.0001; x += sp(0.16)) {
      const jx = (rng() - 0.5) * sp(0.12);
      const jz = (rng() - 0.5) * sp(0.12);
      const y = ridge(x, z);
      const n = groundNormal(x, z);
      b.add(x + jx, y + 0.004, z + jz, n[0], n[1], n[2], 132, 124, 84, 0.5, 0.002);
    }
  }
  b.endFeature();

  b.beginFeature("cliff-face", "edge", "vertical 6 m cliff at x = 12");
  for (let y = -5.5; y <= ridge(12, 0); y += sp(0.018)) {
    for (let z = -10; z <= 10.0001; z += sp(0.018)) {
      const n = norm3(-1, 0.15 * Math.sin(y * 2.2), 0.1 * Math.sin(z * 1.3));
      const shade = 0.45 + 0.25 * Math.sin(y * 3.1 + z * 0.9);
      b.add(12 + Math.sin(y * 4.4) * 0.04, y, z, n[0], n[1], n[2],
        118 * shade + 36, 106 * shade + 30, 88 * shade + 24, 0.5, 0.0015);
    }
  }
  b.endFeature();

  b.beginFeature("terrace", "surface", "raised terrace 3 cm");
  b.plane([17, ridge(17, 0) + 0.06, 0], [1, 0, 0], [0, 0, 1], 5, 10, sp(0.03), [0, 1, 0],
    (su, sv) => [124 + su * 2, 122 + sv * 2, 108], (_su, sv) => 0.5 + 0.15 * Math.sin(sv));
  b.endFeature();

  b.beginFeature("lattice-spheres", "surface", "suspended lattice of spheres");
  for (let ix = 0; ix < 7; ix++) {
    for (let iz = 0; iz < 7; iz++) {
      const cx = -8 + ix * 2.6;
      const cz = -7.8 + iz * 2.6;
      const cy = 3.2 + Math.sin(ix * 1.1 + iz * 0.7) * 0.9;
      b.sphere([cx, cy, cz], 0.32 + 0.05 * Math.sin(ix + iz), sp(0.011),
        (_y, u) => [190 + 40 * u, 160 + 40 * (1 - u), 130 + 30 * u], 0.72);
    }
  }
  b.endFeature();

  b.beginFeature("lattice-rods", "rod", "lattice struts and cables");
  for (let ix = 0; ix < 7; ix++) {
    for (let iz = 0; iz < 7; iz++) {
      const cx = -8 + ix * 2.6;
      const cz = -7.8 + iz * 2.6;
      const cy = 3.2 + Math.sin(ix * 1.1 + iz * 0.7) * 0.9;
      if (iz + 1 < 7) {
        const nz = -7.8 + (iz + 1) * 2.6;
        const ny = 3.2 + Math.sin(ix * 1.1 + (iz + 1) * 0.7) * 0.9;
        b.rod([cx, cy, cz], [cx, ny, nz], 0.018, sp(0.009), [150, 120, 80], 0.3);
      }
      if (ix + 1 < 7) {
        const nx = -8 + (ix + 1) * 2.6;
        const ny = 3.2 + Math.sin((ix + 1) * 1.1 + iz * 0.7) * 0.9;
        b.rod([cx, cy, cz], [nx, ny, cz], 0.014, sp(0.009), [160, 130, 86], 0.3);
      }
      if ((ix + iz) % 3 === 0) {
        b.rod([cx, cy, cz], [cx, -0.5, cz], 0.009, sp(0.009), [90, 92, 98], 0.2);
      }
    }
  }
  b.endFeature();

  b.beginFeature("dense-canopy", "density", "bush canopies at 0.8 cm equivalent");
  b.ellipsoidCluster([4.5, 1.4, -3.2], [1.8, 1.1, 1.6], 140000, [52, 104, 44]);
  b.ellipsoidCluster([-2.0, 1.0, 6.0], [2.2, 1.2, 1.8], 120000, [60, 112, 48]);
  b.endFeature();

  b.beginFeature("scan-shadow", "density", "occluded wedge left by a single station");
  for (let k = 0; k < 26000; k++) {
    const a = Math.PI * 0.15 + rng() * 0.3;
    const r = 2 + rng() * 7;
    const x = 3 + Math.cos(a) * r;
    const z = 2 + Math.sin(a) * r;
    b.add(x, ridge(x, z) + 0.02, z, 0, 1, 0, 90, 112, 74, 0.4, 0.01);
  }
  b.endFeature();

  const bounds: AABB3 = { min: [-10.5, -6.0, -10.5], max: [22.5, 8.0, 10.5] };
  b.surfaces.push(
    { kind: "plane", name: "terrace", origin: [17, ridge(17, 0) + 0.06, 0], normal: [0, 1, 0], half: [5, 10], spacing: 0.03 },
    { kind: "sphere", name: "lattice-sphere-0", center: [-8, 3.2, -7.8], radius: 0.32, spacing: 0.011 },
    { kind: "rod", name: "lattice-cable", a: [-8, 3.2, -7.8], b: [-8, -0.5, -7.8], radius: 0.009, spacing: 0.009 },
    { kind: "lattice", name: "lattice", count: 49, bounds: { min: [-8.5, 2, -8.3], max: [8, 4.5, 8] }, spacing: 0.011 },
  );

  return finish(b, "cliff", bounds, seed, [25.5, 12.5, 21], [6, 0.5, 0]);
};

/** Small but feature-complete dataset for CI / fast browser tests. */
export const generateSmall: GenerateFn = (opts) => {
  const seed = opts.seed >>> 0;
  const b = new Builder(seed, 220_000);
  b.stations = [
    [4.5, 1.7, 0.5],
    [-3.4, 1.7, 2.8],
    [0, 1.7, -4.4],
  ];
  const density = opts.density ?? 1;
  const sp = (s: number) => s / Math.sqrt(density);

  b.beginFeature("floor", "surface", "4 cm floor");
  b.plane([0, -1, 0], [1, 0, 0], [0, 0, 1], 5, 5, sp(0.04), [0, 1, 0],
    (su, sv) => concreteColour(su, sv), () => 0.55);
  b.endFeature();
  b.beginFeature("wall", "surface", "3 cm back wall");
  b.plane([0, 1.2, -5], [1, 0, 0], [0, 1, 0], 5, 2.2, sp(0.03), [0, 0, 1],
    (_su, sv) => [120 + _su * 3, 118 + sv * 4, 112], () => 0.45);
  b.endFeature();
  b.beginFeature("sphere", "surface", "2 cm sphere");
  b.sphere([-1.4, 0.2, -1.2], 0.9, sp(0.02),
    (y, u) => [200 + u * 30, 160 - y * 30, 120 + u * 70], 0.65);
  b.endFeature();
  b.beginFeature("rods", "rod", "thin rods at 1.2 cm");
  b.rod([1.6, -1, -0.6], [2.6, 1.9, -1.2], 0.025, sp(0.012), [176, 44, 38], 0.3);
  b.rod([-3.2, -1, 2.6], [-3.2, 1.6, 2.6], 0.02, sp(0.012), [170, 60, 40], 0.3);
  b.rod([-4.6, 1.9, -4.6], [4.6, 2.3, -4.6], 0.012, sp(0.01), [52, 56, 62], 0.25);
  b.endFeature();
  b.beginFeature("density-cliff", "density", "1.8 cm patch next to 4 cm floor");
  b.plane([2.6, -0.98, 2.4], [1, 0, 0], [0, 0, 1], 1.2, 1.2, sp(0.018), [0, 1, 0],
    (su, sv) => [90 + su * 20, 104 + sv * 30, 92], () => 0.7, 0.0008);
  b.endFeature();
  b.beginFeature("clutter", "noise", "foliage cluster");
  b.ellipsoidCluster([3.2, 0.4, -3.0], [1.1, 0.8, 1.0], 20000, [58, 104, 46]);
  b.endFeature();

  const bounds: AABB3 = { min: [-5.2, -1.3, -5.2], max: [5.2, 2.8, 5.2] };
  b.surfaces.push(
    { kind: "plane", name: "floor", origin: [0, -1, 0], normal: [0, 1, 0], half: [5, 5], spacing: 0.04 },
    { kind: "sphere", name: "sphere", center: [-1.4, 0.2, -1.2], radius: 0.9, spacing: 0.02 },
    { kind: "rod", name: "rod-a", a: [1.6, -1, -0.6], b: [2.6, 1.9, -1.2], radius: 0.025, spacing: 0.012 },
  );
  return finish(b, "small", bounds, seed, [7.5, 4.6, 7.5], [0, 0.2, 0]);
};

function finish(
  b: Builder,
  generator: string,
  declared: AABB3,
  seed: number,
  eye: [number, number, number],
  target: [number, number, number],
): GeneratedScan {
  const store = b.store;
  if (store.n === 0) throw new Error("no points generated");
  const stations = makeStations(b.stations, store);
  // The hierarchy must contain every emitted point (jitter included), so the
  // world bounds are measured, rounded outward, and checked against intent.
  const bounds = measuredBounds(store, declared);
  return {
    store,
    bounds,
    stations,
    surfaces: b.surfaces,
    features: b.features,
    seed,
    generator,
    notes: `synthetic ${generator} scan, ${store.n} points, ${stations.length} stations`,
    home: { eye, target, fov: 55 },
    defaultBudget: generator === "small" ? 240_000 : 1_200_000,
  };
}

export function measuredBounds(store: PointStore, declared: AABB3): AABB3 {
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < store.n; i++) {
    const p = [store.px[i], store.py[i], store.pz[i]];
    for (let a = 0; a < 3; a++) {
      if (p[a] < mn[a]) mn[a] = p[a];
      if (p[a] > mx[a]) mx[a] = p[a];
    }
  }
  const span = Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  const pad = Math.max(span * 0.005, 1e-3);
  const round = (v: number) => Math.round(v * 1000) / 1000;
  const out: AABB3 = {
    min: [round(Math.min(mn[0], declared.min[0]) - pad), round(Math.min(mn[1], declared.min[1]) - pad), round(Math.min(mn[2], declared.min[2]) - pad)],
    max: [round(Math.max(mx[0], declared.max[0]) + pad), round(Math.max(mx[1], declared.max[1]) + pad), round(Math.max(mx[2], declared.max[2]) + pad)],
  };
  return out;
}

export const GENERATORS: Record<string, GenerateFn> = {
  assembly: generateAssembly,
  cliff: generateCliff,
  small: generateSmall,
};

export function generate(name: string, opts: GenerateOptions): GeneratedScan {
  const fn = GENERATORS[name];
  if (!fn) throw new Error(`unknown generator "${name}"; choose ${Object.keys(GENERATORS).join(", ")}`);
  return fn(opts);
}

export function rank01(id: number): number {
  return stableRank(id) / 4294967296;
}
