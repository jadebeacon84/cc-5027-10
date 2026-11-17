/**
 * Shared geometry helpers used by both the browser kernel and Node-side
 * validators/tests. Everything here is pure and dependency-free so the same
 * code decides LOD selection offline and in the page.
 */

export type Vec3 = [number, number, number];
export type AABB3 = { min: Vec3; max: Vec3 };

export type CameraView = {
  eye: Vec3;
  target: Vec3;
  /** Column-major view-projection matrix. */
  viewProjection: Float64Array;
  /** Pixels per world unit at unit distance: viewportHeight / (2 tan(fov/2)). */
  projectionFactor: number;
  near: number;
  fov: number;
};

const EPS = 1e-9;

/** Gribb/Hartmann frustum planes (a,b,c,d), normalized, from a column-major matrix. */
export function extractFrustumPlanes(m: ArrayLike<number>): Float64Array {
  const row = (r: number): [number, number, number, number] => [m[r], m[4 + r], m[8 + r], m[12 + r]];
  const r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);
  const planes = new Float64Array(24);
  const write = (i: number, a: number, b: number, c: number, d: number) => {
    const len = Math.hypot(a, b, c) || 1;
    planes[i * 4] = a / len;
    planes[i * 4 + 1] = b / len;
    planes[i * 4 + 2] = c / len;
    planes[i * 4 + 3] = d / len;
  };
  write(0, r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]);
  write(1, r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]);
  write(2, r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]);
  write(3, r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]);
  write(4, r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2], r3[3] + r2[3]);
  write(5, r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]);
  return planes;
}

export function aabbVisible(box: AABB3, planes: Float64Array): boolean {
  for (let p = 0; p < 6; p++) {
    const a = planes[p * 4], b = planes[p * 4 + 1], c = planes[p * 4 + 2], d = planes[p * 4 + 3];
    const x = a >= 0 ? box.max[0] : box.min[0];
    const y = b >= 0 ? box.max[1] : box.min[1];
    const z = c >= 0 ? box.max[2] : box.min[2];
    if (a * x + b * y + c * z + d < 0) return false;
  }
  return true;
}

export function pointInFrustum(p: Vec3, planes: Float64Array): boolean {
  for (let i = 0; i < 6; i++) {
    if (planes[i * 4] * p[0] + planes[i * 4 + 1] * p[1] + planes[i * 4 + 2] * p[2] + planes[i * 4 + 3] < 0) {
      return false;
    }
  }
  return true;
}

export function distanceToAABB(box: AABB3, p: Vec3): number {
  const dx = Math.max(box.min[0] - p[0], 0, p[0] - box.max[0]);
  const dy = Math.max(box.min[1] - p[1], 0, p[1] - box.max[1]);
  const dz = Math.max(box.min[2] - p[2], 0, p[2] - box.max[2]);
  return Math.hypot(dx, dy, dz);
}

export function aabbCenter(box: AABB3): Vec3 {
  return [
    (box.min[0] + box.max[0]) * 0.5,
    (box.min[1] + box.max[1]) * 0.5,
    (box.min[2] + box.max[2]) * 0.5,
  ];
}

export function aabbRadius(box: AABB3): number {
  return 0.5 * Math.hypot(
    box.max[0] - box.min[0],
    box.max[1] - box.min[1],
    box.max[2] - box.min[2],
  );
}

export function aabbContainsBox(outer: AABB3, inner: AABB3, pad = 0): boolean {
  for (let a = 0; a < 3; a++) {
    if (inner.min[a] < outer.min[a] - pad || inner.max[a] > outer.max[a] + pad) return false;
  }
  return true;
}

export function aabbOverlaps(a: AABB3, b: AABB3, pad = 0): boolean {
  for (let i = 0; i < 3; i++) {
    if (a.max[i] < b.min[i] - pad || a.min[i] > b.max[i] + pad) return false;
  }
  return true;
}

export function multiplyMat4(a: ArrayLike<number>, b: ArrayLike<number>): Float64Array {
  const out = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      let s = 0;
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k];
      out[c * 4 + r] = s;
    }
  }
  return out;
}

export function perspectiveMatrix(fovY: number, aspect: number, near: number, far: number): Float64Array {
  const f = 1 / Math.tan(fovY / 2);
  const nf = 1 / (near - far);
  const out = new Float64Array(16);
  out[0] = f / aspect;
  out[5] = f;
  out[10] = (far + near) * nf;
  out[11] = -1;
  out[14] = 2 * far * near * nf;
  return out;
}

export function lookAtMatrix(eye: Vec3, target: Vec3, up: Vec3 = [0, 1, 0]): Float64Array {
  let zx = eye[0] - target[0], zy = eye[1] - target[1], zz = eye[2] - target[2];
  let zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl; zy /= zl; zz /= zl;
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  let xl = Math.hypot(xx, xy, xz);
  if (xl < EPS) { xx = 1; xy = 0; xz = 0; xl = 1; }
  xx /= xl; xy /= xl; xz /= xl;
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;
  const out = new Float64Array(16);
  out[0] = xx; out[1] = yx; out[2] = zx;
  out[4] = xy; out[5] = yy; out[6] = zy;
  out[8] = xz; out[9] = yz; out[10] = zz;
  out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
  out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
  out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
  out[15] = 1;
  return out;
}

export function projectionFactor(fovY: number, viewportHeightPx: number): number {
  return viewportHeightPx / (2 * Math.tan(fovY / 2));
}

/**
 * Screen-space error in pixels for drawing a node whose region could resolve
 * to `geometricError` world units. Standard Cesium-style metric: world error
 * projected through the perspective factor and divided by view distance.
 */
export function screenSpaceError(
  view: CameraView,
  box: AABB3,
  geometricError: number,
  viewportHeightPx: number,
): number {
  const d = Math.max(distanceToAABB(box, view.eye), view.near);
  return (geometricError * projectionFactor(view.fov, viewportHeightPx)) / d;
}

export function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
