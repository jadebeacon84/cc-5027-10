/**
 * On-disk point-cloud octree format shared by data prep, the static server and
 * the browser loader.
 *
 * A dataset directory contains:
 *   manifest.json      hierarchy, bounds, per-node statistics
 *   meta.json          generator parameters, scan stations, analytic surfaces
 *   nodes/<id>.bin     payload of one octree node
 *
 * =========================== Payload layout ===========================
 *
 * One node payload is a header followed by *parallel attribute arrays*:
 *
 *   header            96 bytes, little endian
 *   positions         count * 3 * float32
 *   colours           count * 4 * uint8     (RGB + 255)
 *   normals           count * 4 * int8      (x,y,z quantised by 127, pad 0)
 *   intensity         count * 1 * float32
 *   ranges            count * 1 * float32   (distance to scan station)
 *   ids               count * 1 * uint32    (stable global point identity)
 *   scans             count * 1 * uint32    (source station index)
 *
 * All arrays start on 4-byte boundaries and have exactly `count` elements,
 * so a fetched ArrayBuffer can be uploaded as typed-array views without any
 * per-point copy, and an off-by-one in one attribute is impossible to hide:
 * each array's checksum covers its own bytes, and `positions` length is
 * derived from `count` in the header.
 *
 * Because positions/colours/normals/ids are indexed identically, "the i-th
 * point" means the same thing in every array. Batch upload code must assert
 * `count` equality and verify the header checksums before creating a geometry.
 * The validator does the same check against the generator's ground truth, so a
 * misaligned colour cannot pass unnoticed.
 */
import type { AABB3 } from "./geo";

export const MAGIC = 0x31544350; // "PCT1"
export const VERSION = 5;
export const HEADER_BYTES = 96;
export type { AABB3 };

/** Header offsets (little endian). */
export const H_MAGIC = 0;
export const H_VERSION = 4;
export const H_HEADER_BYTES = 6;
export const H_COUNT = 8;
export const H_LAYOUT = 12; // 1 = parallel arrays as documented above
export const H_POS_OFF = 16;
export const H_COLOR_OFF = 20;
export const H_NORMAL_OFF = 24;
export const H_INTENSITY_OFF = 28;
export const H_RANGE_OFF = 32;
export const H_ID_OFF = 36;
export const H_SCAN_OFF = 40;
export const H_CHECK_POS = 44;
export const H_CHECK_COLOR = 48;
export const H_CHECK_NORMAL = 52;
export const H_CHECK_ID = 56;
export const H_CHECK_ATTR = 60;
export const H_NODE_HASH = 64;
export const H_MIN = 68;
export const H_MAX = 80;
export const H_PAYLOAD_BYTES = 92;

export const LAYOUT_SOA = 1;

/** Bytes per point across all parallel arrays (allocation guide only). */
export const BYTES_PER_POINT = 12 + 4 + 4 + 4 + 4 + 4 + 4;

export type PayloadHeader = {
  magic: number;
  version: number;
  headerBytes: number;
  count: number;
  layout: number;
  posOff: number;
  colorOff: number;
  normalOff: number;
  intensityOff: number;
  rangeOff: number;
  idOff: number;
  scanOff: number;
  checkPos: number;
  checkColor: number;
  checkNormal: number;
  checkId: number;
  checkAttr: number;
  nodeHash: number;
  min: [number, number, number];
  max: [number, number, number];
  payloadBytes: number;
};

/** Byte offsets of each array for a given point count. */
export function payloadLayout(count: number) {
  const posOff = HEADER_BYTES;
  const colorOff = posOff + count * 12;
  const normalOff = colorOff + count * 4;
  const intensityOff = normalOff + count * 4;
  const rangeOff = intensityOff + count * 4;
  const idOff = rangeOff + count * 4;
  const scanOff = idOff + count * 4;
  const end = scanOff + count * 4;
  return { posOff, colorOff, normalOff, intensityOff, rangeOff, idOff, scanOff, end };
}

export type NodeManifest = {
  id: string;
  path: string;
  depth: number;
  children: string[];
  /** Points stored in this node's own payload. */
  pointCount: number;
  /** Source points in the subtree, including descendants. */
  subtreePointCount: number;
  bounds: AABB3;
  /** Estimated 25th-percentile nearest-neighbour spacing of the payload. */
  spacing: number;
  /** Finest spacing reachable in the subtree; screen error if unrefined. */
  geometricError: number;
  /** Smallest spacing anywhere in the subtree. */
  subtreeMinSpacing: number;
  bytes: number;
  hash: string;
  payloadOffset: number;
  payloadLength: number;
  hasPayload: boolean;
  /** Fraction of subtree points kept in this node (diagnostic). */
  samplingRate: number;
};

export type Manifest = {
  format: "pc-octree";
  version: number;
  dataset: string;
  title: string;
  generator: string;
  bounds: AABB3;
  totalPoints: number;
  nodeCount: number;
  payloadNodeCount: number;
  maxDepth: number;
  finestSpacing: number;
  root: string;
  childOrder: string;
  maxNodeBytes: number;
  nodeCapacity: number;
  nodes: NodeManifest[];
  createdAt: string;
  seed: number;
  notes: string;
};

export type ScanStation = {
  id: number;
  position: [number, number, number];
  yaw: number;
  pitch: number;
  fov: number;
  pointCount: number;
};

export type AnalyticSurface =
  | { kind: "plane"; name: string; origin: [number, number, number]; normal: [number, number, number]; half: [number, number]; spacing: number }
  | { kind: "sphere"; name: string; center: [number, number, number]; radius: number; spacing: number }
  | { kind: "cylinder"; name: string; center: [number, number, number]; radius: number; height: number; spacing: number }
  | { kind: "rod"; name: string; a: [number, number, number]; b: [number, number, number]; radius: number; spacing: number }
  | { kind: "lattice"; name: string; count: number; bounds: AABB3; spacing: number };

export type FeatureRange = {
  name: string;
  kind: "surface" | "rod" | "density" | "noise" | "edge" | "terrain";
  pointCount: number;
  idStart: number;
  idEnd: number;
  notes?: string;
};

export type DatasetMeta = {
  dataset: string;
  title: string;
  generator: string;
  description: string;
  seed: number;
  worldBounds: AABB3;
  totalPoints: number;
  scanCount: number;
  stations: ScanStation[];
  features: FeatureRange[];
  surfaces: AnalyticSurface[];
  home: { eye: [number, number, number]; target: [number, number, number]; fov: number };
  defaultBudget: number;
  notes: string;
};

/** In-memory record used while generating and validating. */
export type PointRecord = {
  x: number; y: number; z: number;
  r: number; g: number; b: number;
  nx: number; ny: number; nz: number;
  intensity: number;
  distance: number;
  pointId: number;
  scan: number;
};

export function fnv1a(bytes: Uint8Array, seed = 0x811c9dc5, start = 0, end = bytes.length): number {
  let h = seed >>> 0;
  for (let i = start; i < end; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/** FNV-1a over the 4-byte little-endian representation of each value. */
export function fnv1aU32(values: Uint32Array, start = 0, end = values.length): number {
  let h = 0x811c9dc5;
  for (let i = start; i < end; i++) {
    const v = values[i];
    h ^= v & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
    h ^= (v >>> 8) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
    h ^= (v >>> 16) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
    h ^= (v >>> 24) & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function hashString(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

/**
 * Sampling rank. Stability is what makes LOD transitions identity-preserving:
 * the same id always receives the same rank, the builder assigns a point to
 * exactly one node, and the renderer never remaps ids.
 */
export function stableRank(pointId: number): number {
  let v = (pointId ^ 0x9e3779b9) >>> 0;
  v = Math.imul(v ^ (v >>> 16), 0x85ebca6b) >>> 0;
  v = Math.imul(v ^ (v >>> 13), 0xc2b2ae35) >>> 0;
  v = (v ^ (v >>> 16)) >>> 0;
  return v;
}

export const RANK_SCALE = 4294967296;

/** Octree path helpers. Root path is "0"; each child appends "/<octant>". */
export function childPath(path: string, index: number): string {
  return `${path}/${index}`;
}

export function nodeFileName(path: string): string {
  return `${path.replaceAll("/", "-")}.bin`;
}

export function depthOfPath(path: string): number {
  return path === "0" ? 0 : path.split("/").length - 1;
}

export function nodeGridCell(path: string, depth: number): [number, number, number] {
  let x = 0, y = 0, z = 0;
  const segs = path.split("/").slice(1);
  for (let i = 0; i < segs.length && i < depth; i++) {
    const c = Number(segs[i]);
    if (c & 1) x |= 1 << (depth - 1 - i);
    if (c & 2) y |= 1 << (depth - 1 - i);
    if (c & 4) z |= 1 << (depth - 1 - i);
  }
  return [x, y, z];
}

export function nodeBoundsAt(path: string, depth: number, world: AABB3): AABB3 {
  const [cx, cy, cz] = nodeGridCell(path, depth);
  const cell = 1 / 2 ** depth;
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  const cells = [cx, cy, cz];
  for (let a = 0; a < 3; a++) {
    const span = world.max[a] - world.min[a];
    min[a] = world.min[a] + cells[a] * cell * span;
    max[a] = world.min[a] + (cells[a] + 1) * cell * span;
  }
  return { min, max };
}

export function writeHeader(bytes: Uint8Array, h: PayloadHeader) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  v.setUint32(H_MAGIC, h.magic, true);
  v.setUint16(H_VERSION, h.version, true);
  v.setUint16(H_HEADER_BYTES, h.headerBytes, true);
  v.setUint32(H_COUNT, h.count, true);
  v.setUint32(H_LAYOUT, h.layout, true);
  v.setUint32(H_POS_OFF, h.posOff, true);
  v.setUint32(H_COLOR_OFF, h.colorOff, true);
  v.setUint32(H_NORMAL_OFF, h.normalOff, true);
  v.setUint32(H_INTENSITY_OFF, h.intensityOff, true);
  v.setUint32(H_RANGE_OFF, h.rangeOff, true);
  v.setUint32(H_ID_OFF, h.idOff, true);
  v.setUint32(H_SCAN_OFF, h.scanOff, true);
  v.setUint32(H_CHECK_POS, h.checkPos, true);
  v.setUint32(H_CHECK_COLOR, h.checkColor, true);
  v.setUint32(H_CHECK_NORMAL, h.checkNormal, true);
  v.setUint32(H_CHECK_ID, h.checkId, true);
  v.setUint32(H_CHECK_ATTR, h.checkAttr, true);
  v.setUint32(H_NODE_HASH, h.nodeHash, true);
  for (let i = 0; i < 3; i++) v.setFloat32(H_MIN + i * 4, h.min[i], true);
  for (let i = 0; i < 3; i++) v.setFloat32(H_MAX + i * 4, h.max[i], true);
  v.setUint32(H_PAYLOAD_BYTES, h.payloadBytes, true);
}

export function readHeader(bytes: Uint8Array): PayloadHeader {
  if (bytes.byteLength < HEADER_BYTES) throw new Error(`payload too short: ${bytes.byteLength}`);
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const min: [number, number, number] = [0, 0, 0];
  const max: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < 3; i++) min[i] = v.getFloat32(H_MIN + i * 4, true);
  for (let i = 0; i < 3; i++) max[i] = v.getFloat32(H_MAX + i * 4, true);
  return {
    magic: v.getUint32(H_MAGIC, true),
    version: v.getUint16(H_VERSION, true),
    headerBytes: v.getUint16(H_HEADER_BYTES, true),
    count: v.getUint32(H_COUNT, true),
    layout: v.getUint32(H_LAYOUT, true),
    posOff: v.getUint32(H_POS_OFF, true),
    colorOff: v.getUint32(H_COLOR_OFF, true),
    normalOff: v.getUint32(H_NORMAL_OFF, true),
    intensityOff: v.getUint32(H_INTENSITY_OFF, true),
    rangeOff: v.getUint32(H_RANGE_OFF, true),
    idOff: v.getUint32(H_ID_OFF, true),
    scanOff: v.getUint32(H_SCAN_OFF, true),
    checkPos: v.getUint32(H_CHECK_POS, true),
    checkColor: v.getUint32(H_CHECK_COLOR, true),
    checkNormal: v.getUint32(H_CHECK_NORMAL, true),
    checkId: v.getUint32(H_CHECK_ID, true),
    checkAttr: v.getUint32(H_CHECK_ATTR, true),
    nodeHash: v.getUint32(H_NODE_HASH, true),
    min,
    max,
    payloadBytes: v.getUint32(H_PAYLOAD_BYTES, true),
  };
}

/** Typed-array views over a fetched payload. Throws on any structural mismatch. */
export type PayloadViews = {
  header: PayloadHeader;
  positions: Float32Array;
  colours: Uint8Array;
  normals: Int8Array;
  intensity: Float32Array;
  ranges: Float32Array;
  ids: Uint32Array;
  scans: Uint32Array;
};

export function payloadViews(bytes: Uint8Array): PayloadViews {
  const header = readHeader(bytes);
  if (header.magic !== MAGIC) throw new Error(`bad magic 0x${header.magic.toString(16)}`);
  if (header.version !== VERSION) throw new Error(`unsupported payload version ${header.version}`);
  if (header.layout !== LAYOUT_SOA) throw new Error(`unsupported payload layout ${header.layout}`);
  const expect = payloadLayout(header.count);
  if (bytes.byteLength !== expect.end) {
    throw new Error(`payload length ${bytes.byteLength} != expected ${expect.end} for ${header.count} points`);
  }
  if (header.posOff !== expect.posOff || header.colorOff !== expect.colorOff
    || header.normalOff !== expect.normalOff || header.intensityOff !== expect.intensityOff
    || header.rangeOff !== expect.rangeOff || header.idOff !== expect.idOff
    || header.scanOff !== expect.scanOff) {
    throw new Error("payload array offsets do not match the documented layout");
  }
  const buf = bytes.buffer;
  const base = bytes.byteOffset;
  const views = {
    header,
    positions: new Float32Array(buf, base + header.posOff, header.count * 3),
    colours: new Uint8Array(buf, base + header.colorOff, header.count * 4),
    normals: new Int8Array(buf, base + header.normalOff, header.count * 4),
    intensity: new Float32Array(buf, base + header.intensityOff, header.count),
    ranges: new Float32Array(buf, base + header.rangeOff, header.count),
    ids: new Uint32Array(buf, base + header.idOff, header.count),
    scans: new Uint32Array(buf, base + header.scanOff, header.count),
  };
  verifyChecksums(views);
  return views;
}

/** Throws if any attribute array does not match the recorded checksum. */
export function verifyChecksums(views: PayloadViews): void {
  const h = views.header;
  const posBytes = new Uint8Array(views.positions.buffer, views.positions.byteOffset, views.positions.byteLength);
  const colorBytes = new Uint8Array(views.colours.buffer, views.colours.byteOffset, views.colours.byteLength);
  const normalBytes = new Uint8Array(views.normals.buffer, views.normals.byteOffset, views.normals.byteLength);
  if (fnv1a(posBytes) !== h.checkPos) throw new Error(`position checksum mismatch (node hash ${h.nodeHash.toString(16)})`);
  if (fnv1a(colorBytes) !== h.checkColor) throw new Error(`colour checksum mismatch (node hash ${h.nodeHash.toString(16)})`);
  if (fnv1a(normalBytes) !== h.checkNormal) throw new Error(`normal checksum mismatch (node hash ${h.nodeHash.toString(16)})`);
  if (fnv1aU32(views.ids) !== h.checkId) throw new Error(`id checksum mismatch (node hash ${h.nodeHash.toString(16)})`);
  let attr = fnv1a(posBytes);
  attr = fnv1a(colorBytes, attr);
  attr = fnv1a(normalBytes, attr);
  if (attr !== h.checkAttr) throw new Error(`combined attribute checksum mismatch (node hash ${h.nodeHash.toString(16)})`);
}
