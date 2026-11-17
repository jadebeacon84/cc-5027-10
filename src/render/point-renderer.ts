/**
 * Three.js point renderer.
 *
 * One THREE.Points object per resident payload node, each bound directly to the
 * views returned by the loader: the same ArrayBuffer that passed checksum
 * verification is the one uploaded, so attributes cannot drift apart and no
 * per-point repack exists to get wrong.
 *
 * Attribute binding:
 *   position   <- views.positions  Float32 x3
 *   aColor     <- views.colours    Uint8   x4 (normalised; alpha unused)
 *   aNormal    <- views.normals    Int8    x3 within a 4-byte stride
 *   aIntensity <- views.intensity  Float32 x1
 *   aRange     <- views.ranges     Float32 x1
 *   aScan      <- views.scans      Uint32  x1 (station index, for palette)
 *   aId        <- views.ids        Uint32  x1 (identity, for picking)
 *   aLod       <- derived          Float32 x1 (depth / maxDepth, debug view)
 */
import * as THREE from "three";
import type { PayloadViews } from "../shared/format";
import type { NodeIndex } from "../kernel/hierarchy";
import type { Hierarchy } from "../kernel/hierarchy";

export type PointCloudDisplay = "colour" | "intensity" | "normal" | "scan" | "lod";

export type RenderStats = {
  drawnNodes: number;
  drawnPoints: number;
  gpuBytes: number;
  objects: number;
  geometriesCreated: number;
  geometriesDisposed: number;
};

export type HoverHit = {
  node: NodeIndex;
  pointIndex: number;
  id: number;
  x: number; y: number; z: number;
  r: number; g: number; b: number;
  scan: number;
  distance: number;
  intensity: number;
};

const VERT = /* glsl */ `
precision highp float;
uniform float uPointSize;
uniform float uPixelRatio;
uniform int uDisplay;
uniform float uScanCount;
attribute vec4 aColor;
attribute vec4 aNormal;
attribute float aIntensity;
attribute float aRange;
attribute float aScan;
attribute float aLod;
varying vec3 vColor;
varying float vAlpha;

vec3 stationColour(float s) {
  float t = fract(s * 0.61803398875 + 0.13);
  return 0.45 + 0.45 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * mv;
  float dist = max(-mv.z, 0.05);
  gl_PointSize = clamp(uPointSize * uPixelRatio / dist, 0.7, 12.0);

  vec3 base = aColor.rgb;
  if (uDisplay == 1) {
    float t = clamp(aIntensity, 0.0, 1.0);
    base = mix(vec3(0.05, 0.10, 0.30), vec3(0.10, 0.72, 0.86), smoothstep(0.0, 0.5, t));
    base = mix(base, vec3(1.00, 0.92, 0.50), smoothstep(0.5, 0.85, t));
    base = mix(base, vec3(1.00, 1.00, 1.00), smoothstep(0.85, 1.0, t));
  } else if (uDisplay == 2) {
    base = normalize(aNormal.xyz) * 0.5 + 0.5;
  } else if (uDisplay == 3) {
    base = stationColour(aScan / max(uScanCount - 1.0, 1.0));
  } else if (uDisplay == 4) {
    float t = clamp(aLod, 0.0, 1.0);
    base = mix(vec3(0.94, 0.36, 0.22), vec3(0.32, 0.86, 1.00), t);
  }

  // Range haze: distant scanner returns read slightly cooler and dimmer.
  float haze = clamp(1.0 - aRange / 90.0, 0.0, 1.0);
  vColor = mix(base * 0.72, base, haze);
  vAlpha = 1.0;
}
`;

const FRAG = /* glsl */ `
precision highp float;
varying vec3 vColor;
varying float vAlpha;
uniform float uBrightness;
void main() {
  vec2 pc = gl_PointCoord - vec2(0.5);
  if (dot(pc, pc) > 0.25) discard;
  gl_FragColor = vec4(vColor * uBrightness, vAlpha);
  #include <colorspace_fragment>
}
`;

export class PointRenderer {
  readonly scene: THREE.Scene;
  private objects = new Map<NodeIndex, THREE.Points>();
  private material: THREE.ShaderMaterial;
  private hierarchy: Hierarchy;
  private display: PointCloudDisplay = "colour";
  private stats: RenderStats = {
    drawnNodes: 0,
    drawnPoints: 0,
    gpuBytes: 0,
    objects: 0,
    geometriesCreated: 0,
    geometriesDisposed: 0,
  };

  constructor(hierarchy: Hierarchy) {
    this.hierarchy = hierarchy;
    this.scene = new THREE.Scene();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERT,
      fragmentShader: FRAG,
      uniforms: {
        uPointSize: { value: 1.4 },
        uPixelRatio: { value: 1 },
        uDisplay: { value: 0 },
        uScanCount: { value: 1 },
        uBrightness: { value: 1 },
      },
      transparent: false,
      depthWrite: true,
      depthTest: true,
    });
  }

  setPixelRatio(ratio: number) { this.material.uniforms.uPixelRatio.value = ratio; }
  setPointSize(px: number) { this.material.uniforms.uPointSize.value = px; }
  setBrightness(v: number) { this.material.uniforms.uBrightness.value = v; }
  setScanCount(n: number) { this.material.uniforms.uScanCount.value = Math.max(1, n); }
  getDisplay(): PointCloudDisplay { return this.display; }
  setDisplay(mode: PointCloudDisplay) {
    this.display = mode;
    this.material.uniforms.uDisplay.value =
      mode === "colour" ? 0 : mode === "intensity" ? 1 : mode === "normal" ? 2 : mode === "scan" ? 3 : 4;
  }

  /** Upload one resident payload. Called once per node per residency. */
  addNode(node: NodeIndex, views: PayloadViews) {
    if (this.objects.has(node)) return;
    const count = views.header.count;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(views.positions, 3));
    geo.setAttribute("aColor", new THREE.Uint8BufferAttribute(views.colours, 4, true));
    // Declared vec3 in the shader, so normalise the signed bytes to float.
    geo.setAttribute("aNormal", new THREE.BufferAttribute(views.normals, 4, true));
    geo.setAttribute("aIntensity", new THREE.BufferAttribute(views.intensity, 1));
    geo.setAttribute("aRange", new THREE.BufferAttribute(views.ranges, 1));
    // WebGL float attributes cannot source UNSIGNED_INT directly; station
    // indices are tiny, so a one-time float view is exact. The id array is
    // kept on the geometry as Uint32 for picking but is not a shader input.
    const scanF = new Float32Array(count);
    for (let i = 0; i < count; i++) scanF[i] = views.scans[i];
    geo.setAttribute("aScan", new THREE.BufferAttribute(scanF, 1));
    geo.setAttribute("aId", new THREE.BufferAttribute(views.ids, 1));
    const lod = new Float32Array(count);
    lod.fill(this.hierarchy.depthMax > 0 ? this.hierarchy.nodes[node].depth / this.hierarchy.depthMax : 0);
    geo.setAttribute("aLod", new THREE.BufferAttribute(lod, 1));
    const { min, max } = views.header;
    geo.boundingBox = new THREE.Box3(new THREE.Vector3(min[0], min[1], min[2]), new THREE.Vector3(max[0], max[1], max[2]));
    geo.boundingSphere = geo.boundingBox.getBoundingSphere(new THREE.Sphere());

    const pts = new THREE.Points(geo, this.material);
    pts.frustumCulled = false; // node-level frustum culling happens in selection
    pts.renderOrder = this.hierarchy.nodes[node].depth;
    pts.userData.node = node;
    this.scene.add(pts);
    this.objects.set(node, pts);
    this.stats.geometriesCreated++;
    this.refreshCounters();
  }

  removeNode(node: NodeIndex) {
    const pts = this.objects.get(node);
    if (!pts) return;
    this.scene.remove(pts);
    pts.geometry.dispose();
    this.objects.delete(node);
    this.stats.geometriesDisposed++;
    this.refreshCounters();
  }

  hasNode(node: NodeIndex): boolean { return this.objects.has(node); }
  residentNodeList(): NodeIndex[] { return [...this.objects.keys()]; }
  getStats(): RenderStats { return { ...this.stats }; }

  private refreshCounters() {
    let points = 0;
    let bytes = 0;
    for (const pts of this.objects.values()) {
      const n = pts.geometry.getAttribute("position").count;
      points += n;
      bytes += n * 36;
    }
    this.stats.drawnPoints = points;
    this.stats.drawnNodes = this.objects.size;
    this.stats.objects = this.objects.size;
    this.stats.gpuBytes = bytes;
  }

  /**
   * Ray pick against the exact records the GPU drew. Bounding boxes are tested
   * first, so only a few nodes require a point scan. `tolerance` is the pick
   * cylinder radius in world units (the caller scales it by scene size and
   * point size); it is deliberately separate from the pick distance.
   */
  pick(origin: THREE.Vector3, dir: THREE.Vector3, maxDistance: number, tolerance = 0.01): HoverHit | null {
    const ray = new THREE.Ray(origin, dir);
    let bestT = maxDistance;
    let best: HoverHit | null = null;
    const hit = new THREE.Vector3();
    const radius = Math.max(1e-5, tolerance);
    for (const [node, pts] of this.objects) {
      if (!pts.geometry.boundingBox) continue;
      if (!ray.intersectBox(pts.geometry.boundingBox, hit)) continue;
      if (hit.distanceTo(origin) > bestT + 0.2) continue;
      const count = pts.geometry.getAttribute("position").count;
      const pos = (pts.geometry.getAttribute("position") as THREE.BufferAttribute).array as Float32Array;
      const col = (pts.geometry.getAttribute("aColor") as THREE.BufferAttribute).array as Uint8Array;
      const id = (pts.geometry.getAttribute("aId") as THREE.BufferAttribute).array as Uint32Array;
      const scanF = (pts.geometry.getAttribute("aScan") as THREE.BufferAttribute).array as Float32Array;
      const inten = (pts.geometry.getAttribute("aIntensity") as THREE.BufferAttribute).array as Float32Array;
      const range = (pts.geometry.getAttribute("aRange") as THREE.BufferAttribute).array as Float32Array;
      for (let i = 0; i < count; i++) {
        const px = pos[i * 3] - origin.x;
        const py = pos[i * 3 + 1] - origin.y;
        const pz = pos[i * 3 + 2] - origin.z;
        const t = px * dir.x + py * dir.y + pz * dir.z;
        if (t < 0 || t > bestT) continue;
        const cx = px - dir.x * t, cy = py - dir.y * t, cz = pz - dir.z * t;
        if (cx * cx + cy * cy + cz * cz > radius * radius) continue;
        bestT = t;
        best = {
          node,
          pointIndex: i,
          id: id[i],
          x: pos[i * 3], y: pos[i * 3 + 1], z: pos[i * 3 + 2],
          r: col[i * 4], g: col[i * 4 + 1], b: col[i * 4 + 2],
          scan: scanF[i] | 0,
          distance: range[i],
          intensity: inten[i],
        };
      }
    }
    return best;
  }

  dispose() {
    for (const node of [...this.objects.keys()]) this.removeNode(node);
    this.material.dispose();
  }
}
