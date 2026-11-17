/**
 * Orbit camera with a few extras a scan viewer needs: damping, zoom-to-cursor,
 * and a "home" pose. Kept framework-free so the trajectory harness can drive
 * it deterministically.
 */
import * as THREE from "three";
import type { Vec3 } from "../shared/geo";

export type Pose = { eye: Vec3; target: Vec3; fov: number };

export class OrbitCamera {
  readonly camera: THREE.PerspectiveCamera;
  target = new THREE.Vector3();
  private spherical = new THREE.Spherical(10, Math.PI / 2.4, Math.PI / 4);
  private desired = new THREE.Spherical(10, Math.PI / 2.4, Math.PI / 4);
  private desiredTarget = new THREE.Vector3();
  damping = 0.18;
  minDistance = 0.05;
  maxDistance = 400;
  private dirty = true;

  constructor(aspect: number) {
    this.camera = new THREE.PerspectiveCamera(55, aspect, 0.02, 900);
  }

  setPose(pose: Pose) {
    this.target.set(pose.target[0], pose.target[1], pose.target[2]);
    this.desiredTarget.copy(this.target);
    const offset = new THREE.Vector3(
      pose.eye[0] - pose.target[0],
      pose.eye[1] - pose.target[1],
      pose.eye[2] - pose.target[2],
    );
    this.spherical.setFromVector3(offset);
    this.desired.copy(this.spherical);
    this.camera.fov = pose.fov;
    this.camera.updateProjectionMatrix();
    this.apply();
    this.dirty = true;
  }

  getPose(): Pose {
    return {
      eye: [this.camera.position.x, this.camera.position.y, this.camera.position.z],
      target: [this.target.x, this.target.y, this.target.z],
      fov: this.camera.fov,
    };
  }

  orbit(dx: number, dy: number) {
    const speed = 0.0045;
    this.desired.theta -= dx * speed;
    this.desired.phi = Math.max(0.04, Math.min(Math.PI - 0.04, this.desired.phi - dy * speed));
    this.dirty = true;
  }

  pan(dx: number, dy: number, viewportHeight: number) {
    // Screen-space pan scaled by distance so it feels consistent at any zoom.
    const dist = this.spherical.radius;
    const scale = (2 * dist * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2)) / viewportHeight;
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
    this.desiredTarget.addScaledVector(right, -dx * scale);
    this.desiredTarget.addScaledVector(up, dy * scale);
    // Keep the pan target loosely inside the dataset bounds.
    this.dirty = true;
  }

  /** Zoom by a multiplicative factor (1.1 zooms out, 1/1.1 zooms in). */
  zoom(factor: number, cursorNdc?: { x: number; y: number }) {
    const before = this.spherical.radius;
    const after = Math.max(this.minDistance, Math.min(this.maxDistance, before * factor));
    if (cursorNdc && Math.abs(after - before) > 1e-9) {
      // Move the orbit target toward the point under the cursor so zooming
      // feels anchored, then adjust radius.
      const ndc = new THREE.Vector3(cursorNdc.x, cursorNdc.y, 0.5);
      ndc.unproject(this.camera);
      const dir = ndc.sub(this.camera.position).normalize();
      const travel = before - after;
      this.desiredTarget.addScaledVector(dir, travel * 0.35);
    }
    this.desired.radius = after;
    this.dirty = true;
  }

  /** Instant zoom with no damping, used by scripted trajectories. */
  zoomImmediate(factor: number) {
    const after = Math.max(this.minDistance, Math.min(this.maxDistance, this.spherical.radius * factor));
    this.desired.radius = after;
    this.spherical.radius = after;
    this.apply();
    this.dirty = true;
  }

  setTargetImmediate(t: Vec3) {
    this.target.set(t[0], t[1], t[2]);
    this.desiredTarget.copy(this.target);
    this.apply();
    this.dirty = true;
  }

  /** Smoothly move to a named pose over `ms` (used by focus presets). */
  private transition: { from: Pose; to: Pose; start: number; ms: number } | null = null;
  flyTo(pose: Pose, ms = 600) {
    this.transition = { from: this.getPose(), to: pose, start: performance.now(), ms };
    this.dirty = true;
  }

  update(now: number) {
    if (this.transition) {
      const t = Math.min(1, (now - this.transition.start) / this.transition.ms);
      const e = t * t * (3 - 2 * t);
      const a = this.transition.from, b = this.transition.to;
      this.target.set(
        a.target[0] + (b.target[0] - a.target[0]) * e,
        a.target[1] + (b.target[1] - a.target[1]) * e,
        a.target[2] + (b.target[2] - a.target[2]) * e,
      );
      this.desiredTarget.copy(this.target);
      const va = new THREE.Vector3(a.eye[0] - a.target[0], a.eye[1] - a.target[1], a.eye[2] - a.target[2]);
      const vb = new THREE.Vector3(b.eye[0] - b.target[0], b.eye[1] - b.target[1], b.eye[2] - b.target[2]);
      const sa = new THREE.Spherical().setFromVector3(va);
      const sb = new THREE.Spherical().setFromVector3(vb);
      const r = sa.radius + (sb.radius - sa.radius) * e;
      let dTheta = sb.theta - sa.theta;
      if (dTheta > Math.PI) dTheta -= Math.PI * 2;
      if (dTheta < -Math.PI) dTheta += Math.PI * 2;
      const s = new THREE.Spherical(r, sa.phi + (sb.phi - sa.phi) * e, sa.theta + dTheta * e);
      this.spherical.copy(s);
      this.desired.copy(s);
      this.apply();
      if (t >= 1) this.transition = null;
      return;
    }
    const k = this.damping;
    this.spherical.radius += (this.desired.radius - this.spherical.radius) * k;
    this.spherical.theta += (this.desired.theta - this.spherical.theta) * k;
    this.spherical.phi += (this.desired.phi - this.spherical.phi) * k;
    this.target.lerp(this.desiredTarget, k);
    this.apply();
  }

  private apply() {
    const offset = new THREE.Vector3().setFromSpherical(this.spherical);
    this.camera.position.copy(this.target).add(offset);
    this.camera.lookAt(this.target);
    this.camera.updateMatrixWorld();
  }

  setViewport(width: number, height: number) {
    this.camera.aspect = width / Math.max(1, height);
    this.camera.near = Math.max(0.01, this.spherical.radius * 0.002);
    this.camera.far = Math.max(200, this.spherical.radius * 20 + 100);
    this.camera.updateProjectionMatrix();
  }

  get radius(): number { return this.spherical.radius; }
  consumeDirty(): boolean {
    const d = this.dirty;
    this.dirty = false;
    return d;
  }
  isTransitioning(): boolean { return this.transition !== null; }
}
