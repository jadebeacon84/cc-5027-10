/**
 * Fixed trajectory harness.
 *
 * Runs the kernel through a deterministic camera path and records, at each
 * waypoint, what the streaming layer actually did: bytes fetched during the
 * segment, resident points, drawn points, pending points, frame time and
 * whether the desired cut was fully resident. This runs in the page (the same
 * UI, same kernel) so it is not a separate rendering path.
 *
 * The path always starts from the dataset's home pose and ends at a list of
 * named targets chosen per dataset: analytic surfaces, thin rods and the
 * density-discontinuity region, so every stop has ground truth in meta.json.
 */
import type { StreamingEngine, FrameStats } from "../kernel/engine";
import type { DatasetMeta } from "../shared/format";

export type TrajectoryWaypoint = {
  index: number;
  label: string;
  eye: [number, number, number];
  target: [number, number, number];
  fov: number;
  requestedSse: number;
  /** Frames rendered while settling at this waypoint. */
  frames: number;
  /** Frames the fixed window ran before adaptive settling began. */
  minFrames: number;
  /** Extra frames rendered until the cut was fully resident (or the cap hit). */
  extraFrames: number;
  /** True if the cut became fully resident before the settle cap. */
  settled: boolean;
  /** Wall-clock time until the settled measurement (loaded + render). */
  settleMs: number;
  /** Bytes fetched from the network during this segment. */
  segmentBytes: number;
  segmentRequests: number;
  segmentAborted: number;
  segmentFailed: number;
  /** Snapshot at the settled frame. */
  drawnPoints: number;
  drawnNodes: number;
  selectedNodes: number;
  selectedPoints: number;
  pendingPoints: number;
  pendingNodes: number;
  residentPoints: number;
  residentBytes: number;
  inflightBytes: number;
  budgetLimited: boolean;
  residency: number;
  unrefinedNodes: number;
  frameMs: number;
  medianFrameMs: number;
  p95FrameMs: number;
  fps: number;
  /** Fraction of selected points that never arrived (loading shortfall). */
  coverageShortfall: number;
  /** Shortfall at the end of the fixed frame window (progressive view). */
  shortfallAtMinFrames: number;
};

export type TrajectoryReport = {
  dataset: string;
  title: string;
  startedAt: string;
  durationMs: number;
  totalMs: number;
  pointBudget: number;
  sseTolerance: number;
  waypoints: TrajectoryWaypoint[];
  totalLoadedBytes: number;
  totalRequests: number;
  totalCompleted: number;
  totalAborted: number;
  totalFailed: number;
  peakResidentBytes: number;
  peakInflightBytes: number;
  peakDrawnPoints: number;
  medianFrameMs: number;
  p95FrameMs: number;
  /** Worst (selected - drawn) / selected across waypoints. */
  maxCoverageShortfall: number;
  /** Worst shortfall measured at the end of the fixed frame window. */
  maxShortfallAtMinFrames: number;
  /** Waypoints that did not become fully resident within the settle cap. */
  waypointsNotSettled: number;
  framesTotal: number;
  notes: string;
};

export type TrajectoryStop = {
  label: string;
  eye: [number, number, number];
  target: [number, number, number];
  fov?: number;
  /** Screen-error tolerance while settling at this stop. */
  sse?: number;
};

export type TrajectorySpec = {
  name: string;
  title: string;
  /** Frames to render at each stop before sampling. */
  settleFrames: number;
  /** Additional frames allowed for in-flight payloads to arrive. */
  maxSettleFrames: number;
  stops: (meta: DatasetMeta) => TrajectoryStop[];
};

function around(
  target: [number, number, number], distance: number, yawDeg: number, pitchDeg: number,
): [number, number, number] {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  return [
    target[0] + distance * Math.cos(pitch) * Math.cos(yaw),
    target[1] + distance * Math.sin(pitch),
    target[2] + distance * Math.cos(pitch) * Math.sin(yaw),
  ];
}

function surfaceCentre(meta: DatasetMeta, name: string): [number, number, number] | null {
  const s = meta.surfaces.find((x) => x.name === name);
  if (!s) return null;
  if (s.kind === "plane") return s.origin;
  if (s.kind === "sphere") return s.center;
  if (s.kind === "cylinder") return s.center;
  if (s.kind === "rod") {
    return [
      (s.a[0] + s.b[0]) / 2,
      (s.a[1] + s.b[1]) / 2,
      (s.a[2] + s.b[2]) / 2,
    ];
  }
  if (s.kind === "lattice") {
    return [
      (s.bounds.min[0] + s.bounds.max[0]) / 2,
      (s.bounds.min[1] + s.bounds.max[1]) / 2,
      (s.bounds.min[2] + s.bounds.max[2]) / 2,
    ];
  }
  return null;
}

/**
 * Stops are built from meta.json, so they exist for every dataset and refer to
 * the same analytic features the validator uses as ground truth.
 */
export const TRAJECTORIES: Record<string, TrajectorySpec> = {
  sweep: {
    name: "sweep",
    title: "Home → surface → thin rod → density edge → home",
    settleFrames: 34,
    maxSettleFrames: 90,
    stops: (meta) => {
      const stops: TrajectoryStop[] = [
        { label: "home", eye: meta.home.eye, target: meta.home.target, fov: meta.home.fov, sse: 2.5 },
      ];
      // Prefer the named analytic features the generators always emit.
      const planeName = meta.surfaces.find((s) => s.kind === "plane")?.name;
      const rodName = meta.surfaces.find((s) => s.kind === "rod")?.name;
      const sphereName = meta.surfaces.find((s) => s.kind === "sphere")?.name;
      const latticeName = meta.surfaces.find((s) => s.kind === "lattice")?.name;
      // Density-discontinuity point sets carry a feature of kind "density"
      // with matching surface metadata; fall back to the last feature's ids.
      const dense = meta.features.find((f) => f.kind === "density" || f.kind === "noise");

      if (planeName) {
        const c = surfaceCentre(meta, planeName)!;
        stops.push({
          label: `surface: ${planeName}`,
          eye: around(c, 4.5, 35, 26),
          target: c,
          sse: 1.1,
        });
      }
      if (rodName) {
        const c = surfaceCentre(meta, rodName)!;
        stops.push({
          label: `thin rod: ${rodName}`,
          eye: around(c, 1.5, 62, 10),
          target: c,
          sse: 0.45,
        });
      }
      if (sphereName) {
        const c = surfaceCentre(meta, sphereName)!;
        stops.push({
          label: `curved surface: ${sphereName}`,
          eye: around(c, 3.2, -40, 22),
          target: c,
          sse: 0.6,
        });
      }
      if (latticeName) {
        const c = surfaceCentre(meta, latticeName)!;
        stops.push({
          label: `suspended lattice: ${latticeName}`,
          eye: around(c, 9, 120, 34),
          target: c,
          sse: 0.9,
        });
      }
      if (dense) {
        // Use the feature's own notes for a label, but aim at the dataset
        // bounds' high-density corner: the generators place density steps on
        // one side of the bounds.
        const b = meta.worldBounds;
        const c: [number, number, number] = [
          b.min[0] + (b.max[0] - b.min[0]) * 0.22,
          b.min[1] + (b.max[1] - b.min[1]) * 0.25,
          b.min[2] + (b.max[2] - b.min[2]) * 0.78,
        ];
        stops.push({
          label: `density edge: ${dense.name}`,
          eye: around(c, 6.5, -125, 20),
          target: c,
          sse: 0.7,
        });
      }
      // Zoom back out with a tight budget to shave detail, then return home.
      stops.push({
        label: "wide view (tight budget)",
        eye: around(meta.home.target, 24, 45, 30),
        target: meta.home.target,
        sse: 4,
      });
      stops.push({
        label: "home again",
        eye: meta.home.eye,
        target: meta.home.target,
        fov: meta.home.fov,
        sse: 2.5,
      });
      return stops;
    },
  },
};

function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(p * (sorted.length - 1))));
  return sorted[i];
}

function nextFrame(): Promise<number> {
  return new Promise((resolve) => requestAnimationFrame(resolve));
}

/**
 * Drive the engine through `spec`. The caller must stop its own render loop
 * for the duration: this function calls `engine.frame` itself so the recorded
 * frames are exactly the ones measured.
 */
export async function runTrajectory(
  engine: StreamingEngine,
  spec: TrajectorySpec,
  onProgress?: (percent: number, note: string) => void,
): Promise<TrajectoryReport> {
  const meta = engine.meta;
  const stops = spec.stops(meta);
  const startedAt = new Date().toISOString();
  const t0 = performance.now();

  engine.orbit.setPose({
    eye: meta.home.eye,
    target: meta.home.target,
    fov: meta.home.fov,
  });

  const waypoints: TrajectoryWaypoint[] = [];
  const allFrames: number[] = [];
  let framesTotal = 0;

  // The UI normally also applies the slider's budget; record what we ran with.
  const pointBudget = engine.pointBudget;
  const baseSse = engine.sseTolerance;

  let prev = engine.cache.snapshot();

  for (let i = 0; i < stops.length; i++) {
    const stop = stops[i];
    const segmentStart = performance.now();
    engine.sseTolerance = stop.sse ?? baseSse;
    engine.orbit.setPose({
      eye: stop.eye,
      target: stop.target,
      fov: stop.fov ?? 55,
    });

    const settleFrames = stop.label === "home" ? 8 : spec.settleFrames;
    const frameTimes: number[] = [];
    let stats: FrameStats | null = null;
    let shortfallAtMinFrames = 1;
    for (let f = 0; f < settleFrames; f++) {
      await nextFrame();
      const now = performance.now();
      const before = performance.now();
      stats = engine.frame(now);
      frameTimes.push(performance.now() - before);
      framesTotal++;
      allFrames.push(stats.dt);
      // Let a few network completions land between frames without relying on
      // wall-clock sleeps: the settle count is the only pacing.
    }
    if (!stats) continue;
    {
      const selected = engine.selectionSnapshot().result.drawnPoints;
      shortfallAtMinFrames = selected > 0
        ? Math.max(0, selected - stats.drawnPoints) / selected
        : 0;
    }
    // Adaptive settle: the measured state is the fully-loaded one, but the
    // per-waypoint record also keeps the progressive value, so a slow network
    // shows up as "arrived late" rather than as missing geometry.
    let extraFrames = 0;
    const maxExtra = Math.max(0, spec.maxSettleFrames - settleFrames);
    const wantResident = () => {
      const pending = engine.selectionSnapshot().result.selected
        .filter((node) => {
          const n = engine.hierarchy.nodes[node];
          return n.hasPayload && !engine.cache.has(node);
        }).length;
      return pending === 0;
    };
    while (extraFrames < maxExtra && !wantResident()) {
      await nextFrame();
      const before = performance.now();
      stats = engine.frame(performance.now());
      frameTimes.push(performance.now() - before);
      framesTotal++;
      allFrames.push(stats.dt);
      extraFrames++;
    }
    const settled = wantResident();
    const settleMs = performance.now() - segmentStart;

    const counters = engine.cache.snapshot();
    const segmentBytes = counters.loadedBytesTotal - prev.loadedBytesTotal;
    const segmentRequests = counters.requestsIssued - prev.requestsIssued;
    const segmentAborted = counters.requestsAborted - prev.requestsAborted;
    const segmentFailed = counters.requestsFailed - prev.requestsFailed;
    prev = counters;

    const selectedPoints = engine.selectionSnapshot().result.drawnPoints;
    const shortfall = selectedPoints > 0
      ? Math.max(0, selectedPoints - stats.drawnPoints) / selectedPoints
      : 0;

    waypoints.push({
      index: i,
      label: stop.label,
      eye: stop.eye,
      target: stop.target,
      fov: stop.fov ?? 55,
      requestedSse: engine.sseTolerance,
      frames: settleFrames + extraFrames,
      minFrames: settleFrames,
      extraFrames,
      settled,
      settleMs: Math.round(settleMs),
      segmentBytes,
      segmentRequests,
      segmentAborted,
      segmentFailed,
      drawnPoints: stats.drawnPoints,
      drawnNodes: stats.drawnNodes,
      selectedNodes: stats.selectedNodes,
      selectedPoints,
      pendingPoints: stats.pendingPoints,
      pendingNodes: stats.pendingNodes,
      residentPoints: stats.residentPoints,
      residentBytes: stats.residentBytes,
      inflightBytes: stats.inflightBytes,
      budgetLimited: stats.budgetLimited,
      residency: stats.residency,
      unrefinedNodes: stats.unrefinedNodes,
      frameMs: Math.round(stats.dt * 100) / 100,
      medianFrameMs: Math.round(percentile(frameTimes, 0.5) * 100) / 100,
      p95FrameMs: Math.round(percentile(frameTimes, 0.95) * 100) / 100,
      fps: stats.fps,
      coverageShortfall: shortfall,
      shortfallAtMinFrames,
    });
    onProgress?.(
      Math.round(((i + 1) / stops.length) * 100),
      stop.label,
    );
  }

  const counters = engine.cache.snapshot();
  const report: TrajectoryReport = {
    dataset: meta.dataset,
    title: meta.title,
    startedAt,
    durationMs: Math.round(performance.now() - t0),
    totalMs: Math.round(performance.now() - t0),
    pointBudget,
    sseTolerance: baseSse,
    waypoints,
    totalLoadedBytes: counters.loadedBytesTotal,
    totalRequests: counters.requestsIssued,
    totalCompleted: counters.requestsCompleted,
    totalAborted: counters.requestsAborted,
    totalFailed: counters.requestsFailed,
    peakResidentBytes: Math.max(...waypoints.map((w) => w.residentBytes)),
    peakInflightBytes: Math.max(...waypoints.map((w) => w.inflightBytes)),
    peakDrawnPoints: Math.max(...waypoints.map((w) => w.drawnPoints)),
    medianFrameMs: Math.round(percentile(allFrames, 0.5) * 100) / 100,
    p95FrameMs: Math.round(percentile(allFrames, 0.95) * 100) / 100,
    maxCoverageShortfall: Math.max(0, ...waypoints.map((w) => w.coverageShortfall)),
    maxShortfallAtMinFrames: Math.max(0, ...waypoints.map((w) => w.shortfallAtMinFrames)),
    waypointsNotSettled: waypoints.filter((w) => !w.settled).length,
    framesTotal,
    notes:
      "Each stop changes the camera and the kernel runs its normal select → reconcile → upload → draw pipeline. "
      + "Bytes are the loader's delta during the settle window; drawn points are what the GPU rendered in the last frame. "
      + "shortfallAtMinFrames is the progressive state at the fixed window; coverageShortfall is after adaptive settling.",
  };
  return report;
}
