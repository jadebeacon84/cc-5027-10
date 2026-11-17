/**
 * UI layer. Every control talks to the streaming kernel; no control
 * reimplements LOD or loading logic. Results flow back from the kernel's
 * per-frame stats, cache events and selection snapshot.
 */
import * as THREE from "three";
import { StreamingEngine, type DrawEvidence, type FrameStats } from "../kernel/engine";
import type { CacheEvent } from "../kernel/cache";
import type { ManifestBundle } from "../kernel/loader";
import type { HoverHit, PointCloudDisplay } from "../render/point-renderer";
import { aabbCenter } from "../shared/geo";
import {
  runTrajectory, TRAJECTORIES,
  type TrajectoryReport, type TrajectorySpec,
} from "./trajectory";

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el as T;
};

function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
function fmtBytes(n: number): string {
  if (!Number.isFinite(n)) return "∞";
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)} kB`;
  return `${Math.round(n)} B`;
}
function fmtDur(n: number): string {
  return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${n.toFixed(1)} ms`;
}

type AppState = {
  dataset: string;
  bundle: ManifestBundle;
  engine: StreamingEngine;
  gl: THREE.WebGLRenderer;
  running: boolean;
  raf: number;
  showUnrefined: boolean;
  autoBudget: boolean;
  pinned: HoverHit | null;
  lastHover: HoverHit | null;
  hoverDirty: boolean;
  report: TrajectoryReport | null;
  busy: boolean;
  unrefinedGroup: THREE.Group;
};

let state: AppState | null = null;
let lastHud = 0;
let toastTimer = 0;
/** Shared across dataset switches: recreating WebGL contexts leaks them. */
let sharedGl: THREE.WebGLRenderer | null = null;

async function boot() {
  const index = await fetch("/data/index.json").then((r) => {
    if (!r.ok) throw new Error(`index.json: HTTP ${r.status}`);
    return r.json();
  }) as {
    datasets: Record<string, { totalPoints: number; nodeCount: number; title: string }>;
  };
  const select = $<HTMLSelectElement>("#dataset");
  const names = Object.keys(index.datasets).sort((a, b) => {
    if (a === "assembly") return -1;
    if (b === "assembly") return 1;
    return a.localeCompare(b);
  });
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `${index.datasets[name].title} — ${fmtInt(index.datasets[name].totalPoints)} pts`;
    select.appendChild(opt);
  }
  const initial = names.includes("assembly") ? "assembly" : names[0];
  select.value = initial;
  select.addEventListener("change", () => {
    if (!state?.busy) void loadDataset(select.value);
  });

  const canvas = $<HTMLCanvasElement>("#gl");
  sharedGl = new THREE.WebGLRenderer({
    canvas, antialias: false, powerPreference: "high-performance",
  });
  sharedGl.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  sharedGl.setClearColor(0x05070a, 1);

  bindControls();
  await loadDataset(initial);
  window.addEventListener("resize", () => { if (state) resizeCanvas(state); });
  requestAnimationFrame(loop);
}

async function loadDataset(name: string) {
  const canvas = $<HTMLCanvasElement>("#gl");
  const gl = sharedGl;
  if (!gl) throw new Error("renderer not initialised");
  // Retain the user's point budget across dataset switches, clamped below.
  const previousBudget = state
    ? Number($<HTMLInputElement>("#point-budget").value)
    : null;
  if (state) {
    state.running = false;
    cancelAnimationFrame(state.raf);
    state.unrefinedGroup.removeFromParent();
    state.engine.dispose();
    state = null;
  }
  setStatus(`Loading ${name}…`);
  setError("");
  let bundle: ManifestBundle;
  try {
    bundle = await fetch(`/data/${name}/manifest.json`).then(async (r) => {
      if (!r.ok) throw new Error(`manifest HTTP ${r.status}`);
      const manifest = await r.json();
      const metaRes = await fetch(`/data/${name}/meta.json`);
      if (!metaRes.ok) throw new Error(`meta HTTP ${metaRes.status}`);
      return { manifest, meta: await metaRes.json(), baseUrl: `/data/${name}` };
    }) as ManifestBundle;
  } catch (err) {
    showToast(`Failed to load dataset ${name}: ${String(err)}`, false);
    setStatus("Load failed");
    setError(String(err));
    return;
  }

  const engine = new StreamingEngine({
    baseUrl: `/data/${name}`,
    manifest: bundle.manifest,
    meta: bundle.meta,
    renderer: gl,
    canvas,
    onEvent: handleCacheEvent,
  });
  const unrefinedGroup = new THREE.Group();
  engine.renderer.scene.add(unrefinedGroup);

  state = {
    dataset: name,
    bundle,
    engine,
    gl,
    running: true,
    raf: 0,
    showUnrefined: false,
    autoBudget: true,
    pinned: null,
    lastHover: null,
    hoverDirty: false,
    report: null,
    busy: false,
    unrefinedGroup,
  };

  // Dataset-specific control ranges and defaults.
  const total = bundle.manifest.totalPoints;
  const pointSlider = $<HTMLInputElement>("#point-budget");
  pointSlider.min = String(Math.min(50_000, total));
  pointSlider.max = String(Math.max(100_000, total));
  pointSlider.step = String(Math.max(10_000, Math.round((total / 500) / 1000) * 1000));
  pointSlider.value = String(
    previousBudget === null
      ? Math.min(bundle.meta.defaultBudget, total)
      : Math.min(previousBudget, total),
  );
  const byteSlider = $<HTMLInputElement>("#byte-budget");
  const sseSlider = $<HTMLInputElement>("#sse");
  const auto = $<HTMLInputElement>("#auto-budget");
  engine.applyBudgets(Number(pointSlider.value), Number.POSITIVE_INFINITY);
  byteSlider.max = String(Math.max(256, Math.ceil((engine.residentByteBudget * 3) / 1e6 / 32) * 32));
  byteSlider.value = String(Math.round(engine.residentByteBudget / 1e6));
  const depthSlider = $<HTMLInputElement>("#force-depth");
  depthSlider.max = String(bundle.manifest.maxDepth);
  depthSlider.value = "-1";
  $("#depth-label").textContent = "auto";
  auto.checked = true;
  byteSlider.disabled = true;
  $<HTMLInputElement>("#show-unrefined").checked = false;
  $<HTMLInputElement>("#latency").value = "0";
  $("#latency-label").textContent = "0 ms";
  // Carry the current tolerance/byte controls over to the new dataset rather
  // than leaving the engine at its constructor defaults.
  engine.setBudgets(
    Number(pointSlider.value),
    auto.checked ? Number.POSITIVE_INFINITY : Number(byteSlider.value) * 1e6,
    Number(sseSlider.value),
  );
  $("#traj-status").textContent = "Idle";
  $("#traj-status").className = "traj-status";
  $("#traj-metrics").innerHTML = "";
  $("#coverage-metrics").innerHTML = "";
  $("#failure-metrics").innerHTML = "";
  $("#integrity-metrics").innerHTML = "";
  lastIntegrityKey = "";
  $("#toast").className = "toast";
  $("#status-errors").textContent = "";
  $<HTMLButtonElement>("#btn-export-traj").disabled = true;

  resizeCanvas(state);
  engine.warmUp(4);
  updateBudgetLabels();
  updateModeUi("colour");
  updateHoverPanel();
  setStatus(
    `${bundle.manifest.title} — ${fmtInt(total)} points · `
    + `${bundle.manifest.nodeCount} nodes · depth ${bundle.manifest.maxDepth}`,
  );
  $("#dataset-meta").textContent =
    `${fmtInt(total)} pts · ${bundle.manifest.payloadNodeCount} payloads · `
    + fmtBytes(bundle.manifest.nodes.reduce((a, n) => a + n.bytes, 0));

  if (state) {
    state.running = true;
    cancelAnimationFrame(state.raf);
    state.raf = requestAnimationFrame(loop);
  }
}

function loop(now: number) {
  const s = state;
  if (!s || !s.running) return;
  s.raf = requestAnimationFrame(loop);
  frame(s, now);
}

function frame(s: AppState, now: number) {
  const stats = s.engine.frame(now);
  if (now - lastHud > 90) {
    lastHud = now;
    updateHud(s, stats);
    updateHoverPanel();
    updateUnrefinedPanel(s);
  }
}

function resizeCanvas(s: AppState) {
  const vp = $<HTMLElement>("#viewport");
  const w = Math.max(1, vp.clientWidth);
  const h = Math.max(1, vp.clientHeight);
  s.gl.setSize(w, h, false);
  s.engine.resize();
}

function updateHud(s: AppState, st: FrameStats) {
  const evidence = s.engine.drawEvidence();
  $("#hud-points").textContent = fmtInt(st.drawnPoints);
  $("#hud-budget").textContent = fmtInt(st.pointBudget);
  $("#hud-resident").textContent = fmtBytes(st.residentBytes);
  $("#hud-inflight").textContent = fmtBytes(st.inflightBytes);
  $("#hud-nodes").textContent = `${fmtInt(st.drawnNodes)} / ${fmtInt(st.selectedNodes)}`;
  $("#hud-cam").textContent = st.cameraDistance.toFixed(2);
  const stream = streamState(st, evidence);
  $("#hud-stream-state").textContent = stream.label;
  $("#hud-stream-state").className = stream.cls;
  $("#hud-fps").textContent = String(st.fps);
  $("#hud-frame").textContent = fmtDur(st.dt);
  $("#hud-sse").textContent = st.sseTolerance.toFixed(2);
  const ratio = st.selectedPointsExpected > 0 ? st.drawnPoints / st.selectedPointsExpected : 1;
  $("#loading-fill").style.width = `${Math.round(Math.min(1, ratio) * 100)}%`;
  $("#status-selection").textContent =
    `cut ${st.selectedNodes} nodes · ${fmtInt(st.selectedPointsExpected)} pts target · `
    + `resident ${fmtInt(st.residentPoints)} pts · inflight ${fmtInt(st.inflightPoints)} pts · `
    + `aborted ${st.requestsAborted} · ${st.budgetLimited ? "budget-limited" : "error-driven"}`;

  $("#budget-metrics").innerHTML = metricRows([
    ["Resident bytes", fmtBytes(st.residentBytes), "ok"],
    ["In-flight bytes", fmtBytes(st.inflightBytes),
      st.inflightBytes > st.residentBytes * 0.5 ? "warn" : ""],
    ["Resident / in-flight nodes", `${st.residentNodes} / ${st.inflightRequests}`, ""],
    ["Loaded this session", fmtBytes(st.loadedBytesTotal), ""],
    ["Requests done / abort / fail",
      `${st.requestsCompleted} / ${st.requestsAborted} / ${st.requestsFailed}`,
      st.requestsFailed > 0 ? "bad" : ""],
    ["Evictions", fmtInt(st.evictions), ""],
    ["Budget-limited", st.budgetLimited ? "yes" : "no", st.budgetLimited ? "warn" : "good"],
  ]);

  const staleSlots = evidence.staleDraws;
  const uncovered = evidence.uncoveredNodes;
  $("#integrity-metrics").innerHTML = metricRows([
    ["Drawn nodes", fmtInt(evidence.drawnNodes), ""],
    ["Stale draw slots", String(staleSlots), staleSlots > 0 ? "bad" : "good"],
    ["Replacing (retained)", String(evidence.drawnRetiring),
      evidence.drawnRetiring > 0 ? "warn" : "good"],
    ["Blocked retirements", String(evidence.unresolvableRetirements),
      evidence.unresolvableRetirements > 0 ? "warn" : "good"],
    ["Coverage holes", String(uncovered), uncovered > 0 ? "bad" : "good"],
    ["Coverage still loading", String(evidence.pendingCoverageNodes),
      evidence.pendingCoverageNodes > 0 ? "warn" : "good"],
    ["Root coverage", evidence.rootResident ? "resident" : evidence.rootInCut ? "loading" : "not in cut",
      evidence.rootResident ? "good" : "warn"],
    ["Retained bytes", fmtBytes(evidence.retainedBytes),
      evidence.retainedBytes > 0 ? "warn" : "good"],
    ["Stale uploads / removals",
      `${evidence.renderer.staleUploads} / ${evidence.renderer.staleRemovals}`,
      evidence.renderer.staleUploads + evidence.renderer.staleRemovals > 0 ? "warn" : "good"],
    ["Late responses dropped", String(evidence.staleResponses), ""],
  ]);
  markIntegrityAnomalies(evidence);

  $("#lod-metrics").innerHTML = metricRows([
    ["Drawn nodes", fmtInt(st.drawnNodes), ""],
    ["Selected (cut)", fmtInt(st.selectedNodes), ""],
    ["Coarse regions", fmtInt(st.unrefinedNodes), st.unrefinedNodes > 0 ? "warn" : "good"],
    ["Pending nodes", fmtInt(st.pendingNodes), st.pendingNodes > 0 ? "warn" : "good"],
    ["Residency", `${(st.residency * 100).toFixed(1)}%`, st.residency > 0.98 ? "good" : "warn"],
    ["GPU buffer", fmtBytes(st.gpuBytes), ""],
    ["Select / upload", `${fmtDur(st.selectMs)} / ${fmtDur(st.uploadMs)}`, ""],
  ]);
  renderDepthChart(st.leafDepthHistogram);

  const loader = s.engine.loader.stats;
  const failed = s.engine.cache.failureCount();
  $("#failure-metrics").innerHTML = metricRows([
    ["Failed nodes (session)", String(failed), failed > 0 ? "bad" : "good"],
    ["Requests failed", String(st.requestsFailed), st.requestsFailed > 0 ? "bad" : "good"],
    ["Verification failures", String(loader.verificationFailures),
      loader.verificationFailures > 0 ? "bad" : "good"],
    ["HTTP failures", String(loader.failures), loader.failures > 0 ? "warn" : "good"],
    ["Average load", fmtDur(loader.avgMs), ""],
    ["Last load", fmtDur(loader.lastMs), ""],
  ]);
}

/**
 * Human-readable scan-stream state: distinguishes "nothing left to load" from
 * "still streaming" and from the anomalous states (holes, stale slots).
 */
function streamState(st: FrameStats, e: DrawEvidence): { label: string; cls: string } {
  if (e.staleDraws > 0 || e.uncoveredNodes > 0) return { label: "ANOMALY", cls: "bad" };
  if (st.inflightRequests > 0 || e.pendingCoverageNodes > 0) {
    return { label: `streaming (${st.inflightRequests} pending)`, cls: "warn" };
  }
  if (st.pendingNodes > 0) return { label: `waiting (${st.pendingNodes})`, cls: "warn" };
  if (st.requestsFailed > 0) return { label: "settled · failures", cls: "warn" };
  return { label: "settled", cls: "good" };
}

/**
 * Surface progressive-point-set / detail-change anomalies outside the metric
 * list: a stale draw slot or a coverage hole is a rendering fault, not a
 * budget hint, so it is also written to the status line and a toast.
 */
let lastIntegrityKey = "";
function markIntegrityAnomalies(e: DrawEvidence) {
  const key = `${e.staleDraws}|${e.uncoveredNodes}|${e.unresolvableRetirements}`;
  if (key === lastIntegrityKey) return;
  lastIntegrityKey = key;
  const anomalies: string[] = [];
  if (e.staleDraws > 0) anomalies.push(`${e.staleDraws} stale draw slot(s)`);
  if (e.uncoveredNodes > 0) anomalies.push(`${e.uncoveredNodes} uncovered region(s)`);
  if (anomalies.length === 0) {
    if ($("#status-errors").textContent.startsWith("draw list:")) setError("");
    return;
  }
  const msg = `draw list: ${anomalies.join(", ")}`;
  setError(msg);
  showToast(`${msg}. Expected to be zero; inspect the point-set integrity panel.`, false);
}

function metricRows(rows: Array<[string, string, string]>): string {
  return rows.map(([k, v, cls]) =>
    `<div class="m ${cls}"><span>${k}</span><b>${v}</b></div>`).join("");
}

function renderDepthChart(counts: number[]) {
  const el = $("#depth-chart");
  const max = Math.max(1, ...counts);
  el.innerHTML = counts.map((c, i) =>
    `<div class="bar" style="height:${Math.max(2, Math.round((c / max) * 100))}%"`
    + ` title="depth ${i}: ${fmtInt(c)} pts"></div>`).join("");
}

function updateHoverPanel() {
  const s = state;
  if (!s) return;
  if (s.pinned) {
    renderHit(s.pinned, "pinned");
    return;
  }
  if (!s.hoverDirty) return;
  s.hoverDirty = false;
  const hit = s.lastHover;
  if (!hit) {
    $("#hover-empty").style.display = "";
    $("#hover-empty").textContent = "Move the cursor over the cloud; click to pin a point.";
    $("#hover-fields").innerHTML = "";
    return;
  }
  renderHit(hit, "hover");
}

function renderHit(hit: HoverHit, mode: string) {
  $("#hover-empty").style.display = "none";
  $("#hover-fields").innerHTML = [
    ["mode", mode],
    ["point id", String(hit.id)],
    ["node", hit.node >= 0 ? `#${hit.node}` : "-"],
    ["position", `${hit.x.toFixed(2)}, ${hit.y.toFixed(2)}, ${hit.z.toFixed(2)}`],
    ["rgb", `${hit.r}, ${hit.g}, ${hit.b}`],
    ["scan station", String(hit.scan)],
    ["range", `${hit.distance.toFixed(2)} m`],
    ["intensity", hit.intensity.toFixed(2)],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function updateUnrefinedPanel(s: AppState) {
  const deferred = s.engine.selectionSnapshot().deferred;
  const summary = $("#unrefined-summary");
  const st = s.engine.lastFrame();
  if (deferred.length === 0) {
    summary.textContent = "All visible regions refined to the error tolerance.";
  } else {
    summary.textContent =
      `${deferred.length} region${deferred.length === 1 ? "" : "s"} held coarse by the budget`
      + `${st && st.pendingNodes > 0 ? `, ${st.pendingNodes} node(s) still loading` : ""}.`;
  }
  summary.className = "muted";
  $<HTMLButtonElement>("#btn-focus-unrefined").disabled = deferred.length === 0;
  if (s.showUnrefined) drawUnrefinedBoxes(s, deferred);
}

/**
 * One wireframe box per deferred region: nodes the error criterion wanted to
 * refine but the point budget refused. This is the visible evidence that the
 * cut is coarse, not that the region is empty.
 */
function drawUnrefinedBoxes(s: AppState, deferred: number[]) {
  const group = s.unrefinedGroup;
  for (const c of [...group.children]) {
    group.remove(c);
    const lines = c as THREE.LineSegments;
    lines.geometry.dispose();
    (lines.material as THREE.Material).dispose();
  }
  for (const node of deferred.slice(0, 160)) {
    const b = s.engine.hierarchy.nodes[node].bounds;
    const box = new THREE.Box3(
      new THREE.Vector3(b.min[0], b.min[1], b.min[2]),
      new THREE.Vector3(b.max[0], b.max[1], b.max[2]),
    );
    const helper = new THREE.Box3Helper(box, 0xffb454);
    (helper.material as THREE.LineBasicMaterial).depthTest = false;
    helper.renderOrder = 10_000;
    group.add(helper);
  }
}

function updateLegendForMode(mode: PointCloudDisplay) {
  const el = $("#legend");
  if (mode === "intensity") {
    el.innerHTML = `<div class="ramp" style="background:linear-gradient(90deg,#0d1a4d,#1ab8db,#ffe680,#ffffff)"></div>
      <div class="muted">Low reflectance → high reflectance</div>`;
  } else if (mode === "lod") {
    el.innerHTML = `<div class="ramp" style="background:linear-gradient(90deg,#f05c38,#52dbff)"></div>
      <div class="muted">Coarse ancestor → leaf-level detail</div>`;
  } else if (mode === "scan") {
    const n = state?.bundle.meta.scanCount ?? 1;
    const dots = Array.from({ length: Math.min(8, Math.max(2, n)) }, (_, i) => {
      const t = i / Math.max(1, Math.min(7, n - 1));
      const r = 0.45 + 0.45 * Math.cos(6.2831853 * t);
      const g = 0.45 + 0.45 * Math.cos(6.2831853 * (t + 0.33));
      const b = 0.45 + 0.45 * Math.cos(6.2831853 * (t + 0.67));
      return `<i class="dot" style="background:rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})"></i>`;
    }).join(" ");
    el.innerHTML = `<div class="swatch">${dots}</div><div class="muted">${n} scan stations</div>`;
  } else if (mode === "normal") {
    el.innerHTML = `<div class="muted">Surface normal quantised to int8 and mapped to RGB.</div>`;
  } else {
    el.innerHTML = state
      ? state.bundle.meta.features.slice(0, 7)
        .map((f) => `<span class="badge">${f.kind}: ${f.name}</span>`).join(" ")
      : "";
  }
}

function updateModeUi(mode: PointCloudDisplay) {
  for (const b of document.querySelectorAll("#display-modes button")) b.classList.remove("active");
  const btn = document.querySelector(`#display-modes button[data-mode="${mode}"]`);
  if (btn) btn.classList.add("active");
  updateLegendForMode(mode);
}

function updateBudgetLabels() {
  if (!state) return;
  const pointSlider = $<HTMLInputElement>("#point-budget");
  const sseSlider = $<HTMLInputElement>("#sse");
  const byteSlider = $<HTMLInputElement>("#byte-budget");
  $("#point-budget-label").textContent = fmtInt(Number(pointSlider.value));
  $("#sse-label").textContent = Number(sseSlider.value).toFixed(2);
  $("#byte-budget-label").textContent = `${Math.round(Number(byteSlider.value))} MB`;
}

function handleCacheEvent(event: CacheEvent) {
  if (!state) return;
  if (event.kind === "failed") {
    const node = state.bundle.manifest.nodes[event.node];
    showToast(
      `Payload failed: ${node ? node.path : event.node} — ${event.error}. Coarse coverage kept.`,
      false,
    );
    setError(`load failure: ${event.error}`);
  }
}

function showToast(message: string, info = true) {
  const el = $("#toast");
  el.textContent = message;
  el.className = `toast show${info ? " info" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => { el.className = "toast"; }, 4500);
}

function setStatus(text: string) { $("#status-text").textContent = text; }
function setError(text: string) { $("#status-errors").textContent = text; }
function clearFocusBox() {
  if (state) state.engine.setFocusBox(null);
}

function bindControls() {
  const canvas = $<HTMLCanvasElement>("#gl");
  const pointSlider = $<HTMLInputElement>("#point-budget");
  const sseSlider = $<HTMLInputElement>("#sse");
  const byteSlider = $<HTMLInputElement>("#byte-budget");
  const auto = $<HTMLInputElement>("#auto-budget");

  const syncBudgets = () => {
    const s = state;
    if (!s || s.busy) return;
    const points = Number(pointSlider.value);
    const sse = Number(sseSlider.value);
    if (auto.checked) {
      s.engine.setBudgets(points, Number.POSITIVE_INFINITY, sse);
      byteSlider.value = String(Math.round(s.engine.residentByteBudget / 1e6));
    } else {
      s.engine.setBudgets(points, Number(byteSlider.value) * 1e6, sse);
    }
    s.autoBudget = auto.checked;
    updateBudgetLabels();
  };
  sseSlider.value = "2.5";
  auto.addEventListener("change", () => {
    byteSlider.disabled = auto.checked;
    syncBudgets();
  });
  pointSlider.addEventListener("input", syncBudgets);
  sseSlider.addEventListener("input", syncBudgets);
  byteSlider.addEventListener("input", syncBudgets);

  const size = $<HTMLInputElement>("#point-size");
  size.addEventListener("input", () => {
    $("#size-label").textContent = Number(size.value).toFixed(1);
    if (state) state.engine.setPointSize(Number(size.value));
  });
  const bright = $<HTMLInputElement>("#brightness");
  bright.addEventListener("input", () => {
    $("#bright-label").textContent = Number(bright.value).toFixed(2);
    if (state) state.engine.setBrightness(Number(bright.value));
  });
  const depth = $<HTMLInputElement>("#force-depth");
  depth.addEventListener("input", () => {
    const v = Number(depth.value);
    $("#depth-label").textContent = v < 0 ? "auto" : String(v);
    if (state) state.engine.setForcedDepth(v);
  });
  $<HTMLInputElement>("#show-unrefined").addEventListener("change", (e) => {
    const s = state;
    if (!s) return;
    s.showUnrefined = (e.target as HTMLInputElement).checked;
    if (!s.showUnrefined) {
      for (const c of [...s.unrefinedGroup.children]) {
        s.unrefinedGroup.remove(c);
        const lines = c as THREE.LineSegments;
        lines.geometry.dispose();
        (lines.material as THREE.Material).dispose();
      }
    } else {
      updateUnrefinedPanel(s);
    }
  });

  $("#display-modes").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest("button") as HTMLElement | null;
    if (!btn || !state) return;
    const mode = btn.dataset.mode as PointCloudDisplay;
    state.engine.setDisplay(mode);
    updateModeUi(mode);
  });

  $("#btn-home").addEventListener("click", () => {
    const s = state;
    if (!s) return;
    clearFocusBox();
    const home = s.bundle.meta.home;
    s.engine.orbit.flyTo({ eye: home.eye, target: home.target, fov: home.fov }, 650);
  });
  $("#btn-fit").addEventListener("click", () => {
    const s = state;
    if (!s) return;
    clearFocusBox();
    const b = s.bundle.manifest.bounds;
    const c = aabbCenter(b);
    const radius = Math.hypot(
      b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2],
    ) / 2;
    s.engine.orbit.flyTo({
      eye: [c[0] + radius * 1.5, c[1] + radius * 1.0, c[2] + radius * 1.5],
      target: c,
      fov: 55,
    }, 650);
  });
  $("#btn-cancel").addEventListener("click", () => {
    if (!state) return;
    state.engine.cancelInflight();
    showToast("Cancelled all in-flight payload requests; residents kept.");
  });

  // Pointer interaction: orbit (drag), pan (shift/right drag), zoom (wheel),
  // hover inspect, click to pin.
  let dragging = false;
  let moved = false;
  let panMode = false;
  let lastX = 0;
  let lastY = 0;

  const ndcAt = (e: PointerEvent | WheelEvent) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * 2 - 1,
      y: -((e.clientY - rect.top) / rect.height) * 2 + 1,
    };
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (!state) return;
    dragging = true;
    moved = false;
    panMode = e.button === 2 || e.shiftKey;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    const s = state;
    if (!s) return;
    if (dragging) {
      const dx = e.clientX - lastX;
      const dy = e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      if (Math.hypot(dx, dy) > 2) moved = true;
      if (panMode) s.engine.orbit.pan(dx, dy, canvas.clientHeight);
      else s.engine.orbit.orbit(dx, dy);
      return;
    }
    const ndc = ndcAt(e);
    s.lastHover = pickAt(s, ndc.x, ndc.y);
    s.hoverDirty = true;
  });
  canvas.addEventListener("pointerup", (e) => {
    const s = state;
    if (!s) { dragging = false; return; }
    const wasDragging = dragging;
    dragging = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* not captured */ }
    if (wasDragging && !moved) {
      const ndc = ndcAt(e);
      const hit = pickAt(s, ndc.x, ndc.y);
      if (hit) {
        s.pinned = hit;
        s.hoverDirty = true;
        updateHoverPanel();
      }
    }
  });
  canvas.addEventListener("pointercancel", () => { dragging = false; });
  canvas.addEventListener("pointerleave", () => {
    if (!state || dragging) return;
    state.lastHover = null;
    state.hoverDirty = true;
  });
  canvas.addEventListener("wheel", (e) => {
    if (!state) return;
    e.preventDefault();
    state.engine.orbit.zoom(e.deltaY > 0 ? 1.12 : 1 / 1.12, ndcAt(e));
  }, { passive: false });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());

  window.addEventListener("keydown", (e) => {
    if (!state) return;
    const target = e.target as HTMLElement | null;
    if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
    if (e.key === "h" || e.key === "H") $("#btn-home").click();
    else if (e.key === "f" || e.key === "F") $("#btn-fit").click();
    else if (e.key === "Escape") {
      state.pinned = null;
      state.hoverDirty = true;
      updateHoverPanel();
    }
  });

  $("#btn-focus-unrefined").addEventListener("click", () => {
    const s = state;
    if (!s) return;
    const deferred = s.engine.selectionSnapshot().deferred;
    if (deferred.length === 0) return;
    let worst = deferred[0];
    for (const n of deferred) {
      if (s.engine.hierarchy.nodes[n].geometricError > s.engine.hierarchy.nodes[worst].geometricError) {
        worst = n;
      }
    }
    const node = s.engine.hierarchy.nodes[worst];
    const c = aabbCenter(node.bounds);
    const r = Math.hypot(
      node.bounds.max[0] - node.bounds.min[0],
      node.bounds.max[1] - node.bounds.min[1],
      node.bounds.max[2] - node.bounds.min[2],
    ) / 2;
    s.engine.setFocusBox(node.bounds);
    s.engine.orbit.flyTo({
      eye: [c[0] + r * 2.2, c[1] + r * 1.5, c[2] + r * 2.2],
      target: c,
      fov: 55,
    }, 700);
    showToast(
      `Focused coarse region at depth ${node.depth}. Press Home or Fit to clear the region focus.`,
    );
  });

  $("#btn-run-traj").addEventListener("click", () => { void runSelectedTrajectory(); });
  $("#btn-export-traj").addEventListener("click", exportTrajectory);
  $("#latency").addEventListener("input", () => {
    const ms = Number($<HTMLInputElement>("#latency").value);
    $("#latency-label").textContent = `${ms} ms`;
    if (state) state.engine.loader.setLatency(ms);
  });
  $("#btn-fail").addEventListener("click", () => {
    const s = state;
    if (!s) return;
    const target = s.engine.injectLoadFailure();
    if (!target) {
      showToast("No resident cut leaf to re-fetch yet; wait for the view to settle and retry.", true);
      return;
    }
    showToast(`Re-fetching ${target.path} with an injected failure. Its ancestor stays on screen.`);
  });
  $("#btn-reload").addEventListener("click", () => {
    const s = state;
    if (!s || s.busy) return;
    void loadDataset(s.dataset);
  });
  $("#btn-coverage").addEventListener("click", () => { if (state) runCoverageProbe(state); });
  updateBudgetLabels();
}

function pickAt(s: AppState, ndcX: number, ndcY: number): HoverHit | null {
  const camera = s.engine.orbit.camera;
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
  // Pick radius scales with view size so thin rods stay selectable up close
  // while a distant pixel does not claim the whole cloud.
  const tolerance = Math.max(1e-3, s.engine.orbit.radius * 0.006);
  return s.engine.renderer.pick(
    raycaster.ray.origin, raycaster.ray.direction,
    s.engine.orbit.radius * 8, tolerance,
  );
}

async function runSelectedTrajectory() {
  const s = state;
  if (!s || s.busy) return;
  const spec: TrajectorySpec = TRAJECTORIES.sweep;
  const runBtn = $<HTMLButtonElement>("#btn-run-traj");
  const status = $("#traj-status");
  s.busy = true;
  s.running = false;
  cancelAnimationFrame(s.raf);
  runBtn.disabled = true;
  status.className = "traj-status running";
  status.textContent = "Running…";
  try {
    const report = await runTrajectory(s.engine, spec, (progress, note) => {
      status.textContent = `${progress}% — ${note}`;
      const last = s.engine.lastFrame();
      if (last) {
        $("#traj-metrics").innerHTML = metricRows([
          ["Waypoint", note, ""],
          ["Drawn points", fmtInt(last.drawnPoints), ""],
          ["Resident", fmtBytes(last.residentBytes), ""],
          ["Frame", fmtDur(last.dt), ""],
          ["Requests done / aborted", `${last.requestsCompleted} / ${last.requestsAborted}`, ""],
        ]);
      }
    });
    s.report = report;
    status.className = "traj-status done";
    status.textContent = `Done — ${report.waypoints.length} waypoints in ${fmtDur(report.durationMs)}`;
    $("#traj-metrics").innerHTML = metricRows([
      ["Waypoints", String(report.waypoints.length), ""],
      ["Total bytes loaded", fmtBytes(report.totalLoadedBytes), ""],
      ["Requests issued", String(report.totalRequests), ""],
      ["Aborted", String(report.totalAborted), report.totalAborted > 0 ? "warn" : "good"],
      ["Failed", String(report.totalFailed), report.totalFailed > 0 ? "bad" : "good"],
      ["Peak resident", fmtBytes(report.peakResidentBytes), ""],
      ["Peak in-flight", fmtBytes(report.peakInflightBytes), ""],
      ["Peak drawn points", fmtInt(report.peakDrawnPoints), ""],
      ["Median frame", fmtDur(report.medianFrameMs), ""],
      ["p95 frame", fmtDur(report.p95FrameMs), report.p95FrameMs > 40 ? "warn" : "good"],
      ["Coverage shortfall", `${(report.maxCoverageShortfall * 100).toFixed(2)}%`,
        report.maxCoverageShortfall > 0.02 ? "warn" : "good"],
    ]);
    $<HTMLButtonElement>("#btn-export-traj").disabled = false;
    showToast(
      `Trajectory complete: ${fmtInt(report.peakDrawnPoints)} peak drawn points, `
      + `p95 frame ${fmtDur(report.p95FrameMs)}.`,
    );
  } catch (err) {
    status.className = "traj-status";
    status.textContent = `Failed — ${String(err)}`;
    setError(`trajectory: ${String(err)}`);
  } finally {
    s.busy = false;
    runBtn.disabled = false;
    // Restore the slider's tolerance, which the trajectory varies per stop.
    s.engine.setBudgets(
      Number($<HTMLInputElement>("#point-budget").value),
      s.autoBudget
        ? Number.POSITIVE_INFINITY
        : Number($<HTMLInputElement>("#byte-budget").value) * 1e6,
      Number($<HTMLInputElement>("#sse").value),
    );
    s.running = true;
    s.raf = requestAnimationFrame(loop);
  }
}

function exportTrajectory() {
  const s = state;
  if (!s?.report) return;
  const blob = new Blob([JSON.stringify(s.report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `trajectory-${s.dataset}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  showToast("Trajectory report downloaded.");
}

/**
 * Screen-space coverage probe: cast a grid of rays, keep the ones that pass
 * through the dataset's bounding sphere, and count how many land on a drawn
 * point. Rays through a region that should be dense but hit nothing are the
 * visual signature of holes.
 */
function runCoverageProbe(s: AppState) {
  const camera = s.engine.orbit.camera;
  const bounds = s.bundle.manifest.bounds;
  const res = 32;
  let hits = 0;
  let misses = 0;
  let total = 0;
  const nodes = new Set<number>();
  const raycaster = new THREE.Raycaster();
  const sphere = new THREE.Sphere(
    new THREE.Vector3(
      (bounds.min[0] + bounds.max[0]) / 2,
      (bounds.min[1] + bounds.max[1]) / 2,
      (bounds.min[2] + bounds.max[2]) / 2,
    ),
    Math.hypot(
      bounds.max[0] - bounds.min[0],
      bounds.max[1] - bounds.min[1],
      bounds.max[2] - bounds.min[2],
    ) / 2,
  );
  const maxDist = s.engine.orbit.radius * 8;
  const tolerance = Math.max(0.01, s.engine.orbit.radius * 0.02);
  for (let y = 0; y < res; y++) {
    for (let x = 0; x < res; x++) {
      raycaster.setFromCamera(
        new THREE.Vector2(((x + 0.5) / res) * 2 - 1, ((y + 0.5) / res) * 2 - 1),
        camera,
      );
      const ray = raycaster.ray;
      if (!ray.intersectSphere(sphere, new THREE.Vector3())) continue;
      total++;
      const hit = s.engine.renderer.pick(ray.origin, ray.direction, maxDist, tolerance);
      if (hit) { hits++; nodes.add(hit.node); } else misses++;
    }
  }
  const pct = total > 0 ? (hits / total) * 100 : 100;
  $("#coverage-metrics").innerHTML = metricRows([
    ["Rays through model", String(total), ""],
    ["Rays hitting points", String(hits), ""],
    ["Rays through empty space", String(misses), misses > total * 0.25 ? "warn" : "good"],
    ["Hit rate", `${pct.toFixed(1)}%`, pct > 60 ? "good" : "warn"],
    ["Distinct nodes hit", String(nodes.size), ""],
  ]);
  showToast(`Coverage probe: ${hits}/${total} model rays hit drawn points (${pct.toFixed(1)}%).`);
}

/**
 * Automation hook used by scripts/trajectory.mjs. It exposes the same kernel
 * instance the UI uses, never a parallel render path.
 */
declare global {
  interface Window {
    scanStream?: {
      ready: boolean;
      dataset: () => string | null;
      stats: () => FrameStats | null;
      evidence: () => import("../kernel/engine").DrawEvidence | null;
      runTrajectory: () => Promise<TrajectoryReport | null>;
      setBudget: (points: number, sse: number) => void;
      loadDataset: (name: string) => Promise<void>;
    };
  }
}

function exposeAutomation() {
  window.scanStream = {
    ready: true,
    dataset: () => state?.dataset ?? null,
    stats: () => state?.engine.lastFrame() ?? null,
    runTrajectory: async () => {
      await runSelectedTrajectory();
      return state?.report ?? null;
    },
    setBudget: (points: number, sse: number) => {
      const s = state;
      if (!s) return;
      $<HTMLInputElement>("#point-budget").value = String(points);
      $<HTMLInputElement>("#sse").value = String(sse);
      s.engine.setBudgets(points, Number.POSITIVE_INFINITY, sse);
      updateBudgetLabels();
    },
    evidence: () => state?.engine.drawEvidence() ?? null,
    loadDataset: async (name: string) => {
      $<HTMLSelectElement>("#dataset").value = name;
      await loadDataset(name);
    },
  };
}

boot()
  .then(() => exposeAutomation())
  .catch((err) => {
    console.error(err);
    setStatus(`Startup failed: ${String(err)}`);
    setError(String(err));
    showToast(`Startup failed: ${String(err)}`, false);
  });
