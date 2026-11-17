#!/usr/bin/env node
/**
 * Auxiliary headless check of the scan viewer's *graphical* path.
 *
 * The primary entry point for the scan point cloud is the browser viewport
 * (`npm run dev` / `npm start`); this script is not a second renderer. It
 * drives that same page in headless Chromium — real wheel zoom, real budget
 * controls, real kernel — and samples `window.scanStream.evidence()` so the
 * progressive point set and detail changes can be checked against the patched
 * reclamation computation.
 *
 * Usage:
 *   npm run build && npm run viewport:check [-- --url http://127.0.0.1:8080]
 *
 * Exit non-zero if a stale draw slot, a coverage hole, or a stale upload /
 * removal is observed (all are expected to be zero).
 */
import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

function arg(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const v = process.argv[i + 1];
  return v === undefined || v.startsWith("--") ? true : v;
}
const useUrl = arg("url", null);
const outDir = path.resolve(root, arg("out", "reports"));
const dataset = arg("dataset", "small");

async function loadPlaywright() {
  const candidates = [
    "playwright",
    "playwright-core",
    "/home/node/.npm/_npx/e41f203b7505f1fb/node_modules/playwright-core/index.mjs",
  ];
  for (const spec of candidates) {
    try {
      const mod = spec.endsWith(".mjs")
        ? await import(pathToFileURL(spec).href)
        : await import(spec);
      if (mod.chromium) return mod;
    } catch { /* try next */ }
  }
  throw new Error("playwright-core not found. Install with `npm i -D playwright`.");
}

function libraryPath() {
  return [
    "/tmp/debs/root/usr/lib/aarch64-linux-gnu",
    "/tmp/debs/root/lib/aarch64-linux-gnu",
    "/tmp/debs/root/usr/lib/x86_64-linux-gnu",
    "/tmp/debs/root/lib/x86_64-linux-gnu",
  ].filter((p) => fs.existsSync(p));
}

async function startServer() {
  const child = spawn(process.execPath, [path.join(root, "server", "index.mjs")], {
    cwd: root,
    env: { ...process.env, PORT: "8127", HOST: "127.0.0.1" },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stderr = "";
  child.stderr.on("data", (d) => { stderr += String(d); });
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`server did not start:\n${stderr}`)), 15000);
    child.stdout.on("data", (d) => {
      if (String(d).includes("serving")) { clearTimeout(timer); resolve(); }
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      reject(new Error(`server exited with ${code}:\n${stderr}`));
    });
  });
  return { child, url: "http://127.0.0.1:8127" };
}

const { chromium } = await loadPlaywright();
const libs = libraryPath();
const env = { ...process.env };
if (libs.length) env.LD_LIBRARY_PATH = [...libs, env.LD_LIBRARY_PATH ?? ""].filter(Boolean).join(":");

let server = null;
let baseUrl = useUrl;
if (!baseUrl) {
  server = await startServer();
  baseUrl = server.url;
  console.log(`started production server on ${baseUrl}`);
}

fs.mkdirSync(outDir, { recursive: true });
const evidencePath = path.join(outDir, "viewport-zoom-budget.json");
const screenshot = path.join(outDir, "viewport-zoom-budget.png");

const browser = await chromium.launch({
  args: [
    "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    "--no-sandbox", "--disable-dev-shm-usage",
  ],
  env,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
let exitCode = 0;
try {
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => window.scanStream?.ready === true, null, { timeout: 60000 });
  await page.evaluate((n) => window.scanStream.loadDataset(n), dataset);
  await page.waitForFunction(
    (n) => window.scanStream.dataset() === n
      && (window.scanStream.stats()?.drawnPoints ?? 0) > 0,
    dataset, { timeout: 90000 },
  );
  // Inject latency through the UI's own control so replacements arrive late.
  await page.evaluate(() => {
    const el = document.querySelector("#latency");
    el.value = "350";
    el.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await page.evaluate(() => window.scanStream.setBudget(1_200_000, 4.0));
  await page.waitForTimeout(700);

  // Record every animation frame, not just the sampled steps: a one-frame
  // regression between samples would otherwise be invisible.
  await page.evaluate(() => {
    window.__frames = [];
    const rec = () => {
      const s = window.scanStream.stats();
      const e = window.scanStream.evidence();
      if (s && e) {
        window.__frames.push({
          t: Math.round(performance.now()),
          drawnPoints: s.drawnPoints,
          selectedNodes: s.selectedNodes,
          residentNodes: s.residentNodes,
          evictions: s.evictions,
          inflightRequests: s.inflightRequests,
          staleDraws: e.staleDraws,
          uncoveredNodes: e.uncoveredNodes,
          pendingCoverageNodes: e.pendingCoverageNodes,
          staleUploads: e.renderer.staleUploads,
          staleRemovals: e.renderer.staleRemovals,
        });
      }
      if (window.__tracing) requestAnimationFrame(rec);
    };
    window.__tracing = true;
    requestAnimationFrame(rec);
  });

  const canvas = await page.$("#gl");
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);

  const rows = [];
  const sample = async (label) => {
    const r = await page.evaluate(() => {
      const s = window.scanStream.stats();
      const e = window.scanStream.evidence();
      return {
        label: "",
        drawnPoints: s.drawnPoints, drawnNodes: s.drawnNodes, selectedNodes: s.selectedNodes,
        residentBytes: s.residentBytes, residentNodes: s.residentNodes,
        inflightRequests: s.inflightRequests, pendingNodes: s.pendingNodes,
        evictions: s.evictions, requestsAborted: s.requestsAborted,
        staleResponses: s.staleResponses,
        retainedBytes: e.retainedBytes, retainedNodes: e.retainedNodes,
        staleDraws: e.staleDraws, drawnRetiring: e.drawnRetiring,
        uncoveredNodes: e.uncoveredNodes, pendingCoverageNodes: e.pendingCoverageNodes,
        rootResident: e.rootResident, rootInCut: e.rootInCut,
        staleUploads: e.renderer.staleUploads, staleRemovals: e.renderer.staleRemovals,
      };
    });
    r.label = label;
    rows.push(r);
    console.log(
      `${label.padEnd(16)} drawn ${String(r.drawnPoints).padStart(7)} `
      + `cut ${String(r.selectedNodes).padStart(4)} inflight ${String(r.inflightRequests).padStart(3)} `
      + `gap ${String(r.uncoveredNodes).padStart(3)} loading ${String(r.pendingCoverageNodes).padStart(3)} `
      + `root ${r.rootResident ? "res" : "-  "} stale ${r.staleDraws}`,
    );
    return r;
  };

  await sample("start wide");
  await page.waitForTimeout(350);
  await sample("wide settled");

  // Rapid zoom in with a mid-flight budget tighten.
  for (let i = 0; i < 5; i++) { await page.mouse.wheel(0, -120); await page.waitForTimeout(40); }
  await page.evaluate(() => window.scanStream.setBudget(220_000, 1.0));
  await sample("zoom in + cut");
  await page.waitForTimeout(200);
  await sample("mid flight");

  // Zoom back out, then shrink hard while the previous view is still arriving.
  for (let i = 0; i < 3; i++) { await page.mouse.wheel(0, 240); await page.waitForTimeout(40); }
  await page.evaluate(() => window.scanStream.setBudget(120_000, 6.0));
  await sample("zoom out");
  await page.waitForTimeout(250);
  await page.evaluate(() => window.scanStream.setBudget(60_000, 0.5));
  await sample("hard shrink");

  // Relax and let everything land.
  await page.evaluate(() => window.scanStream.setBudget(900_000, 1.0));
  await page.waitForTimeout(500);
  for (let i = 0; i < 120; i++) {
    const pending = await page.evaluate(() => {
      const s = window.scanStream.stats();
      return s ? s.inflightRequests + s.pendingNodes : 1;
    });
    if (pending === 0) break;
    await page.waitForTimeout(100);
  }
  await page.waitForTimeout(300);
  const settled = await sample("settled");

  await page.screenshot({ path: screenshot });
  await page.evaluate(() => { window.__tracing = false; });
  const frames = await page.evaluate(() => window.__frames);
  fs.writeFileSync(evidencePath, JSON.stringify({ samples: rows, frames }, null, 2));

  const grew = rows.some((r, i) => i > 0 && r.drawnPoints > rows[i - 1].drawnPoints);
  const anomalies = rows.filter((r) =>
    r.staleDraws > 0 || r.uncoveredNodes > 0 || r.staleUploads > 0 || r.staleRemovals > 0);
  // A frame may legitimately draw nothing immediately after a big zoom while
  // the first payloads are still loading; the invariant is that it is *loading*
  // (pending coverage), not a hole (uncovered with nothing in flight).
  const frameAnomalies = frames.filter((f) =>
    f.staleDraws > 0
    || (f.uncoveredNodes > 0 && f.pendingCoverageNodes === 0 && f.inflightRequests === 0)
    || f.staleUploads > 0 || f.staleRemovals > 0);
  const blankFrames = frames.filter((f) => f.drawnPoints === 0);
  const blankLoading = blankFrames.filter((f) => f.pendingCoverageNodes > 0).length;
  console.log(`\nprogressive point-set growth observed: ${grew ? "yes" : "no"}`);
  console.log(`frames recorded: ${frames.length}; blank frames: ${blankFrames.length} `
    + `(all loading coverage: ${blankLoading === blankFrames.length ? "yes" : "no"})`);
  console.log(`settled: drawn ${settled.drawnPoints} pts, ${settled.drawnNodes} nodes, `
    + `gap ${settled.uncoveredNodes}, stale ${settled.staleDraws}`);
  if (anomalies.length > 0 || frameAnomalies.length > 0) {
    console.error(`\nFAIL: ${anomalies.length} sampled / ${frameAnomalies.length} per-frame `
      + `observation(s) with a stale draw slot, hole, or stale upload/removal`);
    console.error(JSON.stringify({ samples: anomalies, frames: frameAnomalies.slice(0, 20) }, null, 2));
    exitCode = 1;
  } else {
    console.log("PASS: no stale draw slots, coverage holes, or stale uploads/removals");
  }
  console.log(`evidence: ${evidencePath}\nscreenshot: ${screenshot}`);
} finally {
  await browser.close();
  if (server) server.child.kill("SIGTERM");
}
process.exit(exitCode);
