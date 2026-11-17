#!/usr/bin/env node
/**
 * Headless fixed-trajectory runner.
 *
 * Starts the production server (or uses --url) and drives the *real UI* in
 * headless Chromium. The trajectory itself is executed by the page's own
 * kernel via window.scanStream.runTrajectory(), so this script records what
 * the streaming layer did, not what a parallel simulation claims it would do.
 *
 * Usage:
 *   npm run build && npm run trajectory
 *   npm run trajectory -- --url http://127.0.0.1:8080 --out reports --visible
 *
 * Exit code is non-zero when coverage shortfall exceeds --max-shortfall or a
 * waypoint failed to settle, so this can gate a release.
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
const maxShortfall = Number(arg("max-shortfall", "0.02"));
const dataset = arg("dataset", "assembly");
const budget = Number(arg("budget", "1200000"));
const sse = Number(arg("sse", "2.5"));
const pidFile = arg("pid-file", null);

/** Find a usable playwright-core install, preferring a local dependency. */
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
    } catch {
      // try next
    }
  }
  throw new Error(
    "playwright-core not found. Install it with `npm i -D playwright` or run "
    + "`npx playwright install chromium` first.",
  );
}

/** Extra shared libraries are needed when Chromium is not distro-installed. */
function libraryPath() {
  const candidates = [
    "/tmp/debs/root/usr/lib/aarch64-linux-gnu",
    "/tmp/debs/root/lib/aarch64-linux-gnu",
    "/tmp/debs/root/usr/lib/x86_64-linux-gnu",
    "/tmp/debs/root/lib/x86_64-linux-gnu",
  ];
  return candidates.filter((p) => fs.existsSync(p));
}

async function startServer() {
  const child = spawn(process.execPath, [path.join(root, "server", "index.mjs")], {
    cwd: root,
    env: { ...process.env, PORT: "8123", HOST: "127.0.0.1" },
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
  return { child, url: "http://127.0.0.1:8123" };
}

const { chromium } = await loadPlaywright();
const libs = libraryPath();
const env = { ...process.env };
if (libs.length) {
  env.LD_LIBRARY_PATH = [...libs, env.LD_LIBRARY_PATH ?? ""].filter(Boolean).join(":");
}

let server = null;
let baseUrl = useUrl;
if (!baseUrl) {
  try {
    server = await startServer();
    baseUrl = server.url;
    console.log(`started production server on ${baseUrl}`);
  } catch (err) {
    console.error(String(err));
    process.exit(1);
  }
}

fs.mkdirSync(outDir, { recursive: true });
const screenshot = path.join(outDir, "trajectory-final.png");
const reportPath = path.join(outDir, `trajectory-${dataset}.json`);

const browser = await chromium.launch({
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--no-sandbox",
    "--disable-dev-shm-usage",
  ],
  env,
});
const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
page.on("console", (msg) => {
  if (msg.type() === "error") console.log(`  [page error] ${msg.text()}`);
});
page.on("pageerror", (err) => console.log(`  [page exception] ${err.message}`));

let exitCode = 0;
try {
  await page.goto(`${baseUrl}/`, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForFunction(() => window.scanStream?.ready === true, null, { timeout: 60000 });
  // Wait for the first cut to be resident so the start state is stable.
  await page.waitForFunction(
    () => (window.scanStream.stats()?.drawnPoints ?? 0) > 0,
    null,
    { timeout: 60000 },
  );

  const glInfo = await page.evaluate(() => {
    const gl = document.querySelector("#gl").getContext("webgl2");
    return gl
      ? { webgl2: true, renderer: gl.getParameter(gl.RENDERER) }
      : { webgl2: false, renderer: null };
  });
  console.log(`WebGL2: ${glInfo.webgl2 ? "yes" : "NO"} (${glInfo.renderer ?? "-"})`);
  if (!glInfo.webgl2) throw new Error("WebGL2 unavailable in this browser");

  await page.evaluate((name) => window.scanStream.loadDataset(name), dataset);
  await page.waitForFunction(
    (n) => window.scanStream.dataset() === n
      && (window.scanStream.stats()?.drawnPoints ?? 0) > 0,
    dataset,
    { timeout: 90000 },
  );
  await page.evaluate(([points, tolerance]) => window.scanStream.setBudget(points, tolerance), [budget, sse]);

  console.log(`dataset ${dataset}: budget ${budget.toLocaleString()} pts, SSE ${sse}px`);
  const t0 = Date.now();
  const report = await page.evaluate(() => window.scanStream.runTrajectory());
  console.log(`trajectory finished in ${Date.now() - t0} ms`);

  if (!report) throw new Error("trajectory returned no report");
  fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

  console.log("\nwaypoints:");
  for (const w of report.waypoints) {
    console.log(
      `  ${String(w.index).padStart(2)} ${w.label.padEnd(34)} `
      + `drawn ${String(w.drawnPoints).padStart(9)}  `
      + `resident ${String(w.residentPoints).padStart(9)}  `
      + `seg ${(w.segmentBytes / 1e6).toFixed(1).padStart(6)} MB  `
      + `frame ${String(w.frameMs).padStart(7)} ms  `
      + `progressive ${(w.shortfallAtMinFrames * 100).toFixed(1).padStart(5)}% → `
      + `${w.settled ? "settled" : "NOT SETTLED"}`,
    );
  }
  console.log(
    `\npeak drawn ${report.peakDrawnPoints.toLocaleString()} pts, `
    + `peak resident ${(report.peakResidentBytes / 1e6).toFixed(1)} MB, `
    + `loaded ${(report.totalLoadedBytes / 1e6).toFixed(1)} MB in ${report.totalRequests} requests `
    + `(${report.totalAborted} aborted, ${report.totalFailed} failed)`,
  );
  console.log(
    `frame median ${report.medianFrameMs} ms, p95 ${report.p95FrameMs} ms, `
    + `progressive shortfall ${(report.maxShortfallAtMinFrames * 100).toFixed(1)}%, `
    + `settled shortfall ${(report.maxCoverageShortfall * 100).toFixed(2)}%`,
  );

  await page.screenshot({ path: screenshot });
  console.log(`\nreport: ${reportPath}\nscreenshot: ${screenshot}`);

  if (report.maxCoverageShortfall > maxShortfall) {
    console.error(`\nFAIL: settled coverage shortfall ${report.maxCoverageShortfall} > ${maxShortfall}`);
    exitCode = 2;
  }
  if (report.waypointsNotSettled > 0) {
    console.error(`\nFAIL: ${report.waypointsNotSettled} waypoint(s) did not settle within the frame cap`);
    exitCode = 4;
  }
  if (report.totalFailed > 0) {
    console.error(`\nFAIL: ${report.totalFailed} payload requests failed`);
    exitCode = 3;
  }
} finally {
  await browser.close();
  if (server) server.child.kill("SIGTERM");
  if (pidFile) {
    try { fs.unlinkSync(pidFile); } catch { /* nothing */ }
  }
}
process.exit(exitCode);
