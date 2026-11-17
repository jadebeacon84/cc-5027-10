# Scan Stream — out-of-core scan point-cloud viewer

A complete data-preparation + streaming chain for program-generated dense point
sets: an octree of *disjoint* LOD payloads, a screen-error/budget-driven
selector, an epoch-based residency cache, and a Three.js viewport whose controls
are wired to the streaming kernel.

The headless trajectory run writes a screenshot to `reports/trajectory-final.png`.

## Graphical entry point for the scan point cloud

The scan viewer is a **browser application**; the CLI is auxiliary only.

```bash
npm run dev      # development: http://localhost:5173  (scan point cloud viewport)
npm start        # production:  http://127.0.0.1:8080   (after `npm run build`)
```

Open that URL to get the scene-streaming viewport. Pick a scan dataset from the
**Dataset** selector (`small` / `assembly` / `cliff`), then **wheel-zoom** to
refine the point set and drag the **Point budget** / **Screen error tolerance**
sliders to control how much detail streams in. The point set grows
progressively as payloads arrive; the **Point-set integrity** panel marks any
progressive-point-set or detail-change anomaly (stale draw slot, coverage hole,
retained replacement) and the **Scan stream** HUD row shows whether the scan is
still streaming or has settled.

## Quick start

Requirements: Node 20+ and a WebGL2 browser.

```bash
npm install
npm run data:prepare     # generate + validate the three datasets (~1 min)
npm run dev              # http://localhost:5173
```

`npm run data:prepare` writes `public/data/<dataset>/{manifest,meta}.json` and
`nodes/*.bin`, and **fails the build** if any coverage/identity/attribute check
does not pass. Datasets are deterministic (seeded), so the payload checksums are
reproducible.

Production build and server:

```bash
npm run build            # tsc --noEmit && vite build  → dist/
npm start                # http://127.0.0.1:8080
```

### What the UI is for

| Control | What it drives |
|---|---|
| **Dataset** | reloads the manifest and kernel for `small` / `assembly` / `cliff` |
| **Point budget** | hard cap on points in the selected cut; the cut gets coarser, never holey |
| **Screen error tolerance** | screen-space error threshold that decides refinement |
| **Byte budget / auto** | resident-cache capacity; auto derives it from the point budget |
| **Display modes** | colour, intensity, scan station, normals, LOD-depth |
| **Point size / brightness** | renderer uniforms |
| **Force depth** | diagnostic cap on refinement depth |
| **Highlight unrefined regions** | wireframe boxes on every cut node held coarse by the budget |
| **Focus worst region** | focuses the largest-error coarse node and restricts the cut to its box |
| **Run trajectory** | scripted zoom/orbit path; records per-waypoint bytes, resident points, frame time |
| **Latency injection** | delays every payload fetch, to exercise cancellation |
| **Cancel loads** | aborts all in-flight requests, keeps residents |
| **Fail next load** | next two fetches throw; the kernel keeps the coarse ancestor |
| **Coverage probe** | rays through the viewport; reports hits vs rays through empty space |
| **Point-set integrity** | per-frame draw-list evidence: stale slots, blocked retirements, coverage holes, retained bytes |
| **HUD / LOD state / depth chart** | kernel `FrameStats`: drawn/selected/resident/in-flight, residency, per-depth points |

Controls: drag orbits, shift-drag (or right-drag) pans, wheel zooms toward the
cursor, click pins a point into the inspector, `H` home, `F` fit, `Esc` unpin.

## How the streaming works

### 1. Data preparation (`scripts/`)

`generators.ts` produces seeded scans with analytic ground truth: planes,
spheres, cylinders, thin rods and cables, density cliffs, and foliage-like
noise. Each point has a stable `pointId` and co-located colour, normal,
intensity, range and scan-station index.

`octree-builder.ts` builds a **disjoint-strata** hierarchy:

* Each node keeps the `nodeCapacity` lowest-ranked points of its own subtree
  (rank = bit-mixed `stableRank(pointId)`), and recurses the remainder into
  octants. A point belongs to exactly one payload node.
* Therefore **any ancestor-closed cut is duplicate-free by construction**: no
  region is ever drawn at two densities at once.
* It also means refinement is purely additive. A parent payload is never
  deleted to make room for a child; the child is added once it is resident.

Each payload is a struct-of-arrays blob with a 96-byte header and per-array
FNV-1a checksums. The header stores the exact byte offset of every attribute
array, so the loader can prove that the fetched bytes match the manifest's point
count and layout before any GPU upload happens. `validator.ts` independently
replays the partition and compares ids/attributes to generator truth (checks C1
– C7: identity, bounds, attributes, capacity, partition replay, ancestor-closed
cuts, density monotonicity).

### 2. Selection (`src/kernel/hierarchy.ts`)

Two phases per frame:

1. **Error cut** — DFS from the root; a node is refined when its
   screen-space error `spacing · projectionFactor / distance` exceeds the
   tolerance. Selected nodes are pushed *before* their children, so the cut is
   ancestor-closed. `geometricError` is the node's *own* payload spacing, which
   is the error visible while that node is the finest level drawn.
2. **Budget trim** — if the cut exceeds the point or byte budget, childless
   nodes are removed deepest-first and only when no selected child depends on
   them. The budget is a hard cap and the result is still ancestor-closed: a
   budget-limited frame is **coarse, never holey**.

Nodes that wanted to refine but have no selected children are returned as
`deferred`; that is exactly what the UI's unrefined-region panel and wireframes
show.

### 3. Residency and in-flight scheduling (`src/kernel/cache.ts`)

Resident bytes and in-flight bytes are metered separately.

* **Epoch cancellation**: a request is issued under the selection epoch; the
  next reconcile aborts every in-flight request whose node left the cut and
  returns its bytes to the in-flight meter immediately. Late responses are
  dropped. Rapid zooming therefore cannot let stale detail starve the current
  view.
* **Safe eviction**: a resident outside the cut is only evicted once its nearest
  ancestor *in the new cut* is resident. The parent is on screen before the
  child goes away, so there is never a visible hole during a transition. The
  same rule gates the byte-budget pass: while a replacement is still loading the
  cache may sit slightly above its cap rather than tear a hole, and the excess
  is reported as `retainedBytes` instead of hidden. A replacement that keeps
  failing (3 attempts) releases the retention, so reclamation is never disabled
  wholesale.
* **Generations and use completion**: every residency gets a monotonic
  generation id, carried by draw entries, eviction events and release calls. A
  late upload or stale release belonging to a previous residency of a recycled
  slot is rejected (counted as `staleUploads` / `staleRemovals` /
  `staleResponses`), and a payload a draw list still references (`drawUses > 0`)
  cannot be reclaimed.
* **Load order**: coarse-first (by depth), then priority, under a concurrency
  cap and a separate in-flight byte cap.
* **Failure policy**: a failed payload gets exponential backoff and its ancestor
  stays on screen. The failure is surfaced in the UI's Failure feedback panel.

### 4. Upload and draw (`src/render/point-renderer.ts`)

One `THREE.Points` per resident payload, bound directly to the typed-array views
of the fetched buffer. There is no per-point repack, so position, colour, normal
and id cannot drift apart in a batch. `aId` stays a `Uint32` attribute for
picking; station indices are promoted to floats because WebGL2 float attributes
cannot source integer buffers.

The engine uploads *every resident node in the current cut* each frame, then
removes rendered nodes no longer in the cut. Because the cut is ancestor-closed
and the cache protects target nodes, removal can never precede the replacement.

## Reproducible trajectory

The browser UI exposes `window.scanStream.runTrajectory()`, which runs the same
kernel through a fixed camera path built from `meta.json` (home → plane → thin
rod → curved surface → density edge → tight-budget wide view → home), varying
the error tolerance per stop. Each waypoint records:

* bytes fetched and requests issued/aborted/failed in that segment,
* drawn, selected, pending, resident points and resident/in-flight bytes,
* frame time, median and p95 for the settle window,
* coverage shortfall at the end of the fixed window (progressive state) and
  after adaptive settling.

Run it headless (requires `playwright` or `playwright-core`; the script finds a
local or npx install):

```bash
npm run build
npm run trajectory -- --dataset assembly --out reports
```

It writes `reports/trajectory-<dataset>.json` and a screenshot, and exits
non-zero on settled coverage shortfall > 2%, unsettled waypoints, or failed
fetches.

## Verification

```bash
npm run data:validate    # re-validate existing datasets against regenerated truth
npm test                 # kernel selection/cache + dataset integrity
npm run test:reclaim     # headless point-block reclamation evidence (CLI, auxiliary)
npm run build            # strict typecheck + production bundle
npm run trajectory       # real Chromium + WebGL2 end-to-end run
npm run viewport:check   # real Chromium: zoom + budget change in the viewport (after build)
```

`npm run test:reclaim` is the **auxiliary** headless check for the budget-
reclamation fix. It drives the real `ResidencyCache` through the two conditions
that used to flash a hole — a payload retiring while its replacement is still in
flight, and a stale block that must still be reclaimed down to the byte cap —
and prints, per step, the drawn node/slot count, resident and retained bytes,
the resident-byte peak, and the union/fingerprint of the point ids behind the
draw slots. It exits non-zero on a violation. The graphical viewport above is
the primary way to inspect the same behaviour.

`npm run viewport:check` complements it by driving the actual viewport page:
real wheel zoom and budget controls under injected latency, recording every
animation frame's draw evidence. It fails if any frame has a stale draw slot, a
coverage hole, or a stale upload/removal, and writes
`reports/viewport-zoom-budget.json`.

The test suite covers: ancestor closure, monotone refinement under decreasing
tolerance, hard budget caps that stay ancestor-closed, disjoint payloads across
all three datasets, thin-rod coverage, identity/attribute coupling (including
tamper detection), separate resident/in-flight accounting with abort and
backoff behaviour, and **budget reclamation**: a retiring block is retained
until its replacement is resident, a stale block is still reclaimed to the cap,
a late response for a superseded request is dropped, generations are monotonic,
an in-use block is not reclaimed, and retention is released once the
replacement has failed repeatedly.

## Layout

```
src/shared/     geometry + binary format (header, checksums, stable rank)
src/kernel/     hierarchy selection, residency cache, loader, streaming engine
src/render/     Three.js points renderer, orbit camera
src/ui/         app controller + fixed trajectory (runs in the page)
scripts/        generators, octree builder, validator, data prep, dev/trajectory
server/         production static server for dist/
tests/          kernel + dataset integrity tests
public/data/    generated datasets (manifest, meta, nodes/*.bin)
```
