# Scan Stream — out-of-core scan point-cloud viewer

A complete data-preparation + streaming chain for program-generated dense point
sets: an octree of *disjoint* LOD payloads, a screen-error/budget-driven
selector, an epoch-based residency cache, and a Three.js viewport whose controls
are wired to the streaming kernel.

The headless trajectory run writes a screenshot to `reports/trajectory-final.png`.

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
  child goes away, so there is never a visible hole during a transition.
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
npm test                 # 42 tests: kernel selection/cache + dataset integrity
npm run build            # strict typecheck + production bundle
npm run trajectory       # real Chromium + WebGL2 end-to-end run
```

The test suite covers: ancestor closure, monotone refinement under decreasing
tolerance, hard budget caps that stay ancestor-closed, disjoint payloads across
all three datasets, thin-rod coverage, identity/attribute coupling (including
tamper detection), and separate resident/in-flight accounting with abort and
backoff behaviour.

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
