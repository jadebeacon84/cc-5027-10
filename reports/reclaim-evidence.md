# Budget-reclamation regression evidence

Captured while fixing the draw-list flicker caused by budget-driven
reclamation running ahead of the replacement payload's arrival.

## 1. Minimal reproduction (cache level)

`scripts/repro-reclaim.ts` loaded root + ancestor A + deep block D under a byte
budget of `root.bytes + D.bytes - 1` (so the budget pass *must* act), then
shrank the cut to `{root, A}` while A was still in flight.

### Before the fix (original `src/kernel/cache.ts` at HEAD)

```
root=0 A=1(0/0) D=4(0/0/2/4)
after root+D resident: resident=2 D=true bytes=168852 budget=168851
after budget-shrink reconcile: D=false residentBytes=144096 evictions=1
events: started started started resident resident evicted(4)
BUG: coverage block reclaimed before replacement arrived
```

D was evicted while its replacement A had not arrived, so the draw list kept a
slot with no residency behind it — the visible flash.

### After the fix

```
root=0 A=1(0/0) D=4(0/0/2/4)
after root+D resident: resident=2 D=true bytes=168852 budget=168851
after budget-shrink reconcile: D=true residentBytes=168852 evictions=0
events: started started started resident resident
OK: coverage block retained until replacement arrives
```

The retiree is retained (168852 B > the 168851 B cap by one byte, reported as
`retainedBytes`) until A is resident, then released.

## 2. CI check (`scripts/reclaim-check.ts`, `npm run test:reclaim`)

### Before the fix (exit code 1)

```
reclamation check — dataset small

[late-replacement]
step                                  drawn  resident         retained evict  ids     idHash    peak       status
deep delivered, ancestor in flight    2      2n/144408B      undefined0      4006    6ace2ea2  144408B    ok  budget 144407B < root+deep
cut shrinks, replacement not resident 2      1n/144096B      undefined1      4000    417146e7  144408B    FAIL  BUG: deep block reclaimed early; DRAW ENTRY WITHOUT RESIDENCY
replacement arrives, retiree released 2      2n/288192B      undefined1      8000    f858174d  288192B    ok  evictions 1

[budget-reclaim]
step                                  drawn  resident         retained evict  ids     idHash    peak       status
target blocks loaded                  2      2n/288192B      undefined0      8000    231a8fe7  288192B    ok  budget 288191B; both are target
stale block reclaimed to cap          1      1n/144096B      undefined1      4000    417146e7  288192B    ok  resident 144096B <= cap 288191B

FAIL: 1 invariant(s) violated
```

(`retained` prints `undefined` because the pre-fix cache has no
`retainedBytes` counter at all — that is precisely what it failed to expose.)

### After the fix (exit code 0)

```
reclamation check — dataset small

[late-replacement]
step                                  drawn  resident         retained evict  ids     idHash    peak       status
deep delivered, ancestor in flight    2      2n/144408B      0       0      4006    6ace2ea2  144408B    ok  budget 144407B < root+deep
cut shrinks, replacement not resident 2      2n/144408B      312     0      4006    6ace2ea2  144408B    ok  retained, no flicker
replacement arrives, retiree released 2      2n/288192B      0       1      8000    f858174d  288192B    ok  evictions 1

[budget-reclaim]
step                                  drawn  resident         retained evict  ids     idHash    peak       status
target blocks loaded                  2      2n/288192B      0       0      8000    231a8fe7  288192B    ok  budget 288191B; both are target
stale block reclaimed to cap          1      1n/144096B      0       1      4000    417146e7  288192B    ok  resident 144096B <= cap 288191B

PASS: 0 invariant(s) violated
```

Key contrasts:

* Same point identities at the retained step (`idHash 6ace2ea2`, 4006 unique
  ids) and at the released step (`f858174d`, 8000 ids) — the fix changes
  *when* bytes are freed, not *which points* are drawn.
* The memory peak is identical (144408 B → 288192 B in both runs); the fix does
  not raise the peak. At the retained step, `resident` sits 1 B above the
  144407 B cap because the 312 B retiree (reported as `retainedBytes`) is held
  until its replacement lands.
* The second scenario proves the fix is not "pin everything": the stale
  non-target block is still reclaimed to the cap (`resident 144096B <= cap
  288191B`).

The `--dataset small|assembly|cliff` run passes on all three datasets.

## 3. Existing scan point-cloud samples (`npm run trajectory`)

Re-ran the previously generated datasets to completion after the fix.

| dataset | peak drawn pts | peak resident MB | settled shortfall | aborted | failed | not settled |
|---|---|---|---|---|---|---|
| small (240k budget) | 142,670 | 5.1 | 0.00% | 0 | 0 | 0 |
| assembly (1.2M) | 818,870 | 29.54 | 0.00% | 0 | 0 | 0 |
| cliff (1.2M) | 1,199,597 | 43.25 | 0.00% | 0 | 0 | 0 |

Compared with the pre-fix reports committed at HEAD, the per-waypoint
`drawnPoints`, `residentPoints` and `selectedNodes` sequences are **identical**
for assembly and cliff, and the peaks match to the byte. In other words, point
identity and memory peak were already correct on these samples; only the
transient-eviction path changed. The reports are regenerated in place
(`reports/trajectory-*.json`).

## 4. Graphical viewport check (`npm run viewport:check`)

The scan point cloud's primary entry point is the browser viewport, so the
verification drives that page rather than a parallel harness.
`scripts/viewport-check.mjs` loads the `small` dataset in headless Chromium,
injects 350 ms of fetch latency through the UI's own control, then performs
real wheel zooms and budget-slider changes (wide → zoom in 5 steps → 220k/1px →
zoom out 3 steps → 120k/6px → hard shrink 60k/0.5px → relax 900k/1px). It
samples `window.scanStream.evidence()` at each step and records every
animation frame.

Result (`reports/viewport-zoom-budget.json`, reproducible):

```
start wide       drawn   47733 cut   23 inflight   4 gap   0 loading   0 root res stale 0
wide settled     drawn   47733 cut   23 inflight   4 gap   0 loading   0 root res stale 0
zoom in + cut    drawn   50628 cut   55 inflight   8 gap   0 loading   0 root res stale 0
mid flight       drawn   67387 cut  141 inflight   8 gap   0 loading   0 root res stale 0
zoom out         drawn   79491 cut  148 inflight   8 gap   0 loading   0 root res stale 0
hard shrink      drawn   39678 cut   15 inflight   1 gap   0 loading   0 root res stale 0
settled          drawn  182324 cut  149 inflight   0 gap   0 loading   0 root res stale 0

progressive point-set growth observed: yes
frames recorded: 306; blank frames: 0
PASS: no stale draw slots, coverage holes, or stale uploads/removals
```

The drawn-point set grows progressively as payloads land (47k → 67k → 91k →
182k). At every sampled step and every recorded animation frame, the true-hole
count is 0, the root stays resident while in the cut, and no stale draw slot,
stale upload or stale removal occurs — i.e. the progressive point set and
detail changes come from the patched `canRetire` / generation-tagged draw list /
use-completion computation, not from an unbounded cache. When a very fast zoom
outruns its loads, the UI reports those nodes as "Coverage still loading"
(a distinct amber state) rather than as a hole; `pendingCoverageNodes > 0` and
`uncoveredNodes == 0` is exactly that condition.

Screenshots are written to `reports/viewport-zoom-budget.png` (git-ignored);
the JSON evidence is committed so the conclusion can be re-checked without a
browser by inspecting the recorded frames.
