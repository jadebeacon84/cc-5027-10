/**
 * Residency cache and request scheduler.
 *
 * Two budgets are tracked independently because they fail differently:
 *
 *   residentBytes  what is currently held for drawing. Evicting costs a
 *                  re-fetch later, so it uses an LRU with a byte cap.
 *   inflightBytes  what has been requested but has not arrived. Capped
 *                  separately: during a fast zoom the previous view's deep
 *                  nodes are no longer wanted, and their bytes must not keep
 *                  the pipe saturated while the current view starves.
 *
 * Cancellation is epoch-based. A request records the selection epoch it was
 * issued under. On reconcile, requests whose node left the target cut are
 * aborted, their bytes are returned to the in-flight meter, and late responses
 * are dropped on arrival. Stale detail therefore cannot displace current
 * detail.
 *
 * Use completion. A payload that a draw list still references is *in use*. It
 * cannot be reclaimed until the draw list has switched away from it, because
 * freeing the bytes under a live draw entry is exactly the flicker this cache
 * exists to prevent. `acquireDraw`/`releaseDraw` bracket that use; the
 * generation returned by `acquireDraw` must be passed back so a stale release
 * (one belonging to a previous residency of the same node) cannot unpin the
 * current one.
 *
 * Generations. Every residency gets a monotonically increasing generation.
 * Eviction events, draw requests and release calls carry it. When a node slot
 * is recycled, work belonging to the previous occupant is rejected instead of
 * being applied to the new payload — a late upload cannot make the draw list
 * show the old block's bytes in the new block's slot.
 *
 * Eviction safety. The target cut is always ancestor-closed (see hierarchy.ts),
 * so every drawn region has a coarse fallback on its ancestor chain. A
 * non-target payload is reclaimed only when
 *
 *   (a) no draw list uses it (use complete), and
 *   (b) its nearest ancestor in the new target cut is resident, or that
 *       ancestor has failed repeatedly and is no longer plausibly arriving.
 *
 * The budget pass obeys the same rule, so while a replacement is still loading
 * the cache may sit slightly above its byte cap instead of tearing a hole. That
 * excess is bounded by the previous cut (the retiree set) and by load latency,
 * and `retainedNodes`/`retainedBytes` expose it explicitly rather than hiding it
 * in a default. A replacement that keeps failing releases the retention, so
 * reclamation is never disabled wholesale. Eviction proceeds deepest-first, so
 * a whole retiring branch is removed topologically.
 *
 * Failure policy. A failed node gets exponential backoff so a broken payload
 * does not spin. Its ancestor remains resident, so failure degrades detail but
 * never leaves a hole.
 */
import type { PayloadViews } from "../shared/format";

export type NodeIndex = number;

export type CacheEntry = {
  node: NodeIndex;
  views: PayloadViews;
  bytes: number;
  points: number;
  lastUsedFrame: number;
  residentSince: number;
  /** Monotonic id of this residency. Stale users must ignore older values. */
  generation: number;
  /** Live draw-list references. Reclamation waits for this to reach zero. */
  drawUses: number;
};

export type RequestRecord = {
  node: NodeIndex;
  bytes: number;
  points: number;
  epoch: number;
  controller: AbortController;
  startedAt: number;
};

export type CacheCounters = {
  residentBytes: number;
  inflightBytes: number;
  residentPoints: number;
  inflightPoints: number;
  residentNodes: number;
  inflightRequests: number;
  loadedBytesTotal: number;
  loadedPointsTotal: number;
  requestsIssued: number;
  requestsCompleted: number;
  requestsAborted: number;
  requestsFailed: number;
  evictions: number;
  /** Responses that arrived after their request had been superseded. */
  staleResponses: number;
  /** Retiring payloads held for a not-yet-resident replacement. */
  retainedNodes: number;
  retainedBytes: number;
  /** Retirements forced past the replacement rule by retentionMs. */
  retentionExpiries: number;
};

export type FetchFn = (node: NodeIndex, signal: AbortSignal) => Promise<{ views: PayloadViews; bytes: number; ms: number }>;

export type CacheEvent =
  | { kind: "resident"; node: NodeIndex; entry: CacheEntry }
  | { kind: "evicted"; node: NodeIndex; entry: CacheEntry }
  | { kind: "failed"; node: NodeIndex; error: string }
  | { kind: "aborted"; node: NodeIndex }
  | { kind: "started"; node: NodeIndex };

export type RequestSpec = { node: NodeIndex; priority: number; bytes: number; points: number; depth: number };

export type Failure = { count: number; until: number; message: string };

export type CacheHooks = {
  /** Parent index, or -1 for the root. */
  parent: (node: NodeIndex) => NodeIndex;
  depth: (node: NodeIndex) => number;
};

export type CacheOptions = {
  residentByteBudget: number;
  inflightByteBudget: number;
  maxConcurrent: number;
  retryBackoffMs: number;
  /** Upper bound on holding a retiree whose replacement never arrives. */
  retentionMs?: number;
  fetch: FetchFn;
  onChange: (event: CacheEvent) => void;
  hooks: CacheHooks;
};

export class ResidencyCache {
  private opts: CacheOptions;
  private entries = new Map<NodeIndex, CacheEntry>();
  private inflight = new Map<NodeIndex, RequestRecord>();
  private failures = new Map<NodeIndex, Failure>();
  /** Nodes that have failed at least once in this session (UI diagnostic). */
  private failedNodes = new Set<NodeIndex>();
  private target: ReadonlySet<NodeIndex> = new Set();
  private frame = 0;
  private nextGeneration = 1;
  /** When each non-target payload first became a retirement candidate. */
  private retiredSince = new Map<NodeIndex, number>();
  private counters: CacheCounters = {
    residentBytes: 0,
    inflightBytes: 0,
    residentPoints: 0,
    inflightPoints: 0,
    residentNodes: 0,
    inflightRequests: 0,
    loadedBytesTotal: 0,
    loadedPointsTotal: 0,
    requestsIssued: 0,
    requestsCompleted: 0,
    requestsAborted: 0,
    requestsFailed: 0,
    evictions: 0,
    staleResponses: 0,
    retainedNodes: 0,
    retainedBytes: 0,
    retentionExpiries: 0,
  };

  constructor(opts: CacheOptions) {
    this.opts = opts;
  }

  setBudgets(residentByteBudget: number, inflightByteBudget: number) {
    this.opts.residentByteBudget = residentByteBudget;
    this.opts.inflightByteBudget = inflightByteBudget;
    // A budget shrink must reclaim immediately, not one reconcile later.
    this.collect();
  }
  setConcurrency(n: number) { this.opts.maxConcurrent = Math.max(1, n); }

  beginFrame(): number { this.frame++; return this.frame; }
  get currentFrame() { return this.frame; }
  has(node: NodeIndex): boolean { return this.entries.has(node); }
  get(node: NodeIndex): CacheEntry | undefined { return this.entries.get(node); }
  isInflight(node: NodeIndex): boolean { return this.inflight.has(node); }
  failureFor(node: NodeIndex): Failure | undefined { return this.failures.get(node); }
  failureCount(): number { return this.failedNodes.size; }
  residentCount(): number { return this.entries.size; }
  inflightCount(): number { return this.inflight.size; }
  entriesSnapshot(): IterableIterator<[NodeIndex, CacheEntry]> { return this.entries.entries(); }
  generationOf(node: NodeIndex): number { return this.entries.get(node)?.generation ?? 0; }

  touch(node: NodeIndex) {
    const e = this.entries.get(node);
    if (e) e.lastUsedFrame = this.frame;
  }

  snapshot(): CacheCounters & { failures: number } {
    this.updateRetentionCounters();
    return { ...this.counters, failures: this.failedNodes.size };
  }

  /** True if the payload exists and no draw list is currently using it. */
  isUseComplete(node: NodeIndex): boolean {
    const e = this.entries.get(node);
    return !e || e.drawUses === 0;
  }

  /**
   * Bracket a draw-list reference. The caller must pass the returned
   * generation back to `releaseDraw`; 0 means the payload is not resident.
   */
  acquireDraw(node: NodeIndex): number {
    const e = this.entries.get(node);
    if (!e) return 0;
    e.drawUses++;
    return e.generation;
  }

  releaseDraw(node: NodeIndex, generation: number): boolean {
    const e = this.entries.get(node);
    if (!e || e.generation !== generation) return false; // stale release
    if (e.drawUses > 0) e.drawUses--;
    if (e.drawUses === 0 && !this.target.has(node)) this.tryDrop(e);
    return true;
  }

  /**
   * True when a rendered payload may leave the draw list: it is no longer in
   * the target cut and either a coarse replacement is resident or the bounded
   * retention window has expired. The engine consults this before removing a
   * rendered node; the cache itself refuses to drop in-use payloads.
   */
  canRetire(node: NodeIndex): boolean {
    const e = this.entries.get(node);
    if (!e) return false;
    if (this.target.has(node)) return false;
    if (this.replacementReady(node)) return true;
    return this.retentionExpired(e, performance.now());
  }

  /**
   * Reconcile with a new ancestor-closed target cut.
   */
  reconcile(target: ReadonlySet<NodeIndex>, epoch: number, requests: RequestSpec[]) {
    this.target = target;
    // 1. Cancel requests that left the target cut; their bytes stop competing.
    for (const [node, rec] of [...this.inflight]) {
      if (target.has(node)) continue;
      this.inflight.delete(node);
      this.counters.inflightBytes -= rec.bytes;
      this.counters.inflightPoints -= rec.points;
      this.counters.inflightRequests--;
      this.counters.requestsAborted++;
      rec.controller.abort();
      this.opts.onChange({ kind: "aborted", node });
    }

    // 2. Reclaim payloads that left the target, deepest-first and only when
    //    use is complete and their coarse replacement is already resident.
    this.collect();

    // 3. Issue loads coarse-first so coverage is restored before detail.
    const now = performance.now();
    const pending = requests
      .filter((r) => !this.entries.has(r.node) && !this.inflight.has(r.node))
      .filter((r) => {
        const f = this.failures.get(r.node);
        return !f || f.until <= now;
      })
      .sort((a, b) => (a.depth !== b.depth ? a.depth - b.depth : b.priority - a.priority));

    for (const req of pending) {
      if (this.inflight.size >= this.opts.maxConcurrent) break;
      if (this.counters.inflightBytes + req.bytes > this.opts.inflightByteBudget) continue;
      this.startRequest(req.node, req.bytes, req.points, epoch);
    }
  }

  /**
   * Abort every in-flight request, keeping residents. Used by the UI's explicit
   * "cancel loads" control: requests for the current view will be re-issued on
   * the next frame if they are still wanted.
   */
  cancelAllInflight() {
    for (const [node, rec] of [...this.inflight]) {
      this.inflight.delete(node);
      this.counters.inflightBytes -= rec.bytes;
      this.counters.inflightPoints -= rec.points;
      this.counters.inflightRequests--;
      this.counters.requestsAborted++;
      rec.controller.abort();
      this.opts.onChange({ kind: "aborted", node });
    }
  }

  /**
   * Diagnostic only: forget one resident payload so the next frame re-requests
   * it. Callers must choose a node whose parent is resident (the kernel's
   * failure-injection demo picks a cut leaf), so coverage degrades to the
   * ancestor rather than becoming a hole. This intentionally overrides the
   * use-completion pin — the engine's eviction event removes the draw entry.
   */
  invalidate(node: NodeIndex): boolean {
    const entry = this.entries.get(node);
    if (!entry) return false;
    entry.drawUses = 0;
    this.dropEntry(node, entry);
    return true;
  }

  /** Release every residency that is safe to release and enforce the budget. */
  collect() {
    const now = performance.now();
    const retiring = [...this.entries.values()]
      .filter((e) => !this.target.has(e.node))
      .sort((a, b) => this.opts.hooks.depth(b.node) - this.opts.hooks.depth(a.node));
    for (const entry of retiring) {
      this.tryDrop(entry, now);
    }
    this.trimToBudget(now);
  }

  /**
   * A replacement is ready when the nearest target ancestor is resident. If
   * that ancestor has failed repeatedly, it is not plausibly arriving; the
   * retiree is then released so reclamation is never disabled permanently by a
   * broken payload. Merely-slow ancestors keep the retiree on screen.
   */
  private replacementReady(node: NodeIndex): boolean {
    const anchor = this.nearestTargetAncestor(node, this.target);
    if (anchor === null) return true;
    if (this.entries.has(anchor)) return true;
    const failure = this.failures.get(anchor);
    if (failure && failure.count >= 3) return true;
    return false;
  }

  private nearestTargetAncestor(node: NodeIndex, target: ReadonlySet<NodeIndex>): NodeIndex | null {
    let cur = this.opts.hooks.parent(node);
    while (cur >= 0) {
      if (target.has(cur)) return cur;
      cur = this.opts.hooks.parent(cur);
    }
    return null;
  }

  private retentionExpired(entry: CacheEntry, now: number): boolean {
    let since = this.retiredSince.get(entry.node);
    if (since === undefined) {
      since = now;
      this.retiredSince.set(entry.node, since);
    }
    return now - since >= (this.opts.retentionMs ?? 4000);
  }

  /**
   * Reclaim one payload if all three conditions hold. Returns true when the
   * entry was dropped. Never drops a payload a draw list still uses.
   */
  private tryDrop(entry: CacheEntry, now = performance.now()): boolean {
    if (entry.drawUses > 0) return false;
    if (this.target.has(entry.node)) return false;
    if (!this.replacementReady(entry.node)) {
      if (!this.retentionExpired(entry, now)) return false;
      this.counters.retentionExpiries++;
    }
    this.dropEntry(entry.node, entry);
    return true;
  }

  private trimToBudget(now: number) {
    let over = this.counters.residentBytes - this.opts.residentByteBudget;
    if (over <= 0) return;
    // Never evict a target node: it is coverage. Retirees go LRU-first, and
    // an in-use or replacement-pending payload is skipped rather than forced.
    const candidates = [...this.entries.values()]
      .filter((e) => !this.target.has(e.node))
      .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame);
    for (const entry of candidates) {
      if (over <= 0) break;
      if (entry.drawUses > 0) continue;
      if (!this.tryDrop(entry, now)) continue;
      over -= entry.bytes;
    }
  }

  private dropEntry(node: NodeIndex, entry: CacheEntry) {
    this.entries.delete(node);
    this.retiredSince.delete(node);
    this.counters.residentBytes -= entry.bytes;
    this.counters.residentPoints -= entry.points;
    this.counters.residentNodes--;
    this.counters.evictions++;
    this.opts.onChange({ kind: "evicted", node, entry });
  }

  private updateRetentionCounters() {
    let nodes = 0;
    let bytes = 0;
    for (const e of this.entries.values()) {
      if (this.target.has(e.node)) continue;
      nodes++;
      bytes += e.bytes;
    }
    this.counters.retainedNodes = nodes;
    this.counters.retainedBytes = bytes;
  }

  private startRequest(node: NodeIndex, bytes: number, points: number, epoch: number) {
    const controller = new AbortController();
    const rec: RequestRecord = {
      node, bytes, points, epoch, controller, startedAt: performance.now(),
    };
    this.inflight.set(node, rec);
    this.counters.inflightBytes += bytes;
    this.counters.inflightPoints += points;
    this.counters.inflightRequests++;
    this.counters.requestsIssued++;
    this.opts.onChange({ kind: "started", node });

    this.opts.fetch(node, controller.signal)
      .then(({ views, bytes: actualBytes }) => {
        const current = this.inflight.get(node);
        if (!current || current !== rec) {
          this.counters.staleResponses++;
          return;
        }
        this.inflight.delete(node);
        this.counters.inflightBytes -= rec.bytes;
        this.counters.inflightPoints -= rec.points;
        this.counters.inflightRequests--;
        this.counters.requestsCompleted++;
        this.counters.loadedBytesTotal += actualBytes;
        const entry: CacheEntry = {
          node,
          views,
          bytes: actualBytes,
          points: views.header.count,
          lastUsedFrame: this.frame,
          residentSince: performance.now(),
          generation: this.nextGeneration++,
          drawUses: 0,
        };
        this.entries.set(node, entry);
        this.retiredSince.delete(node);
        this.counters.residentBytes += entry.bytes;
        this.counters.residentPoints += entry.points;
        this.counters.residentNodes++;
        this.counters.loadedPointsTotal += entry.points;
        this.failures.delete(node);
        this.opts.onChange({ kind: "resident", node, entry });
        // The replacement for whatever this node supersedes has arrived;
        // reclaim the predecessor now rather than waiting for the next frame.
        this.collect();
      })
      .catch((err: unknown) => {
        const current = this.inflight.get(node);
        if (!current || current !== rec) {
          this.counters.staleResponses++;
          return;
        }
        this.inflight.delete(node);
        this.counters.inflightBytes -= rec.bytes;
        this.counters.inflightPoints -= rec.points;
        this.counters.inflightRequests--;
        if (controller.signal.aborted) {
          this.counters.requestsAborted++;
          this.opts.onChange({ kind: "aborted", node });
          return;
        }
        this.counters.requestsFailed++;
        const message = err instanceof Error ? err.message : String(err);
        const prev = this.failures.get(node);
        const count = (prev?.count ?? 0) + 1;
        this.failedNodes.add(node);
        this.failures.set(node, {
          count,
          until: performance.now() + this.opts.retryBackoffMs * Math.min(16, 2 ** (count - 1)),
          message,
        });
        this.opts.onChange({ kind: "failed", node, error: message });
      });
  }

  clear() {
    for (const rec of this.inflight.values()) rec.controller.abort();
    this.inflight.clear();
    this.entries.clear();
    this.failures.clear();
    this.failedNodes.clear();
    this.retiredSince.clear();
    this.counters = {
      ...this.counters,
      residentBytes: 0,
      inflightBytes: 0,
      residentPoints: 0,
      inflightPoints: 0,
      residentNodes: 0,
      inflightRequests: 0,
      retainedNodes: 0,
      retainedBytes: 0,
    };
  }
}
