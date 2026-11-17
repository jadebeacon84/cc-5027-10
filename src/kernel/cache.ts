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
 * Eviction safety. The target cut is always ancestor-closed (see hierarchy.ts),
 * so every resident node's ancestors are resident too. An eviction removes a
 * node only when its nearest ancestor that is *in the new target* is already
 * resident. That is what prevents "delete the parent before its replacement
 * arrives": the coarse replacement must be on screen first. Eviction proceeds
 * deepest-first, so a whole retiring branch is removed topologically.
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
  };

  constructor(opts: CacheOptions) {
    this.opts = opts;
  }

  setBudgets(residentByteBudget: number, inflightByteBudget: number) {
    this.opts.residentByteBudget = residentByteBudget;
    this.opts.inflightByteBudget = inflightByteBudget;
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

  touch(node: NodeIndex) {
    const e = this.entries.get(node);
    if (e) e.lastUsedFrame = this.frame;
  }

  snapshot(): CacheCounters & { failures: number } {
    return { ...this.counters, failures: this.failedNodes.size };
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

    // 2. Evict residents outside the target, deepest-first and only when the
    //    nearest target ancestor is already resident (coarse replacement）。
    const retiring = [...this.entries.values()]
      .filter((e) => !target.has(e.node))
      .sort((a, b) => this.opts.hooks.depth(b.node) - this.opts.hooks.depth(a.node));
    for (const entry of retiring) {
      const anchor = this.nearestTargetAncestor(entry.node, target);
      if (anchor !== null && !this.entries.has(anchor)) continue;
      this.dropEntry(entry.node, entry);
    }

    // 3. Enforce the resident budget, protecting nodes in the target cut.
    let over = this.counters.residentBytes - this.opts.residentByteBudget;
    if (over > 0) {
      const candidates = [...this.entries.values()]
        .filter((e) => !target.has(e.node))
        .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame);
      for (const entry of candidates) {
        if (over <= 0) break;
        over -= entry.bytes;
        this.dropEntry(entry.node, entry);
      }
    }

    // 4. Issue loads coarse-first so coverage is restored before detail.
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
   * ancestor rather than becoming a hole.
   */
  invalidate(node: NodeIndex): boolean {
    const entry = this.entries.get(node);
    if (!entry) return false;
    this.dropEntry(node, entry);
    return true;
  }

  private nearestTargetAncestor(node: NodeIndex, target: ReadonlySet<NodeIndex>): NodeIndex | null {
    let cur = this.opts.hooks.parent(node);
    while (cur >= 0) {
      if (target.has(cur)) return cur;
      cur = this.opts.hooks.parent(cur);
    }
    return null;
  }

  private dropEntry(node: NodeIndex, entry: CacheEntry) {
    this.entries.delete(node);
    this.counters.residentBytes -= entry.bytes;
    this.counters.residentPoints -= entry.points;
    this.counters.residentNodes--;
    this.counters.evictions++;
    this.opts.onChange({ kind: "evicted", node, entry });
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
        if (!current || current !== rec) return;
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
        };
        this.entries.set(node, entry);
        this.counters.residentBytes += entry.bytes;
        this.counters.residentPoints += entry.points;
        this.counters.residentNodes++;
        this.counters.loadedPointsTotal += entry.points;
        this.failures.delete(node);
        this.opts.onChange({ kind: "resident", node, entry });
        // Trim only non-target nodes; target nodes are coverage.
        this.evictToBudget(this.opts.residentByteBudget);
      })
      .catch((err: unknown) => {
        const current = this.inflight.get(node);
        if (!current || current !== rec) return;
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

  private evictToBudget(budget: number) {
    let over = this.counters.residentBytes - budget;
    if (over <= 0) return;
    // Never evict a node in the current target cut: it is coverage. Stale
    // residents outside the cut are trimmed first; if that is not enough the
    // cache is allowed to exceed its byte budget rather than tear a hole.
    const candidates = [...this.entries.values()]
      .filter((e) => !this.target.has(e.node))
      .sort((a, b) => a.lastUsedFrame - b.lastUsedFrame);
    for (const entry of candidates) {
      if (over <= 0) break;
      over -= entry.bytes;
      this.dropEntry(entry.node, entry);
    }
  }

  clear() {
    for (const rec of this.inflight.values()) rec.controller.abort();
    this.inflight.clear();
    this.entries.clear();
    this.failures.clear();
    this.failedNodes.clear();
    this.counters = {
      ...this.counters,
      residentBytes: 0,
      inflightBytes: 0,
      residentPoints: 0,
      inflightPoints: 0,
      residentNodes: 0,
      inflightRequests: 0,
    };
  }
}
