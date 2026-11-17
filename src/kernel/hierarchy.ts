/**
 * Hierarchy index and LOD traversal.
 *
 * Selection produces an *ancestor-closed cut*: a set of payload nodes such
 * that if a node is selected, every ancestor is selected too. This is the
 * core of the coverage guarantee:
 *
 *   - Every selected node contributes only its own payload (disjoint from all
 *     others), so no point is ever drawn twice.
 *   - Refinement adds nodes; it does not remove them. A child that is still in
 *     flight cannot create a hole, and a failed request leaves the coarse
 *     ancestor visible.
 *   - Nodes are retired only when their screen-space error falls below
 *     tolerance, and retirement is processed child-before-parent. The set
 *     visible at any instant therefore always still covers every region at
 *     least as densely as the previous frame's coarser cut for that region.
 *
 * Budgets cap how many points/bytes are drawn. When the budget is exhausted,
 * the traversal keeps the current level for the node instead of refining; that
 * is a coarser but complete representation, never a partially covered one.
 */
import type { Manifest, NodeManifest } from "../shared/format";
import {
  aabbVisible, distanceToAABB, extractFrustumPlanes, projectionFactor,
  aabbRadius, type CameraView,
} from "../shared/geo";

export type NodeIndex = number;

export type Hierarchy = {
  manifest: Manifest;
  nodes: NodeManifest[];
  byPath: Map<string, NodeManifest>;
  byId: Map<string, NodeIndex>;
  parent: Int32Array;
  children: Int32Array[];
  depthMax: number;
  root: NodeIndex;
  /** Point count of the node and every ancestor, for cut accounting. */
  ancestorPointCount: Int32Array;
  /** Payload bytes of the node and every ancestor. */
  ancestorBytes: Int32Array;
};

export function buildHierarchy(manifest: Manifest): Hierarchy {
  const nodes = manifest.nodes;
  const byPath = new Map<string, NodeManifest>();
  const byId = new Map<string, NodeIndex>();
  for (let i = 0; i < nodes.length; i++) {
    byPath.set(nodes[i].path, nodes[i]);
    byId.set(nodes[i].id, i);
  }
  const parent = new Int32Array(nodes.length).fill(-1);
  const children: Int32Array[] = new Array(nodes.length);
  let depthMax = 0;
  for (let i = 0; i < nodes.length; i++) {
    depthMax = Math.max(depthMax, nodes[i].depth);
    const kids: number[] = [];
    for (const c of nodes[i].children) {
      const ci = byId.get(c);
      if (ci === undefined) continue;
      parent[ci] = i;
      kids.push(ci);
    }
    children[i] = Int32Array.from(kids);
  }
  const root = byId.get(manifest.root) ?? 0;
  // Paths are ordered root-first in the manifest, but do not rely on it.
  const order: number[] = [];
  const stack = [root];
  const seen = new Uint8Array(nodes.length);
  while (stack.length) {
    const i = stack.pop()!;
    if (seen[i]) continue;
    seen[i] = 1;
    order.push(i);
    for (const c of children[i]) stack.push(c);
  }
  const ancestorPointCount = new Int32Array(nodes.length);
  const ancestorBytes = new Int32Array(nodes.length);
  // BFS/DFS parent-first order guarantees ancestors are computed first.
  for (const i of order) {
    const p = parent[i];
    ancestorPointCount[i] = nodes[i].pointCount + (p >= 0 ? ancestorPointCount[p] : 0);
    ancestorBytes[i] = nodes[i].bytes + (p >= 0 ? ancestorBytes[p] : 0);
  }
  return { manifest, nodes, byPath, byId, parent, children, depthMax, root, ancestorPointCount, ancestorBytes };
}

export type SelectionOptions = {
  /** Maximum points the selected cut should draw (including ancestors). */
  pointBudget: number;
  /** Optional hard cap on payload bytes. */
  byteBudget?: number;
  /** Screen-space error tolerance in pixels. Larger = coarser. */
  sseTolerance: number;
  /** Never refine beyond this depth even if error allows. */
  maxDepth?: number;
  /** Only traverse nodes intersecting this box (used for region focus). */
  focusBox?: { min: [number, number, number]; max: [number, number, number] };
  /** Pixels per world unit at distance 1 for this frame. */
  viewportHeight: number;
  /** Ignore nodes closer than this; they are drawing over the camera. */
  near?: number;
};

export type SelectionResult = {
  /** Nodes whose payloads should be drawn, in parent-before-child order. */
  selected: NodeIndex[];
  /** Nodes the traversal wanted but could not refine because of budget. */
  deferred: NodeIndex[];
  /** Estimated points drawn by this cut. */
  drawnPoints: number;
  /** Estimated bytes drawn by this cut. */
  drawnBytes: number;
  /** Nodes culled because they were outside the frustum/focus. */
  culled: number;
  /** Nodes visited during traversal. */
  visited: number;
  /** True if budget prevented at least one refinement. */
  budgetLimited: boolean;
};

/** Real order of a cut: all ancestors before descendants. */
export function sortCut(h: Hierarchy, selected: NodeIndex[]): NodeIndex[] {
  return selected.slice().sort((a, b) => {
    if (h.nodes[a].depth !== h.nodes[b].depth) return h.nodes[a].depth - h.nodes[b].depth;
    return a - b;
  });
}

/**
 * Two-phase selection.
 *
 * Phase 1 (error cut): traverse from the root, refining whenever the node's
 * projected geometric error exceeds tolerance. Every visited node is added to
 * the cut before its children, so the result is ancestor-closed.
 *
 * Phase 2 (budget trim): if the cut exceeds the point or byte budget, remove
 * nodes *deepest-first and only when none of their children remain selected*.
 * Removing a leaf from an ancestor-closed cut leaves its parent in place, so
 * the cut stays ancestor-closed and the region keeps its coarser coverage. This
 * is a hard cap: the final cut is at or below budget unless even the root alone
 * exceeds it. It is also why a budget-limited frame is coarse, never holey.
 */
export function selectNodes(
  h: Hierarchy,
  view: CameraView,
  opts: SelectionOptions,
): SelectionResult {
  const planes = extractFrustumPlanes(view.viewProjection);
  const factor = projectionFactor(view.fov, opts.viewportHeight);
  const maxDepth = Math.min(opts.maxDepth ?? h.depthMax, h.depthMax);
  const near = opts.near ?? view.near;
  const root = h.root;

  let culled = 0;
  let visited = 0;
  const selectedSet = new Set<NodeIndex>();
  const errorAboveTolerance = new Set<NodeIndex>();
  const deferred: NodeIndex[] = [];

  const inFocus = (node: NodeIndex): boolean => {
    if (!opts.focusBox) return true;
    const b = h.nodes[node].bounds;
    const f = opts.focusBox;
    return !(b.max[0] < f.min[0] || b.min[0] > f.max[0]
      || b.max[1] < f.min[1] || b.min[1] > f.max[1]
      || b.max[2] < f.min[2] || b.min[2] > f.max[2]);
  };

  // A node entirely in front of the camera but closer than `near` is skipped.
  // A node whose bounds contain the camera is never skipped: that would blank
  // the view when flying inside the cloud.
  const nearSkip = (node: NodeIndex): boolean => {
    const b = h.nodes[node].bounds;
    const inside = view.eye[0] >= b.min[0] && view.eye[0] <= b.max[0]
      && view.eye[1] >= b.min[1] && view.eye[1] <= b.max[1]
      && view.eye[2] >= b.min[2] && view.eye[2] <= b.max[2];
    if (inside) return false;
    return distanceToAABB(b, view.eye) <= near;
  };

  const stack: NodeIndex[] = [root];
  while (stack.length) {
    const node = stack.pop()!;
    visited++;
    const n = h.nodes[node];
    if (!aabbVisible(n.bounds, planes) || !inFocus(node) || nearSkip(node)) {
      culled++;
      continue;
    }
    selectedSet.add(node);
    const kids = h.children[node];
    const error = n.children.length > 0 ? n.geometricError : 0;
    const dist = Math.max(distanceToAABB(n.bounds, view.eye), near);
    const sse = error > 0 ? (error * factor) / dist : 0;
    const wantRefine = kids.length > 0 && n.depth < maxDepth && sse > opts.sseTolerance;
    if (wantRefine) {
      errorAboveTolerance.add(node);
      for (const child of kids) stack.push(child);
    }
  }

  // Phase 2: trim childless nodes deepest-first until the budgets fit.
  const selectedCount = new Map<NodeIndex, number>();
  for (const node of selectedSet) {
    const p = h.parent[node];
    if (p >= 0) selectedCount.set(p, (selectedCount.get(p) ?? 0) + 1);
  }
  let drawnPoints = 0;
  let drawnBytes = 0;
  for (const node of selectedSet) {
    drawnPoints += h.nodes[node].pointCount;
    drawnBytes += h.nodes[node].bytes;
  }

  const byteBudget = opts.byteBudget ?? Number.POSITIVE_INFINITY;
  let budgetLimited = false;
  if (drawnPoints > opts.pointBudget || drawnBytes > byteBudget) {
    budgetLimited = true;
    const byDepthDesc = [...selectedSet].sort(
      (a, b) => h.nodes[b].depth - h.nodes[a].depth || h.nodes[b].pointCount - h.nodes[a].pointCount,
    );
    const removed = new Set<NodeIndex>();
    for (const node of byDepthDesc) {
      if (node === root) break;
      if (drawnPoints <= opts.pointBudget && drawnBytes <= byteBudget) break;
      // Only removable if no selected child depends on this node staying.
      const selectedChildren = selectedCount.get(node) ?? 0;
      if (selectedChildren > 0) continue;
      const p = h.parent[node];
      if (p >= 0) selectedCount.set(p, (selectedCount.get(p) ?? 1) - 1);
      selectedSet.delete(node);
      removed.add(node);
      drawnPoints -= h.nodes[node].pointCount;
      drawnBytes -= h.nodes[node].bytes;
    }
    // A removed node is not a coarse region: its parent is. The pass below
    // records the survivors with no refined children, which is exactly what
    // the viewer sees held coarse.
    void removed;
  }

  // Deferred = nodes still in the cut that wanted to refine but have no
  // selected children. These are the regions drawn at the ancestor's density.
  for (const node of errorAboveTolerance) {
    if (!selectedSet.has(node)) continue;
    let hasChild = false;
    for (const c of h.children[node]) if (selectedSet.has(c)) hasChild = true;
    if (!hasChild) deferred.push(node);
  }

  const selected = sortCut(h, [...selectedSet]);
  return {
    selected,
    deferred,
    drawnPoints,
    drawnBytes,
    culled,
    visited,
    budgetLimited,
  };
}

/**
 * Nodes that must stay resident for a given cut. Because cuts are
 * ancestor-closed, this is just `selected`. Kept as a separate function so the
 * cache policy is explicit at call sites.
 */
export function requiredForCut(selection: SelectionResult): Set<NodeIndex> {
  return new Set(selection.selected);
}

/** Diagnostics: largest distance from a root-child to any descendant. */
export function treeStats(h: Hierarchy) {
  let leaves = 0;
  let maxDepth = 0;
  let payloadNodes = 0;
  for (const n of h.nodes) {
    if (n.children.length === 0) leaves++;
    if (n.hasPayload) payloadNodes++;
    maxDepth = Math.max(maxDepth, n.depth);
  }
  const rootRadius = aabbRadius(h.nodes[h.root].bounds);
  return { leaves, maxDepth, payloadNodes, totalNodes: h.nodes.length, rootRadius };
}
