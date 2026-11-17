/**
 * Network layer: fetch a node payload, verify its structure and checksums,
 * and expose typed-array views ready for GPU upload.
 *
 * The verification here is the runtime half of the attribute-integrity
 * contract. `payloadViews` re-derives array offsets from the header count and
 * compares them to the header, then validates each attribute checksum. If a
 * payload was truncated, served from a stale build, or written with a shifted
 * layout, the request fails and the ancestor payload stays on screen.
 */
import {
  payloadViews, nodeFileName, type Manifest, type PayloadViews,
} from "../shared/format";
import type { FetchFn } from "./cache";

export type LoaderStats = {
  requests: number;
  bytes: number;
  failures: number;
  verificationFailures: number;
  avgMs: number;
  lastMs: number;
  slowestMs: number;
};

export type NodeLoader = {
  fetch: FetchFn;
  urlFor(node: number): string;
  stats: LoaderStats;
  resetStats(): void;
  /** Change injected latency while running (0 disables the delay). */
  setLatency(ms: number): void;
  /** Force the next N fetches to fail, to exercise the failure path in the UI. */
  failNext(count: number): void;
};

export type LoaderOptions = {
  /** Base URL for the dataset, e.g. "/data/small". */
  baseUrl: string;
  manifest: Manifest;
  /** Per-request delay injection for testing cancellation behaviour. */
  latencyMs?: number;
  /** If true, verify checksums (disable only for profiling). */
  verify?: boolean;
  /** Optional network meter updated as bytes arrive. */
  onBytes?: (bytes: number, url: string) => void;
};

export function createLoader(opts: LoaderOptions): NodeLoader {
  const { baseUrl, manifest } = opts;
  const verify = opts.verify !== false;
  const stats: LoaderStats = {
    requests: 0,
    bytes: 0,
    failures: 0,
    verificationFailures: 0,
    avgMs: 0,
    lastMs: 0,
    slowestMs: 0,
  };
  let totalMs = 0;
  let latency = opts.latencyMs ?? 0;
  let forcedFailures = 0;

  const paths = manifest.nodes.map((n) => n.path);
  const urlFor = (node: number) => `${baseUrl}/nodes/${nodeFileName(paths[node])}`;

  const fetchImpl: FetchFn = async (node, signal) => {
    const url = urlFor(node);
    const t0 = performance.now();
    stats.requests++;
    if (latency) {
      await delay(latency, signal);
    }
    if (forcedFailures > 0) {
      forcedFailures--;
      stats.failures++;
      throw new Error(`injected payload failure for ${url}`);
    }
    let res: Response;
    try {
      res = await fetch(url, { signal });
    } catch (err) {
      stats.failures++;
      throw err;
    }
    if (!res.ok) {
      stats.failures++;
      throw new Error(`HTTP ${res.status} for ${url}`);
    }
    const buf = await res.arrayBuffer();
    const bytes = buf.byteLength;
    stats.bytes += bytes;
    opts.onBytes?.(bytes, url);
    const ms = performance.now() - t0;
    stats.lastMs = ms;
    totalMs += ms;
    stats.avgMs = totalMs / stats.requests;
    stats.slowestMs = Math.max(stats.slowestMs, ms);
    if (signal.aborted) throw makeAbort();
    if (bytes > manifest.maxNodeBytes) {
      stats.verificationFailures++;
      throw new Error(`payload ${url} is ${bytes} bytes, larger than manifest max ${manifest.maxNodeBytes}`);
    }
    let views: PayloadViews;
    try {
      views = payloadViews(new Uint8Array(buf));
      if (!verify) {
        // Still parse, but skip checksum re-verification cost.
      }
    } catch (err) {
      stats.verificationFailures++;
      throw err;
    }
    const expectedCount = manifest.nodes[node].pointCount;
    if (views.header.count !== expectedCount) {
      stats.verificationFailures++;
      throw new Error(`payload ${url} has ${views.header.count} points, manifest says ${expectedCount}`);
    }
    return { views, bytes, ms };
  };

  return {
    fetch: fetchImpl,
    urlFor,
    stats,
    resetStats() {
      stats.requests = 0;
      stats.bytes = 0;
      stats.failures = 0;
      stats.verificationFailures = 0;
      stats.avgMs = 0;
      stats.lastMs = 0;
      stats.slowestMs = 0;
      totalMs = 0;
    },
    setLatency(ms: number) {
      latency = Math.max(0, ms);
    },
    failNext(count: number) {
      forcedFailures += Math.max(0, count);
    },
  };
}

function delay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) { reject(makeAbort()); return; }
    const t = setTimeout(() => { cleanup(); resolve(); }, ms);
    const onAbort = () => { cleanup(); reject(makeAbort()); };
    const cleanup = () => {
      clearTimeout(t);
      signal.removeEventListener("abort", onAbort);
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function makeAbort(): Error {
  const err = new Error("aborted");
  err.name = "AbortError";
  return err;
}

export type ManifestBundle = {
  manifest: Manifest;
  meta: import("../shared/format").DatasetMeta;
  baseUrl: string;
};

export async function loadDataset(baseUrl: string): Promise<ManifestBundle> {
  const [manifestRes, metaRes] = await Promise.all([
    fetch(`${baseUrl}/manifest.json`),
    fetch(`${baseUrl}/meta.json`),
  ]);
  if (!manifestRes.ok) throw new Error(`manifest.json: HTTP ${manifestRes.status}`);
  if (!metaRes.ok) throw new Error(`meta.json: HTTP ${metaRes.status}`);
  const manifest = await manifestRes.json() as Manifest;
  const meta = await metaRes.json() as import("../shared/format").DatasetMeta;
  if (manifest.format !== "pc-octree") throw new Error(`unexpected manifest format ${manifest.format}`);
  if (manifest.nodes.length === 0) throw new Error("manifest has no nodes");
  return { manifest, meta, baseUrl };
}
