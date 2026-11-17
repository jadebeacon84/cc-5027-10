#!/usr/bin/env node
/**
 * Production static server for `dist/` (built UI) plus `public/data` (payloads).
 *
 * Vite copies `public/` into `dist/`, so serving `dist/` alone is sufficient.
 * Supports Range requests because node payloads are fetched individually and
 * range support makes partial re-fetches cheap behind a CDN.
 *
 *   npm run build && npm start      # http://localhost:8080
 *   PORT=9000 npm start
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dist = path.join(root, "dist");
const port = Number(process.env.PORT ?? 8080);
const host = process.env.HOST ?? "127.0.0.1";

if (!fs.existsSync(path.join(dist, "index.html"))) {
  console.error("\ndist/ is missing. Build first:\n\n  npm run build\n");
  process.exit(1);
}

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".map": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".ico": "image/x-icon",
};

function resolveFile(urlPath) {
  let decoded;
  try {
    decoded = decodeURIComponent(urlPath.split("?")[0]);
  } catch {
    return null;
  }
  // Resolve inside dist and refuse anything that escapes it.
  const full = path.resolve(dist, "." + path.posix.normalize(decoded));
  if (full !== dist && !full.startsWith(dist + path.sep)) return null;
  try {
    const stat = fs.statSync(full);
    if (stat.isFile()) return { file: full, size: stat.size };
    if (stat.isDirectory()) {
      const index = path.join(full, "index.html");
      if (fs.existsSync(index)) {
        return { file: index, size: fs.statSync(index).size };
      }
    }
  } catch {
    return null;
  }
  return null;
}

const server = http.createServer((req, res) => {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405, { Allow: "GET, HEAD" });
    res.end("method not allowed");
    return;
  }
  let target = req.url ?? "/";
  if (target === "/" || target === "") target = "/index.html";
  const found = resolveFile(target);

  if (!found) {
    // SPA fallback for unknown non-file paths; payloads and manifests never
    // rely on it, so 404s stay honest.
    if (!path.extname(target)) {
      const index = path.join(dist, "index.html");
      res.writeHead(200, { "Content-Type": MIME[".html"], "Cache-Control": "no-cache" });
      res.end(fs.readFileSync(index));
      return;
    }
    res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("not found");
    return;
  }

  const ext = path.extname(found.file).toLowerCase();
  const type = MIME[ext] ?? "application/octet-stream";
  // Payloads are content-addressed by the manifest; allow long-lived caching.
  const cache = ext === ".bin" ? "public, max-age=31536000, immutable" : "no-cache";
  const headers = {
    "Content-Type": type,
    "Cache-Control": cache,
    "Access-Control-Allow-Origin": "*",
    "Accept-Ranges": "bytes",
  };

  const range = req.headers.range;
  if (range && /^bytes=\d*-\d*$/.test(range)) {
    const [startRaw, endRaw] = range.slice("bytes=".length).split("-");
    let start = startRaw === "" ? NaN : Number(startRaw);
    let end = endRaw === "" ? NaN : Number(endRaw);
    if (Number.isNaN(start)) {
      start = Math.max(0, found.size - (Number.isNaN(end) ? 0 : end));
      end = found.size - 1;
    } else if (Number.isNaN(end)) {
      end = found.size - 1;
    }
    if (start > end || start >= found.size) {
      res.writeHead(416, { "Content-Range": `bytes */${found.size}` });
      res.end();
      return;
    }
    end = Math.min(end, found.size - 1);
    res.writeHead(206, {
      ...headers,
      "Content-Range": `bytes ${start}-${end}/${found.size}`,
      "Content-Length": end - start + 1,
    });
    if (req.method === "HEAD") { res.end(); return; }
    fs.createReadStream(found.file, { start, end }).pipe(res);
    return;
  }

  res.writeHead(200, { ...headers, "Content-Length": found.size });
  if (req.method === "HEAD") { res.end(); return; }
  fs.createReadStream(found.file).pipe(res);
});

server.listen(port, host, () => {
  console.log(`\nScan Stream serving dist/ on http://${host}:${port}\n`);
});
