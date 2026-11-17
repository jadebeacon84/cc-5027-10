#!/usr/bin/env node
/**
 * Dev launcher: start Vite on a fixed port and print exactly what to open.
 * Kept as a script so `npm run dev` is the only instruction a new user needs.
 */
import { createServer } from "vite";
import path from "node:path";
import { fileURLToPath } from "node:url";
import fs from "node:fs";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

if (!fs.existsSync(path.join(root, "public", "data", "index.json"))) {
  console.error(
    "\nNo datasets found in public/data.\n"
    + "Generate them first:\n\n  npm run data:prepare\n",
  );
  process.exit(1);
}

const server = await createServer({
  configFile: path.join(root, "vite.config.ts"),
  server: { port: 5173, strictPort: true },
});
await server.listen();
const urls = server.resolvedUrls?.local ?? [];
console.log("\nScan Stream dev server\n");
for (const url of urls) console.log(`  UI: ${url}`);
console.log(`  Data index: ${urls[0] ?? ""}data/index.json`);
console.log("\nControls: drag orbit · shift-drag pan · wheel zoom · click pin a point · H home · F fit\n");
