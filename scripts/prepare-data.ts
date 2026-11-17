#!/usr/bin/env node
/**
 * Build all deliverable datasets into public/data/<dataset>/.
 *
 * Usage:
 *   npm run data:prepare                 # small + assembly + cliff
 *   npm run data:prepare -- --only small # one dataset
 *   npm run data:prepare -- --capacity 6000 --depth 14
 *
 * Every run writes manifest.json, meta.json, nodes/*.bin and validation.json.
 * The process exits non-zero if any validation check fails.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { generate, GENERATORS } from "./generators.ts";
import { buildOctree, writeDataset } from "./octree-builder.ts";
import { validateDataset, writeReport } from "./validator.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

type Args = {
  only: string[];
  capacity: number;
  depth: number;
  seed: number;
  out: string;
  skipValidate: boolean;
};

function parseArgs(argv: string[]): Args {
  const out: Args = {
    only: [],
    capacity: 4000,
    depth: 12,
    seed: 20261117,
    out: path.join(root, "public", "data"),
    skipValidate: false,
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--only") out.only.push(argv[++i]);
    else if (a === "--capacity") out.capacity = Number(argv[++i]);
    else if (a === "--depth") out.depth = Number(argv[++i]);
    else if (a === "--seed") out.seed = Number(argv[++i]);
    else if (a === "--out") out.out = path.resolve(argv[++i]);
    else if (a === "--skip-validate") out.skipValidate = true;
    else if (a === "--help" || a === "-h") {
      console.log(`Usage: npm run data:prepare -- [--only name] [--capacity n] [--depth n] [--seed n] [--out dir] [--skip-validate]`);
      console.log(`Generators: ${Object.keys(GENERATORS).join(", ")}`);
      process.exit(0);
    } else {
      throw new Error(`unknown argument ${a}`);
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const names = args.only.length ? args.only : ["small", "assembly", "cliff"];
const index: Record<string, unknown> = {};
let failures = 0;

for (const name of names) {
  const t0 = Date.now();
  console.log(`\n[${name}] generating…`);
  const scan = generate(name, { seed: args.seed });
  console.log(`[${name}] ${scan.store.n.toLocaleString()} points in ${Date.now() - t0}ms`);

  const t1 = Date.now();
  const built = buildOctree(scan, {
    nodeCapacity: args.capacity,
    maxDepth: args.depth,
    log: (m) => console.log(`[${name}] ${m}`),
  });
  console.log(`[${name}] octree built in ${Date.now() - t1}ms`);

  const dir = path.join(args.out, name);
  fs.rmSync(dir, { recursive: true, force: true });
  writeDataset(dir, scan, built);
  const bytes = built.manifest.nodes.reduce((a, n) => a + n.bytes, 0);
  console.log(`[${name}] wrote ${built.manifest.payloadNodeCount}/${built.manifest.nodeCount} payloads, ${(bytes / 1e6).toFixed(1)} MB`);

  if (!args.skipValidate) {
    const t2 = Date.now();
    const report = validateDataset(dir, scan, { log: (m) => console.log(`[${name}] ${m}`) });
    writeReport(dir, report);
    if (!report.ok) failures++;
    for (const check of report.checks) {
      console.log(`[${name}]   ${check.ok ? "PASS" : "FAIL"}  ${check.name}: ${check.detail}`);
    }
    console.log(`[${name}] validation took ${Date.now() - t2}ms`);
  }

  index[name] = {
    totalPoints: scan.store.n,
    nodeCount: built.manifest.nodeCount,
    payloadNodeCount: built.manifest.payloadNodeCount,
    maxDepth: built.manifest.maxDepth,
    bytes,
    title: built.manifest.title,
  };
}

fs.mkdirSync(args.out, { recursive: true });
fs.writeFileSync(
  path.join(args.out, "index.json"),
  JSON.stringify({ generatedAt: new Date().toISOString(), datasets: index }, null, 2),
);
fs.writeFileSync(
  path.join(args.out, "README.md"),
  `# Generated scan datasets\n\nRebuild with \`npm run data:prepare\`. Do not edit by hand.\n\n`
  + Object.entries(index).map(([k, v]) => {
    const e = v as { totalPoints: number; nodeCount: number; bytes: number; title: string };
    return `- **${k}** — ${e.title}; ${e.totalPoints.toLocaleString()} points, ${e.nodeCount} nodes, ${(e.bytes / 1e6).toFixed(1)} MB`;
  }).join("\n") + "\n",
);

console.log(`\nindex written to ${path.join(args.out, "index.json")}`);
if (failures > 0) {
  console.error(`\n${failures} dataset(s) failed validation`);
  process.exit(1);
}
console.log("all datasets passed validation");
