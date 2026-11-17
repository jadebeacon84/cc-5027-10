#!/usr/bin/env node
/**
 * Validate datasets that were already written by `data:prepare`.
 *
 *   npm run data:validate                 # validate every built dataset
 *   npm run data:validate -- --only cliff
 *   npm run data:validate -- --dir public/data
 *
 * Regenerates the ground-truth point stores (deterministically, from the seed
 * in meta.json) and compares them against the files on disk.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { generate, GENERATORS } from "./generators.ts";
import { readDataset, validateDataset, writeReport } from "./validator.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

let dir = path.join(root, "public", "data");
const only: string[] = [];
for (let i = 2; i < process.argv.length; i++) {
  const a = process.argv[i];
  if (a === "--dir") dir = path.resolve(process.argv[++i]);
  else if (a === "--only") only.push(process.argv[++i]);
  else if (a === "--help" || a === "-h") {
    console.log("Usage: npm run data:validate -- [--dir public/data] [--only name]");
    process.exit(0);
  } else throw new Error(`unknown argument ${a}`);
}

const names = only.length
  ? only
  : fs.readdirSync(dir, { withFileTypes: true })
    .filter((e) => e.isDirectory() && fs.existsSync(path.join(dir, e.name, "manifest.json")))
    .map((e) => e.name);

if (names.length === 0) {
  console.error(`no datasets found in ${dir}; run npm run data:prepare first`);
  process.exit(1);
}

let failed = 0;
for (const name of names) {
  const datasetDir = path.join(dir, name);
  const { meta } = readDataset(datasetDir);
  if (!GENERATORS[meta.dataset]) {
    console.error(`[${name}] unknown generator "${meta.dataset}" in meta.json`);
    failed++;
    continue;
  }
  const truth = generate(meta.dataset, { seed: meta.seed });
  console.log(`\n[${name}] validating ${truth.store.n} generated points against disk payloads`);
  const report = validateDataset(datasetDir, truth, { log: (m) => console.log(`[${name}] ${m}`) });
  writeReport(datasetDir, report);
  for (const check of report.checks) {
    console.log(`[${name}]   ${check.ok ? "PASS" : "FAIL"}  ${check.name}: ${check.detail}`);
  }
  if (!report.ok) failed++;
  // Also verify meta.json world bounds cover every generated point.
  const b = meta.worldBounds;
  let outside = 0;
  for (let i = 0; i < truth.store.n; i++) {
    const x = truth.store.px[i], y = truth.store.py[i], z = truth.store.pz[i];
    if (x < b.min[0] || x > b.max[0] || y < b.min[1] || y > b.max[1] || z < b.min[2] || z > b.max[2]) outside++;
  }
  console.log(`[${name}]   ${outside === 0 ? "PASS" : "FAIL"}  world bounds cover all generated points (${outside} outside)`);
  if (outside > 0) failed++;
}

if (failed > 0) {
  console.error(`\n${failed} dataset(s) failed validation`);
  process.exit(1);
}
console.log("\nall datasets valid");
