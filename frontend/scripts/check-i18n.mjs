// REQ-L5: fails when pt-BR.json lacks any key present in en.json (the source of truth).
import { readFileSync } from "node:fs";

const load = (name) => JSON.parse(readFileSync(new URL(`../src/locales/${name}`, import.meta.url)));
const flatten = (obj, prefix = "") =>
  Object.entries(obj).flatMap(([key, value]) =>
    value && typeof value === "object" ? flatten(value, `${prefix}${key}.`) : [`${prefix}${key}`],
  );

const en = new Set(flatten(load("en.json")));
const pt = new Set(flatten(load("pt-BR.json")));
const missing = [...en].filter((key) => !pt.has(key));
const extra = [...pt].filter((key) => !en.has(key));
if (extra.length)
  console.warn(`pt-BR.json has ${extra.length} keys not in en.json:\n  ${extra.join("\n  ")}`);
if (missing.length) {
  console.error(`pt-BR.json is missing ${missing.length} keys:\n  ${missing.join("\n  ")}`);
  process.exit(1);
}
console.log(`i18n ok: ${en.size} keys`);

// Every static key used in the source must exist in en.json.
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const files = [];
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walk(path);
    else if (/\.(ts|tsx)$/.test(name) && !name.endsWith(".test.tsx")) files.push(path);
  }
};
walk(new URL("../src", import.meta.url).pathname);
const used = new Set();
for (const file of files) {
  for (const match of readFileSync(file, "utf8").matchAll(/\bt\(\s*"([a-zA-Z0-9_.-]+)"/g))
    used.add(match[1]);
}
const unknown = [...used].filter((key) => !en.has(key));
if (unknown.length) {
  console.error(`keys used in the code but missing in en.json:\n  ${unknown.join("\n  ")}`);
  process.exit(1);
}
