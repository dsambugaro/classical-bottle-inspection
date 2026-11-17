// Client-side checks mirroring the backend catalog validation (the server validates again).
import type { CatalogEntry, ParamValue, Params } from "./types";

export interface ParamIssue {
  code: string;
  params?: Record<string, unknown>;
}

const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export function validateEntry(entry: CatalogEntry, value: ParamValue): ParamIssue | null {
  if (value === null || value === undefined)
    return entry.nullable ? null : { code: "param.required" };
  const checkRange = (v: number) => {
    if (entry.min !== null && v < entry.min)
      return { code: "param.below_min", params: { min: entry.min } };
    if (entry.max !== null && v > entry.max)
      return { code: "param.above_max", params: { max: entry.max } };
    return null;
  };
  switch (entry.type) {
    case "int":
    case "float":
      if (!isNum(value)) return { code: "param.invalid_type" };
      if (entry.type === "int" && !Number.isInteger(value)) return { code: "param.invalid_type" };
      return checkRange(value);
    case "range": {
      const v = value as number[];
      for (const x of v) {
        const issue = checkRange(x);
        if (issue) return issue;
      }
      return v[0] < v[1] ? null : { code: "param.range_order" };
    }
    case "ranges": {
      // Variable-length list of (start, end) pairs (the learned bands of the automatic regions).
      for (const band of value as number[][]) {
        const issue = validateEntry({ ...entry, type: "range" }, band);
        if (issue) return issue;
      }
      return null;
    }
    case "rect": {
      const v = value as number[];
      for (const x of v) {
        const issue = checkRange(x);
        if (issue) return issue;
      }
      return v[0] < v[2] && v[1] < v[3] ? null : { code: "param.rect_order" };
    }
    case "size": {
      const v = value as number[];
      return v.every((x) => Number.isInteger(x) && x > 0)
        ? (v.map(checkRange).find(Boolean) ?? null)
        : { code: "param.invalid_type" };
    }
    case "multiselect":
      return Array.isArray(value) ? null : { code: "param.invalid_type" };
    default:
      return null;
  }
}

export function validateAll(entries: CatalogEntry[], params: Params): Record<string, ParamIssue> {
  const issues: Record<string, ParamIssue> = {};
  for (const entry of entries) {
    if (entry.group === "verdict" || entry.mode === "locked") continue;
    const issue = validateEntry(entry, params[entry.key]);
    if (issue) issues[entry.key] = issue;
  }
  return issues;
}
