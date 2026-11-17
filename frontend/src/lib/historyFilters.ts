// Filters of the card history: the URL query string is the state; this converts it to the
// parameters of GET /api/history. Shared keys (preset, class, ...) are the ones of the reports.

export type ReportView = "dashboard" | "cards";

/** Repeated URL keys that hold several values. */
export const HISTORY_LIST_KEYS = [
  "profile",
  "version",
  "source",
  "camera",
  "verdict",
  "class",
  "truth",
] as const;

export const DEFAULT_PRESET = "7d";

export function viewFromParams(params: URLSearchParams): ReportView {
  return params.get("view") === "cards" ? "cards" : "dashboard";
}

/** The date preset in effect: the URL value, none when a custom start is set, else 7 days. */
export function presetFromParams(params: URLSearchParams): string {
  return params.get("preset") ?? (params.get("start") ? "" : DEFAULT_PRESET);
}

/** Parameters of GET /api/history (without `before` and `limit`) for a URL query string. */
export function historyQuery(params: URLSearchParams): Record<string, unknown> {
  const query: Record<string, unknown> = {
    preset: presetFromParams(params) || undefined,
    start: params.get("start"),
    end: params.get("end"),
    session: params.get("session"),
    only_wrong: params.get("wrong") === "1" ? "true" : undefined,
    disagree: params.get("disagree") === "1" ? "true" : undefined,
    name: params.get("q")?.trim() || undefined,
  };
  for (const key of HISTORY_LIST_KEYS) query[key] = params.getAll(key);
  return query;
}

/** True when any filter besides the date preset is set (to offer "clear filters"). */
export function hasActiveFilters(params: URLSearchParams): boolean {
  return (
    HISTORY_LIST_KEYS.some((key) => params.getAll(key).length > 0) ||
    ["session", "start", "wrong", "disagree", "q"].some((key) => !!params.get(key))
  );
}
