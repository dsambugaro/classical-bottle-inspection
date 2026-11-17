// Locale-aware formatting and parsing with Intl (REQ-L2). The API always uses JSON numbers.
import i18n from "./i18n";

const locale = () => i18n.language || "en";

export function fmtNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale(), {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale(), { maximumFractionDigits: 0 }).format(value);
}

export function fmtPercent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat(locale(), {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(value);
}

export function fmtMs(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${fmtNumber(value, value < 10 ? 2 : 1)} ms`;
}

export function fmtDate(
  value: string | number | Date | null | undefined,
  withSeconds = false,
): string {
  if (value === null || value === undefined) return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(locale(), {
    dateStyle: "short",
    timeStyle: withSeconds ? "medium" : "short",
  }).format(date);
}

export function fmtBytes(bytes: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${fmtNumber(value, unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Decimal separator of the active locale ("." or ","). */
export function decimalSeparator(): string {
  return (
    new Intl.NumberFormat(locale()).formatToParts(1.5).find((p) => p.type === "decimal")?.value ??
    "."
  );
}

/** Parses a number typed with the locale's decimal separator (pt-BR accepts "0,15"). */
export function parseLocaleNumber(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") return null;
  const separator = decimalSeparator();
  const group = separator === "," ? "." : ",";
  let normalized = trimmed.split(group).join("");
  if (separator === ",") normalized = normalized.replace(",", ".");
  if (!/^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(normalized)) return null;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

/** Formats a number for an input field (no grouping, locale decimal separator). */
export function formatInputNumber(value: number, digits: number): string {
  const text = Number(value.toFixed(digits)).toString();
  return decimalSeparator() === "," ? text.replace(".", ",") : text;
}

/** Name of a model as "<feature vector> · <algorithm>" (just the algorithm for old models). */
export function modelLabel(featureVector: string | null | undefined, algorithm: string): string {
  return featureVector ? `${featureVector} · ${algorithm}` : algorithm;
}
