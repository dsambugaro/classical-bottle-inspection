// Catalog-driven parameter controls (§7, §15.5). Sliders are always paired with numeric inputs;
// relative coordinates are shown as percentages but stored as fractions; numbers are parsed
// with the locale's decimal separator (pt-BR accepts the decimal comma).
import { CircleHelp, Lock, RotateCcw } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Button, Checkbox, Select, Slider, Switch, Tip, cn } from "../ui/primitives";
import { formatInputNumber, fmtNumber, parseLocaleNumber } from "@/lib/format";
import type { CatalogEntry, ParamValue } from "@/lib/types";

const isRelative = (entry: CatalogEntry) => entry.unit === "relative";

/** Display scale: relative values are shown as 0–100 %. */
const toDisplay = (entry: CatalogEntry, v: number) => (isRelative(entry) ? v * 100 : v);
const fromDisplay = (entry: CatalogEntry, v: number) =>
  isRelative(entry) ? Math.round(v * 1e4) / 1e6 : v;

function digitsFor(entry: CatalogEntry): number {
  if (entry.type === "int" || entry.type === "size" || entry.type === "int3") return 0;
  const step = isRelative(entry) ? (entry.step ?? 0.01) * 100 : (entry.step ?? 0.01);
  return Math.max(0, Math.min(4, Math.ceil(-Math.log10(step))));
}

export function NumberInput({
  value,
  onChange,
  digits,
  label,
  invalid,
  disabled,
  className,
  min,
  max,
}: {
  value: number;
  onChange: (value: number) => void;
  digits: number;
  label: string;
  invalid?: boolean;
  disabled?: boolean;
  className?: string;
  min?: number;
  max?: number;
}) {
  const formatted = formatInputNumber(value, digits);
  const [text, setText] = useState(formatted);
  const [error, setError] = useState(false);
  const [shown, setShown] = useState(formatted);
  if (shown !== formatted) {
    // The value changed from outside (slider, reset, drag): show it.
    setShown(formatted);
    setText(formatted);
    setError(false);
  }
  const commit = () => {
    const parsed = parseLocaleNumber(text);
    if (
      parsed === null ||
      (min !== undefined && parsed < min) ||
      (max !== undefined && parsed > max)
    ) {
      setError(true);
      return;
    }
    setError(false);
    if (parsed !== value) onChange(parsed);
  };
  return (
    <input
      type="text"
      inputMode="decimal"
      aria-label={label}
      aria-invalid={invalid || error}
      disabled={disabled}
      value={text}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") setText(formatInputNumber(value, digits));
      }}
      className={cn("field-input w-20 text-right", className)}
    />
  );
}

function ValueText({ entry, value }: { entry: CatalogEntry; value: ParamValue }) {
  const { t } = useTranslation();
  if (value === null || value === undefined) return <span className="text-muted">—</span>;
  if (typeof value === "boolean") return <span>{value ? t("common.on") : t("common.off")}</span>;
  if (entry.type === "ranges") {
    // Variable-length list of (start, end) pairs (learned bands): "0–25%, 25–50%, 50–100%".
    const bands = value as number[][];
    if (!Array.isArray(bands) || !bands.length) return <span className="text-muted">—</span>;
    return (
      <span className="font-mono text-[12px]">
        {bands.map((band) => band.map((n) => `${fmtNumber(n * 100, 1)}%`).join("–")).join(", ")}
      </span>
    );
  }
  if (Array.isArray(value)) {
    if (entry.type === "multiselect")
      return <span className="font-mono text-[12px]">{(value as string[]).join(", ")}</span>;
    const numbers = value as number[];
    const join = entry.type === "size" ? "×" : entry.type === "range" ? "–" : ", ";
    return (
      <span className="font-mono text-[12px]">
        {numbers
          .map((n) =>
            isRelative(entry) ? `${fmtNumber(n * 100, 1)}%` : fmtNumber(n, digitsFor(entry)),
          )
          .join(join)}
      </span>
    );
  }
  if (typeof value === "number")
    return (
      <span className="font-mono text-[12px]">
        {isRelative(entry) ? `${fmtNumber(value * 100, 1)}%` : fmtNumber(value, digitsFor(entry))}
      </span>
    );
  return <span className="font-mono text-[12px]">{String(value)}</span>;
}

export function ParamControl({
  entry,
  value,
  onChange,
  disabled,
}: {
  entry: CatalogEntry;
  value: ParamValue;
  onChange: (value: ParamValue, final?: boolean) => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();
  const label = t(entry.label_key);
  const digits = digitsFor(entry);
  const dMin = entry.min !== null ? toDisplay(entry, entry.min) : undefined;
  const dMax = entry.max !== null ? toDisplay(entry, entry.max) : undefined;
  const dStep = entry.step !== null ? toDisplay(entry, entry.step) : 1;

  // A missing value is shown as text.
  if (entry.mode === "locked" || disabled || value === undefined) {
    return <ValueText entry={entry} value={value} />;
  }
  switch (entry.type) {
    case "bool":
      return (
        <Switch
          aria-label={label}
          checked={value === true}
          onCheckedChange={(v) => onChange(v, true)}
        />
      );
    case "enum":
      return (
        <Select
          label={label}
          value={String(value)}
          onValueChange={(v) => onChange(v, true)}
          options={(entry.options ?? []).map((o) => ({
            value: o,
            label: t(`params.option.${o}`, { defaultValue: o }),
          }))}
          className="w-36"
        />
      );
    case "multiselect": {
      const selected = new Set((value as string[]) ?? []);
      return (
        <div className="grid gap-1">
          {(entry.options ?? []).map((option) => (
            <label key={option} className="inline-flex items-center gap-2 font-mono text-[12px]">
              <Checkbox
                checked={selected.has(option)}
                onCheckedChange={(v) => {
                  const next = new Set(selected);
                  if (v) next.add(option);
                  else next.delete(option);
                  onChange(
                    (entry.options ?? []).filter((o) => next.has(o)),
                    true,
                  );
                }}
              />
              {option}
            </label>
          ))}
        </div>
      );
    }
    case "int":
    case "float": {
      if (entry.nullable && value === null) {
        return (
          <div className="flex items-center gap-2">
            <Switch
              aria-label={t("params.enable", { name: label })}
              checked={false}
              onCheckedChange={() => onChange(0.5, true)}
            />
            <span className="text-[12px] text-muted">{t("params.off")}</span>
          </div>
        );
      }
      const v = toDisplay(entry, Number(value));
      return (
        <div className="flex items-center gap-2">
          {entry.nullable ? (
            <Switch
              aria-label={t("params.enable", { name: label })}
              checked
              onCheckedChange={() => onChange(null, true)}
            />
          ) : null}
          {dMin !== undefined && dMax !== undefined ? (
            <Slider
              aria-label={label}
              className="w-40"
              min={dMin}
              max={dMax}
              step={dStep}
              value={[v]}
              onValueChange={([x]) => onChange(fromDisplay(entry, x), false)}
              onValueCommit={([x]) => onChange(fromDisplay(entry, x), true)}
            />
          ) : null}
          <NumberInput
            label={label}
            value={v}
            digits={digits}
            min={dMin}
            max={dMax}
            onChange={(x) => onChange(fromDisplay(entry, x), true)}
          />
        </div>
      );
    }
    case "range": {
      const [a, b] = (value as number[]).map((x) => toDisplay(entry, x));
      return (
        <div className="flex items-center gap-2">
          <NumberInput
            label={`${label} ${t("params.start")}`}
            value={a}
            digits={digits}
            min={dMin}
            max={dMax}
            onChange={(x) => onChange([fromDisplay(entry, x), fromDisplay(entry, b)], true)}
          />
          <Slider
            aria-label={label}
            className="w-72"
            min={dMin ?? 0}
            max={dMax ?? 100}
            step={dStep}
            minStepsBetweenThumbs={1}
            value={[a, b]}
            onValueChange={(xs) =>
              onChange(
                xs.map((x) => fromDisplay(entry, x)),
                false,
              )
            }
            onValueCommit={(xs) =>
              onChange(
                xs.map((x) => fromDisplay(entry, x)),
                true,
              )
            }
          />
          <NumberInput
            label={`${label} ${t("params.end")}`}
            value={b}
            digits={digits}
            min={dMin}
            max={dMax}
            onChange={(x) => onChange([fromDisplay(entry, a), fromDisplay(entry, x)], true)}
          />
        </div>
      );
    }
    case "ranges":
      return <ValueText entry={entry} value={value} />;
    case "rect":
    case "size":
    case "int3": {
      const values = value as number[];
      const names =
        entry.type === "rect"
          ? ["x0", "y0", "x1", "y1"]
          : entry.type === "size"
            ? ["w", "h"]
            : ["1", "2", "3"];
      return (
        <div className="flex flex-wrap items-center gap-1.5">
          {values.map((x, i) => (
            <label key={i} className="inline-flex items-center gap-1">
              <span className="font-mono text-[11px] text-muted">{names[i]}</span>
              <NumberInput
                label={`${label} ${names[i]}`}
                value={toDisplay(entry, x)}
                digits={digits}
                min={dMin}
                max={dMax}
                className="w-16"
                onChange={(n) => {
                  const next = [...values];
                  next[i] = entry.type === "rect" ? fromDisplay(entry, n) : Math.round(n);
                  onChange(next, true);
                }}
              />
            </label>
          ))}
        </div>
      );
    }
  }
}

export function ParamRow({
  entry,
  value,
  training,
  onChange,
  onReset,
  error,
}: {
  entry: CatalogEntry;
  value: ParamValue;
  training: ParamValue | undefined;
  onChange: (value: ParamValue, final?: boolean) => void;
  onReset: () => void;
  error?: string;
}) {
  const { t } = useTranslation();
  const differs = training !== undefined && JSON.stringify(value) !== JSON.stringify(training);
  const locked = entry.mode === "locked";
  return (
    <div
      className={cn(
        "grid grid-cols-[minmax(10rem,14rem)_1fr] items-start gap-3 border-b border-line px-3 py-2.5 last:border-0",
        differs && "bg-review-tint/40",
      )}
    >
      <div className="flex min-w-0 items-start gap-1.5 pt-1.5">
        <span className="text-[13px] font-medium leading-snug">{t(entry.label_key)}</span>
        <Tip
          content={
            <span>
              {t(entry.help_key)}
              <br />
              <span className="font-mono opacity-80">{entry.key}</span>
              {entry.config_constant !== "-" ? (
                <>
                  {" · "}
                  <span className="font-mono opacity-80">config.{entry.config_constant}</span>
                </>
              ) : null}
            </span>
          }
        >
          <button
            type="button"
            aria-label={t("params.help", { name: t(entry.label_key) })}
            className="text-muted hover:text-ink"
          >
            <CircleHelp className="size-3.5" />
          </button>
        </Tip>
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <div className="flex flex-wrap items-center gap-2">
          <ParamControl entry={entry} value={value} onChange={onChange} />
          {entry.unit && entry.unit !== "relative" && entry.unit !== "fraction" ? (
            <span className="text-[12px] text-muted">
              {t(`units.${entry.unit}`, { defaultValue: entry.unit })}
            </span>
          ) : null}
          {entry.unit === "relative" ? <span className="text-[12px] text-muted">%</span> : null}
          {locked ? (
            <Tip content={t("params.lockedHelp")}>
              <span tabIndex={0} className="inline-flex items-center gap-1 text-[12px] text-muted">
                <Lock className="size-3.5" aria-hidden="true" />
                {t("params.definedByTraining")}
              </span>
            </Tip>
          ) : null}
          {differs && !locked ? (
            <>
              <span className="inline-flex items-center gap-1 rounded-[4px] bg-review-tint px-1.5 py-px text-[12px] font-medium text-review">
                ≠ {t("params.training")} <ValueText entry={entry} value={training ?? null} />
              </span>
              <Tip content={t("params.resetField")}>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-6"
                  aria-label={t("params.resetFieldNamed", { name: t(entry.label_key) })}
                  onClick={onReset}
                >
                  <RotateCcw />
                </Button>
              </Tip>
            </>
          ) : null}
        </div>
        {error ? (
          <p role="alert" className="text-[12px] text-reject">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  );
}
