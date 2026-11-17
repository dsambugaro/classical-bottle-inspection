// Report filters shared by the dashboard and the card history: the URL is the state.
import { ChevronDown, Download, X } from "lucide-react";
import { type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router";

import { ClassLabel, VerdictChip } from "@/components/status";
import {
  Button,
  Checkbox,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
  Tip,
  cn,
} from "@/components/ui/primitives";
import { qs } from "@/lib/api";
import {
  DEFAULT_PRESET,
  HISTORY_LIST_KEYS,
  hasActiveFilters,
  historyQuery,
  presetFromParams,
  viewFromParams,
  type ReportView,
} from "@/lib/historyFilters";
import { fmtDate, fmtInt } from "@/lib/format";
import { CLASS_NAMES } from "@/lib/types";

export interface Options {
  profiles: {
    id: number;
    name: string;
    color: string;
    deleted: boolean;
    versions: { version: number; count: number }[];
  }[];
  sources: string[];
  cameras: string[];
  sessions: { id: number; count: number; started_at: string }[];
}

const LIST_KEYS = ["profile", "version", "source", "camera", "verdict", "class"] as const;
type ListKey = (typeof HISTORY_LIST_KEYS)[number];

// ------------------------------------------------------------------ filters (URL state)

export function useFilters() {
  const [params, setParams] = useSearchParams();
  const preset = presetFromParams(params);
  const query: Record<string, unknown> = {
    preset: preset || undefined,
    start: params.get("start"),
    end: params.get("end"),
    session: params.get("session"),
  };
  for (const key of LIST_KEYS) query[key] = params.getAll(key);
  const set = (key: string, values: string | string[] | null) => {
    const next = new URLSearchParams(params);
    next.delete(key);
    if (Array.isArray(values)) values.forEach((v) => next.append(key, v));
    else if (values) next.set(key, values);
    if (key === "start" || key === "end") next.delete("preset");
    setParams(next, { replace: true });
  };
  const view = viewFromParams(params);
  const setView = (next: ReportView) => {
    const updated = new URLSearchParams(params);
    if (next === "cards") updated.set("view", "cards");
    else updated.delete("view");
    setParams(updated, { replace: true });
  };
  const toggleFlag = (key: "wrong" | "disagree") => set(key, params.get(key) === "1" ? null : "1");
  const toggle = (key: ListKey, value: string) => {
    const current = params.getAll(key);
    set(key, current.includes(value) ? current.filter((v) => v !== value) : [...current, value]);
  };
  const clear = () => {
    const next = new URLSearchParams({ preset: preset || DEFAULT_PRESET });
    if (view === "cards") next.set("view", "cards");
    setParams(next, { replace: true });
  };
  const history = historyQuery(params);
  return {
    params,
    query,
    preset,
    set,
    toggle,
    toggleFlag,
    clear,
    key: qs(query),
    view,
    setView,
    historyQuery: history,
    historyKey: qs(history),
  };
}

/** A yes/no filter shown as a toggle button. */
function FlagFilter({
  label,
  hint,
  checked,
  onChange,
}: {
  label: string;
  hint: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <Tip content={hint}>
      <Button
        size="md"
        aria-pressed={checked}
        className={cn(checked && "border-accent bg-sunken text-accent")}
        onClick={onChange}
      >
        {label}
      </Button>
    </Tip>
  );
}

export function MultiFilter({
  label,
  values,
  options,
  onToggle,
}: {
  label: string;
  values: string[];
  options: { value: string; label: ReactNode }[];
  onToggle: (value: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button size="md" className={cn(values.length && "border-accent text-accent")}>
          {label}
          {values.length ? (
            <span className="font-mono">({values.length})</span>
          ) : (
            <span className="text-muted">{t("reports.all")}</span>
          )}
          <ChevronDown className="text-muted" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="max-h-96 w-80 overflow-auto p-1">
        {options.length === 0 ? (
          <p className="p-2 text-[13px] text-muted">{t("reports.noOptions")}</p>
        ) : null}
        {options.map((o) => (
          <label
            key={o.value}
            className="flex cursor-pointer items-center gap-2 rounded-[4px] px-2 py-1.5 text-[13px] hover:bg-sunken"
          >
            <Checkbox
              checked={values.includes(o.value)}
              onCheckedChange={() => onToggle(o.value)}
            />
            {o.label}
          </label>
        ))}
      </PopoverContent>
    </Popover>
  );
}

export function FilterBar({
  filters,
  options,
  cards,
}: {
  filters: ReturnType<typeof useFilters>;
  options?: Options;
  /** The card history: adds its own filters and drops the CSV export. */
  cards?: boolean;
}) {
  const { t, i18n } = useTranslation();
  const p = filters.params;
  const presets = ["last_hour", "today", "7d", "30d", "all", "custom"];
  const exportUrl = `/api/reports/export.csv${qs({ ...filters.query, lang: i18n.language })}`;
  const active = cards
    ? hasActiveFilters(p)
    : LIST_KEYS.some((k) => p.getAll(k).length) || p.get("session") || p.get("start");
  return (
    <div
      className="panel flex flex-wrap items-center gap-2 p-3"
      role="search"
      aria-label={t("reports.filters")}
    >
      <Select
        label={t("reports.range")}
        value={filters.preset || "custom"}
        onValueChange={(v) =>
          v === "custom"
            ? filters.set("start", new Date(Date.now() - 86400000).toISOString())
            : filters.set("preset", v)
        }
        options={presets.map((v) => ({ value: v, label: t(`reports.preset.${v}`) }))}
        className="w-40"
      />
      {!filters.preset ? (
        <>
          <input
            type="datetime-local"
            aria-label={t("reports.start")}
            className="field-input"
            value={p.get("start") ? toLocalInput(p.get("start")!) : ""}
            onChange={(e) =>
              filters.set("start", e.target.value ? new Date(e.target.value).toISOString() : null)
            }
          />
          <input
            type="datetime-local"
            aria-label={t("reports.end")}
            className="field-input"
            value={p.get("end") ? toLocalInput(p.get("end")!) : ""}
            onChange={(e) =>
              filters.set("end", e.target.value ? new Date(e.target.value).toISOString() : null)
            }
          />
        </>
      ) : null}
      <MultiFilter
        label={t("reports.profiles")}
        values={p.getAll("profile")}
        onToggle={(v) => filters.toggle("profile", v)}
        options={(options?.profiles ?? [])
          .filter((x) => x.versions.length)
          .map((x) => ({
            value: String(x.id),
            label: (
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3.5 w-1 rounded-full" style={{ background: x.color }} />
                {x.name}
              </span>
            ),
          }))}
      />
      <MultiFilter
        label={t("reports.versions")}
        values={p.getAll("version")}
        onToggle={(v) => filters.toggle("version", v)}
        options={(options?.profiles ?? []).flatMap((x) =>
          x.versions.map((v) => ({
            value: `${x.id}:${v.version}`,
            label: `${x.name} · v${v.version} (${fmtInt(v.count)})`,
          })),
        )}
      />
      <MultiFilter
        label={t("reports.source")}
        values={[...p.getAll("source"), ...p.getAll("camera").map((c) => `camera:${c}`)]}
        onToggle={(v) =>
          v.startsWith("camera:")
            ? filters.toggle("camera", v.slice(7))
            : filters.toggle("source", v)
        }
        options={[
          ...(options?.sources ?? []).map((s) => ({
            value: s,
            label: t(`reports.sources.${s}`, { defaultValue: s }),
          })),
          ...(options?.cameras ?? []).map((c) => ({
            value: `camera:${c}`,
            label: `${t("reports.camera")}: ${c}`,
          })),
        ]}
      />
      <Select
        label={t("reports.session")}
        value={p.get("session") ?? "all"}
        onValueChange={(v) => filters.set("session", v === "all" ? null : v)}
        options={[
          { value: "all", label: t("reports.allSessions") },
          ...(options?.sessions ?? []).map((s) => ({
            value: String(s.id),
            label: `#${s.id} · ${fmtDate(s.started_at)} (${fmtInt(s.count)})`,
          })),
        ]}
        className="w-48"
      />
      <MultiFilter
        label={t("reports.verdict")}
        values={p.getAll("verdict")}
        onToggle={(v) => filters.toggle("verdict", v)}
        options={(["pass", "reject", "review"] as const).map((v) => ({
          value: v,
          label: <VerdictChip verdict={v} />,
        }))}
      />
      <MultiFilter
        label={t("reports.class")}
        values={p.getAll("class")}
        onToggle={(v) => filters.toggle("class", v)}
        options={CLASS_NAMES.map((c) => ({
          value: c,
          label: <ClassLabel name={c} withDescription />,
        }))}
      />
      {cards ? (
        <>
          <MultiFilter
            label={t("historyView.truth")}
            values={p.getAll("truth")}
            onToggle={(v) => filters.toggle("truth", v)}
            options={CLASS_NAMES.map((c) => ({
              value: c,
              label: <ClassLabel name={c} withDescription />,
            }))}
          />
          <FlagFilter
            label={t("historyView.onlyWrong")}
            hint={t("historyView.onlyWrongHint")}
            checked={p.get("wrong") === "1"}
            onChange={() => filters.toggleFlag("wrong")}
          />
          <FlagFilter
            label={t("historyView.disagree")}
            hint={t("historyView.disagreeHint")}
            checked={p.get("disagree") === "1"}
            onChange={() => filters.toggleFlag("disagree")}
          />
          <input
            type="search"
            aria-label={t("historyView.search")}
            placeholder={t("historyView.search")}
            className="field-input w-44"
            value={p.get("q") ?? ""}
            onChange={(e) => filters.set("q", e.target.value || null)}
          />
        </>
      ) : null}
      {active ? (
        <Button variant="ghost" onClick={filters.clear}>
          <X />
          {t("reports.clear")}
        </Button>
      ) : null}
      {cards ? null : (
        <a
          href={exportUrl}
          className="ml-auto inline-flex h-8 items-center gap-1.5 rounded-[6px] border border-line bg-panel px-3 text-[13px] font-medium hover:bg-sunken"
        >
          <Download className="size-4" aria-hidden="true" />
          {t("reports.exportCsv")}
        </a>
      )}
    </div>
  );
}

export function toLocalInput(iso: string): string {
  const date = new Date(iso);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}
