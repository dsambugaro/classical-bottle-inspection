// Reports (§11, §15.6): filters kept in the URL, KPI tiles with deltas, charts with tooltips,
// table view, PNG export and cross-filtering, analyses table with a side panel, CSV export.
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowUp, X } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import {
  ChartCard,
  DataTable,
  axisStyle,
  baseOption,
  INK,
  type ChartOption,
} from "@/components/charts/Chart";
import { ResultCard } from "@/components/ResultCard";
import { ClassLabel, VerdictChip, VERDICT_HEX } from "@/components/status";
import { Button, Spinner, cn } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import type { ReportView } from "@/lib/historyFilters";
import { fmtDate, fmtInt, fmtMs, fmtNumber, fmtPercent } from "@/lib/format";
import { CLASS_COLORS, CLASS_NAMES, type AnalysisResult, type ClassName } from "@/lib/types";
import { ConfusionMatrix } from "./profiles/ModelTab";
import { ResultViewer } from "./InspectPage";
import { FilterBar, useFilters, type Options } from "./reports/filters";
import { HistoryView, ViewSwitch } from "./reports/HistoryView";

// ------------------------------------------------------------------ types

interface Kpis {
  analyses: number;
  inspected: number;
  pass: number;
  reject: number;
  review: number;
  pass_rate: number | null;
  reject_rate: number | null;
  first_pass_yield: number | null;
  mean_ms: number | null;
  p95_ms: number | null;
  throughput_per_min: number | null;
  rules_fired: number;
}
interface Summary {
  kpis: Kpis;
  previous: Kpis | null;
  class_distribution: { class: ClassName; count: number; share: number }[];
}
interface Pareto {
  total_rejects: number;
  items: { class: ClassName; count: number; share: number; cumulative: number }[];
}
interface Series {
  bucket: "minute" | "hour" | "day";
  series: {
    t: string;
    pass: number;
    reject: number;
    review: number;
    rejects_by_class: Record<ClassName, number>;
  }[];
}
interface ProfilesReport {
  profiles: {
    profile_id: number;
    name: string;
    color: string;
    total: number;
    verdicts: Record<string, number>;
    classes: Record<ClassName, number>;
  }[];
  agreement: {
    profiles: [number, number];
    inspections: number;
    verdict_agreement: number;
    class_agreement: number;
  }[];
}
interface Confidence {
  edges: number[];
  profiles: {
    profile_id: number;
    name: string;
    color: string;
    algorithm: string;
    min_confidence: number | null;
    bins: number[];
  }[];
}
interface TimingsReport {
  steps: string[];
  profiles: {
    profile_id: number;
    name: string;
    color: string;
    count: number;
    mean_total_ms: number;
    steps: Record<string, number>;
  }[];
}
interface Confusion {
  profiles: {
    profile_id: number;
    name: string;
    color: string;
    samples: number;
    accuracy: number | null;
    f1_macro: number;
    matrix: number[][];
  }[];
}
// ------------------------------------------------------------------ KPI tiles

function Delta({
  current,
  previous,
  better,
  format,
}: {
  current: number | null;
  previous: number | null | undefined;
  better: "up" | "down" | "none";
  format: (v: number) => string;
}) {
  const { t } = useTranslation();
  if (current === null || previous === null || previous === undefined)
    return <span className="text-[12px] text-muted">—</span>;
  const diff = current - previous;
  if (Math.abs(diff) < 1e-9)
    return <span className="text-[12px] text-muted">{t("reports.noChange")}</span>;
  const good = better === "none" ? null : diff > 0 === (better === "up");
  const Icon = diff > 0 ? ArrowUp : ArrowDown;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 text-[12px] font-medium",
        good === null ? "text-ink-2" : good ? "text-pass" : "text-reject",
      )}
    >
      <Icon className="size-3" aria-hidden="true" />
      {format(Math.abs(diff))}
      <span className="sr-only">{t("reports.vsPrevious")}</span>
    </span>
  );
}

function Tile({
  label,
  value,
  sub,
  delta,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  delta: ReactNode;
}) {
  return (
    <div className="panel flex min-w-0 flex-col gap-0.5 p-3">
      <span className="text-[12px] text-ink-2">{label}</span>
      <span className="truncate font-mono text-[22px] font-semibold leading-tight">{value}</span>
      <span className="flex items-center gap-2 text-[12px] text-muted">
        {sub}
        {delta}
      </span>
    </div>
  );
}

function KpiTiles({ summary }: { summary: Summary }) {
  const { t } = useTranslation();
  const k = summary.kpis;
  const p = summary.previous;
  const pct = (v: number) => fmtPercent(v, 1);
  return (
    <div
      className="grid grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-8"
      aria-label={t("reports.kpis")}
    >
      <Tile
        label={t("reports.kpi.inspected")}
        value={fmtInt(k.inspected)}
        sub={t("reports.kpi.analyses", { count: k.analyses })}
        delta={
          <Delta current={k.inspected} previous={p?.inspected} better="none" format={fmtInt} />
        }
      />
      <Tile
        label={t("reports.kpi.pass")}
        value={fmtInt(k.pass)}
        sub={pct(k.pass_rate ?? 0)}
        delta={<Delta current={k.pass_rate} previous={p?.pass_rate} better="up" format={pct} />}
      />
      <Tile
        label={t("reports.kpi.reject")}
        value={fmtInt(k.reject)}
        sub={pct(k.reject_rate ?? 0)}
        delta={
          <Delta current={k.reject_rate} previous={p?.reject_rate} better="down" format={pct} />
        }
      />
      <Tile
        label={t("reports.kpi.review")}
        value={fmtInt(k.review)}
        delta={<Delta current={k.review} previous={p?.review} better="down" format={fmtInt} />}
      />
      <Tile
        label={t("reports.kpi.fpy")}
        value={fmtPercent(k.first_pass_yield, 1)}
        delta={
          <Delta
            current={k.first_pass_yield}
            previous={p?.first_pass_yield}
            better="up"
            format={pct}
          />
        }
      />
      <Tile
        label={t("reports.kpi.time")}
        value={fmtMs(k.mean_ms)}
        sub={`p95 ${fmtMs(k.p95_ms)}`}
        delta={<Delta current={k.mean_ms} previous={p?.mean_ms} better="down" format={fmtMs} />}
      />
      <Tile
        label={t("reports.kpi.throughput")}
        value={fmtNumber(k.throughput_per_min, 1)}
        sub={t("reports.kpi.perMinute")}
        delta={
          <Delta
            current={k.throughput_per_min}
            previous={p?.throughput_per_min}
            better="none"
            format={(v) => fmtNumber(v, 1)}
          />
        }
      />
      <Tile
        label={t("reports.kpi.rules")}
        value={fmtInt(k.rules_fired)}
        delta={
          <Delta current={k.rules_fired} previous={p?.rules_fired} better="none" format={fmtInt} />
        }
      />
    </div>
  );
}

// ------------------------------------------------------------------ charts

function classBarOption(data: Summary["class_distribution"]): ChartOption {
  return {
    ...baseOption(),
    legend: { show: false },
    tooltip: {
      ...(baseOption().tooltip as object),
      trigger: "item",
      formatter: (p: { name: string; value: number; dataIndex: number }) =>
        `${p.name}<br/><b>${fmtInt(p.value)}</b> · ${fmtPercent(data[p.dataIndex].share, 1)}`,
    },
    grid: { left: 8, right: 72, top: 8, bottom: 8, containLabel: true },
    xAxis: { type: "value", ...axisStyle },
    yAxis: {
      type: "category",
      inverse: true,
      data: data.map((d) => d.class),
      ...axisStyle,
      axisLabel: { color: INK.primary, fontFamily: "IBM Plex Mono", fontSize: 11 },
    },
    series: [
      {
        type: "bar",
        barMaxWidth: 18,
        data: data.map((d) => ({
          value: d.count,
          itemStyle: { color: CLASS_COLORS[d.class], borderRadius: [0, 4, 4, 0] },
        })),
        label: {
          show: true,
          position: "right",
          color: INK.secondary,
          fontFamily: "IBM Plex Mono",
          fontSize: 11,
          formatter: (p: { dataIndex: number; value: number }) =>
            `${fmtInt(p.value)} · ${fmtPercent(data[p.dataIndex].share, 0)}`,
        },
      },
    ],
  };
}

function paretoOption(pareto: Pareto, t: (k: string) => string): ChartOption {
  const items = pareto.items.filter((i) => i.count > 0);
  // One axis (0–100 %): bars are each class's share of the rejects, the line is cumulative.
  return {
    ...baseOption(),
    tooltip: {
      ...(baseOption().tooltip as object),
      formatter: (params: { dataIndex: number }[]) => {
        const item = items[params[0].dataIndex];
        return `${item.class}<br/>${t("reports.rejects")}: <b>${fmtInt(item.count)}</b> (${fmtPercent(item.share, 1)})<br/>${t("reports.cumulative")}: ${fmtPercent(item.cumulative, 1)}`;
      },
    },
    grid: { left: 8, right: 16, top: 28, bottom: 8, containLabel: true },
    xAxis: {
      type: "category",
      data: items.map((i) => i.class),
      ...axisStyle,
      axisLabel: { color: INK.primary, fontFamily: "IBM Plex Mono", fontSize: 10, rotate: 30 },
    },
    yAxis: {
      type: "value",
      max: 1,
      ...axisStyle,
      axisLabel: { color: INK.muted, formatter: (v: number) => fmtPercent(v, 0) },
    },
    series: [
      {
        name: t("reports.shareOfRejects"),
        type: "bar",
        barMaxWidth: 36,
        data: items.map((i) => ({
          value: i.share,
          itemStyle: { color: CLASS_COLORS[i.class], borderRadius: [4, 4, 0, 0] },
        })),
      },
      {
        name: t("reports.cumulative"),
        type: "line",
        data: items.map((i) => i.cumulative),
        lineStyle: { width: 2, color: INK.primary },
        itemStyle: { color: INK.primary },
        symbolSize: 8,
        label: {
          show: true,
          position: "top",
          color: INK.secondary,
          fontSize: 10,
          formatter: (p: { value: number }) => fmtPercent(p.value, 0),
        },
      },
    ],
  };
}

function timeLabel(iso: string, bucket: Series["bucket"]): string {
  const date = new Date(iso);
  return bucket === "day"
    ? new Intl.DateTimeFormat(undefined, { day: "2-digit", month: "2-digit" }).format(date)
    : new Intl.DateTimeFormat(undefined, {
        day: "2-digit",
        month: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

const BUCKET_MS = { minute: 60_000, hour: 3_600_000, day: 86_400_000 } as const;
const MAX_BUCKETS = 2000;

/** Uniform buckets from the first to the last one, empty buckets filled with zeros, so the
 * x axis spacing is proportional to time. */
export function fillGaps(series: Series): Series["series"] {
  if (series.series.length < 2) return series.series;
  const step = BUCKET_MS[series.bucket];
  const first = Date.parse(series.series[0].t);
  const last = Date.parse(series.series[series.series.length - 1].t);
  if ((last - first) / step > MAX_BUCKETS) return series.series;
  const byTime = new Map(series.series.map((s) => [Date.parse(s.t), s]));
  const zeros = Object.fromEntries(CLASS_NAMES.map((c) => [c, 0])) as Record<ClassName, number>;
  const filled: Series["series"] = [];
  for (let time = first; time <= last; time += step) {
    filled.push(
      byTime.get(time) ?? {
        t: new Date(time).toISOString(),
        pass: 0,
        reject: 0,
        review: 0,
        rejects_by_class: zeros,
      },
    );
  }
  return filled;
}

function goodBadOption(raw: Series, t: (k: string) => string): ChartOption {
  const series = { ...raw, series: fillGaps(raw) };
  const x = series.series.map((s) => timeLabel(s.t, series.bucket));
  const area = (name: string, color: string, data: number[]) => ({
    name,
    type: "line",
    stack: "total",
    smooth: false,
    symbol: "circle",
    symbolSize: 8,
    showSymbol: data.length <= 2,
    lineStyle: { width: 2, color },
    itemStyle: { color },
    areaStyle: { color, opacity: 0.25 },
    data,
  });
  return {
    ...baseOption(),
    xAxis: { type: "category", data: x, boundaryGap: false, ...axisStyle },
    yAxis: { type: "value", ...axisStyle },
    series: [
      area(
        t("verdict.pass"),
        VERDICT_HEX.pass,
        series.series.map((s) => s.pass),
      ),
      area(
        t("reports.defective"),
        VERDICT_HEX.reject,
        series.series.map((s) => s.reject + s.review),
      ),
    ],
  };
}

function rejectsByClassOption(raw: Series): ChartOption {
  const series = { ...raw, series: fillGaps(raw) };
  const x = series.series.map((s) => timeLabel(s.t, series.bucket));
  const classes = CLASS_NAMES.filter((c) => series.series.some((s) => s.rejects_by_class[c]));
  return {
    ...baseOption(),
    legend: {
      ...(baseOption().legend as object),
      type: "scroll",
      itemGap: 14,
      textStyle: { fontFamily: "IBM Plex Mono, monospace", fontSize: 11, color: INK.secondary },
    },
    grid: { left: 8, right: 16, top: 36, bottom: 8, containLabel: true },
    xAxis: { type: "category", data: x, ...axisStyle },
    yAxis: { type: "value", ...axisStyle },
    series: classes.map((c) => ({
      name: c,
      type: "bar",
      stack: "rejects",
      barMaxWidth: 28,
      itemStyle: { color: CLASS_COLORS[c], borderColor: "#FFFFFF", borderWidth: 1 },
      data: series.series.map((s) => s.rejects_by_class[c] ?? 0),
    })),
  };
}

function histogramOption(
  profile: Confidence["profiles"][number],
  edges: number[],
  t: (k: string) => string,
): ChartOption {
  return {
    ...baseOption(),
    legend: { show: false },
    grid: { left: 8, right: 12, top: 12, bottom: 8, containLabel: true },
    tooltip: {
      ...(baseOption().tooltip as object),
      formatter: (p: { dataIndex: number; value: number }[]) =>
        `${fmtNumber(edges[p[0].dataIndex], 2)}–${fmtNumber(edges[p[0].dataIndex + 1], 2)}: <b>${fmtInt(p[0].value)}</b>`,
    },
    xAxis: {
      type: "category",
      data: edges.slice(0, -1).map((e) => fmtNumber(e, 2)),
      ...axisStyle,
      axisLabel: { color: INK.muted, interval: 4 },
    },
    yAxis: { type: "value", ...axisStyle },
    series: [
      {
        type: "bar",
        barCategoryGap: "8%",
        data: profile.bins,
        itemStyle: { color: profile.color, borderRadius: [3, 3, 0, 0] },
        markLine:
          profile.min_confidence !== null
            ? {
                symbol: "none",
                silent: true,
                lineStyle: { color: VERDICT_HEX.review, width: 2, type: "dashed" },
                label: { formatter: t("reports.minConfidence"), color: VERDICT_HEX.review },
                data: [
                  {
                    xAxis: Math.min(
                      edges.length - 2,
                      Math.floor(profile.min_confidence * (edges.length - 1)),
                    ),
                  },
                ],
              }
            : undefined,
      },
    ],
  };
}

// ------------------------------------------------------------------ analyses table + panel

type SortColumn =
  "created_at" | "profile_id" | "predicted_class" | "verdict" | "confidence" | "total_ms";

function AnalysesTable({
  filterKey,
  onOpen,
  onView,
}: {
  filterKey: string;
  onOpen: (a: AnalysisResult) => void;
  onView: (view: ReportView) => void;
}) {
  const { t } = useTranslation();
  const [page, setPage] = useState(0);
  const [sort, setSort] = useState<{ column: string; order: "asc" | "desc" }>({
    column: "created_at",
    order: "desc",
  });
  const pageSize = 25;
  const sep = filterKey ? "&" : "?";
  const data = useQuery({
    queryKey: ["report-analyses", filterKey, page, sort],
    queryFn: () =>
      api.get<{ total: number; items: AnalysisResult[] }>(
        `/api/analyses${filterKey}${sep}limit=${pageSize}&offset=${page * pageSize}&sort=${sort.column}&order=${sort.order}`,
      ),
    // Keeps the current rows on screen while a new page or sort loads, so the table
    // height does not collapse and the scroll position is preserved.
    placeholderData: keepPreviousData,
  });
  const pages = Math.max(1, Math.ceil((data.data?.total ?? 0) / pageSize));
  const header = (column: SortColumn, align?: string) => (
    <th
      className={cn("px-2 py-1.5 font-medium", align)}
      aria-sort={
        sort.column === column ? (sort.order === "asc" ? "ascending" : "descending") : "none"
      }
    >
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:text-ink"
        onClick={() =>
          setSort({
            column,
            order: sort.column === column && sort.order === "desc" ? "asc" : "desc",
          })
        }
      >
        {t(`reports.col.${column}`)}
        {sort.column === column ? (
          sort.order === "asc" ? (
            <ArrowUp className="size-3" />
          ) : (
            <ArrowDown className="size-3" />
          )
        ) : null}
      </button>
    </th>
  );
  return (
    <section className="panel p-3" aria-label={t("reports.analyses")}>
      <div className="mb-2 flex items-center gap-2">
        <h3 className="font-semibold">{t("reports.analyses")}</h3>
        <span className="text-[12px] text-muted">
          {t("reports.analysesCount", { count: data.data?.total ?? 0 })}
        </span>
        {data.isFetching ? <Spinner /> : null}
        <ViewSwitch view="dashboard" onChange={onView} className="ml-auto" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[12px] text-ink-2">
              <th className="px-2 py-1.5 font-medium">
                <span className="sr-only">{t("reports.col.image")}</span>
              </th>
              {header("created_at")}
              {header("profile_id")}
              {header("predicted_class")}
              {header("verdict")}
              {header("confidence", "text-right")}
              {header("total_ms", "text-right")}
            </tr>
          </thead>
          <tbody>
            {data.data?.items.map((a) => (
              <tr
                key={a.analysis_id}
                tabIndex={0}
                onClick={() => onOpen(a)}
                onKeyDown={(e) => e.key === "Enter" && onOpen(a)}
                className="cursor-pointer border-b border-line last:border-0 hover:bg-sunken focus-visible:bg-sunken"
              >
                <td className="px-2 py-1">
                  {a.image.thumb_url ? (
                    <img
                      src={a.image.thumb_url}
                      alt=""
                      loading="lazy"
                      className="h-8 w-12 rounded-[3px] object-cover"
                    />
                  ) : (
                    <span className="block h-8 w-12 rounded-[3px] bg-sunken" />
                  )}
                </td>
                <td className="whitespace-nowrap px-2 py-1">{fmtDate(a.created_at, true)}</td>
                <td className="px-2 py-1">
                  <span className="inline-flex items-center gap-1.5">
                    <span
                      className="h-3.5 w-1 rounded-full"
                      style={{ background: a.profile.color }}
                    />
                    {a.profile.name}{" "}
                    <span className="font-mono text-[11px] text-muted">v{a.profile.version}</span>
                  </span>
                </td>
                <td className="px-2 py-1">
                  <ClassLabel name={a.predicted_class} />
                </td>
                <td className="px-2 py-1">
                  <VerdictChip verdict={a.verdict} />
                </td>
                <td className="px-2 py-1 text-right font-mono">{fmtNumber(a.confidence, 2)}</td>
                <td className="px-2 py-1 text-right font-mono">{fmtMs(a.timings_ms.total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {data.data && !data.data.items.length ? (
          <p className="py-6 text-center text-[13px] text-muted">{t("reports.noData")}</p>
        ) : null}
      </div>
      <div className="mt-2 flex items-center justify-end gap-2 text-[12px]">
        <Button size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
          {t("reports.previous")}
        </Button>
        <span className="font-mono">
          {page + 1} / {pages}
        </span>
        <Button size="sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>
          {t("reports.next")}
        </Button>
      </div>
    </section>
  );
}

function SidePanel({
  analysis,
  onClose,
}: {
  analysis: AnalysisResult | null;
  onClose: () => void;
}) {
  const { t } = useTranslation();
  if (!analysis) return null;
  return (
    <div
      className="fixed inset-0 z-40 flex justify-end bg-ink/20"
      onClick={onClose}
      role="presentation"
    >
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={t("reports.analysisDetail")}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.key === "Escape" && onClose()}
        className="flex h-full w-[min(760px,100vw)] flex-col gap-3 overflow-auto bg-bg p-4 shadow-[var(--shadow-float)]"
      >
        <div className="flex items-center gap-2">
          <h2 className="font-semibold">{t("reports.analysisDetail")}</h2>
          <span className="font-mono text-[12px] text-muted">#{analysis.analysis_id}</span>
          <Button
            className="ml-auto"
            size="icon"
            variant="ghost"
            aria-label={t("common.close")}
            onClick={onClose}
            autoFocus
          >
            <X />
          </Button>
        </div>
        {analysis.image.url ? (
          <ResultViewer
            result={analysis}
            overlays={{ rois: true, bottle: true, patch: false, mask: false }}
            compare={false}
            className="aspect-[16/9] w-full"
          />
        ) : (
          <p className="rounded-[6px] bg-sunken p-4 text-center text-[13px] text-muted">
            {t("reports.imageDeleted")}
          </p>
        )}
        <ResultCard result={analysis} />
      </aside>
    </div>
  );
}

// ------------------------------------------------------------------ page

/** The reports: the charts dashboard with the analyses table, or the card history. */
export default function ReportsPage() {
  const view = useFilters().view;
  return view === "cards" ? <HistoryView /> : <ReportsDashboard />;
}

function ReportsDashboard() {
  const { t } = useTranslation();
  const filters = useFilters();
  const key = filters.key;
  const get =
    <T,>(path: string) =>
    () =>
      api.get<T>(`/api/reports/${path}${key}`);
  const options = useQuery({
    queryKey: ["report-options"],
    queryFn: () => api.get<Options>("/api/reports/options"),
  });
  const summary = useQuery({ queryKey: ["r-summary", key], queryFn: get<Summary>("summary") });
  const pareto = useQuery({ queryKey: ["r-pareto", key], queryFn: get<Pareto>("pareto") });
  const series = useQuery({ queryKey: ["r-series", key], queryFn: get<Series>("timeseries") });
  const profiles = useQuery({
    queryKey: ["r-profiles", key],
    queryFn: get<ProfilesReport>("profiles"),
  });
  const confidence = useQuery({
    queryKey: ["r-confidence", key],
    queryFn: get<Confidence>("confidence"),
  });
  const timings = useQuery({
    queryKey: ["r-timings", key],
    queryFn: get<TimingsReport>("timings"),
  });
  const confusion = useQuery({
    queryKey: ["r-confusion", key],
    queryFn: get<Confusion>("confusion"),
  });
  const [open, setOpen] = useState<AnalysisResult | null>(null);

  const profileName = useMemo(() => {
    const map = new Map<number, string>();
    options.data?.profiles.forEach((p) => map.set(p.id, p.name));
    return (id: number) => map.get(id) ?? `#${id}`;
  }, [options.data]);

  const empty = summary.data?.kpis.analyses === 0;
  const filterClass = (params: { name: string }) =>
    CLASS_NAMES.includes(params.name as ClassName) && filters.toggle("class", params.name);

  return (
    <div className="flex flex-col gap-3 p-4">
      <FilterBar filters={filters} options={options.data} />
      {summary.data ? <KpiTiles summary={summary.data} /> : <Spinner />}
      {empty ? (
        <div className="panel p-8 text-center">
          <p className="font-semibold">{t("reports.emptyTitle")}</p>
          <p className="mt-1 text-[13px] text-ink-2">{t("reports.emptyBody")}</p>
        </div>
      ) : null}
      <div className="grid gap-3 lg:grid-cols-2">
        <ChartCard
          title={t("reports.goodVsDefective")}
          subtitle={series.data ? t(`reports.bucket.${series.data.bucket}`) : undefined}
          empty={!series.data?.series.length}
          option={series.data ? goodBadOption(series.data, t) : undefined}
          table={
            <DataTable
              caption={t("reports.goodVsDefective")}
              head={[
                t("reports.time"),
                t("verdict.pass"),
                t("verdict.reject"),
                t("verdict.review"),
              ]}
              rows={(series.data?.series ?? []).map((s) => [
                fmtDate(s.t),
                fmtInt(s.pass),
                fmtInt(s.reject),
                fmtInt(s.review),
              ])}
            />
          }
        />
        <ChartCard
          title={t("reports.pareto")}
          subtitle={t("reports.paretoHint", { count: pareto.data?.total_rejects ?? 0 })}
          empty={!pareto.data?.total_rejects}
          option={pareto.data ? paretoOption(pareto.data, t) : undefined}
          onClick={filterClass}
          table={
            <DataTable
              caption={t("reports.pareto")}
              head={[
                t("reports.class"),
                t("reports.rejects"),
                t("reports.share"),
                t("reports.cumulative"),
              ]}
              rows={(pareto.data?.items ?? []).map((i) => [
                <ClassLabel key={i.class} name={i.class} />,
                fmtInt(i.count),
                fmtPercent(i.share, 1),
                fmtPercent(i.cumulative, 1),
              ])}
            />
          }
        />
        <ChartCard
          title={t("reports.classDistribution")}
          subtitle={t("reports.clickToFilter")}
          empty={!summary.data?.kpis.analyses}
          height={250}
          option={summary.data ? classBarOption(summary.data.class_distribution) : undefined}
          onClick={filterClass}
          table={
            <DataTable
              caption={t("reports.classDistribution")}
              head={[t("reports.class"), t("reports.count"), t("reports.share")]}
              rows={(summary.data?.class_distribution ?? []).map((d) => [
                <ClassLabel key={d.class} name={d.class} withDescription />,
                fmtInt(d.count),
                fmtPercent(d.share, 1),
              ])}
            />
          }
        />
        <ChartCard
          title={t("reports.rejectsByClass")}
          empty={!series.data?.series.some((s) => s.reject)}
          option={series.data ? rejectsByClassOption(series.data) : undefined}
          onClick={(p) => p.seriesName && filterClass({ name: p.seriesName })}
          table={
            <DataTable
              caption={t("reports.rejectsByClass")}
              head={[
                t("reports.time"),
                ...CLASS_NAMES.map((c) => (
                  <span key={c} className="font-mono text-[11px]">
                    {c}
                  </span>
                )),
              ]}
              rows={(series.data?.series ?? []).map((s) => [
                fmtDate(s.t),
                ...CLASS_NAMES.map((c) => fmtInt(s.rejects_by_class[c])),
              ])}
            />
          }
        />
      </div>

      {/* Per-profile comparison */}
      <section className="panel p-3" aria-label={t("reports.profilesTable")}>
        <h3 className="mb-2 font-semibold">{t("reports.profilesTable")}</h3>
        {profiles.data?.profiles.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-line text-left text-ink-2">
                  <th className="py-1 pr-3 font-medium">{t("reports.profile")}</th>
                  <th className="py-1 pr-3 text-right font-medium">{t("reports.total")}</th>
                  {CLASS_NAMES.map((c) => (
                    <th key={c} className="py-1 pr-3 text-right font-medium">
                      <span className="inline-flex items-center gap-1 font-mono text-[11px]">
                        <span
                          className="size-2 rounded-[2px]"
                          style={{ background: CLASS_COLORS[c] }}
                        />
                        {c}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {profiles.data.profiles.map((p) => (
                  <tr key={p.profile_id} className="border-b border-line last:border-0">
                    <td className="py-1 pr-3">
                      <span className="inline-flex items-center gap-1.5">
                        <span className="h-3.5 w-1 rounded-full" style={{ background: p.color }} />
                        {p.name}
                      </span>
                    </td>
                    <td className="py-1 pr-3 text-right font-mono">{fmtInt(p.total)}</td>
                    {CLASS_NAMES.map((c) => (
                      <td key={c} className="py-1 pr-3 text-right font-mono">
                        {fmtInt(p.classes[c])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            {profiles.data.agreement.length ? (
              <>
                <h4 className="mb-1 mt-4 font-semibold">{t("reports.agreement")}</h4>
                <DataTable
                  caption={t("reports.agreement")}
                  head={[
                    t("reports.pair"),
                    t("reports.sharedInspections"),
                    t("reports.verdictAgreement"),
                    t("reports.classAgreement"),
                  ]}
                  rows={profiles.data.agreement.map((a) => [
                    `${profileName(a.profiles[0])} × ${profileName(a.profiles[1])}`,
                    fmtInt(a.inspections),
                    fmtPercent(a.verdict_agreement, 1),
                    fmtPercent(a.class_agreement, 1),
                  ])}
                />
              </>
            ) : null}
          </div>
        ) : (
          <p className="py-6 text-center text-[13px] text-muted">{t("reports.noData")}</p>
        )}
      </section>

      <div className="grid gap-3 lg:grid-cols-2">
        <section className="panel p-3" aria-label={t("reports.confidence")}>
          <h3 className="font-semibold">{t("reports.confidence")}</h3>
          <p className="mb-2 text-[12px] text-muted">{t("confidence.notComparable")}</p>
          {confidence.data?.profiles.length ? (
            <div className="grid gap-2 sm:grid-cols-2">
              {confidence.data.profiles.map((p) => (
                <ChartCard
                  key={p.profile_id}
                  className="border-0 p-0"
                  title={p.name}
                  subtitle={p.algorithm}
                  height={150}
                  option={histogramOption(p, confidence.data!.edges, t)}
                  table={
                    <DataTable
                      caption={p.name}
                      head={[t("reports.bin"), t("reports.count")]}
                      rows={p.bins.map((n, i) => [
                        `${fmtNumber(confidence.data!.edges[i], 2)}–${fmtNumber(confidence.data!.edges[i + 1], 2)}`,
                        fmtInt(n),
                      ])}
                    />
                  }
                />
              ))}
            </div>
          ) : (
            <p className="py-6 text-center text-[13px] text-muted">{t("reports.noData")}</p>
          )}
        </section>
        <section className="panel p-3" aria-label={t("reports.timings")}>
          <h3 className="mb-2 font-semibold">{t("reports.timings")}</h3>
          {timings.data?.profiles.length ? (
            <div className="overflow-x-auto">
              <table className="w-full text-[12px]">
                <caption className="sr-only">{t("reports.timings")}</caption>
                <thead>
                  <tr className="border-b border-line text-left text-ink-2">
                    <th className="py-1 pr-3 font-medium">{t("reports.step")}</th>
                    {timings.data.profiles.map((p) => (
                      <th key={p.profile_id} className="py-1 pr-3 text-right font-medium">
                        <span className="inline-flex items-center gap-1">
                          <span className="h-3 w-1 rounded-full" style={{ background: p.color }} />
                          {p.name}
                        </span>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {timings.data.steps.map((step) => {
                    const max = Math.max(
                      ...timings.data!.profiles.map((p) => p.steps[step] ?? 0),
                      0.001,
                    );
                    return (
                      <tr key={step} className="border-b border-line">
                        <td className="py-1 pr-3">{t(`steps.${step}`)}</td>
                        {timings.data!.profiles.map((p) => (
                          <td key={p.profile_id} className="py-1 pr-3">
                            <span className="flex items-center justify-end gap-2">
                              <span
                                className="h-1.5 rounded-full bg-ink-2/50"
                                style={{ width: `${((p.steps[step] ?? 0) / max) * 48}px` }}
                              />
                              <span className="w-16 text-right font-mono">
                                {fmtMs(p.steps[step])}
                              </span>
                            </span>
                          </td>
                        ))}
                      </tr>
                    );
                  })}
                  <tr>
                    <td className="py-1 pr-3 font-semibold">{t("reports.totalTime")}</td>
                    {timings.data.profiles.map((p) => (
                      <td
                        key={p.profile_id}
                        className="py-1 pr-3 text-right font-mono font-semibold"
                      >
                        {fmtMs(p.mean_total_ms)}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          ) : (
            <p className="py-6 text-center text-[13px] text-muted">{t("reports.noData")}</p>
          )}
        </section>
      </div>

      {confusion.data?.profiles.length ? (
        <section className="panel p-3" aria-label={t("reports.groundTruth")}>
          <h3 className="font-semibold">{t("reports.groundTruth")}</h3>
          <p className="mb-3 text-[12px] text-muted">{t("reports.groundTruthHint")}</p>
          <div className="grid gap-4 xl:grid-cols-2">
            {confusion.data.profiles.map((p) => (
              <div key={p.profile_id}>
                <div className="mb-1 flex flex-wrap items-center gap-3">
                  <span className="h-4 w-1 rounded-full" style={{ background: p.color }} />
                  <span className="font-medium">{p.name}</span>
                  <span className="text-[12px] text-ink-2">
                    {t("model.accuracy")}{" "}
                    <span className="font-mono font-semibold">{fmtPercent(p.accuracy, 1)}</span>
                  </span>
                  <span className="text-[12px] text-ink-2">
                    {t("model.f1Macro")}{" "}
                    <span className="font-mono font-semibold">{fmtNumber(p.f1_macro, 3)}</span>
                  </span>
                  <span className="text-[12px] text-muted">
                    {t("reports.samples", { count: p.samples })}
                  </span>
                </div>
                <ConfusionMatrix matrix={p.matrix} caption={t("model.confusionCaption")} />
              </div>
            ))}
          </div>
        </section>
      ) : null}

      <AnalysesTable filterKey={key} onOpen={setOpen} onView={filters.setView} />
      <SidePanel analysis={open} onClose={() => setOpen(null)} />
    </div>
  );
}
