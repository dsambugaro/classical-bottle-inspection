// Card history (reports, view=cards): one row per inspection, one card per profile side by side,
// each the same composition as the single result of the Inspect screen. Infinite scroll over
// GET /api/history; the filters live in the URL (see lib/historyFilters.ts).
import { useInfiniteQuery, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import type { Overlays } from "@/components/ImageViewer";
import { ResultCard } from "@/components/ResultCard";
import { ClassLabel } from "@/components/status";
import { Button, Checkbox, Spinner, cn } from "@/components/ui/primitives";
import { api, qs } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import type { ReportView } from "@/lib/historyFilters";
import type { AnalysisResult, HistoryItem, HistoryPage } from "@/lib/types";
import { ResultViewer, errorText } from "../InspectPage";
import { FilterBar, useFilters, type Options } from "./filters";

const PAGE_SIZE = 12;
/** How far outside the screen a row still renders its viewers and a new page is requested. */
const LOOKAHEAD = "900px";

/** Segmented "Table | Cards" switch of the reports analyses. */
export function ViewSwitch({
  view,
  onChange,
  className,
}: {
  view: ReportView;
  onChange: (view: ReportView) => void;
  className?: string;
}) {
  const { t } = useTranslation();
  return (
    <div
      role="group"
      aria-label={t("historyView.view")}
      className={cn("inline-flex rounded-[6px] border border-line bg-panel p-0.5", className)}
    >
      {(["dashboard", "cards"] as const).map((value) => (
        <Button
          key={value}
          size="sm"
          variant={view === value ? "primary" : "ghost"}
          aria-pressed={view === value}
          onClick={() => view !== value && onChange(value)}
        >
          {t(value === "cards" ? "historyView.viewCards" : "historyView.viewTable")}
        </Button>
      ))}
    </div>
  );
}

/** True once the element has come within `margin` of the screen (it then stays true). */
function useSeen<T extends Element>(margin: string) {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(typeof IntersectionObserver === "undefined");
  useEffect(() => {
    const element = ref.current;
    if (seen || !element) return;
    const observer = new IntersectionObserver(
      (entries) => entries.some((e) => e.isIntersecting) && setSeen(true),
      { rootMargin: margin },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [seen, margin]);
  return [ref, seen] as const;
}

function OverlayToggles({
  overlays,
  onChange,
}: {
  overlays: Overlays;
  onChange: (o: Overlays) => void;
}) {
  const { t } = useTranslation();
  return (
    <div
      className="flex flex-wrap items-center gap-3 text-[13px]"
      role="group"
      aria-label={t("inspect.overlays")}
    >
      <span className="font-medium text-ink-2">{t("inspect.overlays")}</span>
      {(["rois", "bottle", "mask"] as const).map((key) => (
        <label key={key} className="inline-flex cursor-pointer items-center gap-1.5">
          <Checkbox
            checked={overlays[key]}
            onCheckedChange={(v) => onChange({ ...overlays, [key]: v === true })}
          />
          {t(`inspect.overlay.${key}`)}
        </label>
      ))}
    </div>
  );
}

function AnalysisCard({ result, overlays }: { result: AnalysisResult; overlays: Overlays }) {
  const { t } = useTranslation();
  return (
    <div className="flex w-[380px] shrink-0 flex-col gap-2">
      {result.image.url ? (
        <ResultViewer
          result={result}
          overlays={overlays}
          compare={false}
          className="aspect-[16/9] w-full"
        />
      ) : (
        <p className="grid aspect-[16/9] w-full place-items-center rounded-[6px] bg-sunken p-4 text-center text-[13px] text-muted">
          {t("reports.imageDeleted")}
        </p>
      )}
      <ResultCard result={result} />
    </div>
  );
}

function HistoryRow({ item, overlays }: { item: HistoryItem; overlays: Overlays }) {
  const { t } = useTranslation();
  const [ref, seen] = useSeen<HTMLElement>(LOOKAHEAD);
  const disagree = new Set(item.analyses.map((a) => a.predicted_class)).size > 1;
  return (
    <article
      ref={ref}
      aria-label={item.original_name || `#${item.inspection_id}`}
      className="flex flex-col gap-2"
    >
      <header className="flex flex-wrap items-center gap-x-4 gap-y-1 px-1 text-[13px]">
        <h3 className="max-w-[28rem] truncate font-mono font-semibold">
          {item.original_name || `#${item.inspection_id}`}
        </h3>
        {item.original_name ? (
          <span className="font-mono text-[12px] text-muted">#{item.inspection_id}</span>
        ) : null}
        <time className="text-ink-2" dateTime={item.created_at}>
          {fmtDate(item.created_at, true)}
        </time>
        <span className="text-ink-2">
          {t(`reports.sources.${item.source}`, { defaultValue: item.source })}
          {item.camera_name ? ` · ${item.camera_name}` : ""}
        </span>
        {item.ground_truth ? (
          <span className="inline-flex items-center gap-1.5 text-ink-2">
            {t("historyView.truth")}
            <ClassLabel name={item.ground_truth} />
          </span>
        ) : null}
        {disagree ? (
          <span className="rounded-[6px] bg-review-tint px-1.5 py-0.5 text-[12px] font-medium text-review">
            {t("historyView.disagree")}
          </span>
        ) : null}
      </header>
      {seen ? (
        <div className="flex items-start gap-3 overflow-x-auto pb-2">
          {item.analyses.map((result) => (
            <AnalysisCard key={result.analysis_id} result={result} overlays={overlays} />
          ))}
        </div>
      ) : (
        <div className="min-h-[640px] rounded-[6px] bg-sunken/50" aria-hidden="true" />
      )}
    </article>
  );
}

export function HistoryView() {
  const { t } = useTranslation();
  const filters = useFilters();
  const [overlays, setOverlays] = useState<Overlays>({
    rois: true,
    bottle: true,
    patch: false,
    mask: false,
  });
  const options = useQuery({
    queryKey: ["report-options"],
    queryFn: () => api.get<Options>("/api/reports/options"),
  });
  const key = filters.historyKey;
  const query = useInfiniteQuery({
    queryKey: ["history", key],
    initialPageParam: null as number | null,
    queryFn: ({ pageParam, signal }) =>
      api.get<HistoryPage>(
        `/api/history${qs({ ...filters.historyQuery, limit: PAGE_SIZE, before: pageParam })}`,
        signal,
      ),
    getNextPageParam: (last) => (last.has_more ? last.next_cursor : undefined),
  });
  const items = query.data?.pages.flatMap((p) => p.items) ?? [];
  const { hasNextPage, isFetchingNextPage, isError, fetchNextPage } = query;

  // Loads the next page when the sentinel below the last row is near the screen. The observer
  // is recreated after every page so that a sentinel still in view triggers the next one.
  const sentinel = useRef<HTMLDivElement>(null);
  const loaded = query.data?.pages.length ?? 0;
  useEffect(() => {
    const element = sentinel.current;
    if (!element || !hasNextPage || isFetchingNextPage || isError) return;
    const observer = new IntersectionObserver(
      (entries) => entries.some((e) => e.isIntersecting) && void fetchNextPage(),
      { rootMargin: LOOKAHEAD },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [hasNextPage, isFetchingNextPage, isError, fetchNextPage, loaded]);

  return (
    <div className="flex flex-col gap-3 p-4">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <h2 className="text-base font-semibold">{t("historyView.title")}</h2>
        <ViewSwitch view="cards" onChange={filters.setView} />
        {query.isFetching && !isFetchingNextPage ? <Spinner /> : null}
        <span className="text-[12px] text-muted" aria-live="polite">
          {query.data ? t("historyView.loaded", { count: items.length }) : null}
        </span>
        <div className="ml-auto">
          <OverlayToggles overlays={overlays} onChange={setOverlays} />
        </div>
      </div>
      <FilterBar filters={filters} options={options.data} cards />
      {query.isPending ? (
        <div className="flex justify-center py-12">
          <Spinner />
        </div>
      ) : null}
      {query.isError && !items.length ? (
        <div className="panel flex flex-col items-center gap-3 p-8 text-center" role="alert">
          <p className="font-semibold">{t("historyView.error")}</p>
          <p className="text-[13px] text-ink-2">{errorText(t, query.error)}</p>
          <Button onClick={() => void query.refetch()}>{t("historyView.retry")}</Button>
        </div>
      ) : null}
      {query.isSuccess && !items.length ? (
        <div className="panel p-8 text-center">
          <p className="font-semibold">{t("historyView.emptyTitle")}</p>
          <p className="mt-1 text-[13px] text-ink-2">{t("historyView.emptyBody")}</p>
        </div>
      ) : null}
      <div className="flex flex-col gap-6">
        {items.map((item) => (
          <HistoryRow key={item.inspection_id} item={item} overlays={overlays} />
        ))}
      </div>
      <div ref={sentinel} className="flex min-h-10 items-center justify-center text-[13px]">
        {isFetchingNextPage ? <Spinner /> : null}
        {query.isError && items.length ? (
          <span className="flex items-center gap-3 text-reject" role="alert">
            {t("historyView.error")}
            <Button size="sm" onClick={() => void fetchNextPage()}>
              {t("historyView.retry")}
            </Button>
          </span>
        ) : null}
        {query.isSuccess && items.length && !hasNextPage ? (
          <span className="text-muted">{t("historyView.end")}</span>
        ) : null}
      </div>
    </div>
  );
}
