// Inspect screen (§15.4): sources, profile selection, single / comparison layouts, viewer
// overlays, comparison table, history filmstrip, session counters and keyboard shortcuts.
import { useQuery } from "@tanstack/react-query";
import { Dices, FolderOpen, Images, Keyboard, Upload } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Link } from "react-router";
import { toast } from "sonner";

import { FIT_VIEW, ImageViewer, type Overlays, type ViewState } from "@/components/ImageViewer";
import { ProfilePicker, useSelectedProfiles } from "@/components/ProfilePicker";
import { ProfileHeader, ResultCard, ResultDetails } from "@/components/ResultCard";
import { ClassLabel, VerdictBanner, VerdictChip, VerdictShape } from "@/components/status";
import {
  CameraSourceBar,
  useCameraSession,
  type CameraSource,
} from "@/components/camera/CameraControls";
import {
  Button,
  Checkbox,
  Dialog,
  Kbd,
  Popover,
  PopoverContent,
  PopoverTrigger,
  Select,
  Spinner,
  cn,
} from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import { fmtMs, fmtNumber, fmtPercent } from "@/lib/format";
import { useProfiles, useServerEvents, useSettings } from "@/lib/queries";
import {
  CLASS_NAMES,
  type AnalysisResult,
  type DatasetSample,
  type InspectionResult,
  type PreviewResult,
  type Profile,
} from "@/lib/types";

const HISTORY_SIZE = 50;

type Source = "upload" | "dataset" | CameraSource;

export function errorText(t: TFunction, error: unknown): string {
  if (error instanceof ApiError) {
    return String(t(`errors.${error.code}`, { ...error.details, defaultValue: error.message }));
  }
  return String(t("errors.unexpected"));
}

function usePreview(result: AnalysisResult | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["inspect-preview", result?.analysis_id],
    enabled: enabled && !!result?.image.id,
    staleTime: Infinity,
    queryFn: () => {
      const form = new FormData();
      form.set("image_id", String(result!.image.id));
      form.set("include", "masks,normalized");
      return api.post<PreviewResult>(`/api/profiles/${result!.profile.id}/preview`, form);
    },
  });
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
      {(["rois", "bottle", "patch", "mask"] as const).map((key) => (
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

export function ResultViewer({
  result,
  overlays,
  compare,
  view,
  onViewChange,
  className,
}: {
  result: AnalysisResult;
  overlays: Overlays;
  compare: boolean;
  view?: ViewState;
  onViewChange?: (v: ViewState) => void;
  className?: string;
}) {
  const preview = usePreview(result, overlays.mask || compare);
  const mask = preview.data?.masks
    ? `data:image/png;base64,${preview.data.masks.foreground}`
    : null;
  const normalized =
    compare && preview.data?.normalized_image
      ? `data:image/jpeg;base64,${preview.data.normalized_image}`
      : null;
  return (
    <ImageViewer
      src={result.image.url}
      width={result.image.width ?? 1280}
      height={result.image.height ?? 720}
      geometry={result.geometry}
      overlays={overlays}
      maskSrc={mask}
      compareSrc={normalized}
      verdict={result.verdict}
      view={view}
      onViewChange={onViewChange}
      className={className}
      label={`${result.profile.name}: ${result.verdict} ${result.predicted_class}`}
    />
  );
}

function ComparisonTable({ results }: { results: AnalysisResult[] }) {
  const { t } = useTranslation();
  // A cell is outlined when its value differs from the most common value of its column.
  const differs = (pick: (r: AnalysisResult) => string) => {
    const counts = new Map<string, number>();
    results.map(pick).forEach((v) => counts.set(v, (counts.get(v) ?? 0) + 1));
    const top = Math.max(...counts.values());
    const modes = [...counts.entries()].filter(([, n]) => n === top).map(([v]) => v);
    if (counts.size === 1) return () => false;
    if (modes.length > 1) return () => true;
    return (value: string) => value !== modes[0];
  };
  const verdictDiff = differs((r) => r.verdict);
  const classDiff = differs((r) => r.predicted_class);
  const paramsDiff = differs((r) => r.profile.status);
  const outline = "outline outline-1 outline-ink-2 -outline-offset-2 rounded-[4px]";
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full text-[13px]">
        <caption className="sr-only">{t("inspect.comparison")}</caption>
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-ink-2">
            <th className="px-3 py-2 font-medium">{t("inspect.col.profile")}</th>
            <th className="px-3 py-2 font-medium">{t("inspect.col.verdict")}</th>
            <th className="px-3 py-2 font-medium">{t("inspect.col.class")}</th>
            <th className="px-3 py-2 text-right font-medium">{t("inspect.col.confidence")}</th>
            <th className="px-3 py-2 text-right font-medium">{t("inspect.col.total")}</th>
            <th className="px-3 py-2 font-medium">{t("inspect.col.params")}</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r) => (
            <tr key={r.analysis_id} className="border-b border-line last:border-0">
              <td className="px-3 py-1.5">
                <span className="inline-flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className="h-3.5 w-1 rounded-full"
                    style={{ background: r.profile.color }}
                  />
                  {r.profile.name}
                </span>
              </td>
              <td className={cn("px-3 py-1.5", verdictDiff(r.verdict) && outline)}>
                <VerdictChip verdict={r.verdict} />
              </td>
              <td className={cn("px-3 py-1.5", classDiff(r.predicted_class) && outline)}>
                <ClassLabel name={r.predicted_class} />
              </td>
              <td className="px-3 py-1.5 text-right font-mono">{fmtNumber(r.confidence, 2)}</td>
              <td className="px-3 py-1.5 text-right font-mono">{fmtMs(r.timings_ms.total)}</td>
              <td className={cn("px-3 py-1.5", paramsDiff(r.profile.status) && outline)}>
                {r.profile.status === "modified"
                  ? t("inspect.paramsModified")
                  : t("inspect.paramsTraining")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-line px-3 py-1.5 text-[12px] text-muted">
        {t("inspect.noCombined")}
      </p>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <span className="inline-block h-4 w-16" />;
  const points = values
    .map((v, i) => `${(i / (values.length - 1)) * 64},${16 - v * 14 - 1}`)
    .join(" ");
  return (
    <svg width="64" height="16" aria-hidden="true" className="inline-block">
      <polyline points={points} fill="none" stroke="#1E7B34" strokeWidth="1.5" />
    </svg>
  );
}

function SessionCounters({
  history,
  profiles,
}: {
  history: InspectionResult[];
  profiles: Profile[];
}) {
  const { t } = useTranslation();
  const stats = useMemo(() => {
    const byProfile = new Map<
      number,
      { total: number; pass: number; reject: number; review: number; rate: number[] }
    >();
    for (const inspection of [...history].reverse()) {
      for (const a of inspection.analyses) {
        const s = byProfile.get(a.profile.id) ?? {
          total: 0,
          pass: 0,
          reject: 0,
          review: 0,
          rate: [],
        };
        s.total += 1;
        s[a.verdict] += 1;
        s.rate.push(s.pass / s.total);
        byProfile.set(a.profile.id, s);
      }
    }
    return byProfile;
  }, [history]);
  if (!stats.size) return null;
  return (
    <div
      className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[12px]"
      aria-label={t("inspect.sessionCounters")}
    >
      {[...stats.entries()].map(([id, s]) => {
        const profile = profiles.find((p) => p.id === id);
        return (
          <span key={id} className="inline-flex items-center gap-1.5">
            <span
              aria-hidden="true"
              className="h-3 w-1 rounded-full"
              style={{ background: profile?.color ?? "#888" }}
            />
            <span className="max-w-32 truncate font-medium">{profile?.name ?? `#${id}`}</span>
            <span className="font-mono">{s.total}</span>
            <span className="inline-flex items-center gap-0.5 font-mono text-pass">
              <VerdictShape verdict="pass" size={10} />
              {s.pass}
            </span>
            <span className="inline-flex items-center gap-0.5 font-mono text-reject">
              <VerdictShape verdict="reject" size={10} />
              {s.reject}
            </span>
            {s.review ? (
              <span className="inline-flex items-center gap-0.5 font-mono text-review">
                <VerdictShape verdict="review" size={10} />
                {s.review}
              </span>
            ) : null}
            <Sparkline values={s.rate.slice(-HISTORY_SIZE)} />
            <span className="font-mono text-muted">{fmtPercent(s.pass / s.total, 0)}</span>
          </span>
        );
      })}
    </div>
  );
}

function Filmstrip({
  history,
  current,
  onSelect,
}: {
  history: InspectionResult[];
  current: number;
  onSelect: (index: number) => void;
}) {
  const { t } = useTranslation();
  const selectedRef = useRef<HTMLButtonElement>(null);
  useEffect(
    () => selectedRef.current?.scrollIntoView({ block: "nearest", inline: "nearest" }),
    [current],
  );
  if (!history.length) return null;
  return (
    <ol className="flex gap-1.5 overflow-x-auto pb-1" aria-label={t("inspect.history")}>
      {history.map((inspection, index) => (
        <li key={inspection.inspection_id}>
          <button
            ref={index === current ? selectedRef : undefined}
            type="button"
            onClick={() => onSelect(index)}
            aria-current={index === current}
            aria-label={t("inspect.historyItem", {
              name: inspection.original_name || `#${inspection.inspection_id}`,
              verdicts: inspection.analyses.map((a) => t(`verdict.${a.verdict}`)).join(", "),
            })}
            className={cn(
              "relative block h-14 w-20 overflow-hidden rounded-[6px] border-2 bg-stage",
              index === current ? "border-accent" : "border-transparent hover:border-line",
            )}
          >
            {(inspection.image?.thumb_url ?? inspection.analyses[0]?.image.url) ? (
              <img
                src={(inspection.image?.thumb_url ?? inspection.analyses[0]?.image.url) as string}
                alt=""
                className="h-full w-full object-cover"
                loading="lazy"
              />
            ) : null}
            <span className="absolute bottom-0.5 left-0.5 flex gap-0.5 rounded bg-white/90 p-0.5">
              {inspection.analyses.map((a) => (
                <VerdictShape key={a.analysis_id} verdict={a.verdict} size={10} />
              ))}
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}

export function DatasetBrowser({ onPick }: { onPick: (refs: string[]) => void }) {
  const { t } = useTranslation();
  const [cls, setCls] = useState("all");
  const [open, setOpen] = useState(false);
  const samples = useQuery({
    queryKey: ["dataset-browse", cls],
    enabled: open,
    queryFn: () =>
      api.get<{ total: number; items: DatasetSample[] }>(
        `/api/dataset/samples?split=val&random=false&n=300${cls !== "all" ? `&class=${cls}` : ""}`,
      ),
  });
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button size="md">
          <Images />
          {t("inspect.browseDataset")}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[640px] max-w-[calc(100vw-32px)]">
        <div className="mb-2 flex items-center gap-2">
          <Select
            label={t("inspect.filterClass")}
            value={cls}
            onValueChange={setCls}
            options={[
              { value: "all", label: t("inspect.allClasses") },
              ...CLASS_NAMES.map((c) => ({ value: c, label: c })),
            ]}
            className="w-64"
          />
          <span className="text-[12px] text-muted">
            {t("inspect.datasetCount", { count: samples.data?.total ?? 0 })}
          </span>
        </div>
        <div className="grid max-h-96 grid-cols-6 gap-1.5 overflow-auto">
          {samples.isLoading ? <Spinner /> : null}
          {samples.data?.items.map((s) => (
            <button
              key={s.ref}
              type="button"
              title={`${s.name} — ${s.label}`}
              onClick={() => {
                onPick([s.ref]);
                setOpen(false);
              }}
              className="overflow-hidden rounded-[4px] border border-line hover:border-accent"
            >
              <img
                src={s.thumb_url}
                alt={`${s.name} (${s.label})`}
                loading="lazy"
                className="aspect-video w-full object-cover"
              />
            </button>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

const SHORTCUTS: [string, string][] = [
  ["Space", "run"],
  ["← / →", "history"],
  ["O", "overlays"],
  ["1–6", "focus"],
  ["F", "fit"],
  ["Esc", "stop"],
  ["?", "sheet"],
];

function EmptyState({
  onFiles,
  onSample,
  onCamera,
  noModels,
}: {
  onFiles: (files: File[]) => void;
  onSample: () => void;
  onCamera: (() => void) | null;
  noModels: boolean;
}) {
  const { t } = useTranslation();
  const input = useRef<HTMLInputElement>(null);
  if (noModels) {
    return (
      <div className="panel mx-auto mt-10 max-w-xl p-8 text-center">
        <h2 className="text-lg font-semibold">{t("empty.noModelsTitle")}</h2>
        <p className="mt-2 text-ink-2">
          {t("empty.noModelsBody1")}{" "}
          <code className="rounded bg-sunken px-1 font-mono">make run</code>{" "}
          {t("empty.noModelsBody2")}
        </p>
        <Button
          className="mt-4"
          variant="primary"
          onClick={() => void api.post("/api/models/refresh").then(() => window.location.reload())}
        >
          {t("empty.refresh")}
        </Button>
      </div>
    );
  }
  return (
    <div className="flex h-full min-h-80 flex-col items-center justify-center gap-4 rounded-[10px] border-2 border-dashed border-line bg-panel/60 p-8 text-center">
      <Upload className="size-10 text-muted" aria-hidden="true" />
      <div>
        <p className="text-lg font-semibold">{t("empty.dropTitle")}</p>
        <p className="mt-1 text-ink-2">{t("empty.dropBody")}</p>
      </div>
      <div className="flex flex-wrap justify-center gap-2">
        <Button variant="primary" onClick={() => input.current?.click()}>
          <FolderOpen />
          {t("inspect.chooseFiles")}
        </Button>
        <Button onClick={onSample}>
          <Dices />
          {t("inspect.trySample")}
        </Button>
        {onCamera ? (
          <Button variant="ghost" onClick={onCamera}>
            {t("empty.connectCamera")}
          </Button>
        ) : null}
      </div>
      <input
        ref={input}
        type="file"
        multiple
        accept="image/jpeg,image/png,image/bmp,image/webp"
        className="hidden"
        onChange={(e) => onFiles([...(e.target.files ?? [])])}
      />
      <p className="text-[12px] text-muted">
        {t("empty.cameraHint")}{" "}
        <Link to="/system" className="text-accent underline">
          {t("nav.system")}
        </Link>
      </p>
    </div>
  );
}

export function InspectPage() {
  const { t } = useTranslation();
  const profilesQuery = useProfiles();
  const settings = useSettings().data;
  const max = settings?.max_profiles ?? 6;
  const profiles = profilesQuery.data ?? [];
  const { ids, setIds } = useSelectedProfiles(profilesQuery.data, max);
  const [source, setSource] = useState<Source>("upload");
  const [history, setHistory] = useState<InspectionResult[]>([]);
  const [current, setCurrent] = useState(0);
  const [overlays, setOverlays] = useState<Overlays>({
    rois: true,
    bottle: true,
    patch: false,
    mask: false,
  });
  const [compare, setCompare] = useState(false);
  const [sync, setSync] = useState(true);
  const [view, setView] = useState<ViewState>(FIT_VIEW);
  const [busy, setBusy] = useState(false);
  const [batch, setBatch] = useState<{ id: number; done: number; total: number } | null>(null);
  const [sampleClass, setSampleClass] = useState("all");
  const [sheet, setSheet] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const resultRefs = useRef<(HTMLElement | null)[]>([]);

  const inspection = history[current];
  const results = inspection?.analyses ?? [];
  const comparison = results.length > 1;
  const noModels = profilesQuery.isSuccess && profiles.length === 0;

  const push = useCallback(
    (item: InspectionResult) => {
      setHistory((h) =>
        [item, ...h.filter((x) => x.inspection_id !== item.inspection_id)].slice(0, HISTORY_SIZE),
      );
      setCurrent(0);
      const text = item.analyses
        .map((a) => `${a.profile.name}: ${t(`verdict.${a.verdict}`)} ${a.predicted_class}`)
        .join("; ");
      setAnnouncement(text);
    },
    [t],
  );

  const camera = useCameraSession({ source, profileIds: ids, onInspection: push });

  const run = useCallback(
    async (form: FormData) => {
      if (!ids.length) {
        toast.error(t("inspect.selectProfileFirst"));
        return;
      }
      form.set("profile_ids", ids.join(","));
      setBusy(true);
      try {
        const response = await fetch("/api/inspections", { method: "POST", body: form });
        const body = await response.json();
        if (!response.ok)
          throw new ApiError(
            response.status,
            body.error?.code,
            body.error?.message,
            body.error?.details ?? {},
          );
        if (response.status === 202) {
          setBatch({ id: body.session_id, done: 0, total: body.total });
          toast.info(t("inspect.batchStarted", { total: body.total }));
        } else {
          (body.inspections as InspectionResult[]).forEach(push);
        }
      } catch (error) {
        toast.error(errorText(t, error));
      } finally {
        setBusy(false);
      }
    },
    [ids, push, t],
  );

  const runFiles = useCallback(
    (files: File[]) => {
      if (!files.length) return;
      const form = new FormData();
      files.forEach((f) => form.append("files", f));
      void run(form);
    },
    [run],
  );

  const runSamples = useCallback(
    (refs: string[]) => {
      const form = new FormData();
      refs.forEach((r) => form.append("dataset_samples", r));
      void run(form);
    },
    [run],
  );

  const randomSample = useCallback(async () => {
    try {
      const data = await api.get<{ items: DatasetSample[] }>(
        `/api/dataset/samples?split=val&n=1${sampleClass !== "all" ? `&class=${sampleClass}` : ""}`,
      );
      if (data.items[0]) runSamples([data.items[0].ref]);
    } catch (error) {
      toast.error(errorText(t, error));
    }
  }, [runSamples, sampleClass, t]);

  const rerun = useCallback(() => {
    if (!inspection?.image?.id) return;
    const form = new FormData();
    form.append("image_ids", String(inspection.image.id));
    void run(form);
  }, [inspection, run]);

  const primaryAction = useCallback(() => {
    if (source === "dataset") void randomSample();
    else if (source === "upload") fileInput.current?.click();
    else camera.trigger();
  }, [source, randomSample, camera]);

  // Batch progress via SSE.
  useServerEvents((event, data) => {
    const payload = data as {
      session_id?: number;
      inspection_id?: number;
      done?: number;
      total?: number;
    };
    if (!batch || payload.session_id !== batch.id) return;
    if (event === "session.progress") setBatch({ ...batch, done: payload.done ?? 0 });
    if (event === "inspection.completed" && payload.inspection_id) {
      void api.get<InspectionResult>(`/api/inspections/${payload.inspection_id}`).then(push);
    }
    if (event === "session.completed") {
      toast.success(t("inspect.batchDone", { total: payload.total }));
      setBatch(null);
    }
  });

  // Keyboard shortcuts (§15.4).
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (
        target.closest(
          "input, textarea, select, [contenteditable], [role=dialog], [role=listbox], [role=menu]",
        )
      )
        return;
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === " ") {
        event.preventDefault();
        primaryAction();
      } else if (event.key === "ArrowLeft") setCurrent((c) => Math.min(history.length - 1, c + 1));
      else if (event.key === "ArrowRight") setCurrent((c) => Math.max(0, c - 1));
      else if (event.key.toLowerCase() === "o") {
        const anyOn = Object.values(overlays).some(Boolean);
        setOverlays({ rois: !anyOn, bottle: !anyOn, patch: false, mask: false });
      } else if (event.key.toLowerCase() === "f") setView(FIT_VIEW);
      else if (event.key === "Escape") camera.stop();
      else if (event.key === "?") setSheet(true);
      else if (/^[1-6]$/.test(event.key)) resultRefs.current[Number(event.key) - 1]?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [primaryAction, history.length, overlays, camera]);

  const sourceTabs: { value: Source; label: string; id?: string }[] = [
    { value: "upload", label: t("inspect.source.upload") },
    { value: "dataset", label: t("inspect.source.dataset") },
    ...(camera.supportsBrowser
      ? [{ value: "browser" as Source, label: t("inspect.source.browser"), id: "source-browser" }]
      : []),
    ...camera.serverCameras.map((c: { id: number; name: string }) => ({
      value: `camera:${c.id}` as Source,
      label: c.name,
    })),
  ];

  return (
    <div
      className={cn("flex h-full flex-col gap-3 p-4", dragOver && "bg-info-tint/40")}
      onDragOver={(e) => {
        if (e.dataTransfer.types.includes("Files")) {
          e.preventDefault();
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        runFiles([...e.dataTransfer.files]);
      }}
    >
      <div
        className="sr-only"
        aria-live={
          camera.running && results.some((r) => r.verdict === "reject") ? "assertive" : "polite"
        }
      >
        {announcement}
      </div>

      {/* Toolbar */}
      <div className="panel flex flex-col gap-2 p-3">
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
          <div
            className="flex items-center gap-2"
            role="radiogroup"
            aria-label={t("inspect.sourceLabel")}
          >
            <span className="text-[13px] font-medium text-ink-2">{t("inspect.sourceLabel")}</span>
            <div className="flex rounded-[6px] border border-line bg-sunken p-0.5">
              {sourceTabs.map((tab) => (
                <button
                  key={tab.value}
                  id={tab.id}
                  type="button"
                  role="radio"
                  aria-checked={source === tab.value}
                  disabled={camera.running && source !== tab.value}
                  onClick={() => setSource(tab.value)}
                  className={cn(
                    "rounded-[4px] px-2.5 py-1 text-[13px] font-medium disabled:opacity-50",
                    source === tab.value
                      ? "bg-panel text-ink shadow-sm"
                      : "text-ink-2 hover:text-ink",
                  )}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>
          <ProfilePicker profiles={profiles} ids={ids} onChange={setIds} max={max} />
          <Button
            variant="ghost"
            size="icon"
            aria-label={t("inspect.shortcuts")}
            className="ml-auto"
            onClick={() => setSheet(true)}
          >
            <Keyboard />
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {source === "upload" ? (
            <>
              <Button
                variant="primary"
                disabled={busy || noModels}
                onClick={() => fileInput.current?.click()}
              >
                <FolderOpen />
                {t("inspect.chooseFiles")} <Kbd>{t("keys.space")}</Kbd>
              </Button>
              <span className="text-[12px] text-muted">
                {t("inspect.uploadHint", {
                  mb: settings?.max_upload_mb ?? 25,
                  n: settings?.max_batch_files ?? 500,
                })}
              </span>
            </>
          ) : null}
          {source === "dataset" ? (
            <>
              <Select
                label={t("inspect.filterClass")}
                value={sampleClass}
                onValueChange={setSampleClass}
                options={[
                  { value: "all", label: t("inspect.allClasses") },
                  ...CLASS_NAMES.map((c) => ({ value: c, label: c })),
                ]}
                className="w-60"
              />
              <Button
                variant="primary"
                disabled={busy || noModels}
                onClick={() => void randomSample()}
              >
                <Dices />
                {t("inspect.randomSample")} <Kbd>{t("keys.space")}</Kbd>
              </Button>
              <DatasetBrowser onPick={runSamples} />
            </>
          ) : null}
          {camera.isCamera ? <CameraSourceBar camera={camera} /> : null}
          {camera.isCamera && inspection ? (
            <div
              className="ml-auto h-24 w-40 overflow-hidden rounded-[6px]"
              aria-label={t("camera.livePreview")}
            >
              {camera.preview}
            </div>
          ) : null}
          {inspection?.image?.id && source !== "browser" && !source.startsWith("camera:") ? (
            <Button disabled={busy} onClick={rerun}>
              {t("inspect.rerun")}
            </Button>
          ) : null}
          {busy ? <Spinner /> : null}
          {batch ? (
            <div
              className="flex min-w-60 items-center gap-2 text-[12px]"
              role="progressbar"
              aria-valuemin={0}
              aria-valuemax={batch.total}
              aria-valuenow={batch.done}
            >
              <span className="h-2 flex-1 overflow-hidden rounded-full bg-sunken">
                <span
                  className="block h-full bg-accent transition-[width]"
                  style={{ width: `${(batch.done / batch.total) * 100}%` }}
                />
              </span>
              <span className="font-mono">
                {batch.done}/{batch.total}
              </span>
            </div>
          ) : null}
          <input
            ref={fileInput}
            type="file"
            multiple
            accept="image/jpeg,image/png,image/bmp,image/webp"
            className="hidden"
            onChange={(e) => {
              runFiles([...(e.target.files ?? [])]);
              e.target.value = "";
            }}
          />
        </div>
        {camera.error ? (
          <p
            role="alert"
            className="rounded-[6px] bg-reject-tint px-3 py-2 text-[13px] text-reject"
          >
            {camera.error}
          </p>
        ) : null}
      </div>

      {/* Body */}
      <div className="min-h-0 flex-1 overflow-auto">
        {camera.isCamera && !inspection ? (
          <div className="h-full min-h-80">{camera.preview}</div>
        ) : !inspection ? (
          <EmptyState
            onFiles={runFiles}
            onSample={() => {
              setSource("dataset");
              void randomSample();
            }}
            onCamera={camera.supportsBrowser ? () => setSource("browser") : null}
            noModels={noModels}
          />
        ) : comparison ? (
          <div className="flex flex-col gap-3">
            <ViewerControls
              overlays={overlays}
              setOverlays={setOverlays}
              compare={compare}
              setCompare={setCompare}
              extra={
                <label className="inline-flex cursor-pointer items-center gap-1.5 text-[13px]">
                  <Checkbox checked={sync} onCheckedChange={(v) => setSync(v === true)} />
                  {t("inspect.syncView")}
                </label>
              }
            />
            <div
              className="grid gap-3"
              style={{
                gridTemplateColumns: `repeat(${Math.min(3, results.length)}, minmax(0, 1fr))`,
              }}
            >
              {results.map((r, i) => (
                <section
                  key={r.analysis_id}
                  ref={(el) => {
                    resultRefs.current[i] = el;
                  }}
                  tabIndex={-1}
                  className="panel flex min-w-0 flex-col gap-2.5 p-3 focus-visible:outline-2"
                  aria-label={r.profile.name}
                >
                  <ProfileHeader result={r} />
                  <VerdictBanner verdict={r.verdict} predictedClass={r.predicted_class} compact />
                  <ResultViewer
                    result={r}
                    overlays={overlays}
                    compare={compare}
                    view={sync ? view : undefined}
                    onViewChange={sync ? setView : undefined}
                    className="aspect-[16/9] w-full"
                  />
                  <ResultDetails result={r} />
                </section>
              ))}
            </div>
            <ComparisonTable results={results} />
          </div>
        ) : (
          <div className="grid h-full min-h-0 gap-3 lg:grid-cols-[minmax(0,1fr)_380px]">
            <div className="flex min-h-0 flex-col gap-2">
              <ResultViewer
                result={results[0]}
                overlays={overlays}
                compare={compare}
                view={view}
                onViewChange={setView}
                className="min-h-80 flex-1"
              />
              <ViewerControls
                overlays={overlays}
                setOverlays={setOverlays}
                compare={compare}
                setCompare={setCompare}
              />
            </div>
            <div
              className="min-h-0 overflow-auto"
              ref={(el) => {
                resultRefs.current[0] = el;
              }}
              tabIndex={-1}
            >
              <ResultCard result={results[0]} />
              {inspection.original_name ? (
                <p
                  className="mt-2 truncate px-1 font-mono text-[12px] text-muted"
                  title={inspection.original_name}
                >
                  {inspection.original_name}
                </p>
              ) : null}
            </div>
          </div>
        )}
      </div>

      {/* History */}
      {history.length ? (
        <div className="panel flex flex-col gap-2 p-2.5">
          <Filmstrip history={history} current={current} onSelect={setCurrent} />
          <SessionCounters history={history} profiles={profiles} />
        </div>
      ) : null}

      <Dialog open={sheet} onOpenChange={setSheet} title={t("inspect.shortcuts")}>
        <table className="w-full text-[13px]">
          <tbody>
            {SHORTCUTS.map(([key, action]) => (
              <tr key={key} className="border-b border-line last:border-0">
                <td className="py-1.5 pr-4">
                  <Kbd>{key === "Space" ? t("keys.space") : key}</Kbd>
                </td>
                <td className="py-1.5">{t(`shortcuts.${action}`)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Dialog>
    </div>
  );
}

function ViewerControls({
  overlays,
  setOverlays,
  compare,
  setCompare,
  extra,
}: {
  overlays: Overlays;
  setOverlays: (o: Overlays) => void;
  compare: boolean;
  setCompare: (v: boolean) => void;
  extra?: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
      <OverlayToggles overlays={overlays} onChange={setOverlays} />
      <label className="inline-flex cursor-pointer items-center gap-1.5 text-[13px]">
        <Checkbox checked={compare} onCheckedChange={(v) => setCompare(v === true)} />
        {t("inspect.compareNormalized")}
      </label>
      {extra}
    </div>
  );
}
