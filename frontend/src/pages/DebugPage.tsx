// Debug mode: one image traced step by step (background subtraction, Otsu threshold, mask,
// bottle box, regions), then run by any number of profiles. The image stays on the server for
// the session, so profiles can be swapped freely; choosing another image starts over.
import { keepPreviousData, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { Dices, FolderOpen, RotateCcw, Upload } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { ProfilePicker, useSelectedProfiles } from "@/components/ProfilePicker";
import { ClassLabel, VerdictChip } from "@/components/status";
import { Button, Select, Spinner, Switch, cn } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { fmtInt, fmtMs, fmtNumber, fmtPercent, modelLabel } from "@/lib/format";
import { useCatalog, useProfiles, useSettings } from "@/lib/queries";
import {
  CLASS_NAMES,
  type DatasetSample,
  type DebugSession,
  type Params,
  type PreprocessingTrace,
  type Profile,
  type ProfileTrace,
} from "@/lib/types";
import { DatasetBrowser, errorText } from "./InspectPage";
import { ProfileTracePanel } from "./debug/ProfileTracePanel";
import {
  DifferenceHistogram,
  Facts,
  Figure,
  LocalizationPlot,
  PixelMatrix,
  REGION_COLORS,
  Step,
} from "./debug/pieces";

const ADJUSTABLE = [
  "localization.close_size",
  "localization.open_size",
  "localization.profile_smoothing",
  "localization.column_fraction",
  "localization.bottle_width",
  "localization.top_fraction",
  "localization.row_fraction",
] as const;
const NORMALIZE = "normalization.enabled";
const PATCH_COLOR = "#ffd43b";
const CHANNELS = ["B", "G", "R"] as const;

function useDebounced<T>(value: T, delay: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return settled;
}

/** Temporary parameter changes applied to the trace and to every profile; never saved. */
function Adjustments({
  base,
  overrides,
  onChange,
}: {
  base: Params;
  overrides: Params;
  onChange: (next: Params) => void;
}) {
  const { t } = useTranslation();
  const catalog = useCatalog().data;
  const set = (key: string, value: number | boolean) => {
    const next = { ...overrides };
    if (value === base[key]) delete next[key];
    else next[key] = value;
    onChange(next);
  };
  const changed = Object.keys(overrides).length;
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
      {ADJUSTABLE.map((key) => {
        const entry = catalog?.entries.find((e) => e.key === key);
        const value = (overrides[key] ?? base[key]) as number;
        return (
          <label key={key} className="flex flex-col gap-1 text-[12px] text-ink-2">
            <span className={cn(key in overrides && "font-semibold text-accent")}>
              {t(`params.${key}.label`)}
            </span>
            <input
              type="number"
              className="field-input w-24"
              min={entry?.min ?? undefined}
              max={entry?.max ?? undefined}
              step={entry?.step ?? undefined}
              value={value ?? ""}
              onChange={(event) => {
                const number = event.target.valueAsNumber;
                if (Number.isFinite(number)) set(key, number);
              }}
            />
          </label>
        );
      })}
      <label className="flex h-8 items-center gap-2 text-[12px] text-ink-2">
        <Switch
          checked={(overrides[NORMALIZE] ?? base[NORMALIZE]) === true}
          onCheckedChange={(checked) => set(NORMALIZE, checked)}
        />
        <span className={cn(NORMALIZE in overrides && "font-semibold text-accent")}>
          {t(`params.${NORMALIZE}.label`)}
        </span>
      </label>
      <Button size="md" variant="ghost" disabled={!changed} onClick={() => onChange({})}>
        <RotateCcw />
        {t("debug.adjust.reset")}
      </Button>
    </div>
  );
}

function Preprocessing({ session, trace }: { session: DebugSession; trace: PreprocessingTrace }) {
  const { t } = useTranslation();
  const { original, normalization, difference, mask, localization, regions } = trace;
  const [height, width, channels] = original.shape;
  const [maskWidth, maskHeight] = localization.size;
  const size = `${maskWidth}×${maskHeight}`;
  const adhoc = regions.filter((r) => r.kind === "adhoc");
  const auto = regions.filter((r) => r.kind === "auto");
  const bottleRect = { rect: localization.box, color: REGION_COLORS.bottle, dashed: true };
  const rectsOf = (items: typeof regions) =>
    items.map((r) => ({ rect: r.rect, color: REGION_COLORS[r.name] ?? "#888", label: r.name }));
  return (
    <>
      <Step n={1} title={t("debug.original.title")} summary={t("debug.original.summary")}>
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
          <Figure
            src={session.image_url}
            caption={session.name}
            note={`${height}×${width}×${channels} · ${original.dtype} · BGR`}
          />
          <PixelMatrix
            patch={original.patch}
            caption={t("debug.original.patch", { x: original.patch.x, y: original.patch.y })}
          />
        </div>
      </Step>

      <Step n={2} title={t("debug.background.title")} summary={t("debug.background.summary")}>
        <div className="max-w-xl">
          <Figure
            src={trace.background.image}
            caption={t("debug.background.caption")}
            note={`${trace.background.shape[0]}×${trace.background.shape[1]}×${trace.background.shape[2]}`}
          />
        </div>
      </Step>

      <Step
        n={3}
        title={t("debug.normalization.title")}
        summary={
          normalization.enabled
            ? t("debug.normalization.summary")
            : t("debug.normalization.disabled")
        }
      >
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
          <Figure
            src={session.image_url}
            caption={t("debug.normalization.before")}
            rects={[
              {
                rect: normalization.reference_patch,
                color: PATCH_COLOR,
                dashed: true,
                label: t("debug.normalization.patch"),
              },
            ]}
          />
          <div className="flex flex-col gap-3">
            <table className="text-[13px]">
              <caption className="sr-only">{t("debug.normalization.title")}</caption>
              <thead>
                <tr className="text-left text-[12px] text-ink-2">
                  <th className="pr-4 font-medium">{t("debug.normalization.channel")}</th>
                  <th className="pr-4 text-right font-medium">
                    {t("debug.normalization.background")}
                  </th>
                  <th className="pr-4 text-right font-medium">{t("debug.normalization.image")}</th>
                  <th className="text-right font-medium">{t("debug.normalization.gain")}</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {CHANNELS.map((channel, index) => (
                  <tr key={channel}>
                    <td className="pr-4">{channel}</td>
                    <td className="pr-4 text-right">
                      {fmtNumber(normalization.patch_mean_background[index], 1)}
                    </td>
                    <td className="pr-4 text-right">
                      {fmtNumber(normalization.patch_mean_image[index], 1)}
                    </td>
                    <td className="text-right font-semibold">
                      ×{fmtNumber(normalization.gain[index], 3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <PixelMatrix
              patch={normalization.patch}
              caption={t("debug.normalization.patchAfter")}
            />
          </div>
          <Figure src={normalization.image} caption={t("debug.normalization.after")} />
        </div>
      </Step>

      <Step n={4} title={t("debug.difference.title")} summary={t("debug.difference.summary")}>
        <div className="grid items-start gap-4 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,0.8fr)]">
          <Figure src={difference.small} caption={t("debug.difference.small", { size })} />
          <Figure
            src={difference.heatmap}
            caption={t("debug.difference.heatmap")}
            note={t("debug.difference.max", { max: difference.max, clip: difference.clip })}
          />
          <div className="flex flex-col gap-3">
            <DifferenceHistogram data={difference} />
            <Facts
              items={[
                [t("debug.difference.otsu"), fmtInt(difference.otsu)],
                [t("debug.difference.floor"), fmtInt(difference.noise_floor)],
                [t("debug.difference.used"), fmtInt(difference.threshold)],
              ]}
            />
          </div>
        </div>
      </Step>

      <Step n={5} title={t("debug.mask.title")} summary={t("debug.mask.summary")}>
        <div className="grid items-start gap-4 md:grid-cols-3">
          <Figure
            src={mask.raw}
            pixelated
            caption={t("debug.mask.raw", { threshold: fmtInt(difference.threshold) })}
            note={t("debug.mask.fraction", { percent: fmtPercent(mask.fraction_raw, 1) })}
          />
          <Figure
            src={mask.closed}
            pixelated
            caption={t("debug.mask.closed", { size: mask.close_size })}
            note={t("debug.mask.fraction", { percent: fmtPercent(mask.fraction_closed, 1) })}
          />
          <Figure
            src={mask.opened}
            pixelated
            caption={t("debug.mask.opened", { size: mask.open_size })}
            note={t("debug.mask.fraction", { percent: fmtPercent(mask.fraction, 1) })}
          />
        </div>
      </Step>

      <Step
        n={6}
        title={t("debug.locate.title")}
        summary={localization.empty ? t("debug.locate.empty") : t("debug.locate.summary")}
      >
        <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <LocalizationPlot data={localization} mask={mask.opened} />
          <div className="flex flex-col gap-3">
            <Figure
              src={normalization.image}
              caption={t("debug.locate.box")}
              rects={[{ ...bottleRect, dashed: false }]}
            />
            <Facts
              items={[
                [t("debug.locate.xRange", { size }), `${localization.x[0]} – ${localization.x[1]}`],
                [t("debug.locate.yRange", { size }), `${localization.y[0]} – ${localization.y[1]}`],
                [
                  t("debug.locate.centre"),
                  `${fmtNumber(localization.centre, 1)} · ${fmtNumber(localization.box_x[0], 1)} – ${fmtNumber(localization.box_x[1], 1)}`,
                ],
                [
                  t("debug.locate.relative"),
                  localization.box.map((v) => fmtNumber(v, 4)).join("  "),
                ],
                [
                  t("debug.locate.pixels", { width, height }),
                  `${localization.box_px[0]}, ${localization.box_px[1]} → ${localization.box_px[2]}, ${localization.box_px[3]}`,
                ],
                [
                  t("debug.locate.boxSize"),
                  `${localization.box_px[2] - localization.box_px[0]}×${localization.box_px[3] - localization.box_px[1]} px`,
                ],
              ]}
            />
          </div>
        </div>
      </Step>

      <Step n={7} title={t("debug.regions.title")} summary={t("debug.regions.summary")}>
        <div className="grid items-start gap-4 lg:grid-cols-2">
          <Figure
            src={normalization.image}
            caption={t("debug.regions.adhoc")}
            rects={[bottleRect, ...rectsOf(adhoc)]}
          />
          {auto.length ? (
            <Figure
              src={normalization.image}
              caption={t("debug.regions.auto")}
              rects={[bottleRect, ...rectsOf(auto)]}
            />
          ) : (
            <p className="text-[13px] text-ink-2">{t("debug.regions.noAuto")}</p>
          )}
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-4">
          {regions.map((region) => (
            <Figure
              key={region.name}
              src={region.image}
              className="w-28"
              imageClassName="max-h-72 object-contain"
              caption={
                <span className="font-mono" style={{ color: REGION_COLORS[region.name] }}>
                  {region.name}
                </span>
              }
              note={
                <>
                  {region.shape[1]}×{region.shape[0]} px
                  <br />
                  {fmtNumber(region.band[0], 3)} – {fmtNumber(region.band[1], 3)}
                  <br />→ {region.canonical_size[0]}×{region.canonical_size[1]}
                </>
              }
            />
          ))}
        </div>
      </Step>
    </>
  );
}

/** One row per selected profile; the traces come from the same cache as the panels below. */
function Comparison({
  sessionId,
  profiles,
  overrides,
}: {
  sessionId: string;
  profiles: Profile[];
  overrides: Params;
}) {
  const { t } = useTranslation();
  const traces = useQueries({
    queries: profiles.map((profile) => ({
      queryKey: ["debug", sessionId, "profile", profile.id, JSON.stringify(overrides)],
      staleTime: Infinity,
      retry: false,
      queryFn: () =>
        api.post<ProfileTrace>(`/api/debug/sessions/${sessionId}/profiles/${profile.id}`, {
          overrides,
        }),
    })),
  });
  return (
    <div className="panel overflow-x-auto">
      <table className="w-full text-[13px]">
        <caption className="sr-only">{t("inspect.comparison")}</caption>
        <thead>
          <tr className="border-b border-line text-left text-[12px] text-ink-2">
            <th className="px-3 py-2 font-medium">{t("inspect.col.profile")}</th>
            <th className="px-3 py-2 font-medium">{t("debug.compare.vector")}</th>
            <th className="px-3 py-2 font-medium">{t("inspect.col.verdict")}</th>
            <th className="px-3 py-2 font-medium">{t("inspect.col.class")}</th>
            <th className="px-3 py-2 text-right font-medium">{t("inspect.col.confidence")}</th>
            <th className="px-3 py-2 text-right font-medium">{t("debug.compare.length")}</th>
            <th className="px-3 py-2 text-right font-medium">{t("inspect.col.total")}</th>
          </tr>
        </thead>
        <tbody>
          {profiles.map((profile, index) => {
            const result = traces[index]?.data?.analysis;
            const wrong = result?.ground_truth && result.ground_truth !== result.predicted_class;
            return (
              <tr key={profile.id} className="border-b border-line last:border-0">
                <td className="px-3 py-1.5">
                  <span className="inline-flex items-center gap-2">
                    <span
                      aria-hidden="true"
                      className="h-3.5 w-1 rounded-full"
                      style={{ background: profile.color }}
                    />
                    {profile.name}
                  </span>
                </td>
                <td className="px-3 py-1.5 font-mono text-[12px]">
                  {modelLabel(profile.model?.feature_vector, profile.algorithm)}
                </td>
                {result ? (
                  <>
                    <td className="px-3 py-1.5">
                      <VerdictChip verdict={result.verdict} />
                    </td>
                    <td
                      className={cn(
                        "px-3 py-1.5",
                        wrong && "rounded-[4px] outline outline-1 -outline-offset-2 outline-reject",
                      )}
                    >
                      <ClassLabel name={result.predicted_class} />
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {fmtNumber(result.confidence, 2)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {fmtInt(traces[index].data!.vector.length)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono">
                      {fmtMs(result.timings_ms.total)}
                    </td>
                  </>
                ) : (
                  <td colSpan={5} className="px-3 py-1.5 text-muted">
                    {traces[index]?.isError ? t("debug.compare.failed") : <Spinner />}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function DebugPage() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const profilesQuery = useProfiles();
  const profiles = profilesQuery.data ?? [];
  const max = useSettings().data?.max_profiles ?? 6;
  const { ids, setIds, selected } = useSelectedProfiles(profilesQuery.data, max);
  const usable = profiles.filter((p) => p.status_detail.can_analyse && !p.deleted);

  const [session, setSession] = useState<DebugSession | null>(null);
  const [busy, setBusy] = useState(false);
  const [sampleClass, setSampleClass] = useState("all");
  const [baseChoice, setBaseChoice] = useState<number | null>(null);
  const [draft, setDraft] = useState<Params>({});
  const overrides = useDebounced(draft, 300);
  const [dragOver, setDragOver] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const sessionRef = useRef<DebugSession | null>(null);

  const base =
    usable.find((p) => p.id === baseChoice) ?? usable.find((p) => p.id === ids[0]) ?? usable[0];

  const forget = useCallback(
    (old: DebugSession | null) => {
      if (!old) return;
      client.removeQueries({ queryKey: ["debug", old.id] });
      void api.del(`/api/debug/sessions/${old.id}`).catch(() => undefined);
    },
    [client],
  );

  // A new image restarts the debug: the previous images and vectors are dropped everywhere.
  const start = useCallback(
    async (form: FormData) => {
      setBusy(true);
      try {
        const next = await api.post<DebugSession>("/api/debug/sessions", form);
        forget(sessionRef.current);
        sessionRef.current = next;
        setSession(next);
      } catch (error) {
        toast.error(errorText(t, error));
      } finally {
        setBusy(false);
      }
    },
    [forget, t],
  );

  useEffect(() => () => forget(sessionRef.current), [forget]);

  const startFile = (file: File | undefined) => {
    if (!file) return;
    const form = new FormData();
    form.set("file", file);
    void start(form);
  };
  const startSample = (ref: string) => {
    const form = new FormData();
    form.set("dataset_ref", ref);
    void start(form);
  };
  const randomSample = async () => {
    try {
      const data = await api.get<{ items: DatasetSample[] }>(
        `/api/dataset/samples?split=val&n=1${sampleClass !== "all" ? `&class=${sampleClass}` : ""}`,
      );
      if (data.items[0]) startSample(data.items[0].ref);
    } catch (error) {
      toast.error(errorText(t, error));
    }
  };

  const preprocessing = useQuery({
    queryKey: ["debug", session?.id, "preprocessing", base?.id, JSON.stringify(overrides)],
    enabled: !!session && !!base,
    staleTime: Infinity,
    retry: false,
    placeholderData: keepPreviousData,
    queryFn: () =>
      api.post<PreprocessingTrace>(`/api/debug/sessions/${session!.id}/preprocessing/${base!.id}`, {
        overrides,
      }),
  });
  const trace = preprocessing.data;
  const noModels = profilesQuery.isSuccess && usable.length === 0;

  return (
    <div
      className={cn("flex min-h-full flex-col gap-3 p-4", dragOver && "bg-info-tint/40")}
      onDragOver={(event) => {
        if (event.dataTransfer.types.includes("Files")) {
          event.preventDefault();
          setDragOver(true);
        }
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragOver(false);
        startFile(event.dataTransfer.files[0]);
      }}
    >
      <div className="panel flex flex-wrap items-center gap-2 p-3">
        <Button variant="primary" disabled={busy} onClick={() => fileInput.current?.click()}>
          <FolderOpen />
          {t("debug.chooseImage")}
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          className="hidden"
          onChange={(event) => {
            startFile(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <span className="mx-1 h-6 w-px bg-line" aria-hidden="true" />
        <Select
          label={t("inspect.filterClass")}
          value={sampleClass}
          onValueChange={setSampleClass}
          options={[
            { value: "all", label: t("inspect.allClasses") },
            ...CLASS_NAMES.map((c) => ({ value: c, label: c })),
          ]}
          className="w-56"
        />
        <Button disabled={busy} onClick={() => void randomSample()}>
          <Dices />
          {t("inspect.randomSample")}
        </Button>
        <DatasetBrowser onPick={(refs) => startSample(refs[0])} />
        {busy ? <Spinner /> : null}
        {session ? (
          <span className="ml-auto flex min-w-0 items-center gap-3 text-[13px]">
            <span className="truncate font-mono">{session.name}</span>
            <span className="shrink-0 font-mono text-muted">
              {session.width}×{session.height}
            </span>
            {session.ground_truth ? (
              <span className="inline-flex shrink-0 items-center gap-1.5">
                <span className="text-ink-2">{t("debug.groundTruth")}</span>
                <ClassLabel name={session.ground_truth} />
              </span>
            ) : null}
          </span>
        ) : null}
      </div>

      {noModels ? (
        <div className="panel mx-auto mt-10 max-w-xl p-8 text-center">
          <h2 className="text-lg font-semibold">{t("empty.noModelsTitle")}</h2>
          <p className="mt-2 text-ink-2">
            {t("empty.noModelsBody1")}{" "}
            <code className="rounded bg-sunken px-1 font-mono">make run</code>{" "}
            {t("empty.noModelsBody2")}
          </p>
        </div>
      ) : !session ? (
        <div className="flex min-h-80 flex-1 flex-col items-center justify-center gap-3 rounded-[10px] border-2 border-dashed border-line bg-panel/60 p-8 text-center">
          <Upload className="size-10 text-muted" aria-hidden="true" />
          <p className="text-lg font-semibold">{t("debug.empty.title")}</p>
          <p className="max-w-xl text-ink-2">{t("debug.empty.body")}</p>
        </div>
      ) : (
        <>
          <div className="panel flex flex-col gap-3 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="text-[15px] font-semibold">{t("debug.preprocessing")}</h2>
              <span className="text-[13px] text-ink-2">{t("debug.baseProfile")}</span>
              <Select
                label={t("debug.baseProfile")}
                value={base ? String(base.id) : ""}
                onValueChange={(value) => setBaseChoice(Number(value))}
                options={usable.map((p) => ({ value: String(p.id), label: p.name }))}
                className="w-72"
              />
              {preprocessing.isFetching ? <Spinner /> : null}
              <p className="basis-full text-[12px] text-muted">{t("debug.baseProfileHelp")}</p>
            </div>
            {base ? (
              <div className="border-t border-line pt-3">
                <p className="mb-2 text-[12px] text-ink-2">{t("debug.adjust.help")}</p>
                <Adjustments base={base.params} overrides={draft} onChange={setDraft} />
              </div>
            ) : null}
            {preprocessing.isError ? (
              <p className="rounded-[6px] bg-reject-tint px-3 py-2 text-[13px] text-reject">
                {errorText(t, preprocessing.error)}
              </p>
            ) : null}
          </div>

          {trace ? <Preprocessing session={session} trace={trace} /> : null}

          <div className="panel sticky top-0 z-10 flex flex-wrap items-center gap-3 p-3 shadow-sm">
            <h2 className="text-[15px] font-semibold">{t("debug.profiles.title")}</h2>
            <ProfilePicker profiles={profiles} ids={ids} onChange={setIds} max={max} />
            <p className="basis-full text-[12px] text-muted">{t("debug.profiles.help")}</p>
          </div>
          {selected.length ? (
            <>
              <Comparison sessionId={session.id} profiles={selected} overrides={overrides} />
              <div className="overflow-x-auto pb-2">
                <div
                  className="grid items-start gap-3"
                  style={{
                    gridTemplateColumns: `repeat(${selected.length}, minmax(${selected.length > 1 ? "420px" : "0"}, 1fr))`,
                  }}
                >
                  {selected.map((profile) => (
                    <ProfileTracePanel
                      key={profile.id}
                      sessionId={session.id}
                      profile={profile}
                      overrides={overrides}
                      baseKey={trace?.key}
                      onUseAsBase={() => setBaseChoice(profile.id)}
                    />
                  ))}
                </div>
              </div>
            </>
          ) : (
            <p className="px-1 text-[13px] text-ink-2">{t("inspect.selectProfileFirst")}</p>
          )}
        </>
      )}
    </div>
  );
}
