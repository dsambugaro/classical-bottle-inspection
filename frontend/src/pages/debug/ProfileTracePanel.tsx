// One profile on the debug image: verdict and scores, the feature vector it was given (raw and
// as the classifier sees it after standardization) and every feature group step by step.
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "react-i18next";

import { Facts, Figure, REGION_COLORS, VectorPlot, VectorValues } from "./pieces";
import { ConfidenceValue, Expander, ScoreBars, TimingBar } from "@/components/ResultCard";
import { TruthMark, VerdictBanner, WarningChip } from "@/components/status";
import { Button, Spinner, cn } from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import { fmtInt, fmtMs, fmtNumber, modelLabel } from "@/lib/format";
import type { GroupTrace, Params, Profile, ProfileTrace } from "@/lib/types";

export function useProfileTrace(sessionId: string, profileId: number, overrides: Params) {
  return useQuery({
    queryKey: ["debug", sessionId, "profile", profileId, JSON.stringify(overrides)],
    staleTime: Infinity,
    retry: false,
    queryFn: () =>
      api.post<ProfileTrace>(`/api/debug/sessions/${sessionId}/profiles/${profileId}`, {
        overrides,
      }),
  });
}

function HistogramBars({
  values,
  colors,
  label,
}: {
  values: number[];
  colors: string | string[];
  label: string;
}) {
  const top = Math.max(...values, 1e-9);
  return (
    <figure className="min-w-0">
      <div className="flex h-16 items-end gap-px rounded-[4px] bg-sunken p-1">
        {values.map((value, index) => (
          <span
            key={index}
            title={fmtNumber(value, 3)}
            className="min-w-0 flex-1 rounded-t-[1px]"
            style={{
              height: `${Math.max((value / top) * 100, 1)}%`,
              background: typeof colors === "string" ? colors : colors[index % colors.length],
            }}
          />
        ))}
      </div>
      <figcaption className="mt-1 text-[12px] text-ink-2">{label}</figcaption>
    </figure>
  );
}

function GroupDetail({ group }: { group: GroupTrace }) {
  const { t } = useTranslation();
  const [cropHeight, cropWidth] = group.crop_shape;
  const size = group.canonical_size;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(88px,1fr))] items-start gap-3">
        <Figure
          src={group.crop}
          caption={t("debug.group.crop")}
          note={`${cropWidth}×${cropHeight} px`}
          imageClassName="max-h-64 object-contain"
        />
        {group.canonical && size ? (
          <Figure
            src={group.canonical}
            pixelated
            caption={t("debug.group.canonical")}
            note={`${size[0]}×${size[1]} px`}
            imageClassName="max-h-64 object-contain"
          />
        ) : null}
        {group.hog ? (
          <>
            <Figure
              src={group.hog.gradient}
              pixelated
              caption={t("debug.group.gradient")}
              imageClassName="max-h-64 object-contain"
            />
            <Figure
              src={group.hog.rendering}
              caption={t("debug.group.hogRendering")}
              imageClassName="max-h-64 object-contain"
            />
          </>
        ) : null}
        {group.lbp ? (
          <Figure
            src={group.lbp.codes}
            pixelated
            caption={t("debug.group.lbpCodes")}
            imageClassName="max-h-64 object-contain"
          />
        ) : null}
        {group.color ? (
          <>
            <Figure
              src={group.color.h}
              caption={t("debug.group.hue")}
              imageClassName="max-h-64 object-contain"
            />
            <Figure
              src={group.color.s}
              caption={t("debug.group.saturation")}
              imageClassName="max-h-64 object-contain"
            />
            <Figure
              src={group.color.v}
              caption={t("debug.group.value")}
              imageClassName="max-h-64 object-contain"
            />
          </>
        ) : null}
      </div>

      {group.hog ? (
        <Facts
          items={[
            [t("debug.group.cell"), `${group.hog.cell[0]}×${group.hog.cell[1]} px`],
            [t("debug.group.cells"), `${group.hog.cells[0]}×${group.hog.cells[1]}`],
            [
              t("debug.group.blocks"),
              `${group.hog.blocks[0]}×${group.hog.blocks[1]} (${group.hog.block[0]}×${group.hog.block[1]})`,
            ],
            [
              t("debug.group.length"),
              `${group.hog.blocks[0]}×${group.hog.blocks[1]}×${group.hog.block[0] * group.hog.block[1]}×${group.hog.orientations} = ${fmtInt(group.length)}`,
            ],
          ]}
        />
      ) : null}
      {group.lbp ? (
        <>
          <HistogramBars
            values={group.lbp.histogram}
            colors={group.lbp.palette}
            label={t("debug.group.lbpHistogram", { n: group.lbp.n_codes })}
          />
          <Facts
            items={[
              [t("debug.group.cell"), `${group.lbp.cell[0]}×${group.lbp.cell[1]} px`],
              [t("debug.group.cells"), `${group.lbp.cells[0]}×${group.lbp.cells[1]}`],
              [
                t("debug.group.length"),
                `${group.lbp.cells[0]}×${group.lbp.cells[1]}×${group.lbp.n_codes} = ${fmtInt(group.length)}`,
              ],
            ]}
          />
        </>
      ) : null}
      {group.color ? (
        <div className="grid grid-cols-3 gap-3">
          {(["h", "s", "v"] as const).map((channel, index) => {
            const start = group.color!.bins.slice(0, index).reduce((a, b) => a + b, 0);
            return (
              <HistogramBars
                key={channel}
                values={group.values.slice(start, start + group.color!.bins[index])}
                colors={["#4dabf7", "#8d96a3", "#4a5361"][index]}
                label={t(`debug.group.bins.${channel}`, { n: group.color!.bins[index] })}
              />
            );
          })}
        </div>
      ) : null}

      <div>
        <p className="mb-1 text-[12px] text-ink-2">
          {t("debug.group.vector", { name: group.name, n: fmtInt(group.length) })}
          <span className="ml-2 font-mono text-muted">
            min {fmtNumber(group.stats.min, 3)} · max {fmtNumber(group.stats.max, 3)} · μ{" "}
            {fmtNumber(group.stats.mean, 3)} · σ {fmtNumber(group.stats.std, 3)}
          </span>
        </p>
        <VectorPlot values={group.values} label={group.name} height={56} />
        <VectorValues values={group.values} />
      </div>
    </div>
  );
}

function TraceBody({
  trace,
  baseKey,
  onUseAsBase,
}: {
  trace: ProfileTrace;
  baseKey: string | undefined;
  onUseAsBase: () => void;
}) {
  const { t } = useTranslation();
  const [selected, setSelected] = useState(0);
  const result = trace.analysis;
  const group = trace.groups[Math.min(selected, trace.groups.length - 1)];
  const bands = trace.groups.map((g) => ({
    start: g.offset,
    length: g.length,
    color: REGION_COLORS[g.region] ?? "#888",
  }));
  const vector = trace.groups.flatMap((g) => g.values);
  const differs = baseKey !== undefined && trace.preprocessing_key !== baseKey;
  return (
    <>
      <VerdictBanner verdict={result.verdict} predictedClass={result.predicted_class} compact />
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ConfidenceValue result={result} />
        <span className="font-mono text-[18px] font-semibold leading-none">
          {fmtMs(result.timings_ms.total)}
        </span>
      </div>
      {result.ground_truth ? (
        <TruthMark predicted={result.predicted_class} truth={result.ground_truth} />
      ) : null}
      {result.warnings.length ? (
        <div className="flex flex-wrap gap-1.5">
          {result.warnings.map((w) => (
            <WarningChip key={w} code={w} />
          ))}
        </div>
      ) : null}
      {differs ? (
        <div className="rounded-[6px] bg-review-tint px-3 py-2 text-[12px] text-ink">
          <p>{t("debug.profile.differentPreprocessing")}</p>
          <Button size="sm" className="mt-1.5" onClick={onUseAsBase}>
            {t("debug.profile.showItsPreprocessing")}
          </Button>
        </div>
      ) : null}

      <div>
        <Expander title={t("result.scores")} defaultOpen>
          <ScoreBars result={result} />
        </Expander>

        <Expander
          title={t("debug.profile.vector")}
          summary={<span className="font-mono text-[12px]">1×{fmtInt(trace.vector.length)}</span>}
          defaultOpen
        >
          <p className="mb-1 text-[12px] text-ink-2">
            {t("debug.profile.vectorHelp", {
              n: fmtInt(trace.vector.length),
              groups: trace.groups.length,
            })}
          </p>
          <VectorPlot values={vector} bands={bands} label={t("debug.profile.vector")} />
          <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[12px]">
            {trace.groups.map((g) => (
              <li key={g.name} className="inline-flex items-center gap-1.5 font-mono">
                <span
                  aria-hidden="true"
                  className="inline-block size-2 rounded-[2px]"
                  style={{ background: REGION_COLORS[g.region] ?? "#888" }}
                />
                {g.name} [{g.offset}:{g.offset + g.length}]
              </li>
            ))}
          </ul>
          <VectorValues values={vector} />
          {trace.scaled ? (
            <div className="mt-3">
              <p className="mb-1 text-[12px] text-ink-2">
                {t("debug.profile.scaledHelp")}
                <span className="ml-2 font-mono text-muted">
                  {t("debug.profile.scaledStats", {
                    mean: fmtNumber(trace.scaled.mean_abs, 2),
                    max: fmtNumber(trace.scaled.max_abs, 1),
                    outliers: fmtInt(trace.scaled.outliers),
                    z: trace.scaled.outlier_z,
                  })}
                </span>
              </p>
              <VectorPlot
                values={trace.scaled.values}
                bands={bands}
                label={t("debug.profile.scaled")}
              />
              <VectorValues values={trace.scaled.values} digits={2} />
            </div>
          ) : null}
        </Expander>

        <Expander
          title={t("debug.profile.groups")}
          summary={<span className="font-mono text-[12px]">{trace.groups.length}</span>}
          defaultOpen
        >
          <div role="tablist" className="mb-3 flex flex-wrap gap-1">
            {trace.groups.map((g, index) => (
              <button
                key={g.name}
                type="button"
                role="tab"
                aria-selected={index === selected}
                onClick={() => setSelected(index)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-[6px] border px-2 py-1 font-mono text-[12px]",
                  index === selected
                    ? "border-ink bg-ink text-white"
                    : "border-line bg-panel text-ink-2 hover:text-ink",
                )}
              >
                <span
                  aria-hidden="true"
                  className="inline-block size-2 rounded-[2px]"
                  style={{ background: REGION_COLORS[g.region] ?? "#888" }}
                />
                {g.name}
                <span className="opacity-70">{fmtInt(g.length)}</span>
              </button>
            ))}
          </div>
          {group ? <GroupDetail key={group.name} group={group} /> : null}
        </Expander>

        <Expander
          title={t("result.timings")}
          summary={<span className="font-mono text-[12px]">{fmtMs(result.timings_ms.total)}</span>}
        >
          <TimingBar timings={result.timings_ms} />
        </Expander>
      </div>
    </>
  );
}

export function ProfileTracePanel({
  sessionId,
  profile,
  overrides,
  baseKey,
  onUseAsBase,
}: {
  sessionId: string;
  profile: Profile;
  overrides: Params;
  baseKey: string | undefined;
  onUseAsBase: () => void;
}) {
  const { t } = useTranslation();
  const trace = useProfileTrace(sessionId, profile.id, overrides);
  return (
    <section className="panel flex min-w-0 flex-col gap-3 p-4" aria-label={profile.name}>
      <header className="flex min-w-0 items-center gap-2">
        <span
          aria-hidden="true"
          className="h-5 w-1.5 shrink-0 rounded-full"
          style={{ background: profile.color }}
        />
        <span className="min-w-0">
          <span className="block truncate font-semibold">{profile.name}</span>
          <span className="block truncate font-mono text-[11px] text-muted">
            {modelLabel(profile.model?.feature_vector, profile.algorithm)} · v{profile.version}
          </span>
        </span>
        {trace.isFetching ? <Spinner className="ml-auto" /> : null}
      </header>
      {trace.isError ? (
        <p className="rounded-[6px] bg-reject-tint px-3 py-2 text-[13px] text-reject">
          {trace.error instanceof ApiError
            ? String(
                t(`errors.${trace.error.code}`, {
                  ...trace.error.details,
                  defaultValue: trace.error.message,
                }),
              )
            : t("errors.unexpected")}
        </p>
      ) : null}
      {trace.data ? (
        <TraceBody trace={trace.data} baseKey={baseKey} onUseAsBase={onUseAsBase} />
      ) : null}
    </section>
  );
}
