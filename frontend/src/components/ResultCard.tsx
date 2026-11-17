// Result of one profile (§15.4): verdict banner, class, confidence, 7 scores, timings, warnings.
import { ChevronRight, Info } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { ClassLabel, ProfileStatusBadge, TruthMark, VerdictBanner, WarningChip } from "./status";
import { Tip, cn } from "./ui/primitives";
import { fmtMs, fmtNumber } from "@/lib/format";
import {
  CLASS_COLORS,
  CLASS_NAMES,
  type AnalysisResult,
  type ClassName,
  type Timings,
} from "@/lib/types";

export function ProfileHeader({ result }: { result: AnalysisResult }) {
  const { t } = useTranslation();
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span
        aria-hidden="true"
        className="h-5 w-1.5 shrink-0 rounded-full"
        style={{ background: result.profile.color }}
      />
      <span className="truncate font-semibold">{result.profile.name}</span>
      <span className="shrink-0 font-mono text-[12px] text-muted">v{result.profile.version}</span>
      {result.profile.status === "modified" ? (
        <Tip content={t("result.modifiedHelp")}>
          <span tabIndex={0}>
            <ProfileStatusBadge status="modified" />
          </span>
        </Tip>
      ) : null}
    </div>
  );
}

export function ConfidenceValue({ result }: { result: AnalysisResult }) {
  const { t } = useTranslation();
  return (
    <span className="inline-flex items-center gap-1.5 text-[13px]">
      <span className="text-ink-2">{t("result.confidence")}</span>
      <span className="font-mono text-[15px] font-semibold">{fmtNumber(result.confidence, 2)}</span>
      <Tip
        content={
          <span>
            {t(`confidence.${result.profile.algorithm}`, { defaultValue: t("confidence.generic") })}
            <br />
            {t("confidence.notComparable")}
          </span>
        }
      >
        <button
          type="button"
          aria-label={t("result.confidenceHelp")}
          className="text-muted hover:text-ink"
        >
          <Info className="size-3.5" />
        </button>
      </Tip>
    </span>
  );
}

export function Expander({
  title,
  summary,
  children,
  defaultOpen,
}: {
  title: ReactNode;
  summary?: ReactNode;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <div className="border-t border-line">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 py-2 text-left text-[13px] font-medium text-ink-2 hover:text-ink"
      >
        <ChevronRight
          className={cn("size-4 transition-transform", open && "rotate-90")}
          aria-hidden="true"
        />
        <span>{title}</span>
        <span className="ml-auto">{summary}</span>
      </button>
      {open ? <div className="pb-3">{children}</div> : null}
    </div>
  );
}

export function ScoreBars({ result }: { result: AnalysisResult }) {
  const { t } = useTranslation();
  const lowerBetter = result.score_kind === "knn_class_distance";
  const values = CLASS_NAMES.map((name) => result.scores[name] ?? 0);
  const min = Math.min(0, ...values);
  const max = Math.max(...values, 1e-9);
  const span = max - min || 1;
  const winner = result.predicted_class;
  return (
    <div>
      <p className="mb-2 text-[12px] text-muted">
        {t(`scoreKind.${result.score_kind}`, { defaultValue: result.score_kind })}
      </p>
      <ul className="flex flex-col gap-1">
        {CLASS_NAMES.map((name) => {
          const value = result.scores[name];
          const v = value ?? 0;
          const left = ((Math.min(0, v) - min) / span) * 100;
          const width = (Math.abs(v) / span) * 100;
          return (
            <li
              key={name}
              className="grid grid-cols-[auto_minmax(3rem,1fr)_3.5rem] items-center gap-2"
            >
              <span className={cn("whitespace-nowrap", name === winner && "font-semibold")}>
                <ClassLabel name={name} />
              </span>
              <span className="relative h-3.5 rounded-[3px] bg-sunken">
                <span
                  className={cn(
                    "absolute top-0 h-full rounded-[3px]",
                    name === winner && "outline-2 outline-ink",
                  )}
                  style={{
                    left: `${left}%`,
                    width: `${Math.max(width, 0.5)}%`,
                    background: CLASS_COLORS[name],
                  }}
                />
              </span>
              <span
                className={cn(
                  "text-right font-mono text-[12px]",
                  name === winner && "font-semibold",
                )}
              >
                {value === null ? "—" : fmtNumber(v, 2)}
              </span>
            </li>
          );
        })}
      </ul>
      {lowerBetter ? (
        <p className="mt-2 text-[12px] font-medium text-ink-2">{t("result.lowerIsBetter")}</p>
      ) : null}
      {result.details.svm_votes ? (
        <p className="mt-2 text-[12px] text-muted">
          {t("result.svmVotes")}{" "}
          <span className="font-mono">
            {CLASS_NAMES.map((n) => result.details.svm_votes?.[n] ?? 0).join(" · ")}
          </span>
        </p>
      ) : null}
      {result.details.neighbors?.length ? (
        <div className="mt-3">
          <p className="mb-1.5 text-[12px] font-medium text-ink-2">
            {t("result.nearest", { k: result.details.k ?? 1 })}
          </p>
          <div className="flex gap-2 overflow-x-auto">
            {result.details.neighbors.map((n, i) => (
              <figure key={n.train_index} className="w-16 shrink-0 text-center">
                {n.image ? (
                  <img
                    src={`/api/dataset/images/train/${n.image}?thumb=1`}
                    alt={n.class}
                    loading="lazy"
                    className="h-16 w-16 rounded-[4px] border-2 object-cover"
                    style={{ borderColor: CLASS_COLORS[n.class as ClassName] }}
                  />
                ) : (
                  <div
                    className="h-16 w-16 rounded-[4px] border-2 bg-sunken"
                    style={{ borderColor: CLASS_COLORS[n.class as ClassName] }}
                  />
                )}
                <figcaption className="mt-0.5 font-mono text-[10px] leading-tight text-ink-2">
                  #{i + 1} d={fmtNumber(n.distance, 1)}
                </figcaption>
              </figure>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export const STEP_COLORS: Record<string, string> = {
  decode: "#8d96a3",
  normalize: "#5f6b7a",
  foreground_mask: "#3d4855",
  localize: "#4f5d9a",
  crop_regions: "#9a6b8f",
  cap_hog: "#c25b5b",
  cap_lbp: "#d57b7b",
  cap_color: "#e8aaaa",
  label_hog: "#3f8f56",
  label_lbp: "#65aa7a",
  label_color: "#9fd0ae",
  body_hog: "#2f6fb5",
  body_lbp: "#5a92c9",
  body_color: "#9cc0e6",
  auto_1_hog: "#d9822b",
  auto_1_lbp: "#e6a25e",
  auto_1_color: "#f3cfa5",
  auto_2_hog: "#6b46c1",
  auto_2_lbp: "#8f70d4",
  auto_2_color: "#c5b4ea",
  auto_3_hog: "#14917a",
  auto_3_lbp: "#4cb09c",
  auto_3_color: "#a0d8cc",
  bottle_hog: "#b5892f",
  bottle_lbp: "#cba654",
  bottle_color: "#e3cd92",
  predict: "#1b1f24",
};

export function timingSteps(timings: Timings): { step: string; ms: number }[] {
  const base = ["decode", "normalize", "foreground_mask", "localize", "crop_regions"] as const;
  return [
    ...base.map((step) => ({ step, ms: timings[step] ?? 0 })),
    ...Object.entries(timings.features ?? {}).map(([step, ms]) => ({ step, ms })),
    { step: "predict", ms: timings.predict ?? 0 },
  ];
}

export function TimingBar({ timings }: { timings: Timings }) {
  const { t } = useTranslation();
  const steps = timingSteps(timings);
  const total = steps.reduce((sum, s) => sum + s.ms, 0) || 1;
  return (
    <div>
      <div
        className="flex h-4 w-full overflow-hidden rounded-[3px]"
        role="img"
        aria-label={t("result.timingBreakdown")}
      >
        {steps.map((s) => (
          <Tip key={s.step} content={`${t(`steps.${s.step}`)}: ${fmtMs(s.ms)}`}>
            <span
              tabIndex={0}
              className="h-full"
              style={{
                width: `${(s.ms / total) * 100}%`,
                background: STEP_COLORS[s.step] ?? "#888",
              }}
            />
          </Tip>
        ))}
      </div>
      <table className="mt-2 w-full text-[12px]">
        <tbody>
          {steps.map((s) => (
            <tr key={s.step}>
              <td className="py-px">
                <span
                  className="mr-1.5 inline-block size-2 rounded-[2px]"
                  style={{ background: STEP_COLORS[s.step] }}
                />
                {t(`steps.${s.step}`)}
              </td>
              <td className="py-px text-right font-mono">{fmtMs(s.ms)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ResultDetails({
  result,
  scoresOpen,
}: {
  result: AnalysisResult;
  scoresOpen?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <ConfidenceValue result={result} />
        <span className="font-mono text-[22px] font-semibold leading-none">
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
      <div>
        <Expander title={t("result.scores")} defaultOpen={scoresOpen}>
          <ScoreBars result={result} />
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

export function ResultCard({ result, compact }: { result: AnalysisResult; compact?: boolean }) {
  return (
    <section className="panel flex flex-col gap-3 p-4" aria-label={result.profile.name}>
      <ProfileHeader result={result} />
      <VerdictBanner
        verdict={result.verdict}
        predictedClass={result.predicted_class}
        compact={compact}
      />
      <ResultDetails result={result} scoresOpen={!compact} />
    </section>
  );
}
