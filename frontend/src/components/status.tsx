// Verdict and status indicators. Colour is never the only cue (WCAG 1.4.1): each verdict has
// its own shape (circle / square / triangle), icon and text label.
import { AlertTriangle, Ban, Check, CircleHelp, Lock, X } from "lucide-react";
import { useTranslation } from "react-i18next";

import { cn } from "./ui/primitives";
import { CLASS_COLORS, type ClassName, type ProfileStatusName, type Verdict } from "@/lib/types";

const VERDICT_STYLE: Record<Verdict, { fg: string; tint: string }> = {
  pass: { fg: "text-pass", tint: "bg-pass-tint" },
  reject: { fg: "text-reject", tint: "bg-reject-tint" },
  review: { fg: "text-review", tint: "bg-review-tint" },
};

export const VERDICT_HEX: Record<Verdict, string> = {
  pass: "#1E7B34",
  reject: "#C62828",
  review: "#B26A00",
};

/** The verdict shape: PASS circle, REJECT square, REVIEW triangle. */
export function VerdictShape({ verdict, size = 16 }: { verdict: Verdict; size?: number }) {
  const color = VERDICT_HEX[verdict];
  const stroke = Math.max(1.6, size / 9);
  return (
    <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden="true" className="shrink-0">
      {verdict === "pass" ? (
        <>
          <circle cx="10" cy="10" r="9" fill={color} />
          <path
            d="M5.5 10.2l3 3 6-6.4"
            fill="none"
            stroke="white"
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </>
      ) : verdict === "reject" ? (
        <>
          <rect x="1" y="1" width="18" height="18" rx="2.5" fill={color} />
          <path
            d="M6.5 6.5l7 7M13.5 6.5l-7 7"
            stroke="white"
            strokeWidth={stroke}
            strokeLinecap="round"
          />
        </>
      ) : (
        <>
          <path d="M10 1.5L19 18H1z" fill={color} strokeLinejoin="round" />
          <path d="M10 7.5v4.5" stroke="white" strokeWidth={stroke} strokeLinecap="round" />
          <circle cx="10" cy="14.8" r="1.1" fill="white" />
        </>
      )}
    </svg>
  );
}

export function VerdictChip({ verdict, className }: { verdict: Verdict; className?: string }) {
  const { t } = useTranslation();
  const style = VERDICT_STYLE[verdict];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-[6px] px-2 py-0.5 text-[12px] font-semibold",
        style.tint,
        style.fg,
        className,
      )}
    >
      <VerdictShape verdict={verdict} size={13} />
      {t(`verdict.${verdict}`)}
    </span>
  );
}

/** Large banner readable from 3 m (§15.1): 48–64 px verdict, then the class identifier. */
export function VerdictBanner({
  verdict,
  predictedClass,
  compact,
}: {
  verdict: Verdict;
  predictedClass: ClassName;
  compact?: boolean;
}) {
  const { t } = useTranslation();
  const style = VERDICT_STYLE[verdict];
  return (
    <div
      key={`${verdict}-${predictedClass}`}
      className={cn(
        "verdict-in flex items-center gap-4 rounded-[10px] px-4 [container-type:inline-size]",
        style.tint,
        compact ? "py-3" : "py-4",
      )}
    >
      <VerdictShape verdict={verdict} size={compact ? 40 : 56} />
      <div className="min-w-0">
        <div
          className={cn(
            "font-semibold leading-none tracking-tight",
            style.fg,
            compact ? "text-[clamp(26px,11cqi,44px)]" : "text-[clamp(32px,13cqi,64px)]",
          )}
        >
          {t(`verdict.${verdict}`)}
        </div>
        <div
          className={cn(
            "mt-1.5 truncate font-mono text-ink",
            compact ? "text-[15px]" : "text-[18px]",
          )}
        >
          {predictedClass}
        </div>
      </div>
    </div>
  );
}

export function ClassDot({ name, size = 10 }: { name: ClassName; size?: number }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block shrink-0 rounded-[2px]"
      style={{ width: size, height: size, background: CLASS_COLORS[name] }}
    />
  );
}

/** Class identifier (never translated) with its localized description next to it (REQ-L3). */
export function ClassLabel({
  name,
  withDescription,
}: {
  name: ClassName;
  withDescription?: boolean;
}) {
  const { t } = useTranslation();
  return (
    <span className="inline-flex min-w-0 items-center gap-1.5">
      <ClassDot name={name} />
      <span className="font-mono text-[13px]">{name}</span>
      {withDescription ? (
        <span className="truncate text-[12px] text-muted">{t(`classes.${name}`)}</span>
      ) : null}
    </span>
  );
}

const STATUS_ICON = {
  ready: Check,
  modified: AlertTriangle,
  model_missing: CircleHelp,
  incompatible: Ban,
} as const;

const STATUS_STYLE: Record<ProfileStatusName, string> = {
  ready: "text-pass bg-pass-tint",
  modified: "text-review bg-review-tint",
  model_missing: "text-ink-2 bg-sunken",
  incompatible: "text-reject bg-reject-tint",
};

export function ProfileStatusBadge({
  status,
  short,
}: {
  status: ProfileStatusName;
  short?: boolean;
}) {
  const { t } = useTranslation();
  const Icon = STATUS_ICON[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-[6px] px-1.5 py-0.5 text-[12px] font-medium",
        STATUS_STYLE[status],
      )}
    >
      <Icon className="size-3.5" aria-hidden="true" />
      {short ? (
        <span className="sr-only">{t(`profileStatus.${status}`)}</span>
      ) : (
        t(`profileStatus.${status}`)
      )}
    </span>
  );
}

export function LockedMark() {
  const { t } = useTranslation();
  return (
    <span className="inline-flex items-center gap-1 text-[12px] text-muted">
      <Lock className="size-3.5" aria-hidden="true" />
      {t("params.lockedShort")}
    </span>
  );
}

export function WarningChip({ code }: { code: string }) {
  const { t } = useTranslation();
  return (
    <span className="inline-flex items-center gap-1 rounded-[6px] bg-review-tint px-1.5 py-0.5 text-[12px] font-medium text-review">
      <AlertTriangle className="size-3.5" aria-hidden="true" />
      {t(`warnings.${code}`, { defaultValue: code })}
    </span>
  );
}

export function TruthMark({ predicted, truth }: { predicted: ClassName; truth: ClassName }) {
  const { t } = useTranslation();
  const ok = predicted === truth;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 text-[13px] font-medium",
        ok ? "text-pass" : "text-reject",
      )}
    >
      {ok ? (
        <Check className="size-4" aria-hidden="true" />
      ) : (
        <X className="size-4" aria-hidden="true" />
      )}
      {ok ? t("result.matchesTruth") : t("result.expected")}
      {!ok ? <span className="font-mono">{truth}</span> : null}
    </span>
  );
}
