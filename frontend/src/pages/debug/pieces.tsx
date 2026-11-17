// Building blocks of the debug screen: step panels, zoomable figures, pixel matrices and the
// small SVG plots (difference histogram, column/row profiles of the mask, feature vectors).
import { Check, Copy } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button, Dialog, cn } from "@/components/ui/primitives";
import { fmtInt, fmtNumber } from "@/lib/format";
import type { PixelPatch, PreprocessingTrace, Rect } from "@/lib/types";

export const REGION_COLORS: Record<string, string> = {
  cap: "#e03131",
  label: "#2f9e44",
  body: "#1c7ed6",
  auto_1: "#f08c00",
  auto_2: "#0c8599",
  auto_3: "#c2255c",
  bottle: "#7048e8",
};

const MARK = "#e03131";

export function Step({
  n,
  title,
  summary,
  children,
}: {
  n: number;
  title: ReactNode;
  summary?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="panel p-4" aria-label={String(title)}>
      <header className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <span className="flex size-6 shrink-0 items-center justify-center self-center rounded-full bg-ink font-mono text-[12px] font-semibold text-white">
          {n}
        </span>
        <h2 className="text-[15px] font-semibold">{title}</h2>
        {summary ? <p className="text-[13px] text-ink-2">{summary}</p> : null}
      </header>
      {children}
    </section>
  );
}

/** Label / value pairs (measurements of a step). */
export function Facts({ items }: { items: [ReactNode, ReactNode][] }) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-[13px]">
      {items.map(([label, value], index) => (
        <div key={index} className="contents">
          <dt className="text-ink-2">{label}</dt>
          <dd className="font-mono">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** An image with a caption; a click opens it enlarged. `rects` are drawn in relative (0..1)
 * image coordinates, so they stay sharp at any size. */
export function Figure({
  src,
  caption,
  note,
  pixelated,
  rects,
  className,
  imageClassName,
}: {
  src: string;
  caption: ReactNode;
  note?: ReactNode;
  pixelated?: boolean;
  rects?: { rect: Rect; color: string; dashed?: boolean; label?: string }[];
  className?: string;
  imageClassName?: string;
}) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const picture = (large: boolean) => (
    <span className="relative block">
      <img
        src={src}
        alt=""
        className={cn(
          "block w-full",
          large ? "max-h-[70vh] object-contain" : imageClassName,
          pixelated && "[image-rendering:pixelated]",
        )}
      />
      {rects?.length ? (
        <svg
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          className="pointer-events-none absolute inset-0 h-full w-full"
          aria-hidden="true"
        >
          {rects.map(({ rect, color, dashed }, index) => (
            <rect
              key={index}
              x={rect[0]}
              y={rect[1]}
              width={rect[2] - rect[0]}
              height={rect[3] - rect[1]}
              fill="none"
              stroke={color}
              strokeWidth={2}
              strokeDasharray={dashed ? "6 4" : undefined}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
      ) : null}
    </span>
  );
  return (
    <figure className={cn("min-w-0", className)}>
      <button
        type="button"
        onClick={() => setOpen(true)}
        aria-label={t("debug.enlarge")}
        className="block w-full cursor-zoom-in overflow-hidden rounded-[6px] bg-stage"
      >
        {picture(false)}
      </button>
      <figcaption className="mt-1.5 text-[12px] leading-snug">
        <span className="font-medium text-ink">{caption}</span>
        {note ? <span className="block font-mono text-muted">{note}</span> : null}
      </figcaption>
      {rects?.some((r) => r.label) ? (
        <ul className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[12px]">
          {rects
            .filter((r) => r.label)
            .map((r) => (
              <li key={r.label} className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="inline-block h-0.5 w-4"
                  style={{ background: r.color }}
                />
                <span className="font-mono">{r.label}</span>
              </li>
            ))}
        </ul>
      ) : null}
      <Dialog open={open} onOpenChange={setOpen} title={caption} wide>
        <div className="overflow-hidden rounded-[6px] bg-stage">{picture(true)}</div>
      </Dialog>
    </figure>
  );
}

/** The pixels of a small square, each cell painted with its own colour and labelled B, G, R. */
export function PixelMatrix({ patch, caption }: { patch: PixelPatch; caption: ReactNode }) {
  return (
    <figure>
      <table className="border-separate border-spacing-0.5 font-mono text-[11px] leading-tight">
        <tbody>
          {patch.values.map((row, r) => (
            <tr key={r}>
              {row.map(([b, g, red], c) => (
                <td
                  key={c}
                  className="rounded-[3px] px-1.5 py-1 text-center"
                  style={{
                    background: `rgb(${red} ${g} ${b})`,
                    color: (red + g + b) / 3 < 140 ? "#fff" : "#1b1f24",
                  }}
                >
                  B{b}
                  <br />G{g}
                  <br />R{red}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <figcaption className="mt-1.5 text-[12px] font-medium">{caption}</figcaption>
    </figure>
  );
}

/** Histogram of the background differences (log scale) with the threshold lines. */
export function DifferenceHistogram({ data }: { data: PreprocessingTrace["difference"] }) {
  const { t } = useTranslation();
  const width = 320;
  const height = 150;
  const bins = data.histogram;
  const top = Math.log10(Math.max(...bins, 1) + 1);
  const barWidth = width / bins.length;
  const xOf = (value: number) => (Math.min(value, data.clip) / (data.clip + 1)) * width;
  const floorUsed = data.threshold > data.otsu;
  return (
    <figure className="min-w-0">
      <svg viewBox={`0 0 ${width} ${height + 18}`} className="w-full" role="img">
        <title>{t("debug.difference.histogram")}</title>
        {bins.map((count, index) => {
          const h = (Math.log10(count + 1) / top) * height;
          return (
            <rect
              key={index}
              x={index * barWidth}
              y={height - h}
              width={barWidth - 0.6}
              height={h}
              fill="#8d96a3"
            />
          );
        })}
        <line x1={0} x2={width} y1={height} y2={height} stroke="#4a5361" strokeWidth={1} />
        <line
          x1={xOf(data.otsu)}
          x2={xOf(data.otsu)}
          y1={0}
          y2={height}
          stroke={MARK}
          strokeWidth={2}
        />
        {floorUsed ? (
          <line
            x1={xOf(data.noise_floor)}
            x2={xOf(data.noise_floor)}
            y1={0}
            y2={height}
            stroke="#b26a00"
            strokeWidth={2}
            strokeDasharray="4 3"
          />
        ) : null}
        <text x={0} y={height + 13} fontSize={10} fill="#4a5361">
          0
        </text>
        <text x={width} y={height + 13} fontSize={10} fill="#4a5361" textAnchor="end">
          {data.clip}
        </text>
      </svg>
      <figcaption className="mt-1 text-[12px] leading-snug">
        <span className="font-medium">{t("debug.difference.histogram")}</span>
        <span className="block font-mono text-muted">
          {t("debug.difference.otsuLine", { value: fmtInt(data.otsu) })}
        </span>
      </figcaption>
    </figure>
  );
}

/** The mask with the box measured on it, the column profile above and the row profile beside
 * it, sharing the axes of the mask (the figure of the "bottle box" slide). */
export function LocalizationPlot({
  data,
  mask,
}: {
  data: PreprocessingTrace["localization"];
  mask: string;
}) {
  const { t } = useTranslation();
  const [width, height] = data.size;
  const top = 96;
  const side = 150;
  const gap = 8;
  const columnMax = Math.max(...data.columns, ...data.columns_raw, 1);
  const rowMax = Math.max(...data.rows, 1);
  const columnY = (value: number) => top - gap - (value / columnMax) * (top - gap - 12);
  const rowX = (value: number) => width + gap + (value / rowMax) * (side - gap - 12);
  const line = (values: number[], point: (v: number, i: number) => string) =>
    values.map((v, i) => point(v, i)).join(" ");
  const [x0, x1] = data.x;
  const [boxX0, boxX1] = data.box_x;
  const [y0, y1] = data.y;
  return (
    <svg
      viewBox={`0 0 ${width + side} ${top + height}`}
      className="w-full rounded-[6px] bg-panel"
      role="img"
    >
      <title>{t("debug.locate.plot")}</title>
      <image href={mask} x={0} y={top} width={width} height={height} />
      {/* column profile */}
      <rect x={x0} y={0} width={x1 - x0} height={top - gap} fill={MARK} opacity={0.12} />
      <polyline
        points={line(data.columns_raw, (v, i) => `${i + 0.5},${columnY(v)}`)}
        fill="none"
        stroke="#b9c0ca"
        strokeWidth={1}
      />
      <polyline
        points={line(data.columns, (v, i) => `${i + 0.5},${columnY(v)}`)}
        fill="none"
        stroke="#1b1f24"
        strokeWidth={1.5}
      />
      <line
        x1={0}
        x2={width}
        y1={columnY(data.column_threshold)}
        y2={columnY(data.column_threshold)}
        stroke={MARK}
        strokeWidth={1.5}
        strokeDasharray="6 4"
      />
      <text x={4} y={11} fontSize={11} fill="#4a5361">
        {t("debug.locate.columns")}
      </text>
      <text
        x={width - 4}
        y={columnY(data.column_threshold) - 3}
        fontSize={11}
        fill={MARK}
        textAnchor="end"
      >
        {t("debug.locate.ofPeak", { percent: Math.round(data.column_fraction * 100) })}
      </text>
      {/* row profile */}
      <rect
        x={width + gap}
        y={top + y0}
        width={side - gap}
        height={y1 - y0}
        fill={MARK}
        opacity={0.12}
      />
      <polyline
        points={line(data.rows, (v, i) => `${rowX(v)},${top + i + 0.5}`)}
        fill="none"
        stroke="#1b1f24"
        strokeWidth={1.5}
      />
      {/* each edge has its own fraction: the top line spans the upper half, the bottom one the lower */}
      <line
        x1={rowX(data.top_threshold)}
        x2={rowX(data.top_threshold)}
        y1={top}
        y2={top + height / 2}
        stroke={MARK}
        strokeWidth={1.5}
        strokeDasharray="6 4"
      />
      <line
        x1={rowX(data.row_threshold)}
        x2={rowX(data.row_threshold)}
        y1={top + height / 2}
        y2={top + height}
        stroke={MARK}
        strokeWidth={1.5}
        strokeDasharray="6 4"
      />
      <text x={width + side - 4} y={top - 14} fontSize={11} fill="#4a5361" textAnchor="end">
        {t("debug.locate.rows")}
      </text>
      <text x={width + side - 4} y={top - 2} fontSize={11} fill={MARK} textAnchor="end">
        {t("debug.locate.topBottom", {
          top: Math.round(data.top_fraction * 100),
          bottom: Math.round(data.row_fraction * 100),
        })}
      </text>
      {/* box */}
      <rect
        x={boxX0}
        y={top + y0}
        width={boxX1 - boxX0}
        height={y1 - y0}
        fill="none"
        stroke={MARK}
        strokeWidth={2.5}
      />
    </svg>
  );
}

/** Every value of a vector as a vertical line from zero; `bands` tint the span of each group. */
export function VectorPlot({
  values,
  bands,
  height = 72,
  label,
}: {
  values: number[];
  bands?: { start: number; length: number; color: string }[];
  height?: number;
  label: string;
}) {
  const path = useMemo(() => {
    const low = Math.min(0, ...values);
    const high = Math.max(0, ...values);
    const span = high - low || 1;
    const y = (value: number) => ((high - value) / span) * 100;
    const zero = y(0);
    return {
      zero,
      d: values.map((v, i) => `M${i + 0.5} ${zero.toFixed(2)}V${y(v).toFixed(2)}`).join(""),
    };
  }, [values]);
  return (
    <svg
      viewBox={`0 0 ${values.length} 100`}
      preserveAspectRatio="none"
      className="w-full rounded-[4px] bg-sunken"
      style={{ height }}
      role="img"
    >
      <title>{label}</title>
      {bands?.map((band) => (
        <rect
          key={band.start}
          x={band.start}
          y={0}
          width={band.length}
          height={100}
          fill={band.color}
          opacity={0.16}
        />
      ))}
      <line
        x1={0}
        x2={values.length}
        y1={path.zero}
        y2={path.zero}
        stroke="#9aa3ae"
        strokeWidth={1}
        vectorEffect="non-scaling-stroke"
      />
      <path d={path.d} stroke="#1b1f24" strokeWidth={1} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** The numbers of a vector, hidden until asked for, with a copy button. */
export function VectorValues({ values, digits = 4 }: { values: number[]; digits?: number }) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);
  const copy = () => {
    void navigator.clipboard?.writeText(JSON.stringify(values)).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  };
  return (
    <div>
      <div className="flex items-center gap-2">
        <Button size="sm" variant="ghost" aria-expanded={open} onClick={() => setOpen(!open)}>
          {open
            ? t("debug.vector.hide")
            : t("debug.vector.show", { count: values.length, n: fmtInt(values.length) })}
        </Button>
        <Button size="sm" variant="ghost" onClick={copy}>
          {copied ? <Check /> : <Copy />}
          {copied ? t("debug.vector.copied") : t("debug.vector.copy")}
        </Button>
      </div>
      {open ? (
        <p className="mt-1 max-h-44 overflow-auto rounded-[6px] bg-sunken p-2 font-mono text-[11px] leading-relaxed break-words">
          [{values.map((v) => fmtNumber(v, digits)).join("  ")}]
        </p>
      ) : null}
    </div>
  );
}
