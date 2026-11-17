// ECharts wrapper (tree-shaken core) and the chart card used by the reports: title, PNG export,
// "view as table" toggle (WCAG), empty state and click-to-filter.
import { BarChart, LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import * as echarts from "echarts/core";
import { CanvasRenderer } from "echarts/renderers";
import { Download, Table2 } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { Button, Tip, cn } from "../ui/primitives";

echarts.use([
  BarChart,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

export type ChartOption = echarts.EChartsCoreOption;

export const INK = {
  primary: "#1B1F24",
  secondary: "#4A5361",
  muted: "#6B7480",
  line: "#D9DDE3",
  grid: "#ECEEF1",
};

/** Base option: recessive axes and grid, Plex fonts, shared tooltip style. */
export function baseOption(): ChartOption {
  return {
    animationDuration: 200,
    textStyle: { fontFamily: "IBM Plex Sans, sans-serif", color: INK.secondary, fontSize: 12 },
    grid: { left: 8, right: 16, top: 28, bottom: 8, containLabel: true },
    tooltip: {
      trigger: "axis",
      backgroundColor: "#FFFFFF",
      borderColor: INK.line,
      textStyle: { color: INK.primary, fontSize: 12 },
      extraCssText: "box-shadow: 0 8px 24px rgb(27 31 36 / 0.16); border-radius: 6px;",
    },
    legend: {
      top: 0,
      left: 0,
      icon: "roundRect",
      itemWidth: 10,
      itemHeight: 10,
      textStyle: { color: INK.secondary },
    },
  };
}

export const axisStyle = {
  axisLine: { lineStyle: { color: INK.line } },
  axisTick: { show: false },
  axisLabel: { color: INK.muted },
  splitLine: { lineStyle: { color: INK.grid } },
};

export function EChart({
  option,
  height,
  onClick,
  onReady,
  label,
}: {
  option: ChartOption;
  height: number;
  onClick?: (params: { name: string; seriesName?: string; data?: unknown }) => void;
  onReady?: (chart: echarts.ECharts) => void;
  label: string;
}) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const clickRef = useRef(onClick);
  useEffect(() => {
    clickRef.current = onClick;
  });
  useEffect(() => {
    if (!element.current) return;
    const instance = echarts.init(element.current, undefined, { renderer: "canvas" });
    chart.current = instance;
    instance.on("click", (params) =>
      clickRef.current?.(params as { name: string; seriesName?: string }),
    );
    const observer = new ResizeObserver(() => instance.resize());
    observer.observe(element.current);
    // Canvas text is measured when drawn: redraw once the self-hosted fonts are loaded.
    void document.fonts?.ready.then(() => {
      if (!instance.isDisposed()) instance.setOption(instance.getOption(), true);
    });
    onReady?.(instance);
    return () => {
      observer.disconnect();
      instance.dispose();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    chart.current?.setOption(option, true);
  }, [option]);
  return <div ref={element} role="img" aria-label={label} style={{ height }} className="w-full" />;
}

export function ChartCard({
  title,
  subtitle,
  option,
  height = 260,
  table,
  empty,
  onClick,
  className,
  children,
}: {
  title: string;
  subtitle?: string;
  option?: ChartOption;
  height?: number;
  table: ReactNode;
  empty?: boolean;
  onClick?: (params: { name: string; seriesName?: string; data?: unknown }) => void;
  className?: string;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  const [asTable, setAsTable] = useState(false);
  const instance = useRef<echarts.ECharts | null>(null);
  const exportPng = () => {
    const url = instance.current?.getDataURL({
      type: "png",
      pixelRatio: 2,
      backgroundColor: "#FFFFFF",
    });
    if (!url) return;
    const link = document.createElement("a");
    link.href = url;
    link.download = `${title.replace(/[^\w-]+/g, "_")}.png`;
    link.click();
  };
  return (
    <section className={cn("panel flex min-w-0 flex-col p-3", className)} aria-label={title}>
      <div className="mb-2 flex items-start gap-2">
        <div className="min-w-0">
          <h3 className="font-semibold">{title}</h3>
          {subtitle ? <p className="text-[12px] text-muted">{subtitle}</p> : null}
        </div>
        <div className="ml-auto flex gap-1">
          {option && !empty ? (
            <Tip content={t("reports.exportPng")}>
              <Button
                size="icon"
                variant="ghost"
                aria-label={t("reports.exportPngNamed", { name: title })}
                onClick={exportPng}
                disabled={asTable}
              >
                <Download />
              </Button>
            </Tip>
          ) : null}
          {!empty ? (
            <Tip content={asTable ? t("reports.viewChart") : t("reports.viewTable")}>
              <Button
                size="icon"
                variant={asTable ? "secondary" : "ghost"}
                aria-pressed={asTable}
                aria-label={t("reports.viewTableNamed", { name: title })}
                onClick={() => setAsTable(!asTable)}
              >
                <Table2 />
              </Button>
            </Tip>
          ) : null}
        </div>
      </div>
      {empty ? (
        <p className="flex flex-1 items-center justify-center py-10 text-[13px] text-muted">
          {t("reports.noData")}
        </p>
      ) : asTable || !option ? (
        <div className="max-h-[420px] overflow-auto">{table}</div>
      ) : (
        <EChart
          option={option}
          height={height}
          onClick={onClick}
          label={title}
          onReady={(c) => (instance.current = c)}
        />
      )}
      {children}
    </section>
  );
}

export function DataTable({
  head,
  rows,
  caption,
}: {
  head: ReactNode[];
  rows: ReactNode[][];
  caption: string;
}) {
  return (
    <table className="w-full text-[12px]">
      <caption className="sr-only">{caption}</caption>
      <thead>
        <tr className="border-b border-line text-left text-ink-2">
          {head.map((h, i) => (
            <th key={i} className={cn("py-1 pr-3 font-medium", i > 0 && "text-right")}>
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, r) => (
          <tr key={r} className="border-b border-line last:border-0">
            {row.map((cell, i) => (
              <td key={i} className={cn("py-1 pr-3", i > 0 && "text-right font-mono")}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
