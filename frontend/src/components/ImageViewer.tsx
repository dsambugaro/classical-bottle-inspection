// Image viewer (§15.4): dark stage, zoom (wheel/buttons/fit/1:1/double-click), pan by drag,
// SVG overlays in image coordinates, mask overlay, raw|normalized compare slider, optional
// synchronised view and draggable ROI edges / reference patch (profile editor).
import { Maximize, Minus, Plus, Scan } from "lucide-react";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";

import { Button, Tip, cn } from "./ui/primitives";
import { VERDICT_HEX } from "./status";
import type { Geometry, Rect, Verdict } from "@/lib/types";

export interface ViewState {
  /** Zoom relative to "fit" (1 = fit). */
  zoom: number;
  /** Centre of the view in relative image coordinates. */
  cx: number;
  cy: number;
}

export const FIT_VIEW: ViewState = { zoom: 1, cx: 0.5, cy: 0.5 };

export interface Overlays {
  rois: boolean;
  bottle: boolean;
  patch: boolean;
  mask: boolean;
}

/** Colour of each region of a model: the ad hoc ROIs and the automatic regions (learned). */
export const ROI_COLORS = {
  cap: "#E03131",
  label: "#2F9E44",
  body: "#1C7ED6",
  auto_1: "#F08C00",
  auto_2: "#7048E8",
  auto_3: "#0CA678",
} as const;
type RegionName = keyof typeof ROI_COLORS;
/** Only the ad hoc ROIs can be dragged: the automatic regions come from training. */
type RoiName = "cap" | "label" | "body";
const EDITABLE_ROIS: readonly string[] = ["cap", "label", "body"];

interface Props {
  src: string | null;
  width: number;
  height: number;
  geometry?: Geometry | null;
  overlays?: Overlays;
  maskSrc?: string | null;
  maskColor?: string;
  compareSrc?: string | null;
  verdict?: Verdict | null;
  view?: ViewState;
  onViewChange?: (view: ViewState) => void;
  editable?: {
    /** `band` is relative to the bottle box (0 = top of the bottle, 1 = bottom). */
    onRoiChange?: (name: RoiName, band: [number, number], final: boolean) => void;
    onPatchChange?: (rect: Rect, final: boolean) => void;
  };
  className?: string;
  label?: string;
  children?: ReactNode;
}

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 16;

type Drag =
  | { kind: "pan"; startX: number; startY: number; view: ViewState }
  | { kind: "roi"; name: RoiName; edge: 0 | 1; band: [number, number] }
  | { kind: "patch"; corner: 0 | 1 | 2 | 3 | -1; rect: Rect; startX: number; startY: number }
  | { kind: "compare" };

export function ImageViewer({
  src,
  width,
  height,
  geometry,
  overlays = { rois: true, bottle: true, patch: false, mask: false },
  maskSrc,
  maskColor = "#FAB005",
  compareSrc,
  verdict,
  view: controlledView,
  onViewChange,
  editable,
  className,
  label,
  children,
}: Props) {
  const { t } = useTranslation();
  const stage = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [localView, setLocalView] = useState<ViewState>(FIT_VIEW);
  const [split, setSplit] = useState(0.5);
  const [drag, setDrag] = useState<Drag | null>(null);
  const [draft, setDraft] = useState<{
    rois?: Partial<Record<RoiName, [number, number]>>;
    patch?: Rect;
  }>({});
  const view = controlledView ?? localView;

  const setView = useCallback(
    (next: ViewState) => {
      const clamped = { ...next, zoom: Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, next.zoom)) };
      if (onViewChange) onViewChange(clamped);
      if (!controlledView) setLocalView(clamped);
    },
    [controlledView, onViewChange],
  );

  useLayoutEffect(() => {
    const element = stage.current;
    if (!element) return;
    const observer = new ResizeObserver(([entry]) =>
      setSize({ w: entry.contentRect.width, h: entry.contentRect.height }),
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const fit = width && height && size.w && size.h ? Math.min(size.w / width, size.h / height) : 1;
  const scale = fit * view.zoom;
  const tx = size.w / 2 - view.cx * width * scale;
  const ty = size.h / 2 - view.cy * height * scale;

  const toImage = useCallback(
    (clientX: number, clientY: number) => {
      const rect = stage.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return {
        x: (clientX - rect.left - tx) / (width * scale),
        y: (clientY - rect.top - ty) / (height * scale),
      };
    },
    [tx, ty, width, height, scale],
  );

  // Wheel zoom around the cursor (non-passive listener).
  useEffect(() => {
    const element = stage.current;
    if (!element) return;
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const point = toImage(event.clientX, event.clientY);
      const factor = Math.exp(-event.deltaY * 0.0015);
      const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, view.zoom * factor));
      const ratio = view.zoom / zoom;
      setView({
        zoom,
        cx: point.x + (view.cx - point.x) * ratio,
        cy: point.y + (view.cy - point.y) * ratio,
      });
    };
    element.addEventListener("wheel", onWheel, { passive: false });
    return () => element.removeEventListener("wheel", onWheel);
  }, [toImage, view, setView]);

  // Region rectangles in image coordinates; a dragged band is relative to the bottle box.
  const rois = { ...(geometry?.rois ?? {}) } as Partial<Record<RegionName, Rect>>;
  const boxTop = geometry?.bottle_box[1] ?? 0;
  const boxHeight = (geometry?.bottle_box[3] ?? 1) - boxTop || 1;
  for (const [name, band] of Object.entries(draft.rois ?? {}) as [RoiName, [number, number]][]) {
    const r = rois[name];
    if (r) rois[name] = [r[0], boxTop + band[0] * boxHeight, r[2], boxTop + band[1] * boxHeight];
  }
  const patch = draft.patch ?? geometry?.reference_patch;

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    (event.target as Element).setPointerCapture?.(event.pointerId);
    const target = (event.target as Element).closest("[data-handle]") as HTMLElement | null;
    const handle = target?.dataset.handle;
    if (handle?.startsWith("roi:") && geometry) {
      const [, name, edge] = handle.split(":");
      const r = rois[name as RoiName];
      if (!r) return;
      const relative = (y: number) => Math.round(((y - boxTop) / boxHeight) * 100) / 100;
      setDrag({
        kind: "roi",
        name: name as RoiName,
        edge: Number(edge) as 0 | 1,
        band: [relative(r[1]), relative(r[3])],
      });
      return;
    }
    if (handle?.startsWith("patch:") && patch) {
      const point = toImage(event.clientX, event.clientY);
      setDrag({
        kind: "patch",
        corner: Number(handle.split(":")[1]) as 0,
        rect: patch,
        startX: point.x,
        startY: point.y,
      });
      return;
    }
    if (handle === "compare") {
      setDrag({ kind: "compare" });
      return;
    }
    setDrag({ kind: "pan", startX: event.clientX, startY: event.clientY, view });
  };

  const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

  const onPointerMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!drag) return;
    if (drag.kind === "pan") {
      const dx = (event.clientX - drag.startX) / (width * scale);
      const dy = (event.clientY - drag.startY) / (height * scale);
      setView({ ...drag.view, cx: drag.view.cx - dx, cy: drag.view.cy - dy });
    } else if (drag.kind === "compare") {
      setSplit(clamp01(toImage(event.clientX, event.clientY).x));
    } else if (drag.kind === "roi") {
      const y = clamp01((toImage(event.clientX, event.clientY).y - boxTop) / boxHeight);
      const band: [number, number] = [...drag.band];
      band[drag.edge] = Math.round(y * 100) / 100;
      if (band[0] >= band[1]) return;
      setDraft((d) => ({ ...d, rois: { ...d.rois, [drag.name]: band } }));
      editable?.onRoiChange?.(drag.name, band, false);
    } else if (drag.kind === "patch") {
      const point = toImage(event.clientX, event.clientY);
      const r = [...drag.rect] as Rect;
      const round = (v: number) => Math.round(clamp01(v) * 100) / 100;
      if (drag.corner === -1) {
        const dx = point.x - drag.startX;
        const dy = point.y - drag.startY;
        const w = r[2] - r[0];
        const h = r[3] - r[1];
        r[0] = round(Math.min(1 - w, Math.max(0, r[0] + dx)));
        r[1] = round(Math.min(1 - h, Math.max(0, r[1] + dy)));
        r[2] = round(r[0] + w);
        r[3] = round(r[1] + h);
      } else {
        if (drag.corner === 0 || drag.corner === 3) r[0] = round(point.x);
        else r[2] = round(point.x);
        if (drag.corner === 0 || drag.corner === 1) r[1] = round(point.y);
        else r[3] = round(point.y);
      }
      if (r[0] >= r[2] || r[1] >= r[3]) return;
      setDraft((d) => ({ ...d, patch: r }));
      editable?.onPatchChange?.(r, false);
    }
  };

  const onPointerUp = () => {
    if (drag?.kind === "roi" && draft.rois?.[drag.name]) {
      editable?.onRoiChange?.(drag.name, draft.rois[drag.name]!, true);
    }
    if (drag?.kind === "patch" && draft.patch) editable?.onPatchChange?.(draft.patch, true);
    setDrag(null);
    setDraft({});
  };

  const W = width || 1;
  const H = height || 1;
  const px = (v: number) => v / scale; // screen px -> image px
  const border = verdict ? VERDICT_HEX[verdict] : "transparent";

  return (
    <div
      className={cn("relative overflow-hidden rounded-[10px] bg-stage", className)}
      style={{ boxShadow: verdict ? `inset 0 0 0 4px ${border}` : undefined }}
    >
      <div
        ref={stage}
        role="img"
        aria-label={label ?? t("viewer.label")}
        className={cn(
          "absolute inset-[4px] touch-none select-none",
          drag?.kind === "pan" ? "cursor-grabbing" : "cursor-grab",
        )}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={() => setView(view.zoom === 1 ? { ...view, zoom: 1 / fit } : FIT_VIEW)}
      >
        {src ? (
          <div
            className="absolute left-0 top-0 origin-top-left"
            style={{
              width: W,
              height: H,
              transform: `translate(${tx}px, ${ty}px) scale(${scale})`,
            }}
          >
            <img
              src={src}
              width={W}
              height={H}
              alt=""
              draggable={false}
              className="block max-w-none"
            />
            {compareSrc ? (
              <img
                src={compareSrc}
                width={W}
                height={H}
                alt=""
                draggable={false}
                className="absolute left-0 top-0 block max-w-none"
                style={{ clipPath: `inset(0 0 0 ${split * 100}%)` }}
              />
            ) : null}
            {overlays.mask && maskSrc ? (
              <div
                className="pointer-events-none absolute left-0 top-0"
                style={{
                  width: W,
                  height: H,
                  background: maskColor,
                  opacity: 0.45,
                  maskImage: `url(${maskSrc})`,
                  WebkitMaskImage: `url(${maskSrc})`,
                  maskSize: "100% 100%",
                  WebkitMaskSize: "100% 100%",
                  maskMode: "luminance",
                }}
              />
            ) : null}
            <svg
              className="absolute left-0 top-0 overflow-visible"
              width={W}
              height={H}
              viewBox={`0 0 ${W} ${H}`}
            >
              {geometry && overlays.patch && patch ? (
                <g>
                  <rect
                    data-handle={editable?.onPatchChange ? "patch:-1" : undefined}
                    x={patch[0] * W}
                    y={patch[1] * H}
                    width={(patch[2] - patch[0]) * W}
                    height={(patch[3] - patch[1]) * H}
                    fill={editable?.onPatchChange ? "rgba(255,255,255,0.08)" : "none"}
                    stroke="white"
                    strokeWidth={2}
                    strokeDasharray="6 4"
                    vectorEffect="non-scaling-stroke"
                    className={editable?.onPatchChange ? "cursor-move" : undefined}
                  />
                  <Tag
                    x={patch[0] * W}
                    y={patch[1] * H}
                    color="#ffffff"
                    text={t("viewer.patch")}
                    s={px}
                    dark
                  />
                  {editable?.onPatchChange
                    ? [
                        [patch[0], patch[1]],
                        [patch[2], patch[1]],
                        [patch[2], patch[3]],
                        [patch[0], patch[3]],
                      ].map(([x, y], i) => (
                        <rect
                          key={i}
                          data-handle={`patch:${i}`}
                          x={x * W - px(6)}
                          y={y * H - px(6)}
                          width={px(12)}
                          height={px(12)}
                          fill="white"
                          stroke="#1E2329"
                          strokeWidth={1}
                          vectorEffect="non-scaling-stroke"
                          className="cursor-nwse-resize"
                        />
                      ))
                    : null}
                </g>
              ) : null}
              {geometry && overlays.bottle ? (
                <g pointerEvents="none">
                  <rect
                    x={geometry.bottle_box[0] * W}
                    y={geometry.bottle_box[1] * H}
                    width={(geometry.bottle_box[2] - geometry.bottle_box[0]) * W}
                    height={(geometry.bottle_box[3] - geometry.bottle_box[1]) * H}
                    fill="none"
                    stroke="#FAB005"
                    strokeWidth={2}
                    strokeDasharray="8 5"
                    vectorEffect="non-scaling-stroke"
                  />
                </g>
              ) : null}
              {geometry && overlays.rois
                ? (Object.keys(ROI_COLORS) as RegionName[]).map((name) => {
                    const r = rois[name];
                    if (!r) return null;
                    const color = ROI_COLORS[name];
                    const draggable = EDITABLE_ROIS.includes(name) && editable?.onRoiChange;
                    const x = r[0] * W;
                    const w = (r[2] - r[0]) * W;
                    return (
                      <g key={name}>
                        <rect
                          x={x}
                          y={r[1] * H}
                          width={w}
                          height={(r[3] - r[1]) * H}
                          fill="none"
                          stroke={color}
                          strokeWidth={2}
                          vectorEffect="non-scaling-stroke"
                          pointerEvents="none"
                        />
                        <Tag
                          x={name === "body" ? x + w : x}
                          y={r[1] * H}
                          color={color}
                          text={name}
                          s={px}
                          alignRight={name === "body"}
                        />
                        {draggable
                          ? ([0, 1] as const).map((edge) => (
                              <line
                                key={edge}
                                data-handle={`roi:${name}:${edge}`}
                                x1={x}
                                x2={x + w}
                                y1={(edge === 0 ? r[1] : r[3]) * H}
                                y2={(edge === 0 ? r[1] : r[3]) * H}
                                stroke="transparent"
                                strokeWidth={14}
                                vectorEffect="non-scaling-stroke"
                                className="cursor-ns-resize"
                              >
                                <title>{t("viewer.dragEdge", { roi: name })}</title>
                              </line>
                            ))
                          : null}
                      </g>
                    );
                  })
                : null}
            </svg>
            {compareSrc ? (
              <div
                data-handle="compare"
                className="absolute top-0 cursor-ew-resize"
                style={{ left: `${split * 100}%`, width: px(16), height: H, marginLeft: -px(8) }}
              >
                <div className="mx-auto h-full bg-white" style={{ width: px(2) }} />
              </div>
            ) : null}
          </div>
        ) : (
          <div className="flex h-full items-center justify-center text-[13px] text-[#9aa3ae]">
            {t("viewer.noImage")}
          </div>
        )}
      </div>
      {compareSrc ? (
        <div className="pointer-events-none absolute left-3 top-3 flex w-[calc(100%-24px)] justify-between text-[11px] font-medium text-white">
          <span className="rounded bg-black/50 px-1.5 py-0.5">{t("viewer.raw")}</span>
          <span className="rounded bg-black/50 px-1.5 py-0.5">{t("viewer.normalized")}</span>
        </div>
      ) : null}
      {src ? (
        <div className="absolute bottom-2 right-2 flex gap-1 rounded-[6px] bg-black/50 p-0.5">
          <ViewerButton
            label={t("viewer.zoomOut")}
            onClick={() => setView({ ...view, zoom: view.zoom / 1.4 })}
          >
            <Minus />
          </ViewerButton>
          <ViewerButton
            label={t("viewer.zoomIn")}
            onClick={() => setView({ ...view, zoom: view.zoom * 1.4 })}
          >
            <Plus />
          </ViewerButton>
          <ViewerButton label={t("viewer.fit")} onClick={() => setView(FIT_VIEW)}>
            <Maximize />
          </ViewerButton>
          <ViewerButton
            label={t("viewer.actualSize")}
            onClick={() => setView({ ...view, zoom: 1 / fit })}
          >
            <Scan />
          </ViewerButton>
        </div>
      ) : null}
      {children}
    </div>
  );
}

function ViewerButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <Tip content={label}>
      <Button
        variant="ghost"
        size="icon"
        aria-label={label}
        onClick={onClick}
        className="size-7 text-white hover:bg-white/15 hover:text-white"
      >
        {children}
      </Button>
    </Tip>
  );
}

function Tag({
  x,
  y,
  color,
  text,
  s,
  dark,
  alignRight,
}: {
  x: number;
  y: number;
  color: string;
  text: string;
  s: (v: number) => number;
  dark?: boolean;
  alignRight?: boolean;
}) {
  const w = s(text.length * 6.6 + 10);
  const h = s(16);
  const left = alignRight ? x - w : x;
  return (
    <g pointerEvents="none">
      <rect x={left} y={y - h} width={w} height={h} fill={color} rx={s(2)} />
      <text
        x={left + s(5)}
        y={y - s(4.5)}
        fontSize={s(11)}
        fontFamily="IBM Plex Sans, sans-serif"
        fontWeight={600}
        fill={dark ? "#1E2329" : "#ffffff"}
      >
        {text}
      </text>
    </g>
  );
}
