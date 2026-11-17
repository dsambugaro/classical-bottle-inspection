// Live preview (§15.5): POST /api/profiles/{id}/preview, debounced 250 ms, reduced resolution
// while dragging and full resolution on release. ROI bands and the reference patch are
// draggable on the image; masks, geometry values and an optional dry-run classification.
import { Dices, Upload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";

import { ImageViewer, type Overlays } from "@/components/ImageViewer";
import { ClassLabel, VerdictChip } from "@/components/status";
import { Button, Checkbox, Select, Spinner } from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import { fmtMs, fmtNumber } from "@/lib/format";
import type { DatasetSample, Params, PreviewResult, Rect } from "@/lib/types";

type SampleSource =
  { kind: "dataset"; ref: string; url: string } | { kind: "file"; file: File; url: string };

export function PreviewPanel({
  profileId,
  params,
  modifiedCount,
  onParam,
}: {
  profileId: number;
  params: Params;
  modifiedCount: number;
  onParam: (key: string, value: number[], final: boolean) => void;
}) {
  const { t } = useTranslation();
  const [sample, setSample] = useState<SampleSource | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [result, setResult] = useState<PreviewResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mask, setMask] = useState<"none" | "foreground">("none");
  const [overlays, setOverlays] = useState<Overlays>({
    rois: true,
    bottle: true,
    patch: true,
    mask: false,
  });
  const [classification, setClassification] = useState<PreviewResult["classification"] | null>(
    null,
  );
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    void api
      .get<{ items: DatasetSample[] }>("/api/dataset/samples?split=val&n=24&seed=7")
      .then((data) => {
        setSamples(data.items);
        if (data.items[0])
          setSample({ kind: "dataset", ref: data.items[0].ref, url: data.items[0].url });
      })
      .catch(() => setSamples([]));
  }, []);

  const request = async (classify: boolean, scale: number) => {
    if (!sample) return null;
    abort.current?.abort();
    const controller = new AbortController();
    abort.current = controller;
    const form = new FormData();
    if (sample.kind === "dataset") form.set("dataset_ref", sample.ref);
    else form.set("file", sample.file);
    form.set("params", JSON.stringify(params));
    form.set("scale", String(scale));
    form.set("include", "masks");
    form.set("classify", String(classify));
    return api.post<PreviewResult>(`/api/profiles/${profileId}/preview`, form, controller.signal);
  };

  const paramsKey = JSON.stringify(params);
  useEffect(() => {
    if (!sample) return;
    const timer = window.setTimeout(() => {
      setLoading(true);
      request(false, dragging ? 0.5 : 1)
        .then((data) => {
          if (data) {
            setResult(data);
            setError(null);
            setClassification(null);
          }
        })
        .catch((e) => {
          if (e instanceof DOMException && e.name === "AbortError") return;
          setError(
            e instanceof ApiError
              ? t(`errors.${e.code}`, { defaultValue: e.message })
              : t("errors.unexpected"),
          );
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramsKey, sample, dragging, profileId]);

  const classify = async () => {
    setLoading(true);
    try {
      const data = await request(true, 1);
      if (data) {
        setResult(data);
        setClassification(data.classification ?? null);
      }
    } catch (e) {
      setError(
        e instanceof ApiError
          ? t(`errors.${e.code}`, { defaultValue: e.message })
          : t("errors.unexpected"),
      );
    } finally {
      setLoading(false);
    }
  };

  const maskSrc =
    result?.masks && mask !== "none" ? `data:image/png;base64,${result.masks[mask]}` : null;

  return (
    <aside
      className="panel sticky top-0 flex flex-col gap-3 p-3"
      aria-label={t("profiles.preview")}
    >
      <div className="flex items-center justify-between gap-2">
        <h3 className="font-semibold">{t("profiles.preview")}</h3>
        {loading ? <Spinner /> : null}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Select
          label={t("profiles.sampleImage")}
          value={sample?.kind === "dataset" ? sample.ref : "__file"}
          onValueChange={(ref) => {
            const s = samples.find((x) => x.ref === ref);
            if (s) setSample({ kind: "dataset", ref: s.ref, url: s.url });
          }}
          options={[
            ...(sample?.kind === "file" ? [{ value: "__file", label: sample.file.name }] : []),
            ...samples.map((s) => ({ value: s.ref, label: `${s.name}` })),
          ]}
          className="min-w-0 flex-1"
        />
        <Button
          size="icon"
          aria-label={t("profiles.randomSample")}
          onClick={() => {
            const s = samples[Math.floor(Math.random() * samples.length)];
            if (s) setSample({ kind: "dataset", ref: s.ref, url: s.url });
          }}
        >
          <Dices />
        </Button>
        <Button
          size="icon"
          aria-label={t("profiles.uploadSample")}
          onClick={() => fileInput.current?.click()}
        >
          <Upload />
        </Button>
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) setSample({ kind: "file", file, url: URL.createObjectURL(file) });
          }}
        />
      </div>
      <ImageViewer
        src={sample?.url ?? null}
        width={result?.image.width ?? 1280}
        height={result?.image.height ?? 720}
        geometry={result?.geometry}
        overlays={{ ...overlays, mask: mask !== "none" }}
        maskSrc={maskSrc}
        maskColor="#FAB005"
        className="aspect-[16/9] w-full"
        editable={{
          onRoiChange: (name, band, final) => {
            setDragging(!final);
            onParam(`rois.${name}`, band, final);
          },
          onPatchChange: (rect: Rect, final) => {
            setDragging(!final);
            onParam("normalization.reference_patch", rect, final);
          },
        }}
      />
      <p className="text-[12px] text-muted">{t("profiles.dragHint")}</p>
      <div className="flex flex-wrap items-center gap-3 text-[13px]">
        {(["rois", "bottle", "patch"] as const).map((key) => (
          <label key={key} className="inline-flex items-center gap-1.5">
            <Checkbox
              checked={overlays[key]}
              onCheckedChange={(v) => setOverlays({ ...overlays, [key]: v === true })}
            />
            {t(`inspect.overlay.${key}`)}
          </label>
        ))}
        <Select
          label={t("profiles.maskOverlay")}
          value={mask}
          onValueChange={(v) => setMask(v as typeof mask)}
          options={[
            { value: "none", label: t("profiles.mask.none") },
            { value: "foreground", label: t("profiles.mask.foreground") },
          ]}
          className="w-44"
        />
      </div>
      {error ? (
        <p
          role="alert"
          className="rounded-[6px] bg-reject-tint px-2 py-1.5 text-[13px] text-reject"
        >
          {error}
        </p>
      ) : null}
      {result ? (
        <dl className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1 text-[13px]">
          <dt className="font-mono text-ink-2">bottle_top_y</dt>
          <dd className="text-right font-mono">{fmtNumber(result.geometry.bottle_top_y, 3)}</dd>
          <dt className="font-mono text-ink-2">foreground_fraction</dt>
          <dd className="text-right font-mono">
            {fmtNumber(result.geometry.foreground_fraction, 3)}
          </dd>
        </dl>
      ) : null}
      {modifiedCount ? (
        <p className="rounded-[6px] bg-review-tint px-2 py-1.5 text-[13px] text-review">
          {t("profiles.differFromTraining", { count: modifiedCount })}
        </p>
      ) : null}
      <div className="flex flex-col gap-2 border-t border-line pt-3">
        <Button onClick={() => void classify()} disabled={!sample || loading}>
          {t("profiles.classifyDryRun")}
        </Button>
        {classification ? (
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <VerdictChip verdict={classification.verdict} />
            <ClassLabel name={classification.predicted_class} />
            <span className="font-mono">{fmtNumber(classification.confidence, 2)}</span>
            <span className="font-mono text-muted">{fmtMs(classification.total_ms)}</span>
            {result?.ground_truth ? (
              <span className="text-[12px] text-muted">
                {t("profiles.groundTruth")} <span className="font-mono">{result.ground_truth}</span>
              </span>
            ) : null}
          </div>
        ) : null}
        <p className="text-[12px] text-muted">{t("profiles.dryRunNote")}</p>
      </div>
    </aside>
  );
}
