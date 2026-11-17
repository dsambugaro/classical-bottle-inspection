// System -> Cameras (REQ-I4): CRUD and a test button that grabs a snapshot or shows the error.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, Plus, Trash2, Video } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { NumberInput } from "../params/ParamControls";
import { Button, Dialog, Field, Select, Spinner } from "../ui/primitives";
import type { ServerCamera } from "./CameraControls";
import { ApiError, api } from "@/lib/api";

type Draft = Omit<ServerCamera, "id" | "active" | "session_id">;

const EMPTY: Draft = {
  name: "",
  kind: "usb",
  device_index: 0,
  url: "",
  width: null,
  height: null,
  fps: 3,
  enabled: true,
};

interface TestResult {
  ok: boolean;
  width?: number;
  height?: number;
  snapshot?: string;
  code?: string;
  message?: string;
}

export function CamerasSection() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const cameras = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<ServerCamera[]>("/api/cameras"),
  });
  const [editing, setEditing] = useState<{ id: number | null; draft: Draft } | null>(null);
  const [tests, setTests] = useState<Record<number, TestResult | "running">>({});

  const save = useMutation({
    mutationFn: ({ id, draft }: { id: number | null; draft: Draft }) => {
      const body = {
        ...draft,
        device_index: draft.kind === "usb" ? draft.device_index : null,
        url: draft.kind === "usb" ? null : draft.url,
      };
      return id === null ? api.post("/api/cameras", body) : api.patch(`/api/cameras/${id}`, body);
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["cameras"] });
      setEditing(null);
      toast.success(t("camera.saved"));
    },
    onError: (e) =>
      toast.error(
        e instanceof ApiError
          ? t(`errors.${e.code}`, { defaultValue: e.message })
          : t("errors.unexpected"),
      ),
  });

  const remove = useMutation({
    mutationFn: (id: number) => api.del(`/api/cameras/${id}`),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["cameras"] }),
  });

  const test = async (id: number) => {
    setTests((all) => ({ ...all, [id]: "running" }));
    try {
      const result = await api.post<TestResult>(`/api/cameras/${id}/test`);
      setTests((all) => ({ ...all, [id]: result }));
    } catch (e) {
      setTests((all) => ({
        ...all,
        [id]: { ok: false, code: e instanceof ApiError ? e.code : "camera.open_failed" },
      }));
    }
  };

  const draft = editing?.draft;
  const setDraft = (patch: Partial<Draft>) =>
    editing && setEditing({ ...editing, draft: { ...editing.draft, ...patch } });

  return (
    <section className="panel p-4">
      <div className="mb-3 flex flex-wrap items-start gap-3">
        <div>
          <h2 className="text-[15px] font-semibold">{t("camera.title")}</h2>
          <p className="mt-0.5 max-w-prose text-[13px] text-ink-2">{t("camera.body")}</p>
        </div>
        <Button
          className="ml-auto"
          variant="primary"
          onClick={() => setEditing({ id: null, draft: { ...EMPTY } })}
        >
          <Plus />
          {t("camera.add")}
        </Button>
      </div>
      {cameras.isLoading ? <Spinner /> : null}
      {cameras.data && !cameras.data.length ? (
        <p className="text-[13px] text-muted">{t("camera.none")}</p>
      ) : null}
      <ul className="flex flex-col gap-2">
        {cameras.data?.map((camera) => {
          const result = tests[camera.id];
          return (
            <li key={camera.id} className="rounded-[6px] border border-line p-3">
              <div className="flex flex-wrap items-center gap-3">
                <Video className="size-4 text-ink-2" aria-hidden="true" />
                <span className="font-medium">{camera.name}</span>
                <span className="font-mono text-[12px] text-muted">
                  {camera.kind === "usb" ? `usb #${camera.device_index}` : camera.url}
                  {camera.width && camera.height
                    ? ` · ${camera.width}×${camera.height}`
                    : ""} · {camera.fps} fps
                </span>
                {camera.session_id ? (
                  <span className="rounded-[4px] bg-info-tint px-1.5 text-[12px] font-medium text-info">
                    {t("camera.inSession")}
                  </span>
                ) : null}
                <div className="ml-auto flex gap-1.5">
                  <Button
                    size="sm"
                    disabled={result === "running"}
                    onClick={() => void test(camera.id)}
                  >
                    {t("camera.test")}
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={t("camera.edit", { name: camera.name })}
                    onClick={() => setEditing({ id: camera.id, draft: { ...camera } })}
                  >
                    <Pencil />
                  </Button>
                  <Button
                    size="icon"
                    variant="ghost"
                    aria-label={t("camera.delete", { name: camera.name })}
                    onClick={() =>
                      window.confirm(t("camera.deleteConfirm", { name: camera.name })) &&
                      remove.mutate(camera.id)
                    }
                  >
                    <Trash2 />
                  </Button>
                </div>
              </div>
              {result === "running" ? <Spinner className="mt-2" /> : null}
              {result && result !== "running" ? (
                result.ok ? (
                  <div className="mt-2 flex items-start gap-3">
                    <img
                      src={`data:image/jpeg;base64,${result.snapshot}`}
                      alt={t("camera.snapshot", { name: camera.name })}
                      className="h-28 rounded-[4px]"
                    />
                    <p className="text-[13px] text-pass">
                      {t("camera.testOk", { width: result.width, height: result.height })}
                    </p>
                  </div>
                ) : (
                  <p
                    role="alert"
                    className="mt-2 rounded-[6px] bg-reject-tint px-2 py-1.5 text-[13px] text-reject"
                  >
                    {t(`errors.${result.code}`, { defaultValue: result.message })}
                    {camera.kind === "usb" ? (
                      <span className="block text-[12px] text-ink-2">{t("camera.wslHint")}</span>
                    ) : null}
                  </p>
                )
              ) : null}
            </li>
          );
        })}
      </ul>
      <Dialog
        open={editing !== null}
        onOpenChange={(open) => !open && setEditing(null)}
        title={editing?.id ? t("camera.editTitle") : t("camera.addTitle")}
        footer={
          <>
            <Button onClick={() => setEditing(null)}>{t("common.cancel")}</Button>
            <Button
              variant="primary"
              disabled={!draft?.name.trim() || save.isPending}
              onClick={() => editing && save.mutate(editing)}
            >
              {t("common.save")}
            </Button>
          </>
        }
      >
        {draft ? (
          <div className="grid gap-3">
            <Field label={t("camera.name")} htmlFor="cam-name">
              <input
                id="cam-name"
                className="field-input font-sans"
                value={draft.name}
                maxLength={60}
                onChange={(e) => setDraft({ name: e.target.value })}
              />
            </Field>
            <Field label={t("camera.kind")}>
              <Select
                label={t("camera.kind")}
                value={draft.kind}
                onValueChange={(kind) => setDraft({ kind })}
                options={[
                  { value: "usb", label: t("camera.kinds.usb") },
                  { value: "rtsp", label: "RTSP" },
                  { value: "http", label: "HTTP" },
                ]}
                className="w-48"
              />
            </Field>
            {draft.kind === "usb" ? (
              <Field label={t("camera.deviceIndex")}>
                <NumberInput
                  label={t("camera.deviceIndex")}
                  value={draft.device_index ?? 0}
                  digits={0}
                  min={0}
                  max={63}
                  onChange={(v) => setDraft({ device_index: Math.round(v) })}
                />
              </Field>
            ) : (
              <Field label="URL" htmlFor="cam-url" hint={t("camera.urlHint", { kind: draft.kind })}>
                <input
                  id="cam-url"
                  className="field-input"
                  value={draft.url ?? ""}
                  onChange={(e) => setDraft({ url: e.target.value })}
                />
              </Field>
            )}
            <div className="flex flex-wrap gap-3">
              <Field label={t("camera.width")}>
                <NumberInput
                  label={t("camera.width")}
                  value={draft.width ?? 1280}
                  digits={0}
                  min={16}
                  max={8192}
                  onChange={(v) => setDraft({ width: Math.round(v) })}
                />
              </Field>
              <Field label={t("camera.height")}>
                <NumberInput
                  label={t("camera.height")}
                  value={draft.height ?? 720}
                  digits={0}
                  min={16}
                  max={8192}
                  onChange={(v) => setDraft({ height: Math.round(v) })}
                />
              </Field>
              <Field label={t("camera.fpsCap")}>
                <NumberInput
                  label={t("camera.fpsCap")}
                  value={draft.fps}
                  digits={0}
                  min={1}
                  max={60}
                  onChange={(v) => setDraft({ fps: v })}
                />
              </Field>
            </div>
          </div>
        ) : null}
      </Dialog>
    </section>
  );
}
