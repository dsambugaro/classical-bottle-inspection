// System screen (§15.8): trained models, storage (REQ-D1), dataset, defaults, about.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { CamerasSection } from "@/components/camera/CamerasSection";
import { NumberInput } from "@/components/params/ParamControls";
import { ClassLabel } from "@/components/status";
import { Button, Dialog, Field, Select, Spinner, cn } from "@/components/ui/primitives";
import { api, ApiError } from "@/lib/api";
import { fmtBytes, fmtDate, fmtInt, fmtPercent } from "@/lib/format";
import { keys, useHealth, useModels, useSettings } from "@/lib/queries";
import { CLASS_NAMES, type Settings } from "@/lib/types";

function Section({
  title,
  description,
  children,
  actions,
}: {
  title: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <section className="panel p-4">
      <div className="mb-3 flex flex-wrap items-start gap-3">
        <div>
          <h2 className="text-[15px] font-semibold">{title}</h2>
          {description ? (
            <p className="mt-0.5 max-w-prose text-[13px] text-ink-2">{description}</p>
          ) : null}
        </div>
        <div className="ml-auto flex gap-2">{actions}</div>
      </div>
      {children}
    </section>
  );
}

function useSettingsMutation() {
  const { t } = useTranslation();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (values: Partial<Settings>) => api.patch<Settings>("/api/settings", values),
    onSuccess: (data) => {
      client.setQueryData(keys.settings, data);
      void client.invalidateQueries({ queryKey: keys.models });
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("system.saved"));
    },
    onError: (e) =>
      toast.error(
        e instanceof ApiError
          ? t(`errors.${e.code}`, { defaultValue: e.message })
          : t("errors.unexpected"),
      ),
  });
}

function ModelsSection({ settings }: { settings: Settings }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const models = useModels().data ?? [];
  const update = useSettingsMutation();
  const stored = settings.model_folders.join("\n");
  const [folders, setFolders] = useState(stored);
  const [shown, setShown] = useState(stored);
  if (shown !== stored) {
    setShown(stored);
    setFolders(stored);
  }
  const refresh = useMutation({
    mutationFn: () => api.post("/api/models/refresh"),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.models });
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("system.modelsRefreshed"));
    },
  });
  return (
    <Section
      title={t("system.models")}
      description={t("system.modelsBody")}
      actions={
        <Button disabled={refresh.isPending} onClick={() => refresh.mutate()}>
          <RefreshCw className={cn(refresh.isPending && "animate-spin")} />
          {t("profiles.refreshModels")}
        </Button>
      }
    >
      <div className="grid gap-4 lg:grid-cols-[18rem_1fr]">
        <Field
          label={t("system.modelFolders")}
          htmlFor="folders"
          hint={t("system.modelFoldersHint")}
        >
          <textarea
            id="folders"
            className="field-input h-24 py-1.5"
            value={folders}
            onChange={(e) => setFolders(e.target.value)}
          />
          <Button
            size="sm"
            className="mt-1 self-start"
            onClick={() =>
              update.mutate({
                model_folders: folders
                  .split("\n")
                  .map((f) => f.trim())
                  .filter(Boolean),
              })
            }
          >
            {t("common.save")}
          </Button>
        </Field>
        <div className="overflow-x-auto" tabIndex={0} role="region" aria-label={t("system.models")}>
          <table className="w-full text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-[12px] text-ink-2">
                <th className="py-1.5 font-medium">{t("system.modelRef")}</th>
                <th className="py-1.5 font-medium">{t("model.featureVector")}</th>
                <th className="py-1.5 font-medium">{t("model.algorithm")}</th>
                <th className="py-1.5 font-medium">{t("model.trainedAt")}</th>
                <th className="py-1.5 text-right font-medium">{t("model.accuracy")}</th>
                <th className="py-1.5 font-medium">{t("system.status")}</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.model_ref} className="border-b border-line last:border-0">
                  <td className="py-1.5 font-mono">{m.model_ref}</td>
                  <td className="py-1.5 font-mono">{m.feature_vector ?? "-"}</td>
                  <td className="py-1.5 font-mono">{m.algorithm}</td>
                  <td className="py-1.5">{fmtDate(m.trained_at)}</td>
                  <td className="py-1.5 text-right font-mono">{fmtPercent(m.accuracy, 1)}</td>
                  <td className="py-1.5">
                    <span
                      className={cn(
                        "text-[12px] font-medium",
                        m.status === "ok" ? "text-pass" : "text-reject",
                      )}
                    >
                      {t(`system.modelStatus.${m.status}`)}
                    </span>
                    {m.version_warning ? (
                      <span className="ml-2 text-[12px] text-review">{t("system.retrain")}</span>
                    ) : null}
                  </td>
                </tr>
              ))}
              {!models.length ? (
                <tr>
                  <td colSpan={5} className="py-4 text-center text-muted">
                    {t("empty.noModelsTitle")}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
          <p className="mt-2 text-[12px] text-muted">{t("system.noTraining")}</p>
        </div>
      </div>
    </Section>
  );
}

interface Storage {
  image_files: number;
  image_bytes: number;
  images_rows: number;
  inspections: number;
  analyses: number;
  db_bytes: number;
  data_dir: string;
}

function StorageSection({ settings }: { settings: Settings }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const storage = useQuery({
    queryKey: ["storage"],
    queryFn: () => api.get<Storage>("/api/storage"),
  });
  const update = useSettingsMutation();
  const [imagesOpen, setImagesOpen] = useState(false);
  const [analysesOpen, setAnalysesOpen] = useState(false);
  const [typed, setTyped] = useState("");
  const afterDelete = () => {
    void client.invalidateQueries();
  };
  const deleteImages = useMutation({
    mutationFn: () =>
      api.del<{ deleted_files: number; freed_bytes: number }>("/api/images", { confirm: true }),
    onSuccess: (r) => {
      toast.success(
        t("system.imagesDeleted", { count: r.deleted_files, size: fmtBytes(r.freed_bytes) }),
      );
      setImagesOpen(false);
      afterDelete();
    },
  });
  const deleteAnalyses = useMutation({
    mutationFn: () => api.del<{ deleted_analyses: number }>("/api/analyses", { confirm: "DELETE" }),
    onSuccess: (r) => {
      toast.success(t("system.analysesDeleted", { count: r.deleted_analyses }));
      setAnalysesOpen(false);
      setTyped("");
      afterDelete();
    },
  });
  const s = storage.data;
  return (
    <Section title={t("system.storage")} description={t("system.storageBody")}>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <Field label={t("system.storagePolicy")} hint={t("system.storagePolicyHint")}>
            <Select
              label={t("system.storagePolicy")}
              value={settings.storage_policy}
              onValueChange={(v) =>
                update.mutate({ storage_policy: v as Settings["storage_policy"] })
              }
              options={(["all", "rejects_only", "none"] as const).map((v) => ({
                value: v,
                label: t(`system.policy.${v}`),
              }))}
              className="w-64"
            />
          </Field>
          <Field label={t("system.retention")}>
            <span className="flex items-center gap-2">
              <NumberInput
                label={t("system.retention")}
                value={settings.image_retention_days}
                digits={0}
                min={1}
                onChange={(v) => update.mutate({ image_retention_days: Math.round(v) })}
              />
              <span className="text-[13px] text-ink-2">{t("system.days")}</span>
            </span>
          </Field>
        </div>
        <div>
          {s ? (
            <dl className="grid grid-cols-2 gap-2 text-[13px]">
              <dt className="text-ink-2">{t("system.imageFiles")}</dt>
              <dd className="text-right font-mono">
                {fmtInt(s.image_files)} · {fmtBytes(s.image_bytes)}
              </dd>
              <dt className="text-ink-2">{t("system.inspections")}</dt>
              <dd className="text-right font-mono">{fmtInt(s.inspections)}</dd>
              <dt className="text-ink-2">{t("system.analyses")}</dt>
              <dd className="text-right font-mono">{fmtInt(s.analyses)}</dd>
              <dt className="text-ink-2">{t("system.database")}</dt>
              <dd className="text-right font-mono">{fmtBytes(s.db_bytes)}</dd>
              <dt className="text-ink-2">{t("system.dataDir")}</dt>
              <dd className="truncate text-right font-mono text-[12px]" title={s.data_dir}>
                {s.data_dir}
              </dd>
            </dl>
          ) : (
            <Spinner />
          )}
          <div className="mt-4 flex flex-wrap gap-2">
            <Button variant="danger-outline" onClick={() => setImagesOpen(true)}>
              <Trash2 />
              {t("system.deleteImages")}
            </Button>
            <Button variant="danger-outline" onClick={() => setAnalysesOpen(true)}>
              <Trash2 />
              {t("system.deleteAnalyses")}
            </Button>
          </div>
        </div>
      </div>
      <Dialog
        open={imagesOpen}
        onOpenChange={setImagesOpen}
        title={t("system.deleteImagesTitle")}
        description={t("system.deleteImagesBody", {
          count: s?.image_files ?? 0,
          size: fmtBytes(s?.image_bytes ?? 0),
        })}
        footer={
          <>
            <Button onClick={() => setImagesOpen(false)}>{t("common.cancel")}</Button>
            <Button
              variant="danger"
              disabled={deleteImages.isPending}
              onClick={() => deleteImages.mutate()}
            >
              {t("system.deleteImages")}
            </Button>
          </>
        }
      />
      <Dialog
        open={analysesOpen}
        onOpenChange={setAnalysesOpen}
        title={t("system.deleteAnalysesTitle")}
        description={t("system.deleteAnalysesBody", { count: s?.analyses ?? 0 })}
        footer={
          <>
            <Button onClick={() => setAnalysesOpen(false)}>{t("common.cancel")}</Button>
            <Button
              variant="danger"
              disabled={typed !== "DELETE" || deleteAnalyses.isPending}
              onClick={() => deleteAnalyses.mutate()}
            >
              {t("system.deleteAnalyses")}
            </Button>
          </>
        }
      >
        <Field label={t("system.typeDelete")} htmlFor="type-delete">
          <input
            id="type-delete"
            className="field-input"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
            autoComplete="off"
          />
        </Field>
      </Dialog>
    </Section>
  );
}

function DatasetSection({ settings }: { settings: Settings }) {
  const { t } = useTranslation();
  const update = useSettingsMutation();
  const [path, setPath] = useState(settings.dataset_path);
  const dataset = useQuery({
    queryKey: ["dataset", settings.dataset_path],
    queryFn: () =>
      api.get<{
        found: boolean;
        resolved: string | null;
        counts: Record<string, Record<string, number>> | null;
      }>("/api/dataset"),
  });
  return (
    <Section title={t("system.dataset")} description={t("system.datasetBody")}>
      <div className="flex flex-wrap items-end gap-2">
        <Field label={t("system.datasetPath")} htmlFor="dataset-path">
          <input
            id="dataset-path"
            className="field-input w-80"
            value={path}
            onChange={(e) => setPath(e.target.value)}
          />
        </Field>
        <Button onClick={() => update.mutate({ dataset_path: path })}>{t("system.check")}</Button>
        <span
          className={cn(
            "text-[13px] font-medium",
            dataset.data?.found ? "text-pass" : "text-reject",
          )}
        >
          {dataset.data?.found
            ? t("system.datasetFound", { path: dataset.data.resolved })
            : t("system.datasetMissing")}
        </span>
      </div>
      {dataset.data?.counts ? (
        <table className="mt-3 text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[12px] text-ink-2">
              <th className="py-1 pr-6 font-medium">{t("model.class")}</th>
              <th className="py-1 pr-6 text-right font-medium">train</th>
              <th className="py-1 text-right font-medium">val</th>
            </tr>
          </thead>
          <tbody>
            {CLASS_NAMES.map((name) => (
              <tr key={name} className="border-b border-line last:border-0">
                <td className="py-1 pr-6">
                  <ClassLabel name={name} withDescription />
                </td>
                <td className="py-1 pr-6 text-right font-mono">
                  {dataset.data!.counts!.train?.[name] ?? 0}
                </td>
                <td className="py-1 text-right font-mono">
                  {dataset.data!.counts!.val?.[name] ?? 0}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </Section>
  );
}

function DefaultsSection({ settings }: { settings: Settings }) {
  const { t } = useTranslation();
  const update = useSettingsMutation();
  const items: { key: keyof Settings; digits: number }[] = [
    { key: "max_profiles", digits: 0 },
    { key: "default_fps", digits: 0 },
    { key: "min_fps", digits: 0 },
    { key: "max_fps", digits: 0 },
    { key: "max_upload_mb", digits: 0 },
    { key: "max_batch_files", digits: 0 },
    { key: "background_capture_frames", digits: 0 },
    { key: "motion_threshold", digits: 3 },
  ];
  return (
    <Section title={t("system.defaults")}>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(({ key, digits }) => (
          <Field key={key} label={t(`system.setting.${key}`)}>
            <NumberInput
              label={t(`system.setting.${key}`)}
              value={settings[key] as number}
              digits={digits}
              min={0}
              onChange={(v) => update.mutate({ [key]: digits === 0 ? Math.round(v) : v })}
            />
          </Field>
        ))}
      </div>
    </Section>
  );
}

function AboutSection() {
  const { t } = useTranslation();
  const health = useHealth().data;
  const settings = useSettings().data;
  const exportSettings = () => {
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "bottle-inspection-settings.json";
    link.click();
  };
  const update = useSettingsMutation();
  return (
    <Section title={t("system.about")}>
      <dl className="grid grid-cols-[10rem_1fr] gap-1 text-[13px]">
        {Object.entries(health?.versions ?? {}).map(([k, v]) => (
          <div key={k} className="contents">
            <dt className="text-ink-2">{k}</dt>
            <dd className="font-mono text-[12px]">{v}</dd>
          </div>
        ))}
      </dl>
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <a
          className="text-[13px] text-accent underline"
          href="/api/docs"
          target="_blank"
          rel="noreferrer"
        >
          {t("system.apiDocs")}
        </a>
        <Button size="sm" onClick={exportSettings}>
          {t("system.exportSettings")}
        </Button>
        <label className="inline-flex cursor-pointer items-center rounded-[6px] border border-line px-2.5 py-1 text-[13px] hover:bg-sunken">
          {t("system.importSettings")}
          <input
            type="file"
            accept="application/json"
            className="hidden"
            onChange={(e) =>
              void e.target.files?.[0]?.text().then((text) => {
                try {
                  update.mutate(JSON.parse(text));
                } catch {
                  toast.error(t("errors.request.invalid_json"));
                }
              })
            }
          />
        </label>
      </div>
    </Section>
  );
}

export default function SystemPage() {
  const settings = useSettings().data;
  if (!settings) return <Spinner />;
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-4 p-4">
      <CamerasSection />
      <ModelsSection settings={settings} />
      <StorageSection settings={settings} />
      <DatasetSection settings={settings} />
      <DefaultsSection settings={settings} />
      <AboutSection />
    </div>
  );
}
