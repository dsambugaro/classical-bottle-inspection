// Profiles screen (§15.5): list, Overview / Parameters / Model / History tabs, catalog-driven
// parameter editor with live preview, reset per field/group/profile, compare, import/export.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ChevronRight,
  Copy,
  Download,
  FileUp,
  Link2,
  MoreHorizontal,
  Plus,
  RefreshCw,
  RotateCcw,
  Search,
  Trash2,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { useBlocker, useNavigate, useParams } from "react-router";
import { toast } from "sonner";

import { ParamRow, ParamControl } from "@/components/params/ParamControls";
import { ProfileStatusBadge } from "@/components/status";
import {
  Button,
  Checkbox,
  Dialog,
  Dropdown,
  DropdownContent,
  DropdownItem,
  DropdownTrigger,
  Field,
  Select,
  Spinner,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
} from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import { fmtDate, fmtPercent, modelLabel } from "@/lib/format";
import { keys, useCatalog, useModels, useProfile, useProfiles } from "@/lib/queries";
import {
  CLASS_NAMES,
  type CatalogEntry,
  type ParamValue,
  type Params,
  type Profile,
  type VerdictSettings,
} from "@/lib/types";
import { validateAll } from "@/lib/validate";

import { ChangeList, HistoryTab } from "./profiles/HistoryTab";
import { ModelTab } from "./profiles/ModelTab";
import { PreviewPanel } from "./profiles/PreviewPanel";

const PROFILE_COLORS = [
  "#2457C5",
  "#0B7A75",
  "#7A4CC2",
  "#B5470F",
  "#1E7B34",
  "#A61E4D",
  "#5C6B7A",
  "#8F6A00",
];

interface Draft {
  name: string;
  description: string;
  color: string;
  model_ref: string;
  params: Params;
  verdict: VerdictSettings;
}

const toDraft = (p: Profile): Draft => ({
  name: p.name,
  description: p.description,
  color: p.color,
  model_ref: p.model_ref,
  params: { ...p.params },
  verdict: { ...p.verdict },
});

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

function errorMessage(t: TFunction, error: unknown): string {
  if (error instanceof ApiError) {
    const errors = (error.details.errors as { key: string; code: string }[] | undefined) ?? [];
    if (errors.length)
      return errors
        .map((e) => `${e.key}: ${t(`errors.${e.code}`, { defaultValue: e.code })}`)
        .join("; ");
    return String(t(`errors.${error.code}`, { ...error.details, defaultValue: error.message }));
  }
  return String(t("errors.unexpected"));
}

// ------------------------------------------------------------------ list pane

function ProfileList({
  profiles,
  selectedId,
  onSelect,
}: {
  profiles: Profile[];
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const filtered = profiles.filter((p) =>
    `${p.name} ${p.model_ref} ${p.algorithm}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <div className="flex min-h-0 flex-col gap-2">
      <label className="relative">
        <Search className="absolute left-2 top-2 size-4 text-muted" aria-hidden="true" />
        <input
          className="field-input w-full pl-8 font-sans"
          placeholder={t("profiles.search")}
          aria-label={t("profiles.search")}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>
      <ul className="min-h-0 flex-1 overflow-auto" aria-label={t("nav.profiles")}>
        {filtered.map((p) => (
          <li key={p.id}>
            <button
              type="button"
              onClick={() => onSelect(p.id)}
              aria-current={p.id === selectedId}
              className={cn(
                "flex w-full items-center gap-2 rounded-[6px] px-2 py-2 text-left hover:bg-sunken",
                p.id === selectedId && "bg-info-tint hover:bg-info-tint",
              )}
            >
              <span
                aria-hidden="true"
                className="h-8 w-1.5 shrink-0 rounded-full"
                style={{ background: p.color }}
              />
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[13px] font-medium">{p.name}</span>
                <span
                  className={cn(
                    "block truncate font-mono text-[11px]",
                    p.id === selectedId ? "text-ink-2" : "text-muted",
                  )}
                >
                  {modelLabel(p.model?.feature_vector, p.algorithm)} ·{" "}
                  {p.model?.folder ?? p.model_ref}
                </span>
              </span>
              <ProfileStatusBadge status={p.status} short />
            </button>
          </li>
        ))}
      </ul>
      <ul className="grid grid-cols-2 gap-1 border-t border-line pt-2 text-[11px] text-ink-2">
        {(["ready", "modified", "incompatible", "model_missing"] as const).map((s) => (
          <li key={s} className="flex items-center gap-1">
            <ProfileStatusBadge status={s} short /> {t(`profileStatus.${s}`)}
          </li>
        ))}
      </ul>
    </div>
  );
}

// ------------------------------------------------------------------ dialogs

function NewProfileDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onCreated: (id: number) => void;
}) {
  const { t } = useTranslation();
  const models = (useModels().data ?? []).filter((m) => m.status === "ok");
  const [name, setName] = useState("");
  const [modelRef, setModelRef] = useState("");
  const client = useQueryClient();
  const create = useMutation({
    mutationFn: () => api.post<Profile>("/api/profiles", { name, model_ref: modelRef }),
    onSuccess: (profile) => {
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("profiles.created", { name: profile.name }));
      onOpenChange(false);
      onCreated(profile.id);
    },
    onError: (e) => toast.error(errorMessage(t, e)),
  });
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t("profiles.newTitle")}
      description={t("profiles.newBody")}
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>{t("common.cancel")}</Button>
          <Button
            variant="primary"
            disabled={!name.trim() || !modelRef || create.isPending}
            onClick={() => create.mutate()}
          >
            {t("profiles.create")}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <Field label={t("profiles.name")} htmlFor="new-name">
          <input
            id="new-name"
            className="field-input font-sans"
            maxLength={60}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <Field label={t("profiles.trainedModel")}>
          <ModelPicker value={modelRef} onChange={setModelRef} />
        </Field>
        {!models.length ? (
          <p className="text-[13px] text-reject">{t("empty.noModelsTitle")}</p>
        ) : null}
      </div>
    </Dialog>
  );
}

function ModelPicker({ value, onChange }: { value: string; onChange: (ref: string) => void }) {
  const { t } = useTranslation();
  const models = useModels().data ?? [];
  return (
    <Select
      label={t("profiles.trainedModel")}
      value={value}
      onValueChange={onChange}
      options={models.map((m) => ({
        value: m.model_ref,
        disabled: m.status !== "ok",
        label: (
          <span className="flex items-center gap-2">
            <span className="font-mono">{modelLabel(m.feature_vector, m.algorithm)}</span>
            <span className="text-ink-2">{m.folder}</span>
            <span className="text-muted">{fmtDate(m.trained_at)}</span>
            <span className="font-mono text-muted">{fmtPercent(m.accuracy, 1)}</span>
          </span>
        ),
      }))}
      className="w-full"
    />
  );
}

interface ImportPreview {
  dry_run: boolean;
  preview: {
    name: string;
    model_ref: string;
    algorithm: string;
    diff_from_training: { key: string; training: unknown; value: unknown }[];
  };
  profile?: Profile;
}

function ImportDialog({
  open,
  onOpenChange,
  onImported,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  onImported: (id: number) => void;
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const [text, setText] = useState("");
  const [preview, setPreview] = useState<ImportPreview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const send = async (dryRun: boolean) => {
    setError(null);
    try {
      const body = JSON.parse(text);
      const result = await api.post<ImportPreview>(`/api/profiles/import?dry_run=${dryRun}`, body);
      if (dryRun) setPreview(result);
      else if (result.profile) {
        void client.invalidateQueries({ queryKey: keys.profiles });
        toast.success(t("profiles.imported", { name: result.profile.name }));
        onOpenChange(false);
        setPreview(null);
        setText("");
        onImported(result.profile.id);
      }
    } catch (e) {
      setPreview(null);
      setError(e instanceof SyntaxError ? t("errors.request.invalid_json") : errorMessage(t, e));
    }
  };
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      wide
      title={t("profiles.importTitle")}
      description={t("profiles.importBody")}
      footer={
        <>
          <Button onClick={() => onOpenChange(false)}>{t("common.cancel")}</Button>
          <Button disabled={!text} onClick={() => void send(true)}>
            {t("profiles.importPreview")}
          </Button>
          <Button variant="primary" disabled={!preview} onClick={() => void send(false)}>
            {t("profiles.importApply")}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-3">
        <input
          type="file"
          accept="application/json,.json"
          aria-label={t("profiles.importFile")}
          className="text-[13px]"
          onChange={(e) =>
            void e.target.files?.[0]?.text().then((value) => {
              setText(value);
              setPreview(null);
            })
          }
        />
        <textarea
          className="field-input h-40 w-full py-2"
          aria-label={t("profiles.importJson")}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setPreview(null);
          }}
        />
        {error ? (
          <p role="alert" className="text-[13px] text-reject">
            {error}
          </p>
        ) : null}
        {preview ? (
          <div className="rounded-[6px] bg-sunken p-3 text-[13px]">
            <p>
              <span className="font-medium">{preview.preview.name}</span> ·{" "}
              <span className="font-mono">{preview.preview.model_ref}</span>
            </p>
            <p className="mt-1 text-ink-2">
              {t("profiles.importDiff", { count: preview.preview.diff_from_training.length })}
            </p>
            <ChangeList
              changes={preview.preview.diff_from_training.map((d) => ({
                key: d.key,
                from: d.training,
                to: d.value,
              }))}
            />
          </div>
        ) : null}
      </div>
    </Dialog>
  );
}

function CompareDialog({
  open,
  onOpenChange,
  profile,
  profiles,
}: {
  open: boolean;
  onOpenChange: (v: boolean) => void;
  profile: Profile;
  profiles: Profile[];
}) {
  const { t } = useTranslation();
  const [target, setTarget] = useState("");
  const diff = useQuery({
    queryKey: ["profile-diff", profile.id, target, profile.version],
    enabled: open && !!target,
    queryFn: () => {
      const [kind, id] = target.split(":");
      const query = kind === "profile" ? `other_profile=${id}` : `from=${id}`;
      return api.get<{ changes: { key: string; from: unknown; to: unknown }[] }>(
        `/api/profiles/${profile.id}/diff?${query}`,
      );
    },
  });
  const options = [
    ...Array.from({ length: profile.version - 1 }, (_, i) => ({
      value: `version:${i + 1}`,
      label: t("profiles.compareVersion", { version: i + 1 }),
    })),
    ...profiles
      .filter((p) => p.id !== profile.id)
      .map((p) => ({ value: `profile:${p.id}`, label: p.name })),
  ];
  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      wide
      title={t("profiles.compareTitle", { name: profile.name })}
    >
      <div className="flex flex-col gap-3">
        <Select
          label={t("profiles.compareWith")}
          value={target}
          onValueChange={setTarget}
          options={options}
          className="w-72"
        />
        {diff.isLoading ? <Spinner /> : null}
        {diff.data ? <ChangeList changes={diff.data.changes} /> : null}
        <p className="text-[12px] text-muted">{t("profiles.compareHint")}</p>
      </div>
    </Dialog>
  );
}

// ------------------------------------------------------------------ parameters tab

function ParamsTab({
  entries,
  groups,
  draft,
  training,
  issues,
  onChange,
}: {
  entries: CatalogEntry[];
  groups: string[];
  draft: Draft;
  training: Params;
  issues: Record<string, string>;
  onChange: (key: string, value: ParamValue, final?: boolean) => void;
}) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [onlyChanged, setOnlyChanged] = useState(false);
  const [closed, setClosed] = useState<Set<string>>(new Set(["hog", "features"]));
  const matches = (e: CatalogEntry) => {
    const q = query.trim().toLowerCase();
    const text = `${e.key} ${t(e.label_key)}`.toLowerCase();
    const changed = !same(draft.params[e.key], training[e.key]);
    return (!q || text.includes(q)) && (!onlyChanged || changed);
  };
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-3">
        <label className="relative">
          <Search className="absolute left-2 top-2 size-4 text-muted" aria-hidden="true" />
          <input
            className="field-input w-64 pl-8 font-sans"
            placeholder={t("profiles.searchParams")}
            aria-label={t("profiles.searchParams")}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <label className="inline-flex items-center gap-1.5 text-[13px]">
          <Checkbox checked={onlyChanged} onCheckedChange={(v) => setOnlyChanged(v === true)} />
          {t("profiles.onlyChanged")}
        </label>
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          onClick={() =>
            entries
              .filter((e) => e.mode === "editable" && e.group !== "verdict")
              .forEach((e) => onChange(e.key, training[e.key], true))
          }
        >
          <RotateCcw />
          {t("profiles.resetAll")}
        </Button>
      </div>
      {groups
        .filter((g) => g !== "verdict")
        .map((group) => {
          const groupEntries = entries.filter((e) => e.group === group && matches(e));
          if (!groupEntries.length) return null;
          const open = !closed.has(group) || !!query || onlyChanged;
          const allLocked = groupEntries.every((e) => e.mode === "locked");
          const changed = groupEntries.filter(
            (e) => !same(draft.params[e.key], training[e.key]),
          ).length;
          return (
            <section key={group} className="panel">
              <div className="flex items-center gap-2 px-3 py-2">
                <button
                  type="button"
                  aria-expanded={open}
                  onClick={() => {
                    const next = new Set(closed);
                    if (next.has(group)) next.delete(group);
                    else next.add(group);
                    setClosed(next);
                  }}
                  className="flex flex-1 items-center gap-2 text-left font-semibold"
                >
                  <ChevronRight
                    className={cn("size-4 transition-transform", open && "rotate-90")}
                    aria-hidden="true"
                  />
                  {t(`groups.${group}`)}
                  {allLocked ? (
                    <span className="text-[12px] font-normal text-muted">
                      {t("params.allLocked")}
                    </span>
                  ) : null}
                  {changed ? (
                    <span className="rounded-[4px] bg-review-tint px-1.5 text-[12px] font-medium text-review">
                      {t("profiles.changedCount", { count: changed })}
                    </span>
                  ) : null}
                </button>
                {changed ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() =>
                      groupEntries
                        .filter((e) => e.mode === "editable")
                        .forEach((e) => onChange(e.key, training[e.key], true))
                    }
                  >
                    <RotateCcw />
                    {t("profiles.resetGroup")}
                  </Button>
                ) : null}
              </div>
              {open ? (
                <div className="border-t border-line">
                  {groupEntries.map((entry) => (
                    <ParamRow
                      key={entry.key}
                      entry={entry}
                      value={draft.params[entry.key]}
                      training={training[entry.key]}
                      error={issues[entry.key]}
                      onChange={(v, final) => onChange(entry.key, v, final)}
                      onReset={() => onChange(entry.key, training[entry.key], true)}
                    />
                  ))}
                </div>
              ) : null}
            </section>
          );
        })}
    </div>
  );
}

// ------------------------------------------------------------------ overview tab

function OverviewTab({
  profile,
  draft,
  setDraft,
  verdictEntries,
}: {
  profile: Profile;
  draft: Draft;
  setDraft: (d: Draft) => void;
  verdictEntries: CatalogEntry[];
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const uploadBackground = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.set("file", file);
      return api.post<Profile>(`/api/profiles/${profile.id}/background/upload`, form);
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("profiles.backgroundSaved"));
    },
    onError: (e) => toast.error(errorMessage(t, e)),
  });
  const cameras = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<{ id: number; name: string }[]>("/api/cameras"),
  });
  const [cameraId, setCameraId] = useState("");
  const captureBackground = useMutation({
    mutationFn: () =>
      api.post<Profile>(`/api/profiles/${profile.id}/background/capture`, {
        camera_id: Number(cameraId),
      }),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("profiles.backgroundSaved"));
    },
    onError: (e) => toast.error(errorMessage(t, e)),
  });
  const setSource = useMutation({
    mutationFn: (source: string) =>
      api.patch<Profile>(`/api/profiles/${profile.id}`, { background_source: source }),
    onSuccess: () => void client.invalidateQueries({ queryKey: keys.profiles }),
    onError: (e) => toast.error(errorMessage(t, e)),
  });
  const verdictEntry = (name: string) => verdictEntries.find((e) => e.key === `verdict.${name}`)!;
  return (
    <div className="grid max-w-3xl gap-4">
      <Field label={t("profiles.name")} htmlFor="p-name">
        <input
          id="p-name"
          className="field-input w-full max-w-md font-sans"
          maxLength={60}
          value={draft.name}
          onChange={(e) => setDraft({ ...draft, name: e.target.value })}
        />
      </Field>
      <Field label={t("profiles.description")} htmlFor="p-desc">
        <textarea
          id="p-desc"
          className="field-input h-20 w-full py-1.5 font-sans"
          value={draft.description}
          onChange={(e) => setDraft({ ...draft, description: e.target.value })}
        />
      </Field>
      <fieldset>
        <legend className="mb-1 text-[13px] font-medium text-ink-2">{t("profiles.color")}</legend>
        <div className="flex gap-2">
          {PROFILE_COLORS.map((color) => (
            <button
              key={color}
              type="button"
              aria-label={color}
              aria-pressed={draft.color === color}
              onClick={() => setDraft({ ...draft, color })}
              className={cn(
                "size-7 rounded-full border-2",
                draft.color === color ? "border-ink" : "border-transparent",
              )}
              style={{ background: color }}
            />
          ))}
        </div>
      </fieldset>
      <Field label={t("profiles.trainedModel")} hint={t("profiles.modelHint")}>
        <ModelPicker
          value={draft.model_ref}
          onChange={(ref) => setDraft({ ...draft, model_ref: ref })}
        />
      </Field>
      {profile.status_detail.model_updated_on ? (
        <p className="text-[13px] text-info">
          {t("profiles.modelUpdated", { date: fmtDate(profile.status_detail.model_updated_on) })}
        </p>
      ) : null}
      <fieldset className="flex flex-col gap-2">
        <legend className="mb-1 text-[13px] font-medium text-ink-2">
          {t("profiles.background")}
        </legend>
        <div className="flex flex-wrap items-center gap-2">
          <Select
            label={t("profiles.background")}
            value={profile.background_source}
            onValueChange={(v) => setSource.mutate(v)}
            options={(["model", "captured", "uploaded"] as const).map((s) => ({
              value: s,
              label: t(`profiles.bg.${s}`),
              disabled: s !== "model" && !profile.has_background,
            }))}
            className="w-56"
          />
          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-[6px] border border-line px-3 py-1.5 text-[13px] hover:bg-sunken">
            <FileUp className="size-4" aria-hidden="true" />
            {t("profiles.uploadBackground")}
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && uploadBackground.mutate(e.target.files[0])}
            />
          </label>
          <a
            className="text-[13px] text-accent underline"
            href={`/api/profiles/${profile.id}/background?v=${profile.version}`}
            target="_blank"
            rel="noreferrer"
          >
            {t("profiles.viewBackground")}
          </a>
        </div>
        {cameras.data?.length ? (
          <div className="flex flex-wrap items-center gap-2">
            <Select
              label={t("profiles.captureCamera")}
              value={cameraId}
              onValueChange={setCameraId}
              options={cameras.data.map((c) => ({ value: String(c.id), label: c.name }))}
              className="w-56"
            />
            <Button
              disabled={!cameraId || captureBackground.isPending}
              onClick={() => captureBackground.mutate()}
            >
              {captureBackground.isPending ? <Spinner /> : null}
              {t("profiles.captureBackground")}
            </Button>
            <span className="text-[12px] text-muted">{t("profiles.captureHint")}</span>
          </div>
        ) : null}
        <p className="text-[12px] text-muted">{t("profiles.backgroundHint")}</p>
      </fieldset>
      <fieldset className="panel flex flex-col gap-3 p-3">
        <legend className="px-1 font-semibold">{t("groups.verdict")}</legend>
        <div>
          <p className="mb-1 text-[13px] font-medium text-ink-2">
            {t("params.verdict.reject_classes.label")}
          </p>
          <div className="grid gap-1 sm:grid-cols-2">
            {CLASS_NAMES.map((name) => (
              <label key={name} className="inline-flex items-center gap-2 text-[13px]">
                <Checkbox
                  checked={draft.verdict.reject_classes.includes(name)}
                  onCheckedChange={(v) => {
                    const next = new Set(draft.verdict.reject_classes);
                    if (v) next.add(name);
                    else next.delete(name);
                    setDraft({
                      ...draft,
                      verdict: {
                        ...draft.verdict,
                        reject_classes: CLASS_NAMES.filter((c) => next.has(c)),
                      },
                    });
                  }}
                />
                <span className="font-mono">{name}</span>
                <span className="text-[12px] text-muted">{t(`classes.${name}`)}</span>
              </label>
            ))}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[13px] font-medium text-ink-2">
            {t("params.verdict.min_confidence.label")}
          </span>
          <ParamControl
            entry={verdictEntry("min_confidence")}
            value={draft.verdict.min_confidence}
            onChange={(v) =>
              setDraft({
                ...draft,
                verdict: { ...draft.verdict, min_confidence: v as number | null },
              })
            }
          />
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-[13px] font-medium text-ink-2">
            {t("params.verdict.low_confidence_verdict.label")}
          </span>
          <ParamControl
            entry={verdictEntry("low_confidence_verdict")}
            value={draft.verdict.low_confidence_verdict}
            onChange={(v) =>
              setDraft({
                ...draft,
                verdict: { ...draft.verdict, low_confidence_verdict: v as "reject" | "review" },
              })
            }
          />
        </div>
      </fieldset>
    </div>
  );
}

// ------------------------------------------------------------------ editor

function ProfileEditor({
  profile,
  profiles,
  onDirtyChange,
  onNavigate,
}: {
  profile: Profile;
  profiles: Profile[];
  onDirtyChange: (dirty: boolean) => void;
  onNavigate: (id: number | null) => void;
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const catalog = useCatalog().data;
  const [draft, setDraft] = useState<Draft>(() => toDraft(profile));
  const [tab, setTab] = useState("parameters");
  const [compare, setCompare] = useState(false);
  const [serverIssues, setServerIssues] = useState<Record<string, string>>({});
  const saved = useMemo(() => toDraft(profile), [profile]);
  const [loadedFor, setLoadedFor] = useState(profile);
  if (loadedFor !== profile) {
    // A new version was loaded (save, restore, relink): start from it.
    setLoadedFor(profile);
    setDraft(toDraft(profile));
    setServerIssues({});
  }

  const training = (profile.training_params ?? profile.params) as Params;
  const entries = useMemo(() => catalog?.entries ?? [], [catalog]);
  const changedFields = (Object.keys(saved) as (keyof Draft)[]).filter(
    (k) => !same(saved[k], draft[k]),
  );
  const changedParams = Object.keys(draft.params).filter(
    (k) => !same(draft.params[k], saved.params[k]),
  );
  const unsavedCount = changedFields.filter((f) => f !== "params").length + changedParams.length;
  const dirty = unsavedCount > 0;
  useEffect(() => onDirtyChange(dirty), [dirty, onDirtyChange]);

  const clientIssues = useMemo(() => validateAll(entries, draft.params), [entries, draft.params]);
  const issues: Record<string, string> = { ...serverIssues };
  for (const [key, issue] of Object.entries(clientIssues))
    issues[key] = t(`errors.${issue.code}`, { ...issue.params });
  const modifiedVsTraining = entries.filter(
    (e) =>
      e.mode === "editable" && e.group !== "verdict" && !same(draft.params[e.key], training[e.key]),
  ).length;

  const save = useMutation({
    mutationFn: () => {
      const body: Record<string, unknown> = {};
      for (const field of changedFields) {
        if (field === "params")
          body.params = Object.fromEntries(changedParams.map((k) => [k, draft.params[k]]));
        else body[field] = draft[field];
      }
      return api.patch<Profile>(`/api/profiles/${profile.id}`, body);
    },
    onSuccess: (updated) => {
      client.setQueryData(keys.profile(profile.id), updated);
      void client.invalidateQueries({ queryKey: keys.profiles });
      setServerIssues({});
      toast.success(t("profiles.saved", { version: updated.version }));
    },
    onError: (e) => {
      if (e instanceof ApiError && Array.isArray(e.details.errors)) {
        const map: Record<string, string> = {};
        for (const err of e.details.errors as {
          key: string;
          code: string;
          params: Record<string, unknown>;
        }[])
          map[err.key] = t(`errors.${err.code}`, { ...err.params });
        setServerIssues(map);
      }
      toast.error(errorMessage(t, e));
    },
  });

  const action = (fn: () => Promise<unknown>, message: string, after?: (data: unknown) => void) =>
    fn()
      .then((data) => {
        void client.invalidateQueries({ queryKey: keys.profiles });
        toast.success(message);
        after?.(data);
      })
      .catch((e) => toast.error(errorMessage(t, e)));

  const exportProfile = async () => {
    const data = await api.get<Record<string, unknown>>(`/api/profiles/${profile.id}/export`);
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${profile.name.replace(/[^\w.-]+/g, "_")}.profile.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const setParam = (key: string, value: ParamValue) => {
    setDraft((d) => ({ ...d, params: { ...d.params, [key]: value } }));
    setServerIssues((s) => {
      const next = { ...s };
      delete next[key];
      return next;
    });
  };

  const readOnly = profile.deleted;
  return (
    <div className="flex min-h-0 flex-col">
      <div className="flex flex-wrap items-center gap-3 border-b border-line px-4 py-3">
        <span
          aria-hidden="true"
          className="h-6 w-1.5 rounded-full"
          style={{ background: profile.color }}
        />
        <h2 className="text-[17px] font-semibold">{profile.name}</h2>
        <span className="font-mono text-[13px] text-ink-2">v{profile.version}</span>
        <ProfileStatusBadge status={profile.status} />
        {profile.is_builtin ? (
          <span className="rounded-[4px] bg-sunken px-1.5 text-[12px] text-ink-2">
            {t("profiles.builtin")}
          </span>
        ) : null}
        <div className="ml-auto flex items-center gap-2">
          {profile.status === "incompatible" ? (
            <Button
              variant="primary"
              onClick={() =>
                void action(
                  () => api.post(`/api/profiles/${profile.id}/relink`),
                  t("profiles.relinked"),
                )
              }
            >
              <Link2 />
              {t("profiles.relink")}
            </Button>
          ) : null}
          <Button
            onClick={() =>
              void action(
                () => api.post<Profile>(`/api/profiles/${profile.id}/duplicate`),
                t("profiles.duplicated"),
                (p) => onNavigate((p as Profile).id),
              )
            }
          >
            <Copy />
            {t("profiles.duplicate")}
          </Button>
          <Button onClick={() => void exportProfile()}>
            <Download />
            {t("profiles.export")}
          </Button>
          <Dropdown>
            <DropdownTrigger asChild>
              <Button size="icon" aria-label={t("common.more")}>
                <MoreHorizontal />
              </Button>
            </DropdownTrigger>
            <DropdownContent>
              <DropdownItem onSelect={() => setCompare(true)}>
                {t("profiles.compareWith")}
              </DropdownItem>
              <DropdownItem
                disabled={profile.is_builtin}
                className="text-reject"
                onSelect={() => {
                  if (window.confirm(t("profiles.deleteConfirm", { name: profile.name }))) {
                    void action(
                      () => api.del(`/api/profiles/${profile.id}`),
                      t("profiles.deleted"),
                      () => onNavigate(null),
                    );
                  }
                }}
              >
                <Trash2 />
                {profile.is_builtin ? t("profiles.builtinNoDelete") : t("profiles.delete")}
              </DropdownItem>
            </DropdownContent>
          </Dropdown>
        </div>
        {profile.status_detail.reason && !profile.status_detail.can_analyse ? (
          <p className="w-full rounded-[6px] bg-review-tint px-3 py-1.5 text-[13px] text-review">
            {t(profile.status_detail.reason)}
          </p>
        ) : null}
      </div>
      <Tabs value={tab} onValueChange={setTab} className="flex min-h-0 flex-1 flex-col">
        <TabsList className="px-4">
          {["overview", "parameters", "model", "history"].map((k) => (
            <TabsTrigger key={k} value={k}>
              {t(`profiles.tab.${k}`)}
            </TabsTrigger>
          ))}
        </TabsList>
        <div className="min-h-0 flex-1 overflow-auto p-4">
          <TabsContent value="overview">
            <OverviewTab
              profile={profile}
              draft={draft}
              setDraft={setDraft}
              verdictEntries={entries.filter((e) => e.group === "verdict")}
            />
          </TabsContent>
          <TabsContent value="parameters">
            {catalog ? (
              <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
                <ParamsTab
                  entries={entries}
                  groups={catalog.groups}
                  draft={draft}
                  training={training}
                  issues={issues}
                  onChange={(k, v) => setParam(k, v)}
                />
                {profile.status_detail.can_analyse ? (
                  <div>
                    <PreviewPanel
                      profileId={profile.id}
                      params={draft.params}
                      modifiedCount={modifiedVsTraining}
                      onParam={(key, value) => setParam(key, value)}
                    />
                  </div>
                ) : null}
              </div>
            ) : (
              <Spinner />
            )}
          </TabsContent>
          <TabsContent value="model">
            <ModelTab modelRef={profile.model_ref} />
          </TabsContent>
          <TabsContent value="history">
            <HistoryTab profileId={profile.id} current={profile.version} readOnly={readOnly} />
          </TabsContent>
        </div>
      </Tabs>
      {dirty ? (
        <div className="sticky bottom-0 flex items-center gap-3 border-t border-line bg-panel px-4 py-2.5 shadow-[0_-4px_12px_rgb(27_31_36/0.06)]">
          <span className="text-[13px] font-medium">
            {t("profiles.unsaved", { count: unsavedCount })}
          </span>
          {Object.keys(clientIssues).length ? (
            <span className="text-[13px] text-reject">
              {t("profiles.fixErrors", { count: Object.keys(clientIssues).length })}
            </span>
          ) : null}
          <div className="ml-auto flex gap-2">
            <Button onClick={() => setDraft(saved)}>{t("profiles.discard")}</Button>
            <Button
              variant="primary"
              disabled={
                save.isPending || Object.keys(clientIssues).length > 0 || !draft.name.trim()
              }
              onClick={() => save.mutate()}
            >
              {t("profiles.save")}
            </Button>
          </div>
        </div>
      ) : null}
      <CompareDialog
        open={compare}
        onOpenChange={setCompare}
        profile={profile}
        profiles={profiles}
      />
    </div>
  );
}

export default function ProfilesPage() {
  const { t } = useTranslation();
  const params = useParams();
  const navigate = useNavigate();
  const client = useQueryClient();
  const profilesQuery = useProfiles();
  const profiles = profilesQuery.data ?? [];
  const selectedId = params.id ? Number(params.id) : (profiles[0]?.id ?? null);
  const profile = useProfile(selectedId);
  const [dirty, setDirty] = useState(false);
  const [newOpen, setNewOpen] = useState(false);
  const [importOpen, setImportOpen] = useState(false);

  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) =>
      dirty && currentLocation.pathname !== nextLocation.pathname,
  );
  useEffect(() => {
    if (blocker.state === "blocked") {
      if (window.confirm(t("profiles.leaveConfirm"))) blocker.proceed();
      else blocker.reset();
    }
  }, [blocker, t]);
  useEffect(() => {
    const onUnload = (event: BeforeUnloadEvent) => {
      if (dirty) event.preventDefault();
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty]);

  const refresh = useMutation({
    mutationFn: () => api.post<{ created_profiles: string[] }>("/api/models/refresh"),
    onSuccess: (data) => {
      void client.invalidateQueries({ queryKey: keys.models });
      void client.invalidateQueries({ queryKey: keys.profiles });
      toast.success(t("profiles.refreshed", { count: data.created_profiles.length }));
    },
  });

  const go = (id: number | null) => navigate(id === null ? "/profiles" : `/profiles/${id}`);

  return (
    <div className="grid h-full min-h-0 grid-cols-[18rem_minmax(0,1fr)]">
      <aside className="flex min-h-0 flex-col gap-2 border-r border-line bg-panel p-3">
        <div className="flex flex-wrap gap-1.5">
          <Button variant="primary" size="sm" onClick={() => setNewOpen(true)}>
            <Plus />
            {t("profiles.new")}
          </Button>
          <Button size="sm" onClick={() => setImportOpen(true)}>
            <FileUp />
            {t("profiles.import")}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={refresh.isPending}
            onClick={() => refresh.mutate()}
          >
            <RefreshCw className={cn(refresh.isPending && "animate-spin")} />
            {t("profiles.refreshModels")}
          </Button>
        </div>
        <ProfileList profiles={profiles} selectedId={selectedId} onSelect={go} />
      </aside>
      <section className="flex min-h-0 flex-col">
        {profilesQuery.isSuccess && !profiles.length ? (
          <div className="panel m-6 max-w-xl p-6">
            <h2 className="font-semibold">{t("empty.noModelsTitle")}</h2>
            <p className="mt-2 text-ink-2">
              {t("empty.noModelsBody1")} <code className="font-mono">make run</code>{" "}
              {t("empty.noModelsBody2")}
            </p>
          </div>
        ) : profile.data ? (
          <ProfileEditor
            key={profile.data.id}
            profile={profile.data}
            profiles={profiles}
            onDirtyChange={setDirty}
            onNavigate={go}
          />
        ) : (
          <div className="p-6">
            <Spinner />
          </div>
        )}
      </section>
      <NewProfileDialog open={newOpen} onOpenChange={setNewOpen} onCreated={go} />
      <ImportDialog open={importOpen} onOpenChange={setImportOpen} onImported={go} />
    </div>
  );
}
