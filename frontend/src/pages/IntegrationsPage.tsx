// Integrations (§12, §15.7): rules (WHEN -> IF -> THEN), targets and the execution log.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Bell,
  ChevronRight,
  FileText,
  Globe,
  Mail,
  Pencil,
  Plus,
  Radio,
  Send,
  Trash2,
  Volume2,
  X,
} from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { useSearchParams } from "react-router";
import { toast } from "sonner";

import { NumberInput } from "@/components/params/ParamControls";
import {
  Button,
  Checkbox,
  Dialog,
  Field,
  Select,
  Spinner,
  Switch,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  cn,
} from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import { fmtDate, fmtInt, fmtMs } from "@/lib/format";
import { useProfiles } from "@/lib/queries";
import { CLASS_NAMES } from "@/lib/types";

// ------------------------------------------------------------------ types

interface Meta {
  conditions: Record<string, Record<string, unknown>>;
  actions: Record<string, Record<string, unknown>>;
  action_targets: Record<string, string>;
  targets: Record<string, Record<string, unknown>>;
  target_secrets: Record<string, string[]>;
  warning_codes: string[];
  variables: string[];
}
interface Target {
  id: number;
  name: string;
  type: "smtp" | "webhook" | "mqtt" | "file";
  config: Record<string, unknown>;
  secrets: Record<string, string | null>;
  enabled: boolean;
  last_test_at: string | null;
  last_test_ok: boolean | null;
  last_test_detail: string | null;
}
interface ConditionDraft {
  type: string;
  params: Record<string, unknown>;
  negate: boolean;
}
interface ActionDraft {
  type: string;
  target_id: number | null;
  params: Record<string, unknown>;
  enabled: boolean;
  timeout_s?: number | null;
  max_retries?: number;
}
interface RuleDraft {
  id?: number;
  name: string;
  description: string;
  enabled: boolean;
  severity: "info" | "warning" | "critical";
  scope: { profiles: "all" | number[] };
  condition_logic: "all" | "any";
  cooldown_s: number;
  dedupe_key: "rule" | "rule+class" | "rule+profile";
  run_mode: "single" | "queued" | "parallel";
  conditions: ConditionDraft[];
  actions: ActionDraft[];
}
interface RuleRow extends RuleDraft {
  id: number;
  fired_24h: number;
  last_fired_at: string | null;
}
interface Execution {
  id: number;
  rule: string | null;
  action_type: string | null;
  target: string | null;
  status: string;
  attempt: number;
  http_status: number | null;
  error: string | null;
  rendered_payload: string | null;
  response_snippet: string | null;
  duration_ms: number | null;
  is_test: boolean;
  suppressed_count: number;
  created_at: string;
}

const TARGET_ICON = { smtp: Mail, webhook: Globe, mqtt: Radio, file: FileText } as const;
const ACTION_ICON: Record<string, typeof Mail> = {
  email: Mail,
  webhook: Globe,
  mqtt_publish: Radio,
  write_file: FileText,
  notify_ui: Bell,
  play_sound: Volume2,
};
const TEMPLATE_FIELDS = new Set([
  "title",
  "message",
  "subject",
  "body",
  "html",
  "path",
  "topic",
  "payload",
]);

function errorText(t: TFunction, e: unknown): string {
  if (e instanceof ApiError)
    return String(t(`errors.${e.code}`, { ...e.details, defaultValue: e.message }));
  return String(t("errors.unexpected"));
}

const useMeta = () =>
  useQuery({
    queryKey: ["integrations-meta"],
    queryFn: () => api.get<Meta>("/api/integrations/meta"),
    staleTime: Infinity,
  });
const useTargets = () =>
  useQuery({ queryKey: ["targets"], queryFn: () => api.get<Target[]>("/api/targets") });

// ------------------------------------------------------------------ rule summary (plain language)

function useSummary() {
  const { t } = useTranslation();
  const targets = useTargets().data ?? [];
  return (rule: RuleDraft) => {
    const conditions = rule.conditions.map((c) => {
      const text = String(
        t(`integrations.cond.summary.${c.type}`, {
          ...(c.params as Record<string, unknown>),
          classes: ((c.params.classes as string[]) ?? []).join(", "),
          codes: ((c.params.codes as string[]) ?? []).join(", "),
          class: c.params.class === "any" ? t("integrations.anyDefect") : c.params.class,
        }),
      );
      return c.negate ? `${t("integrations.not")} ${text}` : text;
    });
    const joiner = ` ${t(`integrations.logic.${rule.condition_logic}Join`)} `;
    const actions = rule.actions.map((a) => {
      const target = targets.find((x) => x.id === a.target_id);
      return target
        ? `${t(`integrations.action.${a.type}`)} ${target.name}`
        : t(`integrations.action.${a.type}`);
    });
    return `${t("integrations.when")} ${conditions.join(joiner) || t("integrations.noConditions")} → ${actions.join(", ") || t("integrations.noActions")}`;
  };
}

// ------------------------------------------------------------------ parameter forms

function ListInput({
  value,
  onChange,
  label,
}: {
  value: string[];
  onChange: (v: string[]) => void;
  label: string;
}) {
  return (
    <input
      className="field-input w-full font-sans"
      aria-label={label}
      value={value.join(", ")}
      onChange={(e) =>
        onChange(
          e.target.value
            .split(",")
            .map((s) => s.trim())
            .filter(Boolean),
        )
      }
    />
  );
}

function MultiCheck({
  options,
  value,
  onChange,
  mono,
}: {
  options: string[];
  value: string[];
  onChange: (v: string[]) => void;
  mono?: boolean;
}) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1">
      {options.map((o) => (
        <label
          key={o}
          className={cn("inline-flex items-center gap-1.5 text-[12px]", mono && "font-mono")}
        >
          <Checkbox
            checked={value.includes(o)}
            onCheckedChange={(v) =>
              onChange(
                v
                  ? options.filter((x) => x === o || value.includes(x))
                  : value.filter((x) => x !== o),
              )
            }
          />
          {o}
        </label>
      ))}
    </div>
  );
}

function TemplateField({
  label,
  value,
  onChange,
  variables,
  multiline,
  ruleId,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  variables: string[];
  multiline?: boolean;
  ruleId?: number;
}) {
  const { t } = useTranslation();
  const [preview, setPreview] = useState<{
    ok: boolean;
    output?: string;
    error?: string;
    source?: string;
  } | null>(null);
  useEffect(() => {
    if (!value) return;
    const timer = window.setTimeout(() => {
      api
        .post<{ ok: boolean; output?: string; error?: string; source?: string }>(
          "/api/templates/preview",
          { template: value, rule_id: ruleId ?? null },
        )
        .then(setPreview)
        .catch(() => setPreview(null));
    }, 300);
    return () => window.clearTimeout(timer);
  }, [value, ruleId]);
  const insert = (variable: string) => onChange(`${value}{{ ${variable} }}`);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <span className="text-[12px] font-medium text-ink-2">{label}</span>
        <Select
          label={t("integrations.insertVariable")}
          value=""
          onValueChange={insert}
          options={variables.map((v) => ({ value: v, label: v }))}
          className="ml-auto h-7 w-48 text-[12px]"
        />
      </div>
      {multiline ? (
        <textarea
          className="field-input h-24 w-full py-1.5"
          aria-label={label}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
      ) : (
        <input
          className="field-input w-full"
          aria-label={label}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          spellCheck={false}
        />
      )}
      {preview && value ? (
        <pre
          className={cn(
            "max-h-28 overflow-auto whitespace-pre-wrap rounded-[6px] px-2 py-1 font-mono text-[11px]",
            preview.ok ? "bg-sunken text-ink-2" : "bg-reject-tint text-reject",
          )}
          aria-label={t("integrations.preview")}
        >
          {preview.ok ? preview.output : preview.error}
          <span className="block text-[10px] text-muted">
            {t(`integrations.previewSource.${preview.source ?? "sample"}`)}
          </span>
        </pre>
      ) : null}
    </div>
  );
}

function ConditionCard({
  condition,
  meta,
  onChange,
  onRemove,
}: {
  condition: ConditionDraft;
  meta: Meta;
  onChange: (c: ConditionDraft) => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation();
  const p = condition.params;
  const set = (key: string, value: unknown) =>
    onChange({ ...condition, params: { ...p, [key]: value } });
  const num = (key: string, digits = 0, min?: number, max?: number) => (
    <label className="inline-flex items-center gap-1.5 text-[12px]">
      {t(`integrations.param.${key}`)}
      <NumberInput
        label={t(`integrations.param.${key}`)}
        value={Number(p[key] ?? 0)}
        digits={digits}
        min={min}
        max={max}
        onChange={(v) => set(key, v)}
        className="w-20"
      />
    </label>
  );
  return (
    <div className="rounded-[6px] border border-line bg-panel p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Select
          label={t("integrations.conditionType")}
          value={condition.type}
          onValueChange={(type) =>
            onChange({ type, params: { ...meta.conditions[type] }, negate: false })
          }
          options={Object.keys(meta.conditions).map((c) => ({
            value: c,
            label: t(`integrations.cond.${c}`),
          }))}
          className="w-64"
        />
        <label className="inline-flex items-center gap-1.5 text-[12px]">
          <Checkbox
            checked={condition.negate}
            onCheckedChange={(v) => onChange({ ...condition, negate: v === true })}
          />
          {t("integrations.negate")}
        </label>
        <Button
          size="icon"
          variant="ghost"
          className="ml-auto"
          aria-label={t("integrations.removeCondition")}
          onClick={onRemove}
        >
          <X />
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {condition.type === "class_is" ? (
          <MultiCheck
            mono
            options={CLASS_NAMES as unknown as string[]}
            value={(p.classes as string[]) ?? []}
            onChange={(v) => set("classes", v)}
          />
        ) : null}
        {condition.type === "verdict_is" ? (
          <Select
            label={t("integrations.param.verdict")}
            value={String(p.verdict)}
            onValueChange={(v) => set("verdict", v)}
            options={["pass", "reject", "review"].map((v) => ({
              value: v,
              label: t(`verdict.${v}`),
            }))}
            className="w-36"
          />
        ) : null}
        {condition.type === "repeated_class" ? (
          <>
            <Select
              label={t("integrations.param.class")}
              value={String(p.class)}
              onValueChange={(v) => set("class", v)}
              options={[
                { value: "any", label: t("integrations.anyDefect") },
                ...CLASS_NAMES.filter((c) => c !== "bottle_no_defect").map((c) => ({
                  value: c,
                  label: c,
                })),
              ]}
              className="w-56"
            />
            {num("n", 0, 1, 1000)}
            {num("m", 0, 1, 1000)}
          </>
        ) : null}
        {condition.type === "consecutive_rejects" ? num("n", 0, 1, 1000) : null}
        {condition.type === "confidence_below" ? num("value", 2, 0, 1) : null}
        {condition.type === "processing_time_above" ? num("ms", 0, 1) : null}
        {condition.type === "reject_rate_above" ? (
          <>
            {num("percent", 1, 0, 100)}
            {num("window_minutes", 0, 1)}
            {num("min_samples", 0, 1)}
          </>
        ) : null}
        {condition.type === "warning_present" ? (
          <MultiCheck
            mono
            options={meta.warning_codes}
            value={(p.codes as string[]) ?? []}
            onChange={(v) => set("codes", v)}
          />
        ) : null}
      </div>
      <p className="mt-1.5 text-[12px] text-muted">
        {t(`integrations.condHelp.${condition.type}`)}
      </p>
    </div>
  );
}

function ActionCard({
  action,
  meta,
  targets,
  ruleId,
  onChange,
  onRemove,
}: {
  action: ActionDraft;
  meta: Meta;
  targets: Target[];
  ruleId?: number;
  onChange: (a: ActionDraft) => void;
  onRemove: () => void;
}) {
  const { t } = useTranslation();
  const p = action.params;
  const set = (key: string, value: unknown) =>
    onChange({ ...action, params: { ...p, [key]: value } });
  const needed = meta.action_targets[action.type];
  const Icon = ACTION_ICON[action.type] ?? Bell;
  return (
    <div
      className={cn(
        "rounded-[6px] border border-line bg-panel p-3",
        !action.enabled && "opacity-60",
      )}
    >
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <Icon className="size-4 text-ink-2" aria-hidden="true" />
        <Select
          label={t("integrations.actionType")}
          value={action.type}
          onValueChange={(type) =>
            onChange({ ...action, type, target_id: null, params: { ...meta.actions[type] } })
          }
          options={Object.keys(meta.actions).map((a) => ({
            value: a,
            label: t(`integrations.action.${a}`),
          }))}
          className="w-52"
        />
        {needed ? (
          <Select
            label={t("integrations.target")}
            value={action.target_id ? String(action.target_id) : ""}
            onValueChange={(v) => onChange({ ...action, target_id: Number(v) })}
            options={targets
              .filter((x) => x.type === needed)
              .map((x) => ({ value: String(x.id), label: x.name }))}
            className="w-52"
          />
        ) : null}
        {needed && !targets.some((x) => x.type === needed) ? (
          <span className="text-[12px] text-reject">
            {t("integrations.createTargetFirst", { type: t(`integrations.targetType.${needed}`) })}
          </span>
        ) : null}
        <label className="ml-auto inline-flex items-center gap-1.5 text-[12px]">
          <Switch
            checked={action.enabled}
            onCheckedChange={(v) => onChange({ ...action, enabled: v })}
            aria-label={t("integrations.actionEnabled")}
          />
          {t("integrations.actionEnabled")}
        </label>
        <Button
          size="icon"
          variant="ghost"
          aria-label={t("integrations.removeAction")}
          onClick={onRemove}
        >
          <X />
        </Button>
      </div>
      <div className="grid gap-2">
        {Object.entries(p).map(([key, value]) => {
          if (TEMPLATE_FIELDS.has(key)) {
            return (
              <TemplateField
                key={key}
                label={t(`integrations.param.${key}`)}
                value={String(value ?? "")}
                onChange={(v) => set(key, v)}
                variables={meta.variables}
                multiline={
                  key === "body" || key === "html" || key === "payload" || key === "message"
                }
                ruleId={ruleId}
              />
            );
          }
          if (key === "to" || key === "fields") {
            return (
              <Field key={key} label={t(`integrations.param.${key}`)}>
                <ListInput
                  label={t(`integrations.param.${key}`)}
                  value={(value as string[]) ?? []}
                  onChange={(v) => set(key, v)}
                />
              </Field>
            );
          }
          if (key === "headers") {
            return (
              <Field
                key={key}
                label={t("integrations.param.headers")}
                hint={t("integrations.headersHint")}
              >
                <textarea
                  className="field-input h-16 w-full py-1.5"
                  aria-label={t("integrations.param.headers")}
                  defaultValue={JSON.stringify(value ?? {}, null, 0)}
                  onBlur={(e) => {
                    try {
                      set(key, JSON.parse(e.target.value || "{}"));
                    } catch {
                      toast.error(t("errors.request.invalid_json"));
                    }
                  }}
                />
              </Field>
            );
          }
          if (key === "level")
            return (
              <Select
                key={key}
                label={t("integrations.param.level")}
                value={String(value)}
                onValueChange={(v) => set(key, v)}
                options={["info", "warning", "critical"].map((v) => ({
                  value: v,
                  label: t(`integrations.severity.${v}`),
                }))}
                className="w-40"
              />
            );
          if (key === "sound")
            return (
              <Select
                key={key}
                label={t("integrations.param.sound")}
                value={String(value)}
                onValueChange={(v) => set(key, v)}
                options={["chime", "alarm", "beep"].map((v) => ({
                  value: v,
                  label: t(`integrations.sound.${v}`),
                }))}
                className="w-40"
              />
            );
          if (key === "method")
            return (
              <Select
                key={key}
                label={t("integrations.param.method")}
                value={String(value)}
                onValueChange={(v) => set(key, v)}
                options={["POST", "PUT", "PATCH"].map((v) => ({ value: v, label: v }))}
                className="w-28"
              />
            );
          if (key === "qos")
            return (
              <Select
                key={key}
                label="QoS"
                value={String(value)}
                onValueChange={(v) => set(key, Number(v))}
                options={["0", "1", "2"].map((v) => ({ value: v, label: `QoS ${v}` }))}
                className="w-28"
              />
            );
          if (typeof value === "boolean")
            return (
              <label key={key} className="inline-flex items-center gap-1.5 text-[12px]">
                <Checkbox checked={value} onCheckedChange={(v) => set(key, v === true)} />
                {t(`integrations.param.${key}`)}
              </label>
            );
          return null;
        })}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ rule editor

const EMPTY_RULE: RuleDraft = {
  name: "",
  description: "",
  enabled: true,
  severity: "warning",
  scope: { profiles: "all" },
  condition_logic: "all",
  cooldown_s: 60,
  dedupe_key: "rule+class",
  run_mode: "single",
  conditions: [
    { type: "repeated_class", params: { class: "defect_no_cap", n: 4, m: 10 }, negate: false },
  ],
  actions: [{ type: "notify_ui", target_id: null, params: {}, enabled: true }],
};

function RuleEditor({ rule, onClose }: { rule: RuleDraft; onClose: () => void }) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const meta = useMeta().data;
  const targets = useTargets().data ?? [];
  const profiles = (useProfiles().data ?? []).filter((p) => !p.deleted);
  const [draft, setDraft] = useState<RuleDraft>(() => ({
    ...rule,
    actions: rule.actions.map((a) => ({
      ...a,
      params: { ...(meta?.actions[a.type] ?? {}), ...a.params },
    })),
  }));
  const [dry, setDry] = useState<{
    evaluated: number;
    would_fire: number;
    suppressed_by_cooldown: number;
    fires: { analysis_id: number; predicted_class: string; created_at: string }[];
  } | null>(null);
  const [tested, setTested] = useState<
    { type: string; status: string; error: string | null; attempt: number }[] | null
  >(null);
  const summary = useSummary();

  const save = useMutation({
    mutationFn: async () => {
      const body = { ...draft };
      delete body.id;
      const saved = draft.id
        ? await api.patch<RuleRow>(`/api/rules/${draft.id}`, body)
        : await api.post<RuleRow>("/api/rules", body);
      return saved;
    },
    onSuccess: (saved) => {
      void client.invalidateQueries({ queryKey: ["rules"] });
      setDraft((d) => ({ ...d, id: saved.id }));
      toast.success(t("integrations.ruleSaved"));
    },
    onError: (e) => toast.error(errorText(t, e)),
  });

  const runDry = async () => {
    const id = draft.id ?? (await save.mutateAsync()).id;
    setDry(await api.post(`/api/rules/${id}/dry-run`, { limit: 100 }));
  };
  const runTest = async () => {
    const id = draft.id ?? (await save.mutateAsync()).id;
    try {
      const result = await api.post<{
        results: { type: string; status: string; error: string | null; attempt: number }[];
      }>(`/api/rules/${id}/test`);
      setTested(result.results);
      void client.invalidateQueries({ queryKey: ["executions"] });
    } catch (e) {
      toast.error(errorText(t, e));
    }
  };

  if (!meta) return <Spinner />;
  const scopeAll = draft.scope.profiles === "all";
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      wide
      title={draft.id ? t("integrations.editRule") : t("integrations.newRule")}
      description={summary(draft)}
      footer={
        <>
          <Button onClick={() => void runDry()} disabled={save.isPending}>
            {t("integrations.dryRun")}
          </Button>
          <Button onClick={() => void runTest()} disabled={save.isPending}>
            <Send />
            {t("integrations.sendTest")}
          </Button>
          <Button onClick={onClose}>{t("common.close")}</Button>
          <Button
            variant="primary"
            disabled={!draft.name.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {t("common.save")}
          </Button>
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <div className="grid gap-3 sm:grid-cols-[1fr_10rem]">
          <Field label={t("integrations.name")} htmlFor="rule-name">
            <input
              id="rule-name"
              className="field-input font-sans"
              maxLength={80}
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            />
          </Field>
          <Field label={t("integrations.severityLabel")}>
            <Select
              label={t("integrations.severityLabel")}
              value={draft.severity}
              onValueChange={(v) => setDraft({ ...draft, severity: v as RuleDraft["severity"] })}
              options={["info", "warning", "critical"].map((v) => ({
                value: v,
                label: t(`integrations.severity.${v}`),
              }))}
            />
          </Field>
        </div>
        <Field label={t("integrations.description")} htmlFor="rule-desc">
          <input
            id="rule-desc"
            className="field-input font-sans"
            value={draft.description}
            onChange={(e) => setDraft({ ...draft, description: e.target.value })}
          />
        </Field>

        <section className="rounded-[10px] bg-sunken p-3">
          <h3 className="mb-1 text-[13px] font-semibold">{t("integrations.whenTitle")}</h3>
          <p className="text-[13px] text-ink-2">{t("integrations.trigger")}</p>
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[13px]">
            <label className="inline-flex items-center gap-1.5">
              <Checkbox
                checked={scopeAll}
                onCheckedChange={(v) => setDraft({ ...draft, scope: { profiles: v ? "all" : [] } })}
              />
              {t("integrations.allProfiles")}
            </label>
            {!scopeAll
              ? profiles.map((p) => (
                  <label key={p.id} className="inline-flex items-center gap-1.5">
                    <Checkbox
                      checked={(draft.scope.profiles as number[]).includes(p.id)}
                      onCheckedChange={(v) => {
                        const ids = draft.scope.profiles as number[];
                        setDraft({
                          ...draft,
                          scope: { profiles: v ? [...ids, p.id] : ids.filter((x) => x !== p.id) },
                        });
                      }}
                    />
                    {p.name}
                  </label>
                ))
              : null}
          </div>
        </section>

        <section className="rounded-[10px] bg-sunken p-3">
          <div className="mb-2 flex items-center gap-2">
            <h3 className="text-[13px] font-semibold">{t("integrations.ifTitle")}</h3>
            <Select
              label={t("integrations.logicLabel")}
              value={draft.condition_logic}
              onValueChange={(v) => setDraft({ ...draft, condition_logic: v as "all" | "any" })}
              options={[
                { value: "all", label: t("integrations.logic.all") },
                { value: "any", label: t("integrations.logic.any") },
              ]}
              className="w-56"
            />
          </div>
          <div className="flex flex-col gap-2">
            {draft.conditions.map((c, i) => (
              <ConditionCard
                key={i}
                condition={c}
                meta={meta}
                onChange={(next) =>
                  setDraft({
                    ...draft,
                    conditions: draft.conditions.map((x, j) => (j === i ? next : x)),
                  })
                }
                onRemove={() =>
                  setDraft({ ...draft, conditions: draft.conditions.filter((_, j) => j !== i) })
                }
              />
            ))}
            <Button
              size="sm"
              className="self-start"
              onClick={() =>
                setDraft({
                  ...draft,
                  conditions: [
                    ...draft.conditions,
                    { type: "any_defect", params: {}, negate: false },
                  ],
                })
              }
            >
              <Plus />
              {t("integrations.addCondition")}
            </Button>
          </div>
        </section>

        <section className="rounded-[10px] bg-sunken p-3">
          <h3 className="mb-2 text-[13px] font-semibold">{t("integrations.thenTitle")}</h3>
          <div className="flex flex-col gap-2">
            {draft.actions.map((a, i) => (
              <ActionCard
                key={i}
                action={a}
                meta={meta}
                targets={targets}
                ruleId={draft.id}
                onChange={(next) =>
                  setDraft({ ...draft, actions: draft.actions.map((x, j) => (j === i ? next : x)) })
                }
                onRemove={() =>
                  setDraft({ ...draft, actions: draft.actions.filter((_, j) => j !== i) })
                }
              />
            ))}
            <Button
              size="sm"
              className="self-start"
              onClick={() =>
                setDraft({
                  ...draft,
                  actions: [
                    ...draft.actions,
                    {
                      type: "notify_ui",
                      target_id: null,
                      params: { ...meta.actions.notify_ui },
                      enabled: true,
                    },
                  ],
                })
              }
            >
              <Plus />
              {t("integrations.addAction")}
            </Button>
          </div>
        </section>

        <div className="grid gap-3 sm:grid-cols-3">
          <Field label={t("integrations.cooldown")} hint={t("integrations.cooldownHint")}>
            <NumberInput
              label={t("integrations.cooldown")}
              value={draft.cooldown_s}
              digits={0}
              min={0}
              max={86400}
              onChange={(v) => setDraft({ ...draft, cooldown_s: Math.round(v) })}
              className="w-28"
            />
          </Field>
          <Field label={t("integrations.dedupe")}>
            <Select
              label={t("integrations.dedupe")}
              value={draft.dedupe_key}
              onValueChange={(v) =>
                setDraft({ ...draft, dedupe_key: v as RuleDraft["dedupe_key"] })
              }
              options={["rule", "rule+class", "rule+profile"].map((v) => ({
                value: v,
                label: t(`integrations.dedupeKey.${v}`),
              }))}
            />
          </Field>
          <Field label={t("integrations.runMode")}>
            <Select
              label={t("integrations.runMode")}
              value={draft.run_mode}
              onValueChange={(v) => setDraft({ ...draft, run_mode: v as RuleDraft["run_mode"] })}
              options={["single", "queued", "parallel"].map((v) => ({
                value: v,
                label: t(`integrations.mode.${v}`),
              }))}
            />
          </Field>
        </div>

        {dry ? (
          <div className="rounded-[6px] border border-line p-3 text-[13px]" role="status">
            <p className="font-medium">
              {t("integrations.dryResult", {
                fired: dry.would_fire,
                evaluated: dry.evaluated,
                suppressed: dry.suppressed_by_cooldown,
              })}
            </p>
            <ul className="mt-1 max-h-32 overflow-auto font-mono text-[12px] text-ink-2">
              {dry.fires.map((f) => (
                <li key={f.analysis_id}>
                  {fmtDate(f.created_at, true)} · #{f.analysis_id} · {f.predicted_class}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        {tested ? (
          <ul className="rounded-[6px] border border-line p-3 text-[13px]" role="status">
            {tested.map((r, i) => (
              <li key={i} className={r.status === "success" ? "text-pass" : "text-reject"}>
                {t(`integrations.action.${r.type}`)}: {t(`integrations.status.${r.status}`)}
                {r.error ? ` — ${r.error}` : ""}
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </Dialog>
  );
}

// ------------------------------------------------------------------ tabs

function RulesTab() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const rules = useQuery({ queryKey: ["rules"], queryFn: () => api.get<RuleRow[]>("/api/rules") });
  const [editing, setEditing] = useState<RuleDraft | null>(null);
  const summary = useSummary();
  const toggle = useMutation({
    mutationFn: (r: RuleRow) => api.patch(`/api/rules/${r.id}`, { enabled: !r.enabled }),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["rules"] }),
  });
  const remove = useMutation({
    mutationFn: (id: number) => api.del(`/api/rules/${id}`),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["rules"] }),
  });
  return (
    <div className="flex flex-col gap-3">
      <Button
        variant="primary"
        className="self-start"
        onClick={() => setEditing({ ...EMPTY_RULE })}
      >
        <Plus />
        {t("integrations.newRule")}
      </Button>
      {rules.isLoading ? <Spinner /> : null}
      {rules.data && !rules.data.length ? (
        <p className="panel p-6 text-center text-[13px] text-muted">{t("integrations.noRules")}</p>
      ) : null}
      <ul className="flex flex-col gap-2">
        {rules.data?.map((r) => (
          <li key={r.id} className="panel flex flex-wrap items-center gap-3 p-3">
            <Switch
              checked={r.enabled}
              onCheckedChange={() => toggle.mutate(r)}
              aria-label={t("integrations.enableRule", { name: r.name })}
            />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <span className="font-medium">{r.name}</span>
                <SeverityChip severity={r.severity} />
              </div>
              <p className="truncate text-[13px] text-ink-2" title={summary(r)}>
                {summary(r)}
              </p>
            </div>
            <div className="text-right text-[12px] text-ink-2">
              <div>
                {r.last_fired_at
                  ? t("integrations.lastFired", { date: fmtDate(r.last_fired_at, true) })
                  : t("integrations.neverFired")}
              </div>
              <div className="font-mono">{t("integrations.fired24h", { count: r.fired_24h })}</div>
            </div>
            <Button
              size="icon"
              variant="ghost"
              aria-label={t("integrations.editRuleNamed", { name: r.name })}
              onClick={() => setEditing(r)}
            >
              <Pencil />
            </Button>
            <Button
              size="icon"
              variant="ghost"
              aria-label={t("integrations.deleteRuleNamed", { name: r.name })}
              onClick={() =>
                window.confirm(t("integrations.deleteRuleConfirm", { name: r.name })) &&
                remove.mutate(r.id)
              }
            >
              <Trash2 />
            </Button>
          </li>
        ))}
      </ul>
      {editing ? <RuleEditor rule={editing} onClose={() => setEditing(null)} /> : null}
    </div>
  );
}

function SeverityChip({ severity }: { severity: string }) {
  const { t } = useTranslation();
  const style =
    severity === "critical"
      ? "bg-reject-tint text-reject"
      : severity === "warning"
        ? "bg-review-tint text-review"
        : "bg-info-tint text-info";
  return (
    <span className={cn("rounded-[4px] px-1.5 text-[12px] font-medium", style)}>
      {t(`integrations.severity.${severity}`)}
    </span>
  );
}

function TargetEditor({
  target,
  onClose,
}: {
  target: Partial<Target> & { type: Target["type"] };
  onClose: () => void;
}) {
  const { t } = useTranslation();
  const client = useQueryClient();
  const meta = useMeta().data;
  const [name, setName] = useState(target.name ?? "");
  const [type, setType] = useState<Target["type"]>(target.type);
  const [config, setConfig] = useState<Record<string, unknown>>(() => ({
    ...(meta?.targets[target.type] ?? {}),
    ...(target.config ?? {}),
  }));
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const save = useMutation({
    mutationFn: () => {
      const body = { name, config, secrets };
      return target.id
        ? api.patch(`/api/targets/${target.id}`, body)
        : api.post("/api/targets", { ...body, type });
    },
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: ["targets"] });
      toast.success(t("integrations.targetSaved"));
      onClose();
    },
    onError: (e) => toast.error(errorText(t, e)),
  });
  if (!meta) return null;
  const fields = Object.keys(meta.targets[type]);
  return (
    <Dialog
      open
      onOpenChange={(o) => !o && onClose()}
      title={target.id ? t("integrations.editTarget") : t("integrations.newTarget")}
      description={t("integrations.secretsNote")}
      footer={
        <>
          <Button onClick={onClose}>{t("common.cancel")}</Button>
          <Button
            variant="primary"
            disabled={!name.trim() || save.isPending}
            onClick={() => save.mutate()}
          >
            {t("common.save")}
          </Button>
        </>
      }
    >
      <div className="grid gap-3">
        <Field label={t("integrations.name")} htmlFor="target-name">
          <input
            id="target-name"
            className="field-input font-sans"
            maxLength={60}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        {!target.id ? (
          <Field label={t("integrations.targetTypeLabel")}>
            <Select
              label={t("integrations.targetTypeLabel")}
              value={type}
              onValueChange={(v) => {
                setType(v as Target["type"]);
                setConfig({ ...meta.targets[v] });
              }}
              options={Object.keys(meta.targets).map((k) => ({
                value: k,
                label: t(`integrations.targetType.${k}`),
              }))}
            />
          </Field>
        ) : null}
        {fields.map((key) => {
          const value = config[key];
          const label = t(`integrations.cfg.${key}`);
          if (typeof value === "boolean")
            return (
              <label key={key} className="inline-flex items-center gap-1.5 text-[13px]">
                <Checkbox
                  checked={value}
                  onCheckedChange={(v) => setConfig({ ...config, [key]: v === true })}
                />
                {label}
              </label>
            );
          if (key === "security" || key === "format" || key === "rotation" || key === "method") {
            const options = {
              security: ["starttls", "ssl", "none"],
              format: ["jsonl", "csv"],
              rotation: ["daily", "size_mb"],
              method: ["POST", "PUT", "PATCH"],
            }[key];
            return (
              <Field key={key} label={label}>
                <Select
                  label={label}
                  value={String(value)}
                  onValueChange={(v) => setConfig({ ...config, [key]: v })}
                  options={options.map((o) => ({ value: o, label: o }))}
                  className="w-40"
                />
              </Field>
            );
          }
          if (typeof value === "number")
            return (
              <Field key={key} label={label}>
                <NumberInput
                  label={label}
                  value={value}
                  digits={0}
                  min={0}
                  onChange={(v) => setConfig({ ...config, [key]: v })}
                  className="w-28"
                />
              </Field>
            );
          if (typeof value === "object")
            return (
              <Field key={key} label={label} hint={t("integrations.headersHint")}>
                <textarea
                  className="field-input h-16 py-1.5"
                  defaultValue={JSON.stringify(value)}
                  aria-label={label}
                  onBlur={(e) => {
                    try {
                      setConfig({ ...config, [key]: JSON.parse(e.target.value || "{}") });
                    } catch {
                      toast.error(t("errors.request.invalid_json"));
                    }
                  }}
                />
              </Field>
            );
          return (
            <Field key={key} label={label} htmlFor={`cfg-${key}`}>
              <input
                id={`cfg-${key}`}
                className="field-input"
                value={String(value ?? "")}
                onChange={(e) => setConfig({ ...config, [key]: e.target.value })}
              />
            </Field>
          );
        })}
        {meta.target_secrets[type].map((key) => (
          <Field
            key={key}
            label={t(`integrations.cfg.${key}`)}
            htmlFor={`secret-${key}`}
            hint={target.secrets?.[key] ? t("integrations.secretStored") : undefined}
          >
            <input
              id={`secret-${key}`}
              type="password"
              autoComplete="new-password"
              className="field-input"
              placeholder={target.secrets?.[key] ? "••••" : ""}
              value={secrets[key] ?? ""}
              onChange={(e) => setSecrets({ ...secrets, [key]: e.target.value })}
            />
          </Field>
        ))}
      </div>
    </Dialog>
  );
}

function TargetsTab() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const targets = useTargets();
  const [editing, setEditing] = useState<(Partial<Target> & { type: Target["type"] }) | null>(null);
  const [testing, setTesting] = useState<number | null>(null);
  const test = async (target: Target) => {
    const to =
      target.type === "smtp"
        ? window.prompt(t("integrations.testRecipient"), String(target.config.from_address ?? ""))
        : null;
    setTesting(target.id);
    try {
      const result = await api.post<{ ok: boolean; detail: string }>(
        `/api/targets/${target.id}/test`,
        to ? { to: [to] } : {},
      );
      if (result.ok) toast.success(t("integrations.testOk"));
      else toast.error(t("integrations.testFailed", { detail: result.detail }));
    } finally {
      setTesting(null);
      void client.invalidateQueries({ queryKey: ["targets"] });
    }
  };
  const remove = useMutation({
    mutationFn: (id: number) => api.del(`/api/targets/${id}`),
    onSuccess: () => void client.invalidateQueries({ queryKey: ["targets"] }),
  });
  return (
    <div className="flex flex-col gap-3">
      <Button
        variant="primary"
        className="self-start"
        onClick={() => setEditing({ type: "webhook" })}
      >
        <Plus />
        {t("integrations.newTarget")}
      </Button>
      {targets.data && !targets.data.length ? (
        <p className="panel p-6 text-center text-[13px] text-muted">
          {t("integrations.noTargets")}
        </p>
      ) : null}
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {targets.data?.map((target) => {
          const Icon = TARGET_ICON[target.type];
          return (
            <section
              key={target.id}
              className="panel flex flex-col gap-2 p-3"
              aria-label={target.name}
            >
              <div className="flex items-center gap-2">
                <Icon className="size-5 text-ink-2" aria-hidden="true" />
                <span className="font-medium">{target.name}</span>
                <span className="text-[12px] text-muted">
                  {t(`integrations.targetType.${target.type}`)}
                </span>
              </div>
              <p className="truncate font-mono text-[12px] text-ink-2">
                {String(
                  target.config.host ?? target.config.base_url ?? target.config.directory ?? "",
                )}
              </p>
              <p
                className={cn(
                  "text-[12px]",
                  target.last_test_ok === null
                    ? "text-muted"
                    : target.last_test_ok
                      ? "text-pass"
                      : "text-reject",
                )}
              >
                {target.last_test_at
                  ? `${target.last_test_ok ? t("integrations.lastTestOk") : t("integrations.lastTestFailed")} · ${fmtDate(target.last_test_at, true)} · ${target.last_test_detail ?? ""}`
                  : t("integrations.notTested")}
              </p>
              <div className="mt-auto flex gap-1.5">
                <Button
                  size="sm"
                  disabled={testing === target.id}
                  onClick={() => void test(target)}
                >
                  {testing === target.id ? <Spinner /> : <Send />}
                  {t("integrations.test")}
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={t("integrations.editTargetNamed", { name: target.name })}
                  onClick={() => setEditing(target)}
                >
                  <Pencil />
                </Button>
                <Button
                  size="icon"
                  variant="ghost"
                  aria-label={t("integrations.deleteTargetNamed", { name: target.name })}
                  onClick={() =>
                    window.confirm(t("integrations.deleteTargetConfirm", { name: target.name })) &&
                    remove.mutate(target.id)
                  }
                >
                  <Trash2 />
                </Button>
              </div>
            </section>
          );
        })}
      </div>
      {editing ? <TargetEditor target={editing} onClose={() => setEditing(null)} /> : null}
    </div>
  );
}

const STATUSES = [
  "success",
  "failed",
  "queued",
  "running",
  "skipped_cooldown",
  "dropped_queue_full",
];

function StatusChip({ status }: { status: string }) {
  const { t } = useTranslation();
  const style =
    status === "success"
      ? "bg-pass-tint text-pass"
      : status === "failed" || status === "dropped_queue_full"
        ? "bg-reject-tint text-reject"
        : status === "skipped_cooldown"
          ? "bg-sunken text-ink-2"
          : "bg-info-tint text-info";
  return (
    <span className={cn("rounded-[4px] px-1.5 py-px text-[12px] font-medium", style)}>
      {t(`integrations.status.${status}`)}
    </span>
  );
}

function LogTab() {
  const { t } = useTranslation();
  const rules = useQuery({ queryKey: ["rules"], queryFn: () => api.get<RuleRow[]>("/api/rules") });
  const targets = useTargets();
  const [rule, setRule] = useState("all");
  const [target, setTarget] = useState("all");
  const [status, setStatus] = useState("all");
  const [open, setOpen] = useState<number | null>(null);
  const query = useMemo(() => {
    const params = new URLSearchParams({ limit: "100" });
    if (rule !== "all") params.set("rule", rule);
    if (target !== "all") params.set("target", target);
    if (status !== "all") params.set("status", status);
    return params.toString();
  }, [rule, target, status]);
  const log = useQuery({
    queryKey: ["executions", query],
    queryFn: () => api.get<{ total: number; items: Execution[] }>(`/api/executions?${query}`),
    refetchInterval: 5000,
  });
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap gap-2">
        <Select
          label={t("integrations.rule")}
          value={rule}
          onValueChange={setRule}
          options={[
            { value: "all", label: t("integrations.allRules") },
            ...(rules.data ?? []).map((r) => ({ value: String(r.id), label: r.name })),
          ]}
          className="w-56"
        />
        <Select
          label={t("integrations.target")}
          value={target}
          onValueChange={setTarget}
          options={[
            { value: "all", label: t("integrations.allTargets") },
            ...(targets.data ?? []).map((x) => ({ value: String(x.id), label: x.name })),
          ]}
          className="w-56"
        />
        <Select
          label={t("integrations.statusLabel")}
          value={status}
          onValueChange={setStatus}
          options={[
            { value: "all", label: t("integrations.allStatuses") },
            ...STATUSES.map((s) => ({ value: s, label: t(`integrations.status.${s}`) })),
          ]}
          className="w-56"
        />
        <span className="self-center text-[12px] text-muted">
          {t("integrations.logCount", { count: log.data?.total ?? 0 })}
        </span>
      </div>
      <div className="panel overflow-x-auto">
        <table className="w-full text-[13px]">
          <thead>
            <tr className="border-b border-line text-left text-[12px] text-ink-2">
              <th className="px-2 py-1.5" />
              <th className="px-2 py-1.5 font-medium">{t("integrations.col.time")}</th>
              <th className="px-2 py-1.5 font-medium">{t("integrations.rule")}</th>
              <th className="px-2 py-1.5 font-medium">{t("integrations.col.action")}</th>
              <th className="px-2 py-1.5 font-medium">{t("integrations.statusLabel")}</th>
              <th className="px-2 py-1.5 text-right font-medium">
                {t("integrations.col.attempts")}
              </th>
              <th className="px-2 py-1.5 text-right font-medium">HTTP</th>
              <th className="px-2 py-1.5 text-right font-medium">
                {t("integrations.col.duration")}
              </th>
            </tr>
          </thead>
          <tbody>
            {log.data?.items.map((e) => (
              <ExecutionRow
                key={e.id}
                execution={e}
                open={open === e.id}
                onToggle={() => setOpen(open === e.id ? null : e.id)}
              />
            ))}
          </tbody>
        </table>
        {log.data && !log.data.items.length ? (
          <p className="py-6 text-center text-[13px] text-muted">{t("integrations.noLog")}</p>
        ) : null}
      </div>
    </div>
  );
}

function ExecutionRow({
  execution: e,
  open,
  onToggle,
}: {
  execution: Execution;
  open: boolean;
  onToggle: () => void;
}): ReactNode {
  const { t } = useTranslation();
  return (
    <>
      <tr className="cursor-pointer border-b border-line hover:bg-sunken" onClick={onToggle}>
        <td className="px-2 py-1">
          <button
            type="button"
            aria-expanded={open}
            aria-label={t("integrations.details")}
            className="text-muted"
          >
            <ChevronRight className={cn("size-4 transition-transform", open && "rotate-90")} />
          </button>
        </td>
        <td className="whitespace-nowrap px-2 py-1">{fmtDate(e.created_at, true)}</td>
        <td className="px-2 py-1">
          {e.rule ?? "—"}
          {e.is_test ? (
            <span className="ml-1.5 rounded-[4px] bg-sunken px-1 text-[11px] text-ink-2">
              {t("integrations.testRun")}
            </span>
          ) : null}
        </td>
        <td className="px-2 py-1">
          {e.action_type ? t(`integrations.action.${e.action_type}`) : "—"}
          {e.target ? <span className="text-muted"> · {e.target}</span> : null}
        </td>
        <td className="px-2 py-1">
          <StatusChip status={e.status} />
          {e.status === "skipped_cooldown" ? (
            <span className="ml-1.5 font-mono text-[12px] text-muted">
              ×{fmtInt(e.suppressed_count)}
            </span>
          ) : null}
        </td>
        <td className="px-2 py-1 text-right font-mono">{e.attempt}</td>
        <td className="px-2 py-1 text-right font-mono">{e.http_status ?? "—"}</td>
        <td className="px-2 py-1 text-right font-mono">{fmtMs(e.duration_ms)}</td>
      </tr>
      {open ? (
        <tr className="border-b border-line bg-sunken/60">
          <td colSpan={8} className="px-4 py-2">
            {e.error ? <p className="mb-2 text-[12px] text-reject">{e.error}</p> : null}
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <p className="mb-1 text-[12px] font-medium text-ink-2">
                  {t("integrations.payload")}
                </p>
                <pre className="max-h-60 overflow-auto whitespace-pre-wrap rounded-[6px] bg-panel p-2 font-mono text-[11px]">
                  {e.rendered_payload || "—"}
                </pre>
              </div>
              <div>
                <p className="mb-1 text-[12px] font-medium text-ink-2">
                  {t("integrations.response")}
                </p>
                <pre className="max-h-60 overflow-auto whitespace-pre-wrap rounded-[6px] bg-panel p-2 font-mono text-[11px]">
                  {e.response_snippet || "—"}
                </pre>
              </div>
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}

export default function IntegrationsPage() {
  const { t } = useTranslation();
  const [params, setParams] = useSearchParams();
  const tab = params.get("tab") ?? "rules";
  return (
    <div className="p-4">
      <Tabs value={tab} onValueChange={(v) => setParams({ tab: v }, { replace: true })}>
        <TabsList className="mb-4">
          <TabsTrigger value="rules">{t("integrations.tab.rules")}</TabsTrigger>
          <TabsTrigger value="targets">{t("integrations.tab.targets")}</TabsTrigger>
          <TabsTrigger value="log">{t("integrations.tab.log")}</TabsTrigger>
        </TabsList>
        <TabsContent value="rules">
          <RulesTab />
        </TabsContent>
        <TabsContent value="targets">
          <TargetsTab />
        </TabsContent>
        <TabsContent value="log">
          <LogTab />
        </TabsContent>
      </Tabs>
    </div>
  );
}
