// Profile multi-select (REQ-I7): only ready/modified profiles are selectable, the selection is
// ordered (drag or Alt+arrow to reorder), remembered in local storage and capped (default 6).
import { GripVertical, Plus, X } from "lucide-react";
import { useCallback, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { ProfileStatusBadge } from "./status";
import { Button, Checkbox, Popover, PopoverContent, PopoverTrigger, cn } from "./ui/primitives";
import { modelLabel } from "@/lib/format";
import type { Profile } from "@/lib/types";

const STORAGE_KEY = "bottle-inspection.selected-profiles";

export function useSelectedProfiles(profiles: Profile[] | undefined, max: number) {
  const [ids, setIds] = useState<number[]>(() => {
    try {
      const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "[]");
      return Array.isArray(parsed) ? parsed.filter((v) => typeof v === "number") : [];
    } catch {
      return [];
    }
  });

  const usable = useMemo(
    () =>
      new Set(
        (profiles ?? []).filter((p) => p.status_detail.can_analyse && !p.deleted).map((p) => p.id),
      ),
    [profiles],
  );

  // Unknown / unusable profiles are dropped; default to the first usable one.
  const effective = useMemo(() => {
    if (!profiles) return ids;
    const kept = ids.filter((id) => usable.has(id)).slice(0, max);
    return kept.length === 0 && usable.size ? [[...usable][0]] : kept;
  }, [ids, profiles, usable, max]);

  const update = useCallback((next: number[]) => {
    setIds(next);
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch {
      // ignore
    }
  }, []);

  const selected = effective
    .map((id) => profiles?.find((p) => p.id === id))
    .filter((p): p is Profile => !!p);
  return { ids: effective, setIds: update, selected };
}

export function ProfilePicker({
  profiles,
  ids,
  onChange,
  max,
}: {
  profiles: Profile[];
  ids: number[];
  onChange: (ids: number[]) => void;
  max: number;
}) {
  const { t } = useTranslation();
  const [dragged, setDragged] = useState<number | null>(null);
  const active = profiles.filter((p) => !p.deleted);

  const toggle = (id: number) => {
    if (ids.includes(id)) onChange(ids.filter((x) => x !== id));
    else if (ids.length < max) onChange([...ids, id]);
  };

  const move = (id: number, delta: number) => {
    const index = ids.indexOf(id);
    const target = index + delta;
    if (index < 0 || target < 0 || target >= ids.length) return;
    const next = [...ids];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next);
  };

  const dropOn = (targetId: number) => {
    if (dragged === null || dragged === targetId) return;
    const next = ids.filter((x) => x !== dragged);
    next.splice(next.indexOf(targetId), 0, dragged);
    onChange(next);
  };

  return (
    <div
      className="flex min-w-0 flex-wrap items-center gap-1.5"
      role="group"
      aria-label={t("inspect.profiles")}
    >
      <span className="text-[13px] font-medium text-ink-2">{t("inspect.profiles")}</span>
      {ids.map((id, index) => {
        const profile = profiles.find((p) => p.id === id);
        if (!profile) return null;
        return (
          <span
            key={id}
            draggable
            tabIndex={0}
            onDragStart={() => setDragged(id)}
            onDragEnd={() => setDragged(null)}
            onDragOver={(e) => e.preventDefault()}
            onDrop={() => dropOn(id)}
            onKeyDown={(e) => {
              if (e.altKey && e.key === "ArrowLeft") move(id, -1);
              if (e.altKey && e.key === "ArrowRight") move(id, 1);
            }}
            aria-label={t("inspect.profileChip", { name: profile.name, position: index + 1 })}
            className={cn(
              "inline-flex h-7 items-center gap-1 rounded-[6px] border border-line bg-panel pl-1 pr-0.5 text-[13px]",
              dragged === id && "opacity-50",
            )}
          >
            <GripVertical className="size-3.5 cursor-grab text-muted" aria-hidden="true" />
            <span
              aria-hidden="true"
              className="h-3.5 w-1 rounded-full"
              style={{ background: profile.color }}
            />
            <span className="max-w-44 truncate font-medium">{profile.name}</span>
            <button
              type="button"
              onClick={() => toggle(id)}
              aria-label={t("inspect.removeProfile", { name: profile.name })}
              className="rounded p-0.5 text-muted hover:bg-sunken hover:text-ink"
            >
              <X className="size-3.5" />
            </button>
          </span>
        );
      })}
      <Popover>
        <PopoverTrigger asChild>
          <Button size="sm" variant="secondary" aria-label={t("inspect.addProfile")}>
            <Plus />
            {ids.length === 0 ? t("inspect.addProfile") : null}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-96 p-1">
          <p className="px-2 py-1.5 text-[12px] text-muted">{t("inspect.maxProfiles", { max })}</p>
          <ul>
            {active.map((profile) => {
              const disabled =
                !profile.status_detail.can_analyse ||
                (!ids.includes(profile.id) && ids.length >= max);
              return (
                <li key={profile.id}>
                  <label
                    className={cn(
                      "flex cursor-pointer items-center gap-2 rounded-[4px] px-2 py-1.5 hover:bg-sunken",
                      disabled && !ids.includes(profile.id) && "cursor-not-allowed opacity-60",
                    )}
                  >
                    <Checkbox
                      checked={ids.includes(profile.id)}
                      disabled={disabled && !ids.includes(profile.id)}
                      onCheckedChange={() => toggle(profile.id)}
                    />
                    <span
                      aria-hidden="true"
                      className="h-4 w-1 rounded-full"
                      style={{ background: profile.color }}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[13px] font-medium">{profile.name}</span>
                      <span className="block truncate font-mono text-[11px] text-muted">
                        {modelLabel(profile.model?.feature_vector, profile.algorithm)} ·{" "}
                        {profile.model?.folder ?? profile.model_ref}
                      </span>
                      {!profile.status_detail.can_analyse ? (
                        <span className="block text-[11px] text-ink-2">
                          {t(`profileStatus.reason.${profile.status}`)}
                        </span>
                      ) : null}
                    </span>
                    <ProfileStatusBadge status={profile.status} short />
                  </label>
                </li>
              );
            })}
          </ul>
        </PopoverContent>
      </Popover>
    </div>
  );
}
