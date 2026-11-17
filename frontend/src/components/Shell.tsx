// App shell (§15.3): left rail, top bar (title, language, notifications, status chips).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Bell,
  Bug,
  ChartColumn,
  Cog,
  ScanLine,
  SlidersHorizontal,
  Video,
  Workflow,
} from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { NavLink, useLocation } from "react-router";

import { Button, Popover, PopoverContent, PopoverTrigger, cn } from "./ui/primitives";
import { api } from "@/lib/api";
import { fmtDate } from "@/lib/format";
import { rememberLanguage, type Language } from "@/lib/i18n";
import { keys, useHealth, useNotifications, useProfiles } from "@/lib/queries";

const NAV = [
  { to: "/", key: "inspect", icon: ScanLine },
  { to: "/profiles", key: "profiles", icon: SlidersHorizontal },
  { to: "/debug", key: "debug", icon: Bug },
  { to: "/reports", key: "reports", icon: ChartColumn },
  { to: "/integrations", key: "integrations", icon: Workflow },
  { to: "/system", key: "system", icon: Cog },
] as const;

function currentTitle(pathname: string): string {
  const match = [...NAV]
    .reverse()
    .find((item) => (item.to === "/" ? pathname === "/" : pathname.startsWith(item.to)));
  return match?.key ?? "inspect";
}

function LanguageSwitch() {
  const { i18n, t } = useTranslation();
  const client = useQueryClient();
  const change = (language: Language) => {
    void i18n.changeLanguage(language);
    rememberLanguage(language);
    void api
      .patch("/api/settings", { language })
      .then(() => client.invalidateQueries({ queryKey: keys.settings }));
  };
  return (
    <div
      role="group"
      aria-label={t("shell.language")}
      className="flex rounded-[6px] border border-line bg-panel p-0.5"
    >
      {(["en", "pt-BR"] as const).map((language) => (
        <button
          key={language}
          type="button"
          aria-pressed={i18n.language === language}
          onClick={() => change(language)}
          className={cn(
            "rounded-[4px] px-2 py-0.5 text-[12px] font-semibold",
            i18n.language === language ? "bg-ink text-white" : "text-ink-2 hover:text-ink",
          )}
        >
          {language === "en" ? "EN" : "PT-BR"}
        </button>
      ))}
    </div>
  );
}

function NotificationsBell() {
  const { t } = useTranslation();
  const { data } = useNotifications();
  const client = useQueryClient();
  const readAll = useMutation({
    mutationFn: () => api.post("/api/notifications/read-all"),
    onSuccess: () => client.invalidateQueries({ queryKey: keys.notifications }),
  });
  const unread = data?.unread ?? 0;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          aria-label={t("shell.notifications", { count: unread })}
          className="relative"
        >
          <Bell />
          {unread ? (
            <span className="absolute -right-0.5 -top-0.5 min-w-4 rounded-full bg-reject px-1 font-mono text-[10px] font-semibold leading-4 text-white">
              {unread > 99 ? "99+" : unread}
            </span>
          ) : null}
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-96 p-0">
        <div className="flex items-center justify-between border-b border-line px-3 py-2">
          <span className="font-semibold">{t("shell.notificationsTitle")}</span>
          <Button variant="ghost" size="sm" disabled={!unread} onClick={() => readAll.mutate()}>
            {t("shell.markAllRead")}
          </Button>
        </div>
        <ul className="max-h-96 overflow-auto">
          {(data?.items ?? []).length === 0 ? (
            <li className="px-3 py-6 text-center text-[13px] text-muted">
              {t("shell.noNotifications")}
            </li>
          ) : (
            data!.items.map((n) => (
              <li
                key={n.id}
                className={cn(
                  "border-b border-line px-3 py-2 last:border-0",
                  !n.read_at && "bg-info-tint/50",
                )}
              >
                <div className="flex items-center gap-2">
                  {n.level === "critical" || n.level === "error" ? (
                    <AlertTriangle className="size-4 text-reject" aria-hidden="true" />
                  ) : n.level === "warning" ? (
                    <AlertTriangle className="size-4 text-review" aria-hidden="true" />
                  ) : null}
                  <span className="text-[13px] font-medium">
                    {n.code
                      ? t(`notifications.${n.code}`, { ...n.params, defaultValue: n.title })
                      : n.title}
                  </span>
                </div>
                {n.message ? <p className="mt-0.5 text-[12px] text-ink-2">{n.message}</p> : null}
                <p className="mt-0.5 text-[11px] text-muted">{fmtDate(n.created_at, true)}</p>
              </li>
            ))
          )}
        </ul>
      </PopoverContent>
    </Popover>
  );
}

function StatusChips() {
  const { t } = useTranslation();
  const health = useHealth().data;
  const profiles = useProfiles().data ?? [];
  const broken = profiles.filter(
    (p) => p.status === "model_missing" || p.status === "incompatible",
  ).length;
  const queue = health?.queues.actions ?? 0;
  return (
    <div className="flex items-center gap-2">
      {health?.active_sessions ? (
        <span className="inline-flex items-center gap-1.5 rounded-[6px] bg-info-tint px-2 py-0.5 text-[12px] font-medium text-info">
          <Video className="size-3.5" aria-hidden="true" />
          {t("shell.activeSessions", { count: health.active_sessions })}
        </span>
      ) : null}
      {queue > 50 ? (
        <span className="rounded-[6px] bg-review-tint px-2 py-0.5 text-[12px] font-medium text-review">
          {t("shell.queueBacklog", { count: queue })}
        </span>
      ) : null}
      {broken ? (
        <NavLink
          to="/profiles"
          className="inline-flex items-center gap-1.5 rounded-[6px] bg-review-tint px-2 py-0.5 text-[12px] font-medium text-review"
        >
          <AlertTriangle className="size-3.5" aria-hidden="true" />
          {t("shell.brokenProfiles", { count: broken })}
        </NavLink>
      ) : null}
    </div>
  );
}

export function Shell({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const location = useLocation();
  const title = currentTitle(location.pathname);
  return (
    <div className="flex h-dvh min-h-0">
      <nav
        aria-label={t("shell.navigation")}
        className="group flex w-14 shrink-0 flex-col gap-1 bg-stage p-2 transition-[width] duration-200 hover:w-48 focus-within:w-48"
      >
        <div className="mb-3 flex h-10 items-center gap-2 px-1.5 text-white">
          <svg viewBox="0 0 32 32" className="size-7 shrink-0" aria-hidden="true">
            <path
              d="M13 5h6v4l2 3v14a2 2 0 0 1-2 2h-6a2 2 0 0 1-2-2V12l2-3z"
              fill="none"
              stroke="#F3F4F6"
              strokeWidth="2"
            />
            <path d="M11 17h10" stroke="#3fb15d" strokeWidth="3" />
          </svg>
          <span className="truncate text-[13px] font-semibold opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
            {t("app.name")}
          </span>
        </div>
        {NAV.map(({ to, key, icon: Icon }) => (
          <NavLink
            key={key}
            to={to}
            end={to === "/"}
            className={({ isActive }) =>
              cn(
                "flex h-10 items-center gap-3 overflow-hidden rounded-[6px] px-2.5 text-[13px] font-medium",
                isActive
                  ? "bg-white/12 text-white"
                  : "text-[#aab2bd] hover:bg-white/6 hover:text-white",
              )
            }
          >
            <Icon className="size-5 shrink-0" aria-hidden="true" />
            <span className="truncate opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100">
              {t(`nav.${key}`)}
            </span>
          </NavLink>
        ))}
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line bg-panel px-4">
          <h1 className="text-[16px] font-semibold">{t(`nav.${title}`)}</h1>
          <div className="ml-auto flex items-center gap-3">
            <StatusChips />
            <LanguageSwitch />
            <NotificationsBell />
          </div>
        </header>
        <main className="relative min-h-0 flex-1 overflow-auto">{children}</main>
      </div>
    </div>
  );
}
