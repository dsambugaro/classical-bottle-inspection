// Server state (TanStack Query) and the SSE event stream.
import { QueryClient, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { api } from "./api";
import type {
  AppNotification,
  Catalog,
  ModelDetail,
  ModelSummary,
  Profile,
  Settings,
} from "./types";

export const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 15_000, retry: 1, refetchOnWindowFocus: false } },
});

export const keys = {
  profiles: ["profiles"] as const,
  profile: (id: number) => ["profiles", id] as const,
  models: ["models"] as const,
  model: (ref: string) => ["models", ref] as const,
  catalog: ["catalog"] as const,
  settings: ["settings"] as const,
  notifications: ["notifications"] as const,
  health: ["health"] as const,
};

export const useProfiles = () =>
  useQuery({ queryKey: keys.profiles, queryFn: () => api.get<Profile[]>("/api/profiles") });

export const useProfile = (id: number | null) =>
  useQuery({
    queryKey: keys.profile(id ?? 0),
    queryFn: () => api.get<Profile>(`/api/profiles/${id}`),
    enabled: id !== null,
  });

export const useModels = () =>
  useQuery({ queryKey: keys.models, queryFn: () => api.get<ModelSummary[]>("/api/models") });

export const useModel = (ref: string | null) =>
  useQuery({
    queryKey: keys.model(ref ?? ""),
    queryFn: () => api.get<ModelDetail>(`/api/models/${ref}`),
    enabled: !!ref,
  });

export const useCatalog = () =>
  useQuery({
    queryKey: keys.catalog,
    queryFn: () => api.get<Catalog>("/api/params/catalog"),
    staleTime: Infinity,
  });

export const useSettings = () =>
  useQuery({ queryKey: keys.settings, queryFn: () => api.get<Settings>("/api/settings") });

export const useNotifications = () =>
  useQuery({
    queryKey: keys.notifications,
    queryFn: () =>
      api.get<{ items: AppNotification[]; unread: number }>("/api/notifications?limit=50"),
  });

export interface Health {
  status: string;
  version: string;
  versions: Record<string, string>;
  dataset_found: boolean;
  models: number;
  active_cameras: number;
  active_sessions: number;
  queues: { actions: number; actions_max: number };
}

export const useHealth = () =>
  useQuery({
    queryKey: keys.health,
    queryFn: () => api.get<Health>("/api/health"),
    refetchInterval: 10_000,
  });

type Handler = (event: string, data: unknown) => void;
const handlers = new Set<Handler>();
let source: EventSource | null = null;

const EVENTS = [
  "analysis.completed",
  "inspection.completed",
  "session.progress",
  "session.completed",
  "models.changed",
  "rule.fired",
  "action.finished",
  "notification.created",
  "camera.status",
];

function ensureSource(): void {
  if (source || typeof EventSource === "undefined") return;
  source = new EventSource("/api/events");
  for (const name of EVENTS) {
    source.addEventListener(name, (message) => {
      const data = JSON.parse((message as MessageEvent).data);
      handlers.forEach((handler) => handler(name, data));
    });
  }
}

/** Subscribes to the server-sent events for the lifetime of the component. */
export function useServerEvents(handler: Handler): void {
  const ref = useRef(handler);
  useEffect(() => {
    ref.current = handler;
  });
  useEffect(() => {
    ensureSource();
    const wrapped: Handler = (event, data) => ref.current(event, data);
    handlers.add(wrapped);
    return () => {
      handlers.delete(wrapped);
    };
  }, []);
}

/** Global cache invalidation on server events. */
export function useGlobalEventInvalidation(): void {
  const client = useQueryClient();
  useServerEvents((event) => {
    if (event === "models.changed") {
      void client.invalidateQueries({ queryKey: keys.models });
      void client.invalidateQueries({ queryKey: keys.profiles });
    }
    if (event === "notification.created" || event === "rule.fired") {
      void client.invalidateQueries({ queryKey: keys.notifications });
    }
  });
}
