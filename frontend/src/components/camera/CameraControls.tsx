// Camera sources (REQ-I3..I5): the browser camera (getUserMedia -> canvas JPEG -> WebSocket,
// one frame in flight) and server cameras (MJPEG preview, results via SSE). Modes: continuous
// (N fps), single (button or Space) and motion (the server analyses when a bottle arrives).
import { useQuery } from "@tanstack/react-query";
import { Camera, Play, Square } from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import { NumberInput } from "../params/ParamControls";
import { Button, Kbd, Select, cn } from "../ui/primitives";
import { ApiError, api } from "@/lib/api";
import { useServerEvents, useSettings } from "@/lib/queries";
import type { InspectionResult } from "@/lib/types";

export type CameraSource = "browser" | `camera:${number}`;
export type CameraMode = "continuous" | "single" | "motion";

export interface ServerCamera {
  id: number;
  name: string;
  kind: string;
  device_index: number | null;
  url: string | null;
  width: number | null;
  height: number | null;
  fps: number;
  enabled: boolean;
  active?: boolean;
  session_id?: number | null;
}

export interface CameraSession {
  serverCameras: ServerCamera[];
  supportsBrowser: boolean;
  isCamera: boolean;
  running: boolean;
  starting: boolean;
  error: string | null;
  preview: ReactNode | null;
  mode: CameraMode;
  setMode: (mode: CameraMode) => void;
  fps: number;
  setFps: (fps: number) => void;
  devices: MediaDeviceInfo[];
  deviceId: string;
  setDeviceId: (id: string) => void;
  analysed: number;
  start: () => void;
  trigger: () => void;
  stop: () => void;
}

const JPEG_QUALITY = 0.85;
const PERMISSION_TIMEOUT_MS = 15_000;

function mediaErrorKey(error: unknown): string {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "camera.errors.denied";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "camera.errors.notFound";
  if (name === "NotReadableError" || name === "AbortError") return "camera.errors.busy";
  if (name === "Timeout") return "camera.errors.noAnswer";
  return "camera.errors.unknown";
}

export function useCameraSession({
  source,
  profileIds,
  onInspection,
}: {
  source: string;
  profileIds: number[];
  onInspection: (inspection: InspectionResult) => void;
}): CameraSession {
  const { t } = useTranslation();
  const settings = useSettings().data;
  const cameras = useQuery({
    queryKey: ["cameras"],
    queryFn: () => api.get<ServerCamera[]>("/api/cameras"),
  });
  const [mode, setMode] = useState<CameraMode>("continuous");
  const [fpsState, setFps] = useState<number | null>(null);
  const fps = fpsState ?? settings?.default_fps ?? 3;
  const [running, setRunning] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [analysed, setAnalysed] = useState(0);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const sessionId = useRef<number | null>(null);
  const socket = useRef<WebSocket | null>(null);
  const inFlight = useRef(false);
  const timer = useRef<number | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const onInspectionRef = useRef(onInspection);
  useEffect(() => {
    onInspectionRef.current = onInspection;
  });

  const isBrowser = source === "browser";
  const serverCameraId = source.startsWith("camera:") ? Number(source.split(":")[1]) : null;
  const supportsBrowser =
    typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia;

  // Browser camera: ask for permission and show the live video as soon as it is selected.
  useEffect(() => {
    if (!isBrowser || !supportsBrowser) return;
    let cancelled = false;
    let opened: MediaStream | null = null;
    const timeout = window.setTimeout(() => {
      if (!opened && !cancelled) setError(t("camera.errors.noAnswer"));
    }, PERMISSION_TIMEOUT_MS);
    navigator.mediaDevices
      .getUserMedia({
        video: deviceId
          ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      .then(async (media) => {
        if (cancelled) {
          media.getTracks().forEach((track) => track.stop());
          return;
        }
        opened = media;
        setStream(media);
        setError(null);
        const all = await navigator.mediaDevices.enumerateDevices();
        setDevices(all.filter((d) => d.kind === "videoinput"));
      })
      .catch((e) => !cancelled && setError(t(mediaErrorKey(e))))
      .finally(() => window.clearTimeout(timeout));
    return () => {
      cancelled = true;
      window.clearTimeout(timeout);
      opened?.getTracks().forEach((track) => track.stop());
      setStream(null);
    };
  }, [isBrowser, supportsBrowser, deviceId, t]);

  // The <video> moves between the stage and the picture-in-picture slot, so the stream is
  // attached whenever an element mounts (a callback ref), not only when the stream changes.
  const attachVideo = useCallback(
    (element: HTMLVideoElement | null) => {
      videoRef.current = element;
      if (element && stream && element.srcObject !== stream) element.srcObject = stream;
    },
    [stream],
  );

  const receive = useCallback(async (inspectionId: number) => {
    const inspection = await api.get<InspectionResult>(`/api/inspections/${inspectionId}`);
    onInspectionRef.current(inspection);
  }, []);

  // Server-camera results arrive through SSE.
  useServerEvents((event, data) => {
    const payload = data as {
      session_id?: number;
      inspection_id?: number;
      status?: string;
      code?: string;
    };
    if (sessionId.current === null || payload.session_id !== sessionId.current) return;
    if (event === "inspection.completed" && serverCameraId !== null && payload.inspection_id) {
      setAnalysed((n) => n + 1);
      void receive(payload.inspection_id);
    }
    if (event === "camera.status" && payload.status === "error") {
      setError(t(`errors.${payload.code}`, { defaultValue: t("camera.errors.unknown") }));
      setRunning(false);
      sessionId.current = null;
    }
  });

  const sendFrame = useCallback(() => {
    const video = videoRef.current;
    const ws = socket.current;
    if (!video || !ws || ws.readyState !== WebSocket.OPEN || inFlight.current || !video.videoWidth)
      return;
    const c = canvas.current ?? (canvas.current = document.createElement("canvas"));
    c.width = video.videoWidth;
    c.height = video.videoHeight;
    c.getContext("2d")!.drawImage(video, 0, 0);
    inFlight.current = true;
    c.toBlob(
      (blob) => {
        if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        else inFlight.current = false;
      },
      "image/jpeg",
      JPEG_QUALITY,
    );
  }, []);

  const cleanup = useCallback(() => {
    if (timer.current !== null) window.clearInterval(timer.current);
    timer.current = null;
    socket.current?.close();
    socket.current = null;
    inFlight.current = false;
  }, []);

  const stop = useCallback(() => {
    const id = sessionId.current;
    cleanup();
    sessionId.current = null;
    setRunning(false);
    if (id !== null) void api.post(`/api/sessions/${id}/stop`).catch(() => undefined);
  }, [cleanup]);

  const start = useCallback(async () => {
    if (!profileIds.length) {
      setError(t("inspect.selectProfileFirst"));
      return;
    }
    setError(null);
    setStarting(true);
    setAnalysed(0);
    try {
      const session = await api.post<{ id: number }>("/api/sessions", {
        source: isBrowser ? "browser_camera" : "server_camera",
        camera_id: serverCameraId,
        profile_ids: profileIds,
        mode,
        fps,
      });
      sessionId.current = session.id;
      if (isBrowser) {
        const protocol = window.location.protocol === "https:" ? "wss" : "ws";
        const ws = new WebSocket(
          `${protocol}://${window.location.host}/ws/camera/browser?session_id=${session.id}`,
        );
        ws.binaryType = "arraybuffer";
        ws.onmessage = (message) => {
          inFlight.current = false;
          const data = JSON.parse(message.data as string);
          if (data.type === "inspection") {
            setAnalysed((n) => n + 1);
            onInspectionRef.current(data.inspection as InspectionResult);
          } else if (data.type === "error") {
            setError(t(`errors.${data.code}`, { defaultValue: t("camera.errors.unknown") }));
          }
        };
        ws.onclose = () => {
          if (socket.current === ws) {
            cleanup();
            sessionId.current = null;
            setRunning(false);
          }
        };
        await new Promise<void>((resolve, reject) => {
          ws.onopen = () => resolve();
          ws.onerror = () => reject(new Error("websocket"));
        });
        socket.current = ws;
        if (mode !== "single") timer.current = window.setInterval(sendFrame, 1000 / fps);
      }
      setRunning(true);
    } catch (e) {
      setError(
        e instanceof ApiError
          ? t(`errors.${e.code}`, { ...e.details, defaultValue: e.message })
          : t("camera.errors.unknown"),
      );
      stop();
    } finally {
      setStarting(false);
    }
  }, [profileIds, isBrowser, serverCameraId, mode, fps, sendFrame, cleanup, stop, t]);

  const trigger = useCallback(() => {
    if (!running) {
      void start();
      return;
    }
    if (mode !== "single") return;
    if (isBrowser) sendFrame();
    else if (sessionId.current !== null)
      void api.post(`/api/sessions/${sessionId.current}/trigger`);
  }, [running, mode, isBrowser, sendFrame, start]);

  // Leaving a camera source (or the page) stops the session.
  useEffect(() => () => stop(), [source, stop]);

  const preview: ReactNode | null = isBrowser ? (
    <video
      ref={attachVideo}
      autoPlay
      muted
      playsInline
      aria-label={t("camera.livePreview")}
      className="h-full max-h-full w-full rounded-[10px] bg-stage object-contain"
    />
  ) : serverCameraId !== null ? (
    <img
      src={`/api/cameras/${serverCameraId}/mjpeg`}
      alt={t("camera.livePreview")}
      className="h-full max-h-full w-full rounded-[10px] bg-stage object-contain"
    />
  ) : null;

  return {
    serverCameras: (cameras.data ?? []).filter((c) => c.enabled),
    supportsBrowser,
    isCamera: isBrowser || serverCameraId !== null,
    running,
    starting,
    error,
    preview,
    mode,
    setMode,
    fps,
    setFps,
    devices,
    deviceId,
    setDeviceId,
    analysed,
    start: () => void start(),
    trigger,
    stop,
  };
}

export function CameraSourceBar({ camera }: { camera: CameraSession }) {
  const { t } = useTranslation();
  const settings = useSettings().data;
  return (
    <div className="flex flex-wrap items-center gap-3">
      {camera.devices.length > 1 ? (
        <Select
          label={t("camera.device")}
          value={camera.deviceId || camera.devices[0].deviceId}
          onValueChange={camera.setDeviceId}
          disabled={camera.running}
          options={camera.devices.map((d, i) => ({
            value: d.deviceId,
            label: d.label || `${t("camera.device")} ${i + 1}`,
          }))}
          className="w-56"
        />
      ) : null}
      <div
        role="radiogroup"
        aria-label={t("camera.mode")}
        className="flex items-center gap-3 text-[13px]"
      >
        <span className="font-medium text-ink-2">{t("camera.mode")}</span>
        {(["single", "continuous", "motion"] as const).map((m) => (
          <label
            key={m}
            className={cn("inline-flex items-center gap-1.5", camera.running && "opacity-60")}
          >
            <input
              type="radio"
              name="camera-mode"
              checked={camera.mode === m}
              disabled={camera.running}
              onChange={() => camera.setMode(m)}
              className="accent-[#2457C5]"
            />
            {t(`camera.modes.${m}`)}
          </label>
        ))}
      </div>
      {camera.mode !== "single" ? (
        <label className="inline-flex items-center gap-1.5 text-[13px]">
          <NumberInput
            label={t("camera.fps")}
            value={camera.fps}
            digits={0}
            min={settings?.min_fps ?? 1}
            max={settings?.max_fps ?? 15}
            disabled={camera.running}
            onChange={(v) => camera.setFps(Math.round(v))}
            className="w-14"
          />
          fps
        </label>
      ) : null}
      {camera.running ? (
        <>
          {camera.mode === "single" ? (
            <Button variant="primary" onClick={camera.trigger}>
              <Camera />
              {t("camera.capture")} <Kbd>{t("keys.space")}</Kbd>
            </Button>
          ) : null}
          <Button variant="danger-outline" onClick={camera.stop}>
            <Square />
            {t("camera.stop")} <Kbd>Esc</Kbd>
          </Button>
          <span
            className="inline-flex items-center gap-1.5 text-[12px] font-medium text-pass"
            role="status"
          >
            <span
              className="size-2 animate-pulse rounded-full bg-pass motion-reduce:animate-none"
              aria-hidden="true"
            />
            {t("camera.running", { count: camera.analysed })}
          </span>
        </>
      ) : (
        <Button variant="primary" disabled={camera.starting} onClick={camera.start}>
          <Play />
          {t("camera.start")} <Kbd>{t("keys.space")}</Kbd>
        </Button>
      )}
      <span className="text-[12px] text-muted">{t(`camera.modeHelp.${camera.mode}`)}</span>
    </div>
  );
}
