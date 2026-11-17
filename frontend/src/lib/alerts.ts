// Rule alerts in the UI: sounds (synthesized with WebAudio, works offline) and persistent
// toasts for critical rules (§15.3).
import { useTranslation } from "react-i18next";
import { toast } from "sonner";

import { useServerEvents } from "./queries";

type Sound = "chime" | "alarm" | "beep";

const PATTERNS: Record<
  Sound,
  { freq: number; start: number; length: number; type: OscillatorType }[]
> = {
  chime: [
    { freq: 880, start: 0, length: 0.18, type: "sine" },
    { freq: 1318.5, start: 0.16, length: 0.3, type: "sine" },
  ],
  beep: [{ freq: 1000, start: 0, length: 0.15, type: "square" }],
  alarm: [0, 0.3, 0.6].flatMap((start) => [
    { freq: 960, start, length: 0.14, type: "sawtooth" as OscillatorType },
    { freq: 720, start: start + 0.15, length: 0.14, type: "sawtooth" as OscillatorType },
  ]),
};

let context: AudioContext | null = null;

export function playSound(sound: Sound): void {
  try {
    context ??= new AudioContext();
    const now = context.currentTime;
    for (const tone of PATTERNS[sound] ?? PATTERNS.chime) {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = tone.type;
      oscillator.frequency.value = tone.freq;
      gain.gain.setValueAtTime(0.0001, now + tone.start);
      gain.gain.exponentialRampToValueAtTime(0.2, now + tone.start + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + tone.start + tone.length);
      oscillator.connect(gain).connect(context.destination);
      oscillator.start(now + tone.start);
      oscillator.stop(now + tone.start + tone.length + 0.02);
    }
  } catch {
    // audio unavailable (no user gesture yet, or no output device)
  }
}

export function useRuleAlerts(): void {
  const { t } = useTranslation();
  useServerEvents((event, data) => {
    const payload = data as {
      sound?: Sound;
      severity?: string;
      rule?: string;
      predicted_class?: string;
      level?: string;
      title?: string;
      message?: string;
    };
    if (event === "rule.sound" && payload.sound) playSound(payload.sound);
    if (event === "rule.fired" && payload.severity === "critical") {
      toast.error(t("alerts.critical", { rule: payload.rule }), {
        description: payload.predicted_class,
        duration: Infinity,
      });
    }
    if (event === "notification.created" && payload.level === "error") {
      toast.error(payload.title ?? t("errors.unexpected"), { description: payload.message });
    }
  });
}
