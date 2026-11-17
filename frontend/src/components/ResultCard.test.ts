import { describe, expect, it } from "vitest";

import en from "@/locales/en.json";
import ptBR from "@/locales/pt-BR.json";
import type { Timings } from "@/lib/types";
import { STEP_COLORS, timingSteps } from "./ResultCard";

const REGIONS = ["cap", "label", "body", "auto_1", "auto_2", "auto_3", "bottle"];
const DESCRIPTORS = ["hog", "lbp", "color"];
const GROUPS = REGIONS.flatMap((region) => DESCRIPTORS.map((d) => `${region}_${d}`));
const BASE_STEPS = ["decode", "normalize", "foreground_mask", "localize", "crop_regions"];

describe("timing steps", () => {
  it("follows the pipeline: normalize, mask, box, region crops, groups, predict", () => {
    const timings = {
      decode: 1,
      normalize: 2,
      foreground_mask: 3,
      localize: 4,
      crop_regions: 5,
      features: { auto_1_hog: 6, auto_2_hog: 7 },
      predict: 8,
      total: 36,
    } as Timings;
    expect(timingSteps(timings)).toEqual([
      { step: "decode", ms: 1 },
      { step: "normalize", ms: 2 },
      { step: "foreground_mask", ms: 3 },
      { step: "localize", ms: 4 },
      { step: "crop_regions", ms: 5 },
      { step: "auto_1_hog", ms: 6 },
      { step: "auto_2_hog", ms: 7 },
      { step: "predict", ms: 8 },
    ]);
  });

  it("has a colour and a translated label for the 21 feature groups and the base steps", () => {
    for (const step of [...BASE_STEPS, ...GROUPS, "predict"]) {
      expect(STEP_COLORS[step], `colour of ${step}`).toBeTruthy();
      expect((en.steps as Record<string, string>)[step], `en ${step}`).toBeTruthy();
      expect((ptBR.steps as Record<string, string>)[step], `pt-BR ${step}`).toBeTruthy();
    }
    expect(STEP_COLORS).not.toHaveProperty("find_top");
  });
});
