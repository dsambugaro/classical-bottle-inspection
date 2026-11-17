import { afterEach, describe, expect, it } from "vitest";

import { fmtNumber, formatInputNumber, modelLabel, parseLocaleNumber } from "./format";
import i18n from "./i18n";

afterEach(async () => {
  await i18n.changeLanguage("en");
});

describe("locale-aware numbers (REQ-L2)", () => {
  it("parses the English decimal point", async () => {
    await i18n.changeLanguage("en");
    expect(parseLocaleNumber("0.15")).toBe(0.15);
    expect(parseLocaleNumber("1,234.5")).toBe(1234.5);
    expect(parseLocaleNumber("abc")).toBeNull();
    expect(formatInputNumber(0.15, 2)).toBe("0.15");
  });

  it("parses the pt-BR decimal comma", async () => {
    await i18n.changeLanguage("pt-BR");
    expect(parseLocaleNumber("0,15")).toBe(0.15);
    expect(parseLocaleNumber("1.234,5")).toBe(1234.5);
    expect(formatInputNumber(0.15, 2)).toBe("0,15");
    expect(fmtNumber(33.8, 1)).toBe("33,8");
  });
});

import { fillGaps } from "@/pages/ReportsPage";

describe("time series buckets", () => {
  it("fills empty buckets so time spacing is uniform", () => {
    const zero = { bottle_no_defect: 0 } as never;
    const filled = fillGaps({
      bucket: "minute",
      series: [
        { t: "2026-11-09T10:00:00Z", pass: 1, reject: 0, review: 0, rejects_by_class: zero },
        { t: "2026-11-09T10:03:00Z", pass: 2, reject: 1, review: 0, rejects_by_class: zero },
      ],
    });
    expect(filled.map((s) => s.pass)).toEqual([1, 0, 0, 2]);
  });
});

describe("model names", () => {
  it("joins the feature vector and the algorithm", () => {
    expect(modelLabel("rois_hog", "svm")).toBe("rois_hog · svm");
    expect(modelLabel(null, "svm")).toBe("svm");
  });
});
