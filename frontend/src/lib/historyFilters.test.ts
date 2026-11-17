import { describe, expect, it } from "vitest";

import { hasActiveFilters, historyQuery, presetFromParams, viewFromParams } from "./historyFilters";
import { qs } from "./api";

const params = (text: string) => new URLSearchParams(text);

describe("historyFilters", () => {
  it("defaults to the 7 day preset and the dashboard view", () => {
    expect(presetFromParams(params(""))).toBe("7d");
    expect(viewFromParams(params(""))).toBe("dashboard");
    expect(viewFromParams(params("view=cards"))).toBe("cards");
  });

  it("a custom start replaces the default preset", () => {
    expect(presetFromParams(params("start=2030-01-01T00:00:00Z"))).toBe("");
    expect(presetFromParams(params("preset=all"))).toBe("all");
  });

  it("converts the URL to API parameters", () => {
    const query = historyQuery(
      params(
        "view=cards&class=defect_no_cap&class=defect_debris&truth=bottle_no_defect&wrong=1&disagree=1&q=%20beta%20&source=dataset",
      ),
    );
    expect(qs(query)).toBe(
      "?preset=7d&only_wrong=true&disagree=true&name=beta&source=dataset" +
        "&class=defect_no_cap&class=defect_debris&truth=bottle_no_defect",
    );
  });

  it("detects active filters but ignores view and preset", () => {
    expect(hasActiveFilters(params("view=cards&preset=30d"))).toBe(false);
    expect(hasActiveFilters(params("wrong=1"))).toBe(true);
    expect(hasActiveFilters(params("class=defect_debris"))).toBe(true);
  });
});
