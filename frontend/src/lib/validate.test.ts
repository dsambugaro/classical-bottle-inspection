import { describe, expect, it } from "vitest";

import type { CatalogEntry } from "./types";
import { validateAll, validateEntry } from "./validate";

const entry = (over: Partial<CatalogEntry>): CatalogEntry => ({
  key: "rois.cap",
  group: "rois",
  type: "range",
  mode: "editable",
  config_constant: "",
  min: 0,
  max: 1,
  step: 0.01,
  unit: "relative",
  affects_preview: true,
  options: null,
  constraints: [],
  nullable: false,
  label_key: "",
  help_key: "",
  default: [0.08, 0.19],
  ...over,
});

describe("client-side parameter validation", () => {
  it("checks ranges, order and types", () => {
    expect(validateEntry(entry({}), [0.1, 0.2])).toBeNull();
    expect(validateEntry(entry({}), [0.3, 0.2])?.code).toBe("param.range_order");
    expect(validateEntry(entry({}), [0.1, 1.2])?.code).toBe("param.above_max");
    expect(validateEntry(entry({ type: "int", min: 1, max: 765 }), 1.5)?.code).toBe(
      "param.invalid_type",
    );
  });

  it("checks every range of a list of ranges", () => {
    const bands = entry({ key: "auto_rois.bands", type: "ranges", default: [] });
    expect(validateEntry(bands, [])).toBeNull();
    expect(
      validateEntry(bands, [
        [0, 0.25],
        [0.25, 1],
      ]),
    ).toBeNull();
    expect(
      validateEntry(bands, [
        [0, 0.25],
        [0.6, 0.5],
      ])?.code,
    ).toBe("param.range_order");
    expect(validateEntry(bands, [[0, 1.5]])?.code).toBe("param.above_max");
  });

  it("skips locked and verdict entries", () => {
    const locked = entry({ key: "lbp.points", type: "int", mode: "locked", min: 1, max: 24 });
    expect(validateAll([locked, entry({})], { "lbp.points": 99, "rois.cap": [0.3, 0.2] })).toEqual({
      "rois.cap": { code: "param.range_order" },
    });
  });
});
