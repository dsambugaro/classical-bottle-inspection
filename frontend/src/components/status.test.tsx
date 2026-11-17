import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import "@/lib/i18n";
import type { AnalysisResult } from "@/lib/types";
import { CLASS_NAMES } from "@/lib/types";

import { ScoreBars } from "./ResultCard";
import { VerdictBanner, VerdictChip } from "./status";

describe("verdict indicators", () => {
  it("never use colour alone: text label and class identifier are shown", () => {
    render(<VerdictBanner verdict="reject" predictedClass="defect_no_cap" />);
    expect(screen.getByText("REJECT")).toBeInTheDocument();
    expect(screen.getByText("defect_no_cap")).toBeInTheDocument();
    render(<VerdictChip verdict="review" />);
    expect(screen.getByText("REVIEW")).toBeInTheDocument();
  });
});

describe("score bars", () => {
  it("list all 7 classes in id order", () => {
    const scores = Object.fromEntries(
      CLASS_NAMES.map((c, i) => [c, i]),
    ) as AnalysisResult["scores"];
    const result = {
      predicted_class: "defect_label_damage",
      score_kind: "svm_ovr_score",
      scores,
      details: {},
    } as unknown as AnalysisResult;
    const { container } = render(<ScoreBars result={result} />);
    const names = [...container.querySelectorAll("li .font-mono")]
      .map((e) => e.textContent ?? "")
      .filter((text) => (CLASS_NAMES as readonly string[]).includes(text));
    expect(names).toEqual([...CLASS_NAMES]);
  });
});
