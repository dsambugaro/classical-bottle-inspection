import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

// WCAG 2.2 AA automated checks of every screen (with real data on the Inspect screen).
const IMAGE = "../../archive/images/val/damaged_label_0058_20260210_151601.jpg";
const TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

const PAGES = [
  { path: "/", name: "Inspect" },
  { path: "/profiles", name: "Profiles" },
  { path: "/reports", name: "Reports" },
  { path: "/integrations?tab=rules", name: "Integrations" },
  { path: "/system", name: "System" },
];

for (const language of ["en", "pt-BR"]) {
  for (const { path, name } of PAGES) {
    test(`${name} has no WCAG A/AA violations (${language})`, async ({ page, request }) => {
      const profiles = await (await request.get("/api/profiles")).json();
      await page.addInitScript(
        ([lang, ids]) => {
          localStorage.setItem("bottle-inspection.language", lang as string);
          localStorage.setItem("bottle-inspection.selected-profiles", JSON.stringify(ids));
        },
        [language, profiles.slice(0, 3).map((p: { id: number }) => p.id)],
      );
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      if (path === "/") {
        await page.locator('input[type="file"]').first().setInputFiles(IMAGE);
        await page.waitForSelector("table", { timeout: 30_000 });
      }
      await page.waitForTimeout(800);
      const results = await new AxeBuilder({ page }).withTags(TAGS).analyze();
      const summary = results.violations.map(
        (v) => `${v.id} (${v.impact}): ${v.nodes.length}× ${v.nodes[0]?.target.join(" ")}`,
      );
      expect(summary).toEqual([]);
    });
  }
}
