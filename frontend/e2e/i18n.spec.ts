import { readFileSync } from "node:fs";

import { expect, test } from "@playwright/test";

// Acceptance 2: in pt-BR no English UI string from en.json remains visible.
const flatten = (obj: object, prefix = ""): [string, string][] =>
  Object.entries(obj).flatMap(([k, v]) =>
    typeof v === "object" ? flatten(v, `${prefix}${k}.`) : [[`${prefix}${k}`, String(v)]],
  );
const en = new Map(flatten(JSON.parse(readFileSync("src/locales/en.json", "utf8"))));
const pt = new Map(flatten(JSON.parse(readFileSync("src/locales/pt-BR.json", "utf8"))));
// English sentences (at least two words, no placeholders) that differ in Portuguese.
const english = [...en.entries()]
  .filter(
    ([key, value]) =>
      pt.get(key) !== value && /\s/.test(value) && !value.includes("{") && value.length >= 8,
  )
  .map(([, value]) => value);

const IMAGE = "../../archive/images/val/damaged_label_0058_20260210_151601.jpg";
const PAGES = [
  "/",
  "/profiles",
  "/reports",
  "/integrations?tab=rules",
  "/integrations?tab=targets",
  "/integrations?tab=log",
  "/system",
];

for (const path of PAGES) {
  test(`no English text on ${path} in pt-BR`, async ({ page, request }) => {
    const profiles = await (await request.get("/api/profiles")).json();
    await page.addInitScript(
      (ids) => {
        localStorage.setItem("bottle-inspection.language", "pt-BR");
        localStorage.setItem("bottle-inspection.selected-profiles", JSON.stringify(ids));
      },
      profiles.slice(0, 3).map((p: { id: number }) => p.id),
    );
    await page.goto(path);
    await page.waitForLoadState("networkidle");
    if (path === "/") {
      await page.locator('input[type="file"]').first().setInputFiles(IMAGE);
      await page.waitForSelector("table", { timeout: 30_000 });
      for (const button of await page.getByRole("button", { name: /Pontuações|Tempos/ }).all())
        await button.click();
    }
    await page.waitForTimeout(600);
    const text = await page.locator("body").innerText();
    const found = english.filter((sentence) => text.includes(sentence));
    expect(found).toEqual([]);
  });
}
