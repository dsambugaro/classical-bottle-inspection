import { createServer, type IncomingMessage, type Server } from "node:http";

import { expect, test } from "@playwright/test";

const IMAGE = "../../archive/images/val/damaged_label_0058_20260210_151601.jpg";

let server: Server;
let received: { url: string; headers: IncomingMessage["headers"]; body: string }[] = [];
let hookUrl = "";

test.beforeAll(async () => {
  // Local webhook receiver.
  server = createServer((request, response) => {
    let body = "";
    request.on("data", (chunk) => (body += chunk));
    request.on("end", () => {
      received.push({ url: request.url ?? "", headers: request.headers, body });
      response.writeHead(200, { "content-type": "application/json" }).end("{}");
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  hookUrl = `http://127.0.0.1:${typeof address === "object" && address ? address.port : 0}`;
});

test.afterAll(() => server.close());

async function selectOption(
  page: import("@playwright/test").Page,
  label: string | RegExp,
  option: string | RegExp,
) {
  await page.getByRole("combobox", { name: label, exact: true }).first().click();
  await page.getByRole("option", { name: option }).first().click();
}

test("inspect an image, switch language, and deliver a webhook rule", async ({ page, request }) => {
  const profiles = await (await request.get("/api/profiles")).json();
  const svm = profiles.find((p: { name: string }) => p.name.startsWith("SVM rois_hog ("));
  expect(
    svm,
    "run `make quick` first: the rois_hog/svm model of the results is needed",
  ).toBeTruthy();
  await page.addInitScript((id) => {
    localStorage.setItem("bottle-inspection.language", "en");
    localStorage.setItem("bottle-inspection.selected-profiles", JSON.stringify([id]));
  }, svm.id);

  // 1. Upload a validation image: verdict, overlays and timings.
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Inspect" })).toBeVisible();
  await page.locator('input[type="file"]').first().setInputFiles(IMAGE);
  await expect(page.getByText("REJECT", { exact: true }).first()).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("defect_label_damage").first()).toBeVisible();
  for (const roi of ["cap", "label", "body"])
    await expect(page.locator("svg text", { hasText: roi })).toHaveCount(1);
  await page.getByRole("button", { name: /^Timings/ }).click();
  await expect(page.getByText("Normalize lighting")).toBeVisible();
  await expect(page.getByText("Body HOG")).toBeVisible();

  // 2. Switch to Portuguese and back.
  await page.getByRole("button", { name: "PT-BR", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Inspecionar" })).toBeVisible();
  await expect(page.getByText("REPROVADO", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Inspect" })).toBeVisible();

  // 3. Create a webhook target and a rule in the UI.
  await page.goto("/integrations?tab=targets");
  await page.getByRole("button", { name: "New target" }).click();
  await page.getByLabel("Name").fill("Local receiver");
  await page.getByLabel("Base URL").fill(hookUrl);
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByText("Local receiver")).toBeVisible();

  await page.goto("/integrations?tab=rules");
  await page.getByRole("button", { name: "New rule" }).click();
  await page.getByLabel("Name").fill("Any reject to MES");
  await selectOption(page, "Condition", "Any defect (reject)");
  await selectOption(page, "Action", "Webhook");
  await selectOption(page, "Target", "Local receiver");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.getByText("Rule saved.")).toBeVisible();
  await page.getByRole("button", { name: "Close", exact: true }).last().click();
  await expect(page.getByText("Any reject to MES")).toBeVisible();

  // 4. Inspect again: the rule fires and the delivery appears in the log.
  received = [];
  await page.goto("/");
  await page.locator('input[type="file"]').first().setInputFiles(IMAGE);
  await expect(page.getByText("REJECT", { exact: true }).first()).toBeVisible({ timeout: 30_000 });
  await expect.poll(() => received.length, { timeout: 15_000 }).toBeGreaterThan(0);
  expect(JSON.parse(received[0].body).predicted_class).toBe("defect_label_damage");
  await page.goto("/integrations?tab=log");
  const row = page
    .getByRole("row")
    .filter({ hasText: "Any reject to MES" })
    .filter({ hasText: "Webhook" });
  await expect(row.getByText("Sent", { exact: true })).toBeVisible();
  await expect(row.getByText("200")).toBeVisible();
});
