import { defineConfig } from "@playwright/test";

// End-to-end test against the real app (uses the models trained by `make quick` / `make run`).
// The app runs on a fresh data folder so the test never touches webdata/.
const PORT = 8790;

export default defineConfig({
  testDir: "e2e",
  timeout: 120_000,
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: `http://127.0.0.1:${PORT}`, viewport: { width: 1500, height: 950 } },
  webServer: {
    command: `rm -rf .e2e-data && .venv/bin/python -m bottle_inspection_web --data .e2e-data --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  },
});
