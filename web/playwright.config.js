import { defineConfig } from "@playwright/test";

const PORT = 4174;
const LIVE_URL = process.env.SINHALALM_E2E_URL;
export const BASE_URL = LIVE_URL ?? `http://127.0.0.1:${PORT}`;
export const DOWNLOAD_MINUTES = LIVE_URL ? 60 : 15;

export default defineConfig({
  testDir: "e2e",
  timeout: (DOWNLOAD_MINUTES + 15) * 60 * 1000,
  expect: { timeout: 30 * 1000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: BASE_URL },
  webServer: LIVE_URL
    ? undefined
    : {
        command: `npx vite build && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
        url: BASE_URL,
        reuseExistingServer: false,
        timeout: 120 * 1000,
      },
});
