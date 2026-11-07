import { defineConfig } from "@playwright/test";

const PORT = 4174;
export const BASE_URL = `http://127.0.0.1:${PORT}`;

export default defineConfig({
  testDir: "e2e",
  timeout: 30 * 60 * 1000,
  expect: { timeout: 30 * 1000 },
  workers: 1,
  reporter: [["list"]],
  use: { baseURL: BASE_URL },
  webServer: {
    command: `npx vite build && npx vite preview --host 127.0.0.1 --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: false,
    timeout: 120 * 1000,
  },
});
