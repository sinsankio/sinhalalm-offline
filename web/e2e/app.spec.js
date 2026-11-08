import {
  EXPECTED_GREEDY,
  MAX_NEW_TOKENS,
  ask,
  expect,
  expectedAnswers,
  setSlider,
  status,
  test,
} from "./fixtures.js";
import { DOWNLOAD_MINUTES } from "../playwright.config.js";

const READY = /Ready on/;
const LOAD_TIMEOUT = { timeout: DOWNLOAD_MINUTES * 60 * 1000 };

test.describe.configure({ mode: "serial" });
test.skip(!EXPECTED_GREEDY, "SINHALALM_EXPECTED_GREEDY is not set");

let cases;

test.beforeAll(async () => {
  cases = await expectedAnswers();
});

test("downloads the model with progress", async ({ app: { page } }) => {
  await page.goto("/");
  await page.evaluate(() => navigator.serviceWorker.ready);
  await expect(status(page)).toHaveText(/not on this device/);
  await page.getByRole("button", { name: /Download model/ }).click();
  await expect(page.locator("progress.progress")).toBeVisible();
  await expect(status(page)).toHaveText(READY, LOAD_TIMEOUT);
  await expect(page.locator(".status-facts")).toContainText("Isolatedyes");
});

test("gate 4: greedy answers match ONNX Runtime in Python", async ({ app: { page } }) => {
  await setSlider(page, "Temperature", 0);
  await setSlider(page, "Repetition penalty", 1);
  await setSlider(page, "Maximum answer length", MAX_NEW_TOKENS);
  const toggles = [
    page.getByRole("checkbox", { name: /joiner in questions/ }),
    page.getByRole("checkbox", { name: /joiner in answers/ }),
  ];
  for (const toggle of toggles) await toggle.uncheck();
  for (const { key, question, answer } of cases) {
    expect(await ask(page, question), key).toBe(answer);
  }
  for (const toggle of toggles) await toggle.check();
});

test("streams an answer and stops it", async ({ app: { page } }) => {
  await setSlider(page, "Temperature", 0.7);
  await page.locator("#question").fill(cases[2].question);
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.locator(".answer-text")).toHaveText(/\S/, { timeout: 120 * 1000 });
  await expect(page.locator(".answer")).toHaveAttribute("aria-busy", "true");
  await page.getByRole("button", { name: "Stop" }).click();
  await expect(page.locator(".answer")).toContainText("Stopped.");
});

test("rejects a question longer than the context", async ({ app: { page } }) => {
  await page.locator("#question").fill("ගස් ".repeat(1200));
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.locator(".answer-error")).toContainText("too long");
});

test("keeps settings and the model after a reload", async ({ app: { page } }) => {
  await setSlider(page, "Temperature", 0);
  await page.reload();
  await expect(status(page)).toHaveText(READY, LOAD_TIMEOUT);
  await expect(page.locator(".status-facts")).toContainText("this device");
  await expect(page.locator(".setting", { hasText: "Temperature" }).locator("output")).toHaveText(
    "0",
  );
});

test("answers offline", async ({ app: { context, page } }) => {
  await context.setOffline(true);
  await page.reload();
  await expect(status(page)).toHaveText(READY, LOAD_TIMEOUT);
  expect(await ask(page, cases[0].question)).toBe(cases[0].answer);
});

test("deletes the model and needs a connection to download it again", async ({
  app: { context, page },
}) => {
  await page.getByRole("button", { name: "Delete downloaded model" }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(status(page)).toHaveText(/not on this device/);
  const download = page.getByRole("button", { name: /Download model/ });
  await expect(download).toBeDisabled();
  await expect(page.locator(".status-detail")).toContainText("You are offline");
  expect(await page.evaluate(() => caches.keys())).not.toContain("transformers-cache");
  await context.setOffline(false);
  await expect(download).toBeEnabled();
});
