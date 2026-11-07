import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { test as base, expect } from "@playwright/test";
import { BASE_URL } from "../playwright.config.js";
import { decodeEntities } from "../src/text/entities.js";

export const EXPECTED_GREEDY = process.env.SINHALALM_EXPECTED_GREEDY;
export const MAX_NEW_TOKENS = 32;

export async function expectedAnswers() {
  env.localModelPath = "public/models/";
  env.allowRemoteModels = false;
  const expected = JSON.parse(readFileSync(EXPECTED_GREEDY, "utf8"));
  const tokenizer = await AutoTokenizer.from_pretrained(expected.model);
  const { questions } = JSON.parse(readFileSync("../golden/prompts.json", "utf8"));
  return expected.cases.map((item, index) => ({
    key: item.key,
    question: questions[index],
    answer: decodeEntities(
      tokenizer.decode(item.ids.slice(0, MAX_NEW_TOKENS), { skip_special_tokens: true }),
    ).trim(),
  }));
}

export const test = base.extend({
  app: [
    async ({ playwright }, use) => {
      const profile = mkdtempSync(join(tmpdir(), "sinhalallm-e2e-"));
      const context = await playwright.chromium.launchPersistentContext(profile, {
        channel: "chrome",
        headless: true,
        baseURL: BASE_URL,
      });
      const page = context.pages()[0] ?? (await context.newPage());
      try {
        await use({ context, page });
      } finally {
        await context.close();
        rmSync(profile, { recursive: true, force: true });
      }
    },
    { scope: "worker" },
  ],
});

export { expect };

export function status(page) {
  return page.locator(".status-message");
}

export async function ask(page, question) {
  await page.locator("#question").fill(question);
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(page.locator(".answer")).toHaveAttribute("aria-busy", "false", {
    timeout: 10 * 60 * 1000,
  });
  return (await page.locator(".answer-text").innerText()).trim();
}

export async function setSlider(page, label, value) {
  await page
    .locator(".setting", { hasText: label })
    .locator('input[type="range"]')
    .fill(`${value}`);
}
