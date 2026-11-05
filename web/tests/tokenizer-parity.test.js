import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { beforeAll, describe, expect, it } from "vitest";

const goldenPath = (path) => fileURLToPath(new URL(`../../golden/${path}`, import.meta.url));
const readText = (path) => readFileSync(goldenPath(path), "utf8");
const readJson = (path) => JSON.parse(readText(path));

const edgeCases = readJson("tokenizer_cases.json");
const questions = readJson("reference.json").cases;

let tokenizer;

beforeAll(async () => {
  env.allowRemoteModels = false;
  env.localModelPath = goldenPath("generated/tokenizer/");
  tokenizer = await AutoTokenizer.from_pretrained("sinhalallm");
});

const encode = (text) => tokenizer.encode(text, { add_special_tokens: false });

describe("gate 3: tokenizer parity with Python", () => {
  it("uses the corpus described by the manifest", () => {
    const manifest = readJson("tokenizer_manifest.json");
    const corpus = readText("generated/tokenizer_corpus.json");
    expect(createHash("sha256").update(corpus).digest("hex")).toBe(manifest.corpus_sha256);
  });

  it("encodes the 10,000-sentence corpus identically", () => {
    const { cases } = readJson("generated/tokenizer_corpus.json");
    const mismatches = cases
      .filter((item) => JSON.stringify(encode(item.text)) !== JSON.stringify(item.ids))
      .map((item) => `${item.source}: ${item.text}`);
    expect(cases).toHaveLength(10000);
    expect(mismatches).toEqual([]);
  });

  it.each(edgeCases.map((item) => [item.name, item]))("encodes %s identically", (_, item) => {
    expect(encode(item.text)).toEqual(item.ids);
  });

  it.each(questions.map((item) => [item.key, item]))("builds the chat prompt for %s", (_, item) => {
    const prompt = tokenizer.apply_chat_template([{ role: "user", content: item.question }], {
      add_generation_prompt: true,
      tokenize: false,
    });
    expect(prompt).toBe(item.prompt);
    expect(encode(prompt)).toEqual(item.input_ids);
  });
});
