import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AutoTokenizer, env } from "@huggingface/transformers";
import { beforeAll, describe, expect, it } from "vitest";
import { createEntityDecoder, decodeEntities } from "../src/text/entities.js";
import { ZWJ_CLUSTERS, normalizeZwj } from "../src/text/zwj.js";

const ZWJ = "‍";
const goldenPath = (path) => fileURLToPath(new URL(`../../golden/${path}`, import.meta.url));
const edgeCases = Object.fromEntries(
  JSON.parse(readFileSync(goldenPath("tokenizer_cases.json"), "utf8")).map((item) => [
    item.name,
    item,
  ]),
);

let tokenizer;

beforeAll(async () => {
  env.allowRemoteModels = false;
  env.localModelPath = goldenPath("generated/tokenizer/");
  tokenizer = await AutoTokenizer.from_pretrained("sinhalallm");
});

const encode = (text) => tokenizer.encode(text, { add_special_tokens: false });
const join = (cluster) => cluster.slice(0, 2) + ZWJ + cluster.slice(2);

describe("normalizeZwj", () => {
  it.each([
    ["ශ්රී ලංකාව", "ශ්‍රී ලංකාව"],
    ["ප්රධාන", "ප්‍රධාන"],
    ["විද්යාව", "විද්‍යාව"],
    ["ක්රියාව", "ක්‍රියාව"],
  ])("adds the ZWJ to a learned cluster in %s", (input, expected) => {
    expect(normalizeZwj(input)).toBe(expected);
  });

  it.each([
    ["ව්‍යාපාරය", "ව්යාපාරය"],
    ["සත්‍ය", "සත්ය"],
    ["බ්‍රිතාන්‍ය", "බ්රිතාන්ය"],
  ])("removes the ZWJ from an unlearned cluster in %s", (input, expected) => {
    expect(normalizeZwj(input)).toBe(expected);
  });

  it.each([
    ["text that is already normalized", "ශ්‍රී ලංකාවේ ප්‍රධාන විද්‍යාලය"],
    ["a repaya without the ZWJ", "කාර්යය"],
    ["the zero-width non-joiner", "ක්‌ෂ"],
    ["a virama before another consonant", "සිංහලෙන් ලියන්න"],
    ["English and digits", "SinhalaLLM 1.7B"],
    ["empty text", ""],
  ])("leaves %s unchanged", (_, input) => {
    expect(normalizeZwj(input)).toBe(input);
  });

  it("is idempotent", () => {
    const text = "ශ්රී ව්‍යාපාරය විද්යාව සත්‍ය";
    expect(normalizeZwj(normalizeZwj(text))).toBe(normalizeZwj(text));
  });

  it.each(ZWJ_CLUSTERS)("joins %s as the vocabulary learned it", (cluster) => {
    expect(normalizeZwj(cluster)).toBe(join(cluster));
    expect(encode(join(cluster)).length).toBeLessThanOrEqual(encode(cluster).length);
  });

  it("tokenizes text without the ZWJ like text with it", () => {
    const repaired = normalizeZwj(edgeCases.missing_zwj.text);
    expect(repaired).toBe(edgeCases.zwj_conjuncts.text);
    expect(encode(repaired)).toEqual(edgeCases.zwj_conjuncts.ids);
  });
});

describe("decodeEntities", () => {
  it.each([
    ["&quot;සිංහල&quot;", '"සිංහල"'],
    ["A &amp; B", "A & B"],
    ["&lt;b&gt;", "<b>"],
    ["it&apos;s", "it's"],
    ["&#3523;&#x0DD2;", "සි"],
    ["&amp;quot;", "&quot;"],
  ])("decodes %s", (input, expected) => {
    expect(decodeEntities(input)).toBe(expected);
  });

  it.each([
    ["an unknown name", "&unknown;"],
    ["a missing semicolon", "&quot"],
    ["a surrogate code point", "&#xD800;"],
    ["a bare ampersand", "A & B"],
  ])("leaves %s unchanged", (_, input) => {
    expect(decodeEntities(input)).toBe(input);
  });
});

describe("createEntityDecoder", () => {
  const stream = (chunks) => {
    const decoder = createEntityDecoder();
    return chunks.map((chunk) => decoder.push(chunk)).join("") + decoder.flush();
  };

  it("decodes an entity split across chunks", () => {
    expect(stream(["එය &qu", "ot;ප්‍රමාණවත්&", "quot; වේ."])).toBe('එය "ප්‍රමාණවත්" වේ.');
  });

  it("holds back only a possible entity", () => {
    const decoder = createEntityDecoder();
    expect(decoder.push("සිංහල &am")).toBe("සිංහල ");
    expect(decoder.push("p; භාෂාව")).toBe("& භාෂාව");
  });

  it("releases an unfinished entity at the end", () => {
    expect(stream(["A &am"])).toBe("A &am");
  });

  it("keeps text without entities unchanged", () => {
    const chunks = ["ශ්‍රී ලංකාවේ ", "අගනුවර ", "ශ්‍රී ජයවර්ධනපුර කෝට්ටේ වේ."];
    expect(stream(chunks)).toBe(chunks.join(""));
  });
});
