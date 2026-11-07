import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { AutoModelForCausalLM, Tensor, env } from "@huggingface/transformers";
import { describe, expect, it } from "vitest";

const expectedPath = process.env.SINHALALM_EXPECTED_GREEDY;
const modelsPath = fileURLToPath(new URL("../public/models/", import.meta.url));
const golden = JSON.parse(
  readFileSync(new URL("../../golden/reference.json", import.meta.url), "utf8"),
);
const STOP_IDS = [2, 0];

describe.skipIf(!expectedPath)("packaged model parity with Python", () => {
  it("generates the same greedy answers in transformers.js", { timeout: 600_000 }, async () => {
    const expected = JSON.parse(readFileSync(expectedPath, "utf8"));
    env.allowRemoteModels = false;
    env.localModelPath = modelsPath;
    const model = await AutoModelForCausalLM.from_pretrained(expected.model, {
      dtype: "q4",
      device: "cpu",
    });
    const byKey = Object.fromEntries(golden.cases.map((item) => [item.key, item]));
    for (const { key, ids } of expected.cases) {
      const prompt = byKey[key].input_ids;
      const inputIds = new Tensor("int64", BigInt64Array.from(prompt, BigInt), [1, prompt.length]);
      const mask = new Tensor("int64", new BigInt64Array(prompt.length).fill(1n), [
        1,
        prompt.length,
      ]);
      const output = await model.generate({
        input_ids: inputIds,
        attention_mask: mask,
        max_new_tokens: golden.max_new_tokens,
        do_sample: false,
        eos_token_id: STOP_IDS,
      });
      const generated = output.tolist()[0].slice(prompt.length).map(Number);
      expect(generated, key).toEqual(ids);
    }
    await model.dispose();
  });
});
