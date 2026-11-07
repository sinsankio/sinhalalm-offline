import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { AutoModelForCausalLM, Tensor, env } from "@huggingface/transformers";
import * as ort from "onnxruntime-web";
import { describe, expect, it } from "vitest";

const expectedPath = process.env.SINHALALM_EXPECTED_GREEDY;
const modelsPath = fileURLToPath(new URL("../public/models/", import.meta.url));
const golden = JSON.parse(
  readFileSync(new URL("../../golden/reference.json", import.meta.url), "utf8"),
);
const STOP_IDS = [2, 0];
const ortDist = join(
  dirname(createRequire(import.meta.url).resolve("onnxruntime-web")),
  "..",
  "dist",
);

function argmaxLast(logits) {
  const [, length, vocab] = logits.dims;
  const last = logits.data.subarray((length - 1) * vocab, length * vocab);
  let best = 0;
  for (let i = 1; i < vocab; i += 1) if (last[i] > last[best]) best = i;
  return best;
}

describe.skipIf(!expectedPath)("packaged model parity with Python", () => {
  it("generates the same greedy answers in transformers.js", { timeout: 600_000 }, async () => {
    const expected = JSON.parse(readFileSync(expectedPath, "utf8"));
    env.allowRemoteModels = false;
    env.localModelPath = modelsPath;
    const model = await AutoModelForCausalLM.from_pretrained(expected.model, {
      dtype: "q4",
      device: "cpu",
      use_external_data_format: 1,
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

  it(
    "loads in the browser WASM runtime and predicts the first answer token",
    { timeout: 600_000 },
    async () => {
      const expected = JSON.parse(readFileSync(expectedPath, "utf8"));
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = {
        mjs: join(ortDist, "ort-wasm-simd-threaded.asyncify.mjs"),
        wasm: join(ortDist, "ort-wasm-simd-threaded.asyncify.wasm"),
      };
      const onnxDir = join(modelsPath, expected.model, "onnx");
      const session = await ort.InferenceSession.create(join(onnxDir, "model_q4.onnx"), {
        executionProviders: ["wasm"],
        externalData: [
          { path: "model_q4.onnx_data", data: readFileSync(join(onnxDir, "model_q4.onnx_data")) },
        ],
      });
      const pastNames = session.inputNames.filter((name) => name.startsWith("past_key_values"));
      const byKey = Object.fromEntries(golden.cases.map((item) => [item.key, item]));
      for (const { key, ids } of expected.cases) {
        const prompt = byKey[key].input_ids;
        const feeds = {
          input_ids: new ort.Tensor("int64", BigInt64Array.from(prompt, BigInt), [
            1,
            prompt.length,
          ]),
          attention_mask: new ort.Tensor("int64", new BigInt64Array(prompt.length).fill(1n), [
            1,
            prompt.length,
          ]),
          position_ids: new ort.Tensor(
            "int64",
            BigInt64Array.from(prompt, (_, index) => BigInt(index)),
            [1, prompt.length],
          ),
        };
        for (const name of pastNames) {
          feeds[name] = new ort.Tensor("float32", new Float32Array(0), [1, 32, 0, 64]);
        }
        const { logits } = await session.run(feeds, ["logits"]);
        expect(argmaxLast(logits), key).toBe(ids[0]);
      }
      await session.release();
    },
  );
});
