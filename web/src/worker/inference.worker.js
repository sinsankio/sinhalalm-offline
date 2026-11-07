import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import { createEntityDecoder, decodeEntities } from "../text/entities.js";
import { buildMessages, generationOptions, generationStats, stopReason } from "./generation.js";
import { describeAdapter, isSoftwareAdapter, wasmThreads } from "./runtime.js";
import { createStopSignal } from "./stop-signal.js";

const MODEL_ID = "sinhalallm-1.7b/v2.1";
const MODEL_FILE = `/models/${MODEL_ID}/onnx/model_q4.onnx`;
const WARMUP_QUESTION = "ආයුබෝවන්";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/models/";
env.backends.onnx.wasm.wasmPaths = {
  mjs: "/ort/ort-wasm-simd-threaded.asyncify.mjs",
  wasm: "/ort/ort-wasm-simd-threaded.asyncify.wasm",
};

let tokenizer = null;
let model = null;
let runtime = null;
let generating = false;
let stopSignal = createStopSignal(null);

class StopOnSignal extends InterruptableStoppingCriteria {
  _call(inputIds) {
    return new Array(inputIds.length).fill(stopSignal.requested);
  }
}

const post = (type, payload = {}) => self.postMessage({ type, ...payload });

async function chooseDevice(preferred) {
  if (preferred !== "wasm" && self.navigator.gpu) {
    const adapter = await self.navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (adapter && !isSoftwareAdapter(adapter.info)) {
      return { device: "webgpu", adapter: describeAdapter(adapter.info) };
    }
    if (preferred === "webgpu") {
      throw new Error("No hardware WebGPU adapter is available in this browser.");
    }
    return { device: "wasm", adapter: adapter ? describeAdapter(adapter.info) : null };
  }
  return { device: "wasm", adapter: null };
}

async function isCached() {
  try {
    const cache = await caches.open(env.cacheKey);
    const keys = [MODEL_FILE, new URL(MODEL_FILE, self.location.origin).href];
    for (const key of keys) {
      if (await cache.match(key)) return true;
    }
  } catch {
    return false;
  }
  return false;
}

function onProgress(event) {
  if (event.status !== "progress" || !event.file?.endsWith(".onnx")) return;
  post("progress", {
    file: event.file,
    loaded: event.loaded,
    total: event.total,
    progress: event.total ? event.loaded / event.total : null,
  });
}

async function warmUp() {
  const inputs = tokenizer.apply_chat_template([{ role: "user", content: WARMUP_QUESTION }], {
    add_generation_prompt: true,
    return_dict: true,
  });
  await model.generate({ ...inputs, max_new_tokens: 2, do_sample: false });
}

async function load({ preferred = "auto" } = {}) {
  await unload();
  const started = performance.now();
  const fromCache = await isCached();
  const { device, adapter } = await chooseDevice(preferred);
  const threads = wasmThreads(self.navigator.hardwareConcurrency, self.crossOriginIsolated);
  env.backends.onnx.wasm.numThreads = threads;
  post("status", { state: "loading", device, fromCache });

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    device,
    dtype: "q4",
    session_options: {
      graphOptimizationLevel: "all",
      freeDimensionOverrides: { batch_size: 1 },
    },
    progress_callback: onProgress,
  });
  const loadedAt = performance.now();
  post("status", { state: "warming", device });
  await warmUp();

  runtime = {
    model: MODEL_ID,
    dtype: "q4",
    device,
    adapter,
    wasmThreads: device === "wasm" ? threads : null,
    crossOriginIsolated: self.crossOriginIsolated,
    fromCache,
    loadMs: Math.round(loadedAt - started),
    warmupMs: Math.round(performance.now() - loadedAt),
  };
  post("ready", { runtime });
}

function eosIds() {
  const ids = model.generation_config?.eos_token_id ?? [];
  return (Array.isArray(ids) ? ids : [ids]).map(Number);
}

async function generate({ id, question, settings }) {
  if (!model) throw new Error("The model is not loaded.");
  if (generating) throw new Error("An answer is already being generated.");
  generating = true;
  try {
    const inputs = tokenizer.apply_chat_template(buildMessages(question, settings), {
      add_generation_prompt: true,
      return_dict: true,
    });
    const promptTokens = inputs.input_ids.dims.at(-1);
    const options = generationOptions(settings, promptTokens);
    const decoder = createEntityDecoder();
    const startedMs = performance.now();
    let firstTokenMs = null;
    let newTokens = 0;
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      token_callback_function: () => {
        newTokens += 1;
        firstTokenMs ??= performance.now();
      },
      callback_function: (text) => {
        const chunk = decoder.push(text);
        if (chunk) post("chunk", { id, text: chunk });
      },
    });

    stopSignal.reset();
    post("generating", { id, promptTokens, maxNewTokens: options.max_new_tokens });
    const sequences = await model.generate({
      ...inputs,
      ...options,
      streamer,
      stopping_criteria: [new StopOnSignal()],
    });
    const endedMs = performance.now();
    const rest = decoder.flush();
    if (rest) post("chunk", { id, text: rest });

    const answerIds = sequences.slice(null, [promptTokens, null]);
    const text = decodeEntities(
      tokenizer.batch_decode(answerIds, { skip_special_tokens: true })[0],
    ).trim();
    const lastToken = sequences.data.at(-1);
    post("answer", {
      id,
      text,
      reason: stopReason({
        interrupted: stopSignal.requested,
        lastToken,
        eosIds: eosIds(),
        newTokens,
        maxNewTokens: options.max_new_tokens,
      }),
      stats: generationStats({ promptTokens, newTokens, startedMs, firstTokenMs, endedMs }),
    });
  } finally {
    generating = false;
  }
}

async function unload() {
  if (model) {
    await model.dispose();
    model = null;
    tokenizer = null;
    runtime = null;
    post("status", { state: "idle" });
  }
}

async function removeDownloaded() {
  await unload();
  const removed = await caches.delete(env.cacheKey);
  post("removed", { removed });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "init") stopSignal = createStopSignal(data.stopBuffer ?? null);
    else if (data.type === "probe") post("probe", { fromCache: await isCached() });
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopSignal.request();
    else if (data.type === "load") await load(data);
    else if (data.type === "unload") await unload();
    else if (data.type === "remove") await removeDownloaded();
  } catch (error) {
    const message = error?.message ?? String(error);
    if (data.type === "generate") post("answer-error", { id: data.id, message });
    else post("error", { message });
  }
});
