import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import { createEntityDecoder, decodeEntities } from "../text/entities.js";
import { buildMessages, generationOptions, generationStats, stopReason } from "./generation.js";
import { describeAdapter, gpuSupport, wasmThreads, webgpuUnavailableMessage } from "./runtime.js";
import { createStopSignal } from "./stop-signal.js";
import {
  DEFAULT_VARIANT,
  MODEL_ID,
  VARIANTS,
  combinedProgress,
  variantFileUrls,
} from "./variants.js";

const WARMUP_QUESTION = "ආයුබෝවන්";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/models/";
env.backends.onnx.wasm.wasmPaths = undefined;

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
  const adapter =
    preferred !== "wasm" && self.navigator.gpu
      ? await self.navigator.gpu.requestAdapter({ powerPreference: "high-performance" })
      : null;
  const gpu = preferred === "wasm" ? null : gpuSupport(adapter?.info ?? null);
  const described = adapter ? describeAdapter(adapter.info) : null;
  if (gpu === "hardware") return { device: "webgpu", adapter: described, gpu };
  if (preferred === "webgpu") throw new Error(webgpuUnavailableMessage(gpu));
  return { device: "wasm", adapter: described, gpu };
}

async function cachedVariants() {
  const cached = Object.fromEntries(Object.keys(VARIANTS).map((dtype) => [dtype, false]));
  try {
    if (!(await caches.has(env.cacheKey))) return cached;
    const cache = await caches.open(env.cacheKey);
    for (const dtype of Object.keys(VARIANTS)) {
      const urls = variantFileUrls(dtype, self.location.origin);
      const found = await Promise.all(urls.map((url) => cache.match(url)));
      cached[dtype] = found.every(Boolean);
    }
  } catch {
    return cached;
  }
  return cached;
}

let downloads = {};

function onProgress(event) {
  if (event.status !== "progress" || !/\.onnx(_data)?$/.test(event.file ?? "")) return;
  downloads[event.file] = { loaded: event.loaded, total: event.total };
  post("progress", combinedProgress(downloads));
}

async function warmUp() {
  const inputs = tokenizer.apply_chat_template([{ role: "user", content: WARMUP_QUESTION }], {
    add_generation_prompt: true,
    return_dict: true,
  });
  await model.generate({ ...inputs, max_new_tokens: 2, do_sample: false });
}

async function load({ preferred = "auto", dtype = DEFAULT_VARIANT } = {}) {
  if (!VARIANTS[dtype]) throw new Error(`Unknown model variant: ${dtype}`);
  await unload();
  const started = performance.now();
  const fromCache = (await cachedVariants())[dtype];
  const { device, adapter, gpu } = await chooseDevice(preferred);
  const threads = wasmThreads(self.navigator.hardwareConcurrency, self.crossOriginIsolated);
  env.backends.onnx.wasm.numThreads = threads;
  downloads = {};
  post("status", { state: "loading", device, dtype, fromCache });

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    device,
    dtype,
    use_external_data_format: VARIANTS[dtype].externalData,
    session_options: {
      graphOptimizationLevel: "all",
      freeDimensionOverrides: { batch_size: 1 },
    },
    progress_callback: onProgress,
  });
  const loadedAt = performance.now();
  post("status", { state: "warming", device, dtype });
  await warmUp();

  runtime = {
    model: MODEL_ID,
    dtype,
    device,
    adapter,
    gpu,
    wasmThreads: device === "wasm" ? threads : null,
    crossOriginIsolated: self.crossOriginIsolated,
    fromCache,
    loadMs: Math.round(loadedAt - started),
    warmupMs: Math.round(performance.now() - loadedAt),
  };
  post("ready", { runtime, cached: await cachedVariants() });
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

async function removeDownloaded({ dtype = null } = {}) {
  if (!dtype || runtime?.dtype === dtype) await unload();
  if (dtype) {
    const cache = await caches.open(env.cacheKey);
    await Promise.all(variantFileUrls(dtype, self.location.origin).map((url) => cache.delete(url)));
  }
  const cached = await cachedVariants();
  if (!dtype || !Object.values(cached).some(Boolean)) await caches.delete(env.cacheKey);
  post("removed", { dtype, cached: await cachedVariants(), loaded: runtime });
}

self.addEventListener("message", async ({ data }) => {
  try {
    if (data.type === "init") stopSignal = createStopSignal(data.stopBuffer ?? null);
    else if (data.type === "probe") post("probe", { cached: await cachedVariants() });
    else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopSignal.request();
    else if (data.type === "load") await load(data);
    else if (data.type === "unload") await unload();
    else if (data.type === "remove") await removeDownloaded(data);
  } catch (error) {
    const message = error?.message ?? String(error);
    if (data.type === "generate") post("answer-error", { id: data.id, message });
    else post("error", { message });
  }
});
