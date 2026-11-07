import { AutoModelForCausalLM, AutoTokenizer, env } from "@huggingface/transformers";
import { describeAdapter, isSoftwareAdapter, wasmThreads } from "./runtime.js";

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
    if (data.type === "probe") post("probe", { fromCache: await isCached() });
    else if (data.type === "load") await load(data);
    else if (data.type === "unload") await unload();
    else if (data.type === "remove") await removeDownloaded();
  } catch (error) {
    post("error", { message: error?.message ?? String(error) });
  }
});
