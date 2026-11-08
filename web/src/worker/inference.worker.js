import {
  AutoModelForCausalLM,
  AutoTokenizer,
  InterruptableStoppingCriteria,
  TextStreamer,
  env,
} from "@huggingface/transformers";
import { createSHA256 } from "hash-wasm";
import { createEntityDecoder, decodeEntities } from "../text/entities.js";
import {
  DOWNLOAD_ATTEMPTS,
  DamagedFileError,
  DownloadStatusError,
  downloadError,
  shouldRetry,
  storageMessage,
} from "./download.js";
import { buildMessages, generationOptions, generationStats, stopReason } from "./generation.js";
import { describeAdapter, gpuSupport, wasmThreads, webgpuUnavailableMessage } from "./runtime.js";
import { HUB_HOST, MODEL_SOURCE } from "./source.js";
import { createStopSignal } from "./stop-signal.js";
import {
  COMMON_FILES,
  DEFAULT_VARIANT,
  FALLBACK_VARIANT,
  VARIANTS,
  combinedProgress,
  modelFileUrl,
  variantFileUrls,
} from "./variants.js";

const WARMUP_QUESTION = "ආයුබෝවන්";

env.allowRemoteModels = MODEL_SOURCE.remote;
env.allowLocalModels = !MODEL_SOURCE.remote;
env.localModelPath = "/models/";
env.remoteHost = HUB_HOST;
env.remotePathTemplate = "{model}/resolve/{revision}/";
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

function trackedResponse(response, hasher, onBytes) {
  let loaded = 0;
  const tracker = new TransformStream({
    transform(chunk, controller) {
      loaded += chunk.byteLength;
      hasher.update(chunk);
      onBytes(loaded);
      controller.enqueue(chunk);
    },
  });
  return new Response(response.body.pipeThrough(tracker), {
    status: response.status,
    headers: response.headers,
  });
}

async function readManifest() {
  try {
    const response = await fetch(modelFileUrl("manifest.json", self.location.origin), {
      cache: "no-store",
    });
    if (!response.ok) throw new DownloadStatusError("manifest.json", response.status);
    return (await response.json()).files;
  } catch (error) {
    throw downloadError(error, { name: "manifest.json", needed: 0 });
  }
}

async function removeOldVersions() {
  if (!(await caches.has(env.cacheKey))) return;
  const cache = await caches.open(env.cacheKey);
  const base = modelFileUrl("", self.location.origin);
  const keys = await cache.keys();
  await Promise.all(
    keys.filter((key) => !key.url.startsWith(base)).map((key) => cache.delete(key)),
  );
  if ((await cache.keys()).length === 0) await caches.delete(env.cacheKey);
}

async function download(dtype) {
  const cache = await caches.open(env.cacheKey);
  const common = COMMON_FILES.map((file) => modelFileUrl(file, self.location.origin));
  const urls = [...common, ...variantFileUrls(dtype, self.location.origin)];
  const missing = [];
  for (const url of urls) if (!(await cache.match(url))) missing.push(url);
  const base = modelFileUrl("", self.location.origin);
  const manifest = await readManifest();
  const files = Object.fromEntries(
    missing.map((url) => [
      url,
      { loaded: 0, total: manifest[url.slice(base.length)]?.bytes ?? null },
    ]),
  );
  const needed = Object.values(files).reduce((sum, file) => sum + (file.total ?? 0), 0);
  const { usage = 0, quota = Infinity } = (await self.navigator.storage?.estimate?.()) ?? {};
  if (quota - usage < needed) throw new Error(storageMessage(needed, quota - usage));
  let name = null;
  try {
    for (const url of missing) {
      name = url.slice(base.length);
      await downloadFile(cache, url, name, manifest[name]?.sha256, (loaded) => {
        files[url].loaded = loaded;
        post("progress", combinedProgress(files));
      });
    }
  } catch (error) {
    await Promise.all(missing.map((url) => cache.delete(url)));
    if ((await cache.keys()).length === 0) await caches.delete(env.cacheKey);
    throw downloadError(error, { name, needed });
  }
}

async function downloadFile(cache, url, name, expected, onBytes) {
  for (let attempt = 1; ; attempt += 1) {
    try {
      const response = await fetch(url);
      if (!response.ok) throw new DownloadStatusError(name, response.status);
      const hasher = await createSHA256();
      hasher.init();
      await cache.put(url, trackedResponse(response, hasher, onBytes));
      if (expected && hasher.digest("hex") !== expected) throw new DamagedFileError(name);
      return;
    } catch (error) {
      await cache.delete(url);
      if (attempt >= DOWNLOAD_ATTEMPTS || !shouldRetry(error)) throw error;
      console.warn(`Retrying ${name}:`, error);
      onBytes(0);
    }
  }
}

async function warmUp() {
  const inputs = tokenizer.apply_chat_template([{ role: "user", content: WARMUP_QUESTION }], {
    add_generation_prompt: true,
    return_dict: true,
  });
  await model.generate({ ...inputs, max_new_tokens: 2, do_sample: false });
}

async function load(options = {}) {
  const dtype = options.dtype ?? DEFAULT_VARIANT;
  try {
    await loadVariant({ ...options, dtype });
  } catch (error) {
    if (dtype !== DEFAULT_VARIANT) throw error;
    const reason = error?.message ?? String(error);
    post("fallback", { from: dtype, to: FALLBACK_VARIANT, reason });
    await loadVariant({ ...options, dtype: FALLBACK_VARIANT, fallback: { from: dtype, reason } });
  }
}

async function loadVariant({ preferred = "auto", dtype, fallback = null }) {
  if (!VARIANTS[dtype]) throw new Error(`Unknown model variant: ${dtype}`);
  await unload();
  const started = performance.now();
  const fromCache = (await cachedVariants())[dtype];
  const { device, adapter, gpu } = await chooseDevice(preferred);
  const threads = wasmThreads(self.navigator.hardwareConcurrency, self.crossOriginIsolated);
  env.backends.onnx.wasm.numThreads = threads;
  post("status", { state: "loading", device, dtype, fromCache });
  if (!fromCache) await download(dtype);

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_SOURCE.id, {
    revision: MODEL_SOURCE.revision,
  });
  model = await AutoModelForCausalLM.from_pretrained(MODEL_SOURCE.id, {
    revision: MODEL_SOURCE.revision,
    device,
    dtype,
    use_external_data_format: VARIANTS[dtype].externalData,
    session_options: {
      graphOptimizationLevel: "all",
      freeDimensionOverrides: { batch_size: 1 },
    },
  });
  const loadedAt = performance.now();
  post("status", { state: "warming", device, dtype });
  await warmUp();

  runtime = {
    model: MODEL_SOURCE.id,
    dtype,
    device,
    adapter,
    gpu,
    wasmThreads: device === "wasm" ? threads : null,
    crossOriginIsolated: self.crossOriginIsolated,
    fromCache,
    loadMs: Math.round(loadedAt - started),
    warmupMs: Math.round(performance.now() - loadedAt),
    fallback,
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
    else if (data.type === "probe") {
      await removeOldVersions();
      const adapter = await self.navigator.gpu
        ?.requestAdapter({ powerPreference: "high-performance" })
        .catch(() => null);
      post("probe", { cached: await cachedVariants(), gpu: gpuSupport(adapter?.info ?? null) });
    } else if (data.type === "generate") await generate(data);
    else if (data.type === "stop") stopSignal.request();
    else if (data.type === "load") await load(data);
    else if (data.type === "unload") await unload();
    else if (data.type === "remove") await removeDownloaded(data);
  } catch (error) {
    const message = error?.message ?? String(error);
    if (error?.cause) console.error(error.cause);
    if (data.type === "generate") post("answer-error", { id: data.id, message });
    else post("error", { message });
  }
});
