import { VARIANTS } from "../../worker/variants.js";
import useOnline from "../useOnline.js";
import GpuHint from "./GpuHint.jsx";
import VariantPicker from "./VariantPicker.jsx";

const BACKENDS = [
  { value: "auto", label: "Auto" },
  { value: "webgpu", label: "WebGPU" },
  { value: "wasm", label: "WASM" },
];

const megabytes = (bytes) => `${Math.round(bytes / 1e6)} MB`;

function describe(model) {
  const { state, runtime, progress } = model;
  if (state === "checking") return "Checking whether the model is already on this device…";
  if (state === "download-required")
    return `The model (${VARIANTS[model.dtype].downloadGb} GB) is not on this device yet.`;
  if (state === "loading") {
    if (model.fromCache) return "Loading the model from this device…";
    if (progress?.total)
      return `Downloading the model: ${megabytes(progress.loaded)} of ${megabytes(progress.total)}`;
    return "Downloading the model…";
  }
  if (state === "warming") return "Preparing the model for the first question…";
  if (state === "removing") return "Deleting the downloaded model…";
  if (state === "ready") {
    const where =
      runtime.device === "webgpu"
        ? `WebGPU (${runtime.adapter?.vendor ?? "GPU"})`
        : `WASM (${runtime.wasmThreads} threads)`;
    return `Ready on ${where}`;
  }
  if (state === "error") return "The model could not be loaded.";
  return "The model is not loaded.";
}

export default function StatusBar({ model, locked, onLoad, onSelect }) {
  const { state, runtime, progress, error } = model;
  const online = useOnline();
  const busy = state === "checking" || state === "loading" || state === "warming";
  const percent =
    state === "loading" && !model.fromCache && progress?.progress != null
      ? progress.progress
      : null;
  const tone =
    state === "ready" ? "ready" : state === "error" ? "error" : busy ? "loading" : "idle";

  return (
    <section className={`panel status status-${tone}`} aria-live="polite">
      <h2 className="panel-label">Model</h2>
      {onSelect && (
        <VariantPicker
          value={model.dtype}
          cached={model.cached}
          disabled={locked || busy || state === "removing"}
          onChange={(dtype) => onSelect(dtype, model.cached?.[dtype], runtime?.device ?? "auto")}
        />
      )}
      {state === "ready" && runtime.fallback && (
        <p className="status-detail">
          The 8-bit model could not be used, so the 4-bit model is loaded instead. Reason:{" "}
          {runtime.fallback.reason}
        </p>
      )}
      {onSelect && model.dtype === "q8" && runtime?.device === "wasm" && (
        <p className="status-detail">
          The 8-bit model is very slow without WebGPU. The 4-bit model is recommended in this
          browser.
        </p>
      )}
      <p className="status-message">
        <span className="status-dot" aria-hidden="true" />
        {describe(model)}
      </p>
      {percent != null && <progress className="progress" max={1} value={percent} />}
      {state === "download-required" && (
        <>
          <button
            type="button"
            className="primary status-action"
            disabled={!online}
            onClick={() => onLoad("auto")}
          >
            Download model ({VARIANTS[model.dtype].downloadGb} GB)
          </button>
          {!online && (
            <p className="status-detail">
              You are offline. Connect once to download the model; after that, the application works
              without a connection.
            </p>
          )}
        </>
      )}
      {state === "error" && (
        <>
          <p className="status-detail">{error}</p>
          <button type="button" className="primary status-action" onClick={() => onLoad("auto")}>
            Try again
          </button>
        </>
      )}
      {state === "ready" && (
        <dl className="status-facts">
          <dt>Loaded from</dt>
          <dd>{runtime.fromCache ? "this device" : "the network"}</dd>
          <dt>Load time</dt>
          <dd>{(runtime.loadMs / 1000).toFixed(1)} s</dd>
          <dt>Warm-up</dt>
          <dd>{(runtime.warmupMs / 1000).toFixed(1)} s</dd>
          <dt>Isolated</dt>
          <dd>{runtime.crossOriginIsolated ? "yes" : "no"}</dd>
        </dl>
      )}
      {state === "ready" && runtime.device === "wasm" && runtime.gpu && (
        <GpuHint gpu={runtime.gpu} />
      )}
      {(state === "ready" || state === "error") && (
        <div className="backend-switch" role="group" aria-label="Backend">
          <span className="hint">Run on</span>
          {BACKENDS.map(({ value, label }) => (
            <button
              key={value}
              type="button"
              className="chip"
              aria-pressed={state === "ready" && value === runtime?.device}
              disabled={locked}
              onClick={() => onLoad(value)}
            >
              {label}
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
