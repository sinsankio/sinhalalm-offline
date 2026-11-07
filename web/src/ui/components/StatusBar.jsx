import { MODEL } from "../content.js";
import GpuHint from "./GpuHint.jsx";

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
    return `The model (${MODEL.downloadGb} GB) is not on this device yet.`;
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

export default function StatusBar({ model, locked, onLoad }) {
  const { state, runtime, progress, error } = model;
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
      <p className="status-message">
        <span className="status-dot" aria-hidden="true" />
        {describe(model)}
      </p>
      {percent != null && <progress className="progress" max={1} value={percent} />}
      {state === "download-required" && (
        <button type="button" className="primary status-action" onClick={() => onLoad("auto")}>
          Download model ({MODEL.downloadGb} GB)
        </button>
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
