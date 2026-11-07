const SOFTWARE_ARCHITECTURES = ["swiftshader", "llvmpipe", "lavapipe", "warp"];
const MAX_WASM_THREADS = 8;

export function isSoftwareAdapter(info = {}) {
  if (info.isFallbackAdapter) return true;
  const text = [info.architecture, info.description, info.device, info.vendor]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return SOFTWARE_ARCHITECTURES.some((name) => text.includes(name));
}

export function wasmThreads(hardwareConcurrency, crossOriginIsolated) {
  if (!crossOriginIsolated) return 1;
  const cores = Number.isFinite(hardwareConcurrency) ? hardwareConcurrency : 1;
  return Math.max(1, Math.min(cores - 1, MAX_WASM_THREADS));
}

export function describeAdapter(info = {}) {
  return {
    vendor: info.vendor || "unknown",
    architecture: info.architecture || "unknown",
    description: info.description || "",
    isFallbackAdapter: Boolean(info.isFallbackAdapter),
  };
}
