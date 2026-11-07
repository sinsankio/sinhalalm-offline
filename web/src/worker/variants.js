export const MODEL_ID = "sinhalallm-1.7b/v2.1";
export const DEFAULT_VARIANT = "q8";
export const FALLBACK_VARIANT = "q4";
export const COMMON_FILES = [
  "config.json",
  "generation_config.json",
  "tokenizer.json",
  "tokenizer_config.json",
];

export const VARIANTS = Object.freeze({
  q4: {
    label: "4-bit",
    summary: "Faster, smaller download",
    strength: 2,
    downloadGb: 1.19,
    files: ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data"],
    externalData: 1,
  },
  q8: {
    label: "8-bit",
    summary: "More accurate, larger download",
    strength: 4,
    downloadGb: 2.48,
    files: [
      "onnx/model_quantized.onnx",
      "onnx/model_quantized.onnx_data",
      "onnx/model_quantized.onnx_data_1",
    ],
    externalData: 2,
  },
});

export function variantFileUrls(dtype, origin) {
  return VARIANTS[dtype].files.map((file) => new URL(`/models/${MODEL_ID}/${file}`, origin).href);
}

export function combinedProgress(files) {
  const items = Object.values(files);
  const loaded = items.reduce((sum, item) => sum + (item.loaded ?? 0), 0);
  const total = items.every((item) => item.total)
    ? items.reduce((sum, item) => sum + item.total, 0)
    : null;
  return { loaded, total, progress: total ? loaded / total : null };
}

export function initialVariant({ saved, cached = {}, gpu }) {
  if (VARIANTS[saved]) return saved;
  if (gpu !== "hardware") return FALLBACK_VARIANT;
  if (!cached[DEFAULT_VARIANT] && cached[FALLBACK_VARIANT]) return FALLBACK_VARIANT;
  return DEFAULT_VARIANT;
}
