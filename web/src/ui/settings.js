import { DEFAULT_SETTINGS } from "./content.js";

const STORAGE_KEY = "sinhalallm-settings";

export const SLIDERS = [
  { key: "temperature", label: "Temperature", min: 0, max: 1.5, step: 0.05 },
  { key: "topK", label: "Top-k", min: 1, max: 100, step: 1 },
  { key: "repetitionPenalty", label: "Repetition penalty", min: 1, max: 1.5, step: 0.01 },
  { key: "maxNewTokens", label: "Maximum answer length (tokens)", min: 32, max: 512, step: 16 },
];

export function sanitizeSettings(value) {
  const settings = { ...DEFAULT_SETTINGS };
  if (!value || typeof value !== "object") return settings;
  for (const { key, min, max } of SLIDERS) {
    const number = value[key];
    if (typeof number === "number" && Number.isFinite(number)) {
      settings[key] = Math.min(max, Math.max(min, number));
    }
  }
  if (typeof value.normalizeZwj === "boolean") settings.normalizeZwj = value.normalizeZwj;
  return settings;
}

export function readSettings() {
  try {
    return sanitizeSettings(JSON.parse(localStorage.getItem(STORAGE_KEY)));
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    return;
  }
}

export function isDefault(settings) {
  return Object.keys(DEFAULT_SETTINGS).every((key) => settings[key] === DEFAULT_SETTINGS[key]);
}
