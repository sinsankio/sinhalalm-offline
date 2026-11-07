import { DEFAULT_SETTINGS, SAMPLE_QUESTIONS } from "./content.js";

export const BENCHMARK_QUESTIONS = SAMPLE_QUESTIONS.slice(0, 5);
export const BENCHMARK_SETTINGS = { ...DEFAULT_SETTINGS, temperature: 0, maxNewTokens: 64 };

export function median(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const middle = sorted.length >> 1;
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function summarizeBenchmark(results) {
  const answered = results.filter((result) => result.stats);
  return {
    questions: results.length,
    answered: answered.length,
    medianFirstTokenMs: median(answered.map((result) => result.stats.firstTokenMs)),
    medianTokensPerSecond: median(answered.map((result) => result.stats.tokensPerSecond)),
    tokens: answered.reduce((sum, result) => sum + result.stats.newTokens, 0),
  };
}

export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "not available";
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  return `${Math.round(bytes / 1e6)} MB`;
}

export function formatMs(ms) {
  if (!Number.isFinite(ms)) return "–";
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}
