import { normalizeZwj } from "../text/zwj.js";

export const CONTEXT_LIMIT = 1024;
export const MIN_ANSWER_TOKENS = 16;

export class ContextLimitError extends Error {
  constructor(promptTokens) {
    super(
      `The question is too long: it uses ${promptTokens} of the ${CONTEXT_LIMIT} tokens the model can read.`,
    );
    this.name = "ContextLimitError";
    this.promptTokens = promptTokens;
  }
}

export function buildMessages(question, { normalizeZwj: normalize = true } = {}) {
  const text = question.trim();
  return [{ role: "user", content: normalize ? normalizeZwj(text) : text }];
}

export function generationOptions(settings, promptTokens) {
  const room = CONTEXT_LIMIT - promptTokens;
  if (room < MIN_ANSWER_TOKENS) throw new ContextLimitError(promptTokens);
  const sample = settings.temperature > 0;
  return {
    max_new_tokens: Math.min(settings.maxNewTokens, room),
    do_sample: sample,
    ...(sample && { temperature: settings.temperature, top_k: settings.topK }),
    repetition_penalty: settings.repetitionPenalty,
  };
}

export function stopReason({ interrupted, lastToken, eosIds, newTokens, maxNewTokens }) {
  if (interrupted) return "user";
  if (lastToken != null && eosIds.includes(Number(lastToken))) return "end";
  return newTokens >= maxNewTokens ? "length" : "end";
}

export function generationStats({ promptTokens, newTokens, startedMs, firstTokenMs, endedMs }) {
  const decodeMs = firstTokenMs == null ? 0 : endedMs - firstTokenMs;
  return {
    promptTokens,
    newTokens,
    firstTokenMs: firstTokenMs == null ? null : Math.round(firstTokenMs - startedMs),
    totalMs: Math.round(endedMs - startedMs),
    tokensPerSecond: newTokens > 1 && decodeMs > 0 ? ((newTokens - 1) * 1000) / decodeMs : null,
  };
}
