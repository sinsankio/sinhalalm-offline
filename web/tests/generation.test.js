import { describe, expect, it } from "vitest";
import {
  CONTEXT_LIMIT,
  ContextLimitError,
  buildMessages,
  generationOptions,
  generationStats,
  stopReason,
} from "../src/worker/generation.js";
import { createStopSignal } from "../src/worker/stop-signal.js";

const SETTINGS = { temperature: 0.7, topK: 50, repetitionPenalty: 1.1, maxNewTokens: 256 };

describe("buildMessages", () => {
  it("normalizes the ZWJ and trims the question", () => {
    expect(buildMessages("  ශ්රී ලංකාව ", { normalizeZwj: true })).toEqual([
      { role: "user", content: "ශ්‍රී ලංකාව" },
    ]);
  });

  it("keeps the question unchanged when normalization is off", () => {
    expect(buildMessages("ශ්රී", { normalizeZwj: false })[0].content).toBe("ශ්රී");
  });
});

describe("generationOptions", () => {
  it("samples with the given settings", () => {
    expect(generationOptions(SETTINGS, 20)).toEqual({
      max_new_tokens: 256,
      do_sample: true,
      temperature: 0.7,
      top_k: 50,
      repetition_penalty: 1.1,
    });
  });

  it("decodes greedily at temperature 0", () => {
    const options = generationOptions({ ...SETTINGS, temperature: 0 }, 20);
    expect(options).toEqual({ max_new_tokens: 256, do_sample: false, repetition_penalty: 1.1 });
  });

  it("keeps prompt and answer within the context limit", () => {
    expect(generationOptions(SETTINGS, CONTEXT_LIMIT - 100).max_new_tokens).toBe(100);
  });

  it("rejects a prompt that leaves no room for an answer", () => {
    expect(() => generationOptions(SETTINGS, CONTEXT_LIMIT - 4)).toThrow(ContextLimitError);
  });
});

describe("stopReason", () => {
  const base = { interrupted: false, eosIds: [2, 0], newTokens: 10, maxNewTokens: 256 };

  it("reports an interruption", () => {
    expect(stopReason({ ...base, interrupted: true, lastToken: 2n })).toBe("user");
  });

  it("reports an end token", () => {
    expect(stopReason({ ...base, lastToken: 2n })).toBe("end");
  });

  it("reports the length limit", () => {
    expect(stopReason({ ...base, lastToken: 5n, newTokens: 256 })).toBe("length");
  });
});

describe("generationStats", () => {
  it("measures decoding speed after the first token", () => {
    expect(
      generationStats({
        promptTokens: 12,
        newTokens: 11,
        startedMs: 0,
        firstTokenMs: 500,
        endedMs: 2500,
      }),
    ).toEqual({
      promptTokens: 12,
      newTokens: 11,
      firstTokenMs: 500,
      totalMs: 2500,
      tokensPerSecond: 5,
    });
  });

  it("reports no speed without generated tokens", () => {
    const stats = generationStats({ promptTokens: 12, newTokens: 0, startedMs: 0, endedMs: 100 });
    expect(stats.firstTokenMs).toBeNull();
    expect(stats.tokensPerSecond).toBeNull();
  });
});

describe("createStopSignal", () => {
  it("shares a stop request through the buffer", () => {
    const sender = createStopSignal(new SharedArrayBuffer(4));
    const receiver = createStopSignal(sender.buffer);
    expect(receiver.requested).toBe(false);
    sender.request();
    expect(receiver.requested).toBe(true);
    receiver.reset();
    expect(receiver.requested).toBe(false);
  });

  it("works without shared memory", () => {
    const signal = createStopSignal(null);
    signal.request();
    expect(signal.requested).toBe(true);
    signal.reset();
    expect(signal.requested).toBe(false);
  });
});
