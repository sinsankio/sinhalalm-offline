import { describe, expect, it } from "vitest";
import { formatBytes, formatMs, median, summarizeBenchmark } from "../src/ui/diagnostics.js";
import { combinedProgress, variantFileUrls } from "../src/worker/variants.js";

describe("median", () => {
  it("handles odd, even and empty inputs", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([null, Number.NaN])).toBeNull();
  });
});

describe("summarizeBenchmark", () => {
  it("summarizes answered questions only", () => {
    const stats = (firstTokenMs, tokensPerSecond, newTokens) => ({
      firstTokenMs,
      tokensPerSecond,
      newTokens,
    });
    expect(
      summarizeBenchmark([
        { stats: stats(100, 20, 10) },
        { stats: stats(300, 30, 20) },
        { error: "failed" },
      ]),
    ).toEqual({
      questions: 3,
      answered: 2,
      medianFirstTokenMs: 200,
      medianTokensPerSecond: 25,
      tokens: 30,
    });
  });
});

describe("formatting", () => {
  it("formats sizes and durations", () => {
    expect(formatBytes(1.19e9)).toBe("1.19 GB");
    expect(formatBytes(250e6)).toBe("250 MB");
    expect(formatBytes(undefined)).toBe("not available");
    expect(formatMs(800)).toBe("800 ms");
    expect(formatMs(3200)).toBe("3.2 s");
  });
});

describe("variants", () => {
  it("lists every file of a variant", () => {
    expect(variantFileUrls("q8", "http://localhost")).toEqual([
      "http://localhost/models/sinhalallm-1.7b/v2.1/onnx/model_quantized.onnx",
      "http://localhost/models/sinhalallm-1.7b/v2.1/onnx/model_quantized.onnx_data",
      "http://localhost/models/sinhalallm-1.7b/v2.1/onnx/model_quantized.onnx_data_1",
    ]);
  });

  it("combines the progress of several files", () => {
    expect(
      combinedProgress({ a: { loaded: 50, total: 100 }, b: { loaded: 50, total: 300 } }),
    ).toEqual({ loaded: 100, total: 400, progress: 0.25 });
    expect(combinedProgress({ a: { loaded: 5 } }).progress).toBeNull();
  });
});
