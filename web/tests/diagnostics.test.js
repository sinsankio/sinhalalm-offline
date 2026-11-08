import { describe, expect, it } from "vitest";
import { formatBytes, formatMs, median, summarizeBenchmark } from "../src/ui/diagnostics.js";
import { modelSource } from "../src/worker/source.js";
import {
  combinedProgress,
  initialVariant,
  modelFileUrl,
  variantFileUrls,
} from "../src/worker/variants.js";

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
      totalMs: firstTokenMs * 10,
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
      totalMs: 4000,
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

describe("initialVariant", () => {
  it("prefers the 8-bit model on hardware WebGPU", () => {
    expect(initialVariant({ saved: null, cached: {}, gpu: "hardware" })).toBe("q8");
  });

  it("falls back to the 4-bit model without hardware WebGPU", () => {
    expect(initialVariant({ saved: null, cached: {}, gpu: "software" })).toBe("q4");
    expect(initialVariant({ saved: null, cached: {}, gpu: "none" })).toBe("q4");
  });

  it("keeps a downloaded 4-bit model instead of downloading the 8-bit one", () => {
    expect(initialVariant({ saved: null, cached: { q4: true }, gpu: "hardware" })).toBe("q4");
  });

  it("respects the reader's choice", () => {
    expect(initialVariant({ saved: "q8", cached: {}, gpu: "none" })).toBe("q8");
    expect(initialVariant({ saved: "q4", cached: { q8: true }, gpu: "hardware" })).toBe("q4");
  });
});

describe("modelSource", () => {
  it("uses the local package without a repository", () => {
    expect(modelSource({})).toEqual({
      remote: false,
      id: "sinhalallm-1.7b/v2.1",
      revision: "main",
      base: "/models/sinhalallm-1.7b/v2.1/",
      pathTemplate: "{model}/resolve/{revision}/",
    });
  });

  it("pins a Hub repository to a revision", () => {
    const source = modelSource({ repo: "user/model", revision: "abc123" });
    expect(modelFileUrl("onnx/model_q4.onnx", "https://app.example", source)).toBe(
      "https://huggingface.co/user/model/resolve/abc123/onnx/model_q4.onnx",
    );
  });

  it("keeps the revision when transformers.js leaves it out", () => {
    const { pathTemplate } = modelSource({ repo: "user/model", revision: "abc123" });
    expect(pathTemplate).toBe("{model}/resolve/abc123/");
    expect(pathTemplate.replaceAll("{revision}", "main")).toBe("{model}/resolve/abc123/");
  });
});
