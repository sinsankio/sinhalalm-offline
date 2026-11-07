import { describe, expect, it } from "vitest";
import { describeAdapter, isSoftwareAdapter, wasmThreads } from "../src/worker/runtime.js";

describe("isSoftwareAdapter", () => {
  it.each([
    [{ vendor: "google", architecture: "swiftshader" }],
    [{ description: "llvmpipe (LLVM 17.0.6, 256 bits)" }],
    [{ vendor: "microsoft", architecture: "warp" }],
    [{ vendor: "nvidia", architecture: "ampere", isFallbackAdapter: true }],
  ])("rejects %o", (info) => {
    expect(isSoftwareAdapter(info)).toBe(true);
  });

  it.each([
    [{ vendor: "nvidia", architecture: "ampere" }],
    [{ vendor: "intel", architecture: "gen-12lp" }],
    [{ vendor: "apple", architecture: "metal-3" }],
  ])("accepts %o", (info) => {
    expect(isSoftwareAdapter(info)).toBe(false);
  });
});

describe("wasmThreads", () => {
  it("uses one thread without cross-origin isolation", () => {
    expect(wasmThreads(16, false)).toBe(1);
  });

  it("leaves one core for the page", () => {
    expect(wasmThreads(8, true)).toBe(7);
  });

  it("caps the thread count", () => {
    expect(wasmThreads(32, true)).toBe(8);
  });

  it("keeps at least one thread", () => {
    expect(wasmThreads(1, true)).toBe(1);
    expect(wasmThreads(undefined, true)).toBe(1);
  });
});

describe("describeAdapter", () => {
  it("fills missing fields", () => {
    expect(describeAdapter({})).toEqual({
      vendor: "unknown",
      architecture: "unknown",
      description: "",
      isFallbackAdapter: false,
    });
  });
});
