import { describe, expect, it } from "vitest";
import {
  describeAdapter,
  gpuSupport,
  isLinuxDesktop,
  isSoftwareAdapter,
  wasmThreads,
  webgpuUnavailableMessage,
} from "../src/worker/runtime.js";

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

describe("gpuSupport", () => {
  it("distinguishes hardware, software and missing adapters", () => {
    expect(gpuSupport({ vendor: "nvidia", architecture: "ampere" })).toBe("hardware");
    expect(gpuSupport({ vendor: "google", architecture: "swiftshader" })).toBe("software");
    expect(gpuSupport(null)).toBe("none");
  });

  it("explains why WebGPU cannot be used", () => {
    expect(webgpuUnavailableMessage("software")).toMatch(/only in software/);
    expect(webgpuUnavailableMessage("none")).toMatch(/does not offer WebGPU/);
  });
});

describe("isLinuxDesktop", () => {
  it.each([
    ["Linux", "Mozilla/5.0 (X11; Linux x86_64)", true],
    ["Linux armv8l", "Mozilla/5.0 (Linux; Android 14)", false],
    ["Linux", "Mozilla/5.0 (X11; CrOS x86_64 16000.0.0)", false],
    ["Windows", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", false],
  ])("%s / %s", (platform, userAgent, expected) => {
    expect(isLinuxDesktop(platform, userAgent)).toBe(expected);
  });
});
