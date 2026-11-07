import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS } from "../src/ui/content.js";
import { isDefault, readSettings, sanitizeSettings } from "../src/ui/settings.js";

describe("sanitizeSettings", () => {
  it("returns the defaults for missing or invalid input", () => {
    expect(sanitizeSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(sanitizeSettings("text")).toEqual(DEFAULT_SETTINGS);
  });

  it("keeps valid values", () => {
    const stored = { ...DEFAULT_SETTINGS, temperature: 0, maxNewTokens: 128, normalizeZwj: false };
    expect(sanitizeSettings(stored)).toEqual(stored);
  });

  it("clamps values to the slider ranges", () => {
    const settings = sanitizeSettings({ temperature: 9, topK: 0, maxNewTokens: 5000 });
    expect(settings.temperature).toBe(1.5);
    expect(settings.topK).toBe(1);
    expect(settings.maxNewTokens).toBe(512);
  });

  it("ignores values of the wrong type and unknown keys", () => {
    const settings = sanitizeSettings({ temperature: "hot", topK: Number.NaN, extra: 1 });
    expect(settings).toEqual(DEFAULT_SETTINGS);
  });
});

describe("readSettings", () => {
  it("falls back to the defaults without local storage", () => {
    expect(readSettings()).toEqual(DEFAULT_SETTINGS);
  });
});

describe("isDefault", () => {
  it("detects changed settings", () => {
    expect(isDefault(DEFAULT_SETTINGS)).toBe(true);
    expect(isDefault({ ...DEFAULT_SETTINGS, topK: 10 })).toBe(false);
  });
});
