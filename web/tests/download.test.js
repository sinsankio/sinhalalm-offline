import { describe, expect, it } from "vitest";
import {
  DamagedFileError,
  DownloadStatusError,
  downloadError,
  shouldRetry,
} from "../src/worker/download.js";

const quota = Object.assign(new Error("quota"), { name: "QuotaExceededError" });
const network = new TypeError("Cache.put() encountered a network error");

describe("download errors", () => {
  it("reports a full quota as a storage problem", () => {
    const error = downloadError(quota, { name: "onnx/model_q4.onnx_data", needed: 1.19e9 });
    expect(error.message).toMatch(/1\.19 GB\) could not be stored/);
    expect(error.cause).toBe(quota);
  });

  it("reports a broken stream as an interrupted download, not a full disk", () => {
    const error = downloadError(network, { name: "onnx/model_q4.onnx_data", needed: 1.19e9 });
    expect(error.message).toMatch(/download of onnx\/model_q4\.onnx_data stopped/);
    expect(error.message).not.toMatch(/could not be stored/);
    expect(error.cause).toBe(network);
  });

  it("keeps hash and status errors as they are", () => {
    const damaged = new DamagedFileError("tokenizer.json");
    const missing = new DownloadStatusError("config.json", 404);
    expect(downloadError(damaged, { name: "tokenizer.json", needed: 1 })).toBe(damaged);
    expect(downloadError(missing, { name: "config.json", needed: 1 })).toBe(missing);
  });

  it("retries network and server errors only", () => {
    expect(shouldRetry(network)).toBe(true);
    expect(shouldRetry(new DownloadStatusError("config.json", 503))).toBe(true);
    expect(shouldRetry(new DownloadStatusError("config.json", 404))).toBe(false);
    expect(shouldRetry(quota)).toBe(false);
    expect(shouldRetry(new DamagedFileError("tokenizer.json"))).toBe(false);
  });
});
