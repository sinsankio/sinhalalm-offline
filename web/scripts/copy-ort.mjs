import { copyFileSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const source = join(dirname(require.resolve("onnxruntime-web")), "..", "dist");
const target = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "ort");
const FILES = ["ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.wasm"];

mkdirSync(target, { recursive: true });
for (const name of FILES) {
  copyFileSync(join(source, name), join(target, name));
}
