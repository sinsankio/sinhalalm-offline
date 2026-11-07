import { cpSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const PUBLIC_ASSETS = ["ort"];

function copyPublicAssets() {
  return {
    name: "copy-public-assets",
    apply: "build",
    writeBundle(options) {
      for (const name of PUBLIC_ASSETS) {
        const source = fileURLToPath(new URL(`./public/${name}`, import.meta.url));
        if (existsSync(source)) cpSync(source, `${options.dir}/${name}`, { recursive: true });
      }
    },
  };
}

export default defineConfig({
  plugins: [react(), copyPublicAssets()],
  build: { copyPublicDir: false },
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  worker: { format: "es" },
  test: {
    include: ["tests/**/*.test.js"],
    environment: "node",
  },
});
