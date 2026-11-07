import { createHash } from "node:crypto";
import { cpSync, createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const crossOriginIsolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "require-corp",
};

const publicDir = fileURLToPath(new URL("./public", import.meta.url));
const PUBLIC_ASSETS = ["icon.svg", "manifest.webmanifest"];
const CONTENT_TYPES = { ".json": "application/json", ".txt": "text/plain; charset=utf-8" };

function copyPublicAssets() {
  return {
    name: "copy-public-assets",
    apply: "build",
    writeBundle(options) {
      for (const name of PUBLIC_ASSETS) {
        const source = join(publicDir, name);
        if (existsSync(source)) cpSync(source, join(options.dir, name), { recursive: true });
      }
    },
  };
}

function serviceWorker() {
  const template = fileURLToPath(new URL("./src/sw/service-worker.js", import.meta.url));
  return {
    name: "service-worker",
    apply: "build",
    generateBundle(_, bundle) {
      const built = Object.keys(bundle)
        .filter((name) => !name.endsWith(".map") && name !== "index.html")
        .map((name) => `/${name}`);
      const files = ["/", ...built, ...PUBLIC_ASSETS.map((name) => `/${name}`)].sort();
      const version = createHash("sha256").update(JSON.stringify(files)).digest("hex").slice(0, 16);
      this.emitFile({
        type: "asset",
        fileName: "sw.js",
        source: `self.__SW_CONFIG = ${JSON.stringify({ version, files })};\n${readFileSync(template, "utf8")}`,
      });
    },
  };
}

function serveModelsInPreview() {
  const root = join(publicDir, "models");
  return {
    name: "serve-models-in-preview",
    configurePreviewServer(server) {
      server.middlewares.use("/models", (request, response, next) => {
        const path = normalize(join(root, decodeURIComponent(request.url.split("?")[0])));
        if (!path.startsWith(root) || !existsSync(path) || !statSync(path).isFile()) return next();
        response.setHeader(
          "Content-Type",
          CONTENT_TYPES[extname(path)] ?? "application/octet-stream",
        );
        response.setHeader("Content-Length", statSync(path).size);
        if (request.method === "HEAD") return response.end();
        createReadStream(path).pipe(response);
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), copyPublicAssets(), serviceWorker(), serveModelsInPreview()],
  build: { copyPublicDir: false },
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  worker: { format: "es" },
  test: {
    include: ["tests/**/*.test.js"],
    environment: "node",
  },
});
