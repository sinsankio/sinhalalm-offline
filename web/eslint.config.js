import js from "@eslint/js";
import globals from "globals";

export default [
  { ignores: ["dist/", "dev-dist/", "public/models/", "playwright-report/", "test-results/"] },
  js.configs.recommended,
  {
    files: ["src/**/*.js"],
    languageOptions: { globals: { ...globals.browser, ...globals.worker } },
  },
  {
    files: ["tests/**/*.js", "*.config.js"],
    languageOptions: { globals: { ...globals.node } },
  },
];
