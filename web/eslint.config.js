import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";

export default [
  {
    ignores: ["dist/", "dev-dist/", "public/models/", "playwright-report/", "test-results/"],
  },
  js.configs.recommended,
  {
    files: ["src/**/*.{js,jsx}"],
    languageOptions: {
      globals: { ...globals.browser, ...globals.worker },
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
    plugins: { "react-hooks": reactHooks, "react-refresh": reactRefresh },
    rules: {
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": ["warn", { allowConstantExport: true }],
    },
  },
  {
    files: ["tests/**/*.js", "scripts/**/*.mjs", "*.config.js"],
    languageOptions: { globals: { ...globals.node } },
  },
  {
    files: ["e2e/**/*.js"],
    languageOptions: { globals: { ...globals.node, ...globals.browser } },
  },
];
