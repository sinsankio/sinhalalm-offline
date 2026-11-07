import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import { registerServiceWorker } from "./registerServiceWorker.js";
import { applyTheme, readTheme } from "./theme.js";
import "./styles.css";

applyTheme(readTheme());
if (import.meta.env.PROD) registerServiceWorker();

createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
