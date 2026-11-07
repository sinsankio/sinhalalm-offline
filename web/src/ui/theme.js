const STORAGE_KEY = "sinhalallm-theme";
export const THEMES = ["auto", "light", "dark"];

export function readTheme() {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return THEMES.includes(stored) ? stored : "auto";
  } catch {
    return "auto";
  }
}

export function applyTheme(theme) {
  const root = document.documentElement;
  if (theme === "auto") {
    delete root.dataset.theme;
  } else {
    root.dataset.theme = theme;
  }
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    return;
  }
}
