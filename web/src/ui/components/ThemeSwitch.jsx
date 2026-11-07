import { useState } from "react";
import { THEMES, applyTheme, readTheme } from "../theme.js";

const LABELS = { auto: "Auto", light: "Light", dark: "Dark" };

export default function ThemeSwitch() {
  const [theme, setTheme] = useState(readTheme);

  const choose = (next) => {
    setTheme(next);
    applyTheme(next);
  };

  return (
    <div className="theme-switch" role="group" aria-label="Theme">
      {THEMES.map((option) => (
        <button
          key={option}
          type="button"
          aria-pressed={theme === option}
          onClick={() => choose(option)}
        >
          {LABELS[option]}
        </button>
      ))}
    </div>
  );
}
