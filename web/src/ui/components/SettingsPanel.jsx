import { DEFAULT_SETTINGS } from "../content.js";
import { SLIDERS, isDefault } from "../settings.js";

export default function SettingsPanel({ settings, onChange }) {
  const update = (key, value) => onChange({ ...settings, [key]: value });

  return (
    <section className="panel settings">
      <h2 className="panel-label">Settings</h2>
      {SLIDERS.map(({ key, label, min, max, step }) => (
        <label key={key} className="setting">
          <span className="setting-row">
            <span>{label}</span>
            <output>{settings[key]}</output>
          </span>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={settings[key]}
            onChange={(event) => update(key, Number(event.target.value))}
          />
        </label>
      ))}
      <label className="setting toggle">
        <input
          type="checkbox"
          checked={settings.normalizeZwj}
          onChange={(event) => update("normalizeZwj", event.target.checked)}
        />
        <span>Normalize the zero-width joiner in questions</span>
      </label>
      <label className="setting toggle">
        <input
          type="checkbox"
          checked={settings.restoreZwj}
          onChange={(event) => update("restoreZwj", event.target.checked)}
        />
        <span>Join conjuncts with the zero-width joiner in answers</span>
      </label>
      <div className="settings-actions">
        <button
          type="button"
          className="secondary"
          disabled={isDefault(settings)}
          onClick={() => onChange({ ...DEFAULT_SETTINGS })}
        >
          Reset to defaults
        </button>
      </div>
      <p className="hint">
        Settings apply to the next question. A temperature of 0 always picks the most likely word.
      </p>
    </section>
  );
}
