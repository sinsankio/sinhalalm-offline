const SLIDERS = [
  { key: "temperature", label: "Temperature", min: 0, max: 1.5, step: 0.05 },
  { key: "topK", label: "Top-k", min: 1, max: 100, step: 1 },
  { key: "repetitionPenalty", label: "Repetition penalty", min: 1, max: 1.5, step: 0.01 },
  { key: "maxNewTokens", label: "Maximum answer length (tokens)", min: 32, max: 512, step: 16 },
];

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
      <p className="hint">
        Settings apply to the next question. A temperature of 0 always picks the most likely word.
      </p>
    </section>
  );
}
