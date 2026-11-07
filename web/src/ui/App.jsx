import { useCallback, useEffect, useState } from "react";
import AnswerPanel from "./components/AnswerPanel.jsx";
import AskPanel from "./components/AskPanel.jsx";
import Credits from "./components/Credits.jsx";
import Diagnostics from "./components/Diagnostics.jsx";
import Header from "./components/Header.jsx";
import Notice from "./components/Notice.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import StorageControl from "./components/StorageControl.jsx";
import { addEntry } from "./sessionLog.js";
import { readSettings, saveSettings } from "./settings.js";
import useModel from "./useModel.js";

const TABS = [
  { id: "ask", label: "Ask" },
  { id: "diagnostics", label: "Diagnostics" },
];

function logEntry(event) {
  const { runtime } = event;
  const model = runtime ? { dtype: runtime.dtype, device: runtime.device } : {};
  if (event.type === "load") {
    return {
      type: "load",
      ...model,
      adapter: runtime.adapter,
      fromCache: runtime.fromCache,
      loadMs: runtime.loadMs,
      warmupMs: runtime.warmupMs,
    };
  }
  return {
    type: "answer",
    source: event.source,
    ...model,
    reason: event.reason ?? null,
    stats: event.stats ?? null,
    error: event.error ?? null,
  };
}

export default function App() {
  const [tab, setTab] = useState("ask");
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState(readSettings);
  const [logVersion, setLogVersion] = useState(0);
  const onEvent = useCallback((event) => {
    addEntry(logEntry(event)).then(() => setLogVersion((version) => version + 1));
  }, []);
  const { model, storage, answer, load, remove, ask, stop } = useModel({ onEvent });
  const ready = model.state === "ready";

  useEffect(() => saveSettings(settings), [settings]);

  return (
    <div className="app">
      <Header />
      <nav className="tabs" role="tablist" aria-label="Sections">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            role="tab"
            className="tab"
            aria-selected={tab === id}
            onClick={() => setTab(id)}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "ask" ? (
        <main className="layout">
          <section className="column column-main">
            <AskPanel
              question={question}
              onChange={setQuestion}
              ready={ready}
              answering={answer.streaming}
              hint="Available once the model is loaded"
              onAsk={(text) => ask(text, settings)}
              onStop={stop}
            />
            <AnswerPanel answer={answer} joinConjuncts={settings.restoreZwj} />
            <Notice />
          </section>
          <aside className="column column-side">
            <StatusBar model={model} locked={answer.streaming} onLoad={load} />
            <StorageControl
              storage={storage}
              model={model}
              locked={answer.streaming}
              onRemove={() => remove()}
            />
            <SettingsPanel settings={settings} onChange={setSettings} />
            <Credits />
          </aside>
        </main>
      ) : (
        <main>
          <StatusBar model={model} locked={answer.streaming} onLoad={load} />
          <Diagnostics
            model={model}
            storage={storage}
            locked={answer.streaming}
            onLoad={load}
            onRemove={remove}
            ask={ask}
            logVersion={logVersion}
          />
        </main>
      )}
    </div>
  );
}
