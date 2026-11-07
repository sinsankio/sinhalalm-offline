import { useState } from "react";
import AnswerPanel from "./components/AnswerPanel.jsx";
import AskPanel from "./components/AskPanel.jsx";
import Credits from "./components/Credits.jsx";
import Header from "./components/Header.jsx";
import Notice from "./components/Notice.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import StorageControl from "./components/StorageControl.jsx";
import { DEFAULT_SETTINGS } from "./content.js";
import useModel from "./useModel.js";

export default function App() {
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const { model, storage, answer, load, remove, ask, stop } = useModel();
  const ready = model.state === "ready";

  return (
    <div className="app">
      <Header />
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
          <AnswerPanel answer={answer} />
          <Notice />
        </section>
        <aside className="column column-side">
          <StatusBar model={model} locked={answer.streaming} onLoad={load} />
          <StorageControl
            storage={storage}
            model={model}
            locked={answer.streaming}
            onRemove={remove}
          />
          <SettingsPanel settings={settings} onChange={setSettings} />
          <Credits />
        </aside>
      </main>
    </div>
  );
}
