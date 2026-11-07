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
  const { model, storage, load, remove } = useModel();

  return (
    <div className="app">
      <Header />
      <main className="layout">
        <section className="column column-main">
          <AskPanel
            question={question}
            onChange={setQuestion}
            ready={false}
            hint={
              model.state === "ready"
                ? "Answering is connected in the next step"
                : "Available once the model is loaded"
            }
          />
          <AnswerPanel answer="" streaming={false} />
          <Notice />
        </section>
        <aside className="column column-side">
          <StatusBar model={model} onLoad={load} />
          <StorageControl storage={storage} model={model} onRemove={remove} />
          <SettingsPanel settings={settings} onChange={setSettings} />
          <Credits />
        </aside>
      </main>
    </div>
  );
}
