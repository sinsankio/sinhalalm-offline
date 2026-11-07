import { useState } from "react";
import AnswerPanel from "./components/AnswerPanel.jsx";
import AskPanel from "./components/AskPanel.jsx";
import Credits from "./components/Credits.jsx";
import Header from "./components/Header.jsx";
import Notice from "./components/Notice.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import { DEFAULT_SETTINGS } from "./content.js";

export default function App() {
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const status = { state: "idle", message: "Model loading is not connected yet" };

  return (
    <div className="app">
      <Header />
      <main className="layout">
        <section className="column column-main">
          <AskPanel question={question} onChange={setQuestion} ready={false} />
          <AnswerPanel answer="" streaming={false} />
          <Notice />
        </section>
        <aside className="column column-side">
          <StatusBar status={status} />
          <SettingsPanel settings={settings} onChange={setSettings} />
          <Credits />
        </aside>
      </main>
    </div>
  );
}
