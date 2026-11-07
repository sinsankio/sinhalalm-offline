import { MODEL } from "../content.js";
import ThemeSwitch from "./ThemeSwitch.jsx";

export default function Header() {
  return (
    <header className="header">
      <div>
        <h1 className="title">
          Offline සිංහල <span className="title-latin">Offline Q&amp;A in Sinhala Language</span>
        </h1>
        <p className="subtitle">
          Ask a question in Sinhala. {MODEL.name} answers it inside your browser, with no server and
          no connection once the model is downloaded.
        </p>
      </div>
      <ThemeSwitch />
    </header>
  );
}
