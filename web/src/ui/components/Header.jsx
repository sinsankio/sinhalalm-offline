import { MODEL } from "../content.js";

export default function Header() {
  return (
    <header className="header">
      <div>
        <h1 className="title">
          සිංහල ප්‍රශ්න <span className="title-latin">Offline Q&amp;A</span>
        </h1>
        <p className="subtitle">
          Ask a question in Sinhala. {MODEL.name} answers it inside your browser, with no server and
          no connection once the model is downloaded.
        </p>
      </div>
    </header>
  );
}
