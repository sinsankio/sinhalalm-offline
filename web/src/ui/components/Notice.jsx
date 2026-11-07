import { LIMITATIONS } from "../content.js";

export default function Notice() {
  return (
    <section className="panel notice" aria-label="Limitations">
      <h2 className="panel-label">Before you rely on an answer</h2>
      <ul>
        {LIMITATIONS.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
