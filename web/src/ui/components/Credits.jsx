import { MODEL } from "../content.js";

export default function Credits() {
  return (
    <section className="panel credits">
      <h2 className="panel-label">About the model</h2>
      <p>
        <a href={MODEL.card} target="_blank" rel="noreferrer">
          {MODEL.name}
        </a>{" "}
        ({MODEL.version}) by {MODEL.author}, released under the {MODEL.license}. It is built on{" "}
        <a href={MODEL.baseCard} target="_blank" rel="noreferrer">
          {MODEL.base}
        </a>
        .
      </p>
      <p>
        This page runs a 4-bit version of the model, about {MODEL.downloadGb} GB, which is
        downloaded once and then kept in your browser.
      </p>
    </section>
  );
}
