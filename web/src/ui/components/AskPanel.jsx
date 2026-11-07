import { useState } from "react";
import { SAMPLE_QUESTIONS, VISIBLE_SAMPLES } from "../content.js";

export default function AskPanel({ question, onChange, ready }) {
  const [showAll, setShowAll] = useState(false);
  const submit = (event) => event.preventDefault();
  const samples = showAll ? SAMPLE_QUESTIONS : SAMPLE_QUESTIONS.slice(0, VISIBLE_SAMPLES);

  return (
    <form className="panel ask" onSubmit={submit}>
      <label className="panel-label" htmlFor="question">
        Your question
      </label>
      <textarea
        id="question"
        className="question"
        lang="si"
        rows={3}
        placeholder="ප්‍රශ්නය මෙහි ලියන්න…"
        value={question}
        onChange={(event) => onChange(event.target.value)}
      />
      <div className="samples" aria-label="Sample questions">
        {samples.map((sample) => (
          <button
            key={sample}
            type="button"
            className="chip"
            lang="si"
            onClick={() => onChange(sample)}
          >
            {sample}
          </button>
        ))}
        <button type="button" className="chip chip-more" onClick={() => setShowAll(!showAll)}>
          {showAll
            ? "Fewer examples"
            : `More examples (${SAMPLE_QUESTIONS.length - VISIBLE_SAMPLES})`}
        </button>
      </div>
      <div className="actions">
        <button type="submit" className="primary" disabled={!ready || !question.trim()}>
          Ask
        </button>
        {!ready && <span className="hint">Available once the model is loaded</span>}
      </div>
    </form>
  );
}
