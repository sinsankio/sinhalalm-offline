import { useState } from "react";
import { SAMPLE_QUESTIONS, VISIBLE_SAMPLES } from "../content.js";

export default function AskPanel({ question, onChange, ready, answering, hint, onAsk, onStop }) {
  const [showAll, setShowAll] = useState(false);
  const canAsk = ready && !answering && question.trim().length > 0;
  const submit = (event) => {
    event.preventDefault();
    if (canAsk) onAsk(question);
  };
  const submitOnShortcut = (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) submit(event);
  };
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
        onKeyDown={submitOnShortcut}
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
        {answering ? (
          <button type="button" className="secondary" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="primary" disabled={!canAsk}>
            Ask
          </button>
        )}
        <span className="hint">{ready ? "Ctrl + Enter to ask" : hint}</span>
      </div>
    </form>
  );
}
