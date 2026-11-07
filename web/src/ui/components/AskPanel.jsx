import { SAMPLE_QUESTIONS } from "../content.js";

export default function AskPanel({ question, onChange, ready }) {
  const submit = (event) => event.preventDefault();

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
        {SAMPLE_QUESTIONS.map((sample) => (
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
