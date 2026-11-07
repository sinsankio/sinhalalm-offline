export default function AnswerPanel({ answer, streaming }) {
  return (
    <section className="panel answer" aria-live="polite">
      <h2 className="panel-label">Answer</h2>
      {answer ? (
        <p className="answer-text" lang="si">
          {answer}
          {streaming && <span className="cursor" aria-hidden="true" />}
        </p>
      ) : (
        <p className="placeholder">
          The answer will appear here, word by word, as it is generated.
        </p>
      )}
    </section>
  );
}
