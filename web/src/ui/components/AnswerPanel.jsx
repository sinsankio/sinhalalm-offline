import { restoreZwj } from "../../text/zwj.js";

const REASONS = {
  length: "Stopped at the maximum answer length.",
  user: "Stopped.",
};

function Stats({ stats }) {
  const parts = [`${stats.newTokens} tokens`];
  if (stats.tokensPerSecond != null) parts.push(`${stats.tokensPerSecond.toFixed(1)} tokens/s`);
  if (stats.firstTokenMs != null)
    parts.push(`first token after ${(stats.firstTokenMs / 1000).toFixed(1)} s`);
  return <p className="answer-stats">{parts.join(" · ")}</p>;
}

export default function AnswerPanel({ answer, joinConjuncts }) {
  const { question, text, streaming, reason, stats, error } = answer;
  const trimmed = text.trimStart();
  const shown = joinConjuncts ? restoreZwj(trimmed) : trimmed;

  return (
    <section className="panel answer" aria-live="polite" aria-busy={streaming}>
      <h2 className="panel-label">Answer</h2>
      {question && (
        <p className="answer-question" lang="si">
          {question}
        </p>
      )}
      {shown || streaming ? (
        <p className="answer-text" lang="si">
          {shown}
          {streaming && <span className="cursor" aria-hidden="true" />}
        </p>
      ) : (
        !error && (
          <p className="placeholder">
            The answer will appear here, word by word, as it is generated.
          </p>
        )
      )}
      {error && <p className="answer-error">{error}</p>}
      {REASONS[reason] && <p className="hint">{REASONS[reason]}</p>}
      {stats && <Stats stats={stats} />}
    </section>
  );
}
