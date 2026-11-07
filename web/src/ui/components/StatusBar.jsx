export default function StatusBar({ status }) {
  return (
    <section className={`panel status status-${status.state}`} aria-live="polite">
      <h2 className="panel-label">Model</h2>
      <p className="status-message">
        <span className="status-dot" aria-hidden="true" />
        {status.message}
      </p>
      {typeof status.progress === "number" && (
        <progress className="progress" max={1} value={status.progress} />
      )}
    </section>
  );
}
