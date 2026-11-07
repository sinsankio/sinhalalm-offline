import { useState } from "react";

const gigabytes = (bytes) => `${(bytes / 1e9).toFixed(2)} GB`;

export default function StorageControl({ storage, model, locked, onRemove }) {
  const [confirming, setConfirming] = useState(false);
  const busy = locked || ["checking", "loading", "warming", "removing"].includes(model.state);
  const hasModel = model.state === "ready" || model.state === "error";

  return (
    <section className="panel storage">
      <h2 className="panel-label">Storage</h2>
      <p className="storage-usage">
        {storage
          ? `This site uses ${gigabytes(storage.usage)} of the ${gigabytes(storage.quota)} the browser allows.`
          : "The browser does not report storage usage."}
      </p>
      {storage && hasModel && (
        <p className="hint">
          {storage.persisted
            ? "The browser keeps the model until you delete it."
            : "The browser may delete the model when the disk runs low on space."}
        </p>
      )}
      {model.state === "removing" && <p className="hint">Deleting the downloaded model…</p>}
      {hasModel && !confirming && (
        <button
          type="button"
          className="secondary"
          disabled={busy}
          onClick={() => setConfirming(true)}
        >
          Delete downloaded model
        </button>
      )}
      {hasModel && confirming && (
        <div className="confirm">
          <p>
            Delete the model from this browser? You will need to download it again (about 1.19 GB)
            before you can ask questions.
          </p>
          <div className="confirm-actions">
            <button
              type="button"
              className="danger"
              onClick={() => {
                setConfirming(false);
                onRemove();
              }}
            >
              Delete
            </button>
            <button type="button" className="secondary" onClick={() => setConfirming(false)}>
              Cancel
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
