import { useCallback, useEffect, useState } from "react";
import { VARIANTS } from "../../worker/variants.js";
import {
  BENCHMARK_QUESTIONS,
  BENCHMARK_SETTINGS,
  formatBytes,
  formatMs,
  summarizeBenchmark,
} from "../diagnostics.js";
import { addEntry, clearEntries, listEntries } from "../sessionLog.js";
import useMemory from "../useMemory.js";

const BACKENDS = [
  { value: "auto", label: "Auto" },
  { value: "webgpu", label: "WebGPU" },
  { value: "wasm", label: "WASM" },
];

const speed = (stats) =>
  stats?.tokensPerSecond != null ? `${stats.tokensPerSecond.toFixed(1)} tokens/s` : "–";

function Facts({ rows }) {
  return (
    <dl className="facts">
      {rows.map(([label, value]) => (
        <div key={label} className="fact">
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function Runtime({ model }) {
  const { runtime } = model;
  if (model.state !== "ready") {
    return <p className="hint">Runtime details appear once a model is loaded.</p>;
  }
  const adapter = runtime.adapter
    ? `${runtime.adapter.vendor} ${runtime.adapter.architecture}`
    : "none";
  return (
    <Facts
      rows={[
        ["Variant", `${runtime.dtype} (${VARIANTS[runtime.dtype].label})`],
        ["Backend", runtime.device === "webgpu" ? "WebGPU" : "WASM"],
        ["GPU adapter", adapter],
        ["WebGPU support", runtime.gpu ?? "not checked"],
        ["WASM threads", runtime.wasmThreads ?? "–"],
        ["Cross-origin isolated", runtime.crossOriginIsolated ? "yes" : "no"],
        ["Loaded from", runtime.fromCache ? "this device" : "the network"],
        ["Load time", formatMs(runtime.loadMs)],
        ["Warm-up", formatMs(runtime.warmupMs)],
      ]}
    />
  );
}

function Memory({ memory, storage }) {
  const { heap, heapPeak, page, measure } = memory;
  const pageValue =
    page.state === "measuring"
      ? "measuring…"
      : page.state === "unavailable"
        ? "not available in this browser"
        : formatBytes(page.bytes);
  return (
    <>
      <Facts
        rows={[
          ["Page memory", pageValue],
          ["Page memory, peak", formatBytes(page.peak)],
          ["JavaScript heap", heap ? formatBytes(heap.used) : "not available"],
          ["JavaScript heap, peak", formatBytes(heapPeak)],
          ["Storage used", storage ? formatBytes(storage.usage) : "not available"],
          ["Storage quota", storage ? formatBytes(storage.quota) : "not available"],
          ["Storage kept", storage ? (storage.persisted ? "yes" : "may be cleared") : "–"],
        ]}
      />
      <div className="row-actions">
        <button
          type="button"
          className="secondary"
          disabled={page.state === "measuring"}
          onClick={measure}
        >
          Measure page memory
        </button>
      </div>
      <p className="hint">
        Page memory covers the page, its workers and the WebAssembly memory, but not GPU memory. The
        browser can take several seconds to measure it.
      </p>
    </>
  );
}

function Variants({ model, locked, onLoad, onRemove }) {
  const loaded = model.state === "ready" ? model.runtime.dtype : null;
  return (
    <>
      <ul className="variant-list">
        {Object.entries(VARIANTS).map(([dtype, variant]) => (
          <li key={dtype} className="variant">
            <div>
              <strong>
                {dtype} ({variant.label})
              </strong>
              {loaded === dtype && <span className="badge">loaded</span>}
              <div className="hint">
                {variant.downloadGb} GB ·{" "}
                {model.cached?.[dtype] ? "on this device" : "not downloaded"}
              </div>
            </div>
            <div className="cell-actions">
              <button
                type="button"
                className="chip"
                disabled={locked || loaded === dtype}
                onClick={() => onLoad("auto", dtype)}
              >
                {model.cached?.[dtype] ? "Load" : "Download"}
              </button>
              <button
                type="button"
                className="chip"
                disabled={locked || !model.cached?.[dtype]}
                onClick={() => onRemove(dtype)}
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>
      <div className="backend-switch" role="group" aria-label="Backend">
        <span className="hint">Run on</span>
        {BACKENDS.map(({ value, label }) => (
          <button
            key={value}
            type="button"
            className="chip"
            aria-pressed={model.state === "ready" && value === model.runtime?.device}
            disabled={locked}
            onClick={() => onLoad(value, model.dtype)}
          >
            {label}
          </button>
        ))}
      </div>
      <p className="hint">
        The 8-bit variant answers more accurately but is about twice the size. It is practical only
        on WebGPU.
      </p>
    </>
  );
}

function Benchmark({ benchmark, ready, locked, onRun }) {
  const { running, results, summary } = benchmark;
  return (
    <>
      <p className="hint">
        Asks {BENCHMARK_QUESTIONS.length} sample questions with greedy decoding and up to{" "}
        {BENCHMARK_SETTINGS.maxNewTokens} tokens each, on the loaded variant and backend.
      </p>
      <div className="row-actions">
        <button type="button" className="primary" disabled={!ready || locked} onClick={onRun}>
          {running ? `Running ${results.length + 1} of ${BENCHMARK_QUESTIONS.length}…` : "Run"}
        </button>
      </div>
      {results.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Tokens</th>
              <th>First token</th>
              <th>Speed</th>
            </tr>
          </thead>
          <tbody>
            {results.map((result) => (
              <tr key={result.question}>
                <td lang="si" className="cell-question">
                  {result.question}
                </td>
                <td>{result.stats?.newTokens ?? "–"}</td>
                <td>{formatMs(result.stats?.firstTokenMs)}</td>
                <td>{result.error ? "failed" : speed(result.stats)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {summary && (
        <p className="benchmark-summary">
          Median first token {formatMs(summary.medianFirstTokenMs)} · median speed{" "}
          {summary.medianTokensPerSecond != null
            ? `${summary.medianTokensPerSecond.toFixed(1)} tokens/s`
            : "–"}{" "}
          · {summary.tokens} tokens
        </p>
      )}
    </>
  );
}

function describeEntry(entry) {
  if (entry.type === "load") {
    return `Loaded in ${formatMs(entry.loadMs)}, warm-up ${formatMs(entry.warmupMs)}`;
  }
  if (entry.type === "benchmark") {
    return `Benchmark: median ${formatMs(entry.summary.medianFirstTokenMs)} to first token, ${
      entry.summary.medianTokensPerSecond?.toFixed(1) ?? "–"
    } tokens/s`;
  }
  if (entry.error) return `Failed: ${entry.error}`;
  return `${entry.stats?.newTokens ?? 0} tokens, ${speed(entry.stats)}, first token ${formatMs(
    entry.stats?.firstTokenMs,
  )}`;
}

function SessionLog({ entries, onClear }) {
  const exportLog = () => {
    const blob = new Blob([JSON.stringify(entries, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `sinhalallm-sessions-${new Date().toISOString().slice(0, 10)}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };
  return (
    <>
      <div className="row-actions">
        <button
          type="button"
          className="secondary"
          disabled={entries.length === 0}
          onClick={exportLog}
        >
          Export as JSON
        </button>
        <button
          type="button"
          className="secondary"
          disabled={entries.length === 0}
          onClick={onClear}
        >
          Clear log
        </button>
      </div>
      {entries.length === 0 ? (
        <p className="hint">Loads, answers and benchmark runs are recorded here.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Event</th>
              <th>Model</th>
              <th>Details</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.id}>
                <td>{new Date(entry.time).toLocaleTimeString()}</td>
                <td>{entry.source === "benchmark" ? "benchmark question" : entry.type}</td>
                <td>{entry.dtype ? `${entry.dtype} · ${entry.device}` : "–"}</td>
                <td>{describeEntry(entry)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="hint">The log is kept in this browser only. Questions are not stored.</p>
    </>
  );
}

export default function Diagnostics({ model, storage, locked, onLoad, onRemove, ask, logVersion }) {
  const memory = useMemory(true);
  const [entries, setEntries] = useState([]);
  const [benchmark, setBenchmark] = useState({ running: false, results: [], summary: null });
  const refresh = useCallback(() => listEntries().then(setEntries), []);

  useEffect(() => {
    refresh();
  }, [refresh, logVersion]);

  const runBenchmark = async () => {
    setBenchmark({ running: true, results: [], summary: null });
    const results = [];
    for (const question of BENCHMARK_QUESTIONS) {
      const result = await ask(question, BENCHMARK_SETTINGS, { show: false, source: "benchmark" });
      results.push({ question, ...result });
      setBenchmark({ running: true, results: [...results], summary: null });
    }
    const summary = summarizeBenchmark(results);
    const pageBytes = await memory.measure();
    setBenchmark({ running: false, results, summary });
    await addEntry({
      type: "benchmark",
      dtype: model.runtime.dtype,
      device: model.runtime.device,
      adapter: model.runtime.adapter,
      summary,
      pageBytes,
      userAgent: navigator.userAgent,
    });
    refresh();
  };

  const busy = locked || benchmark.running || model.state !== "ready";
  return (
    <div className="diagnostics">
      <section className="panel">
        <h2 className="panel-label">Runtime</h2>
        <Runtime model={model} />
      </section>
      <section className="panel">
        <h2 className="panel-label">Memory</h2>
        <Memory memory={memory} storage={storage} />
      </section>
      <section className="panel">
        <h2 className="panel-label">Model and backend</h2>
        <Variants
          model={model}
          locked={
            locked || benchmark.running || ["loading", "warming", "removing"].includes(model.state)
          }
          onLoad={onLoad}
          onRemove={onRemove}
        />
      </section>
      <section className="panel panel-wide">
        <h2 className="panel-label">Benchmark</h2>
        <Benchmark
          benchmark={benchmark}
          ready={model.state === "ready"}
          locked={busy}
          onRun={runBenchmark}
        />
      </section>
      <section className="panel panel-wide">
        <h2 className="panel-label">Session log</h2>
        <SessionLog
          entries={entries}
          onClear={async () => {
            await clearEntries();
            refresh();
          }}
        />
      </section>
    </div>
  );
}
