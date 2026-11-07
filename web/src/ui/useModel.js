import { useCallback, useEffect, useRef, useState } from "react";
import { DEFAULT_VARIANT, VARIANTS } from "../worker/variants.js";
import { createStopSignal } from "../worker/stop-signal.js";

const VARIANT_KEY = "sinhalallm-variant";
const INITIAL = {
  state: "checking",
  dtype: DEFAULT_VARIANT,
  device: null,
  progress: null,
  runtime: null,
  error: null,
  cached: {},
};
const NO_ANSWER = {
  id: 0,
  question: "",
  text: "",
  streaming: false,
  reason: null,
  stats: null,
  error: null,
};

function readVariant() {
  try {
    const stored = localStorage.getItem(VARIANT_KEY);
    return VARIANTS[stored] ? stored : DEFAULT_VARIANT;
  } catch {
    return DEFAULT_VARIANT;
  }
}

function saveVariant(dtype) {
  try {
    localStorage.setItem(VARIANT_KEY, dtype);
  } catch {
    return;
  }
}

async function estimateStorage() {
  try {
    const { usage, quota } = await navigator.storage.estimate();
    const persisted = (await navigator.storage.persisted?.()) ?? false;
    return { usage, quota, persisted };
  } catch {
    return null;
  }
}

async function requestPersistence() {
  try {
    if (!(await navigator.storage.persisted())) await navigator.storage.persist();
  } catch {
    return;
  }
}

export default function useModel({ onEvent } = {}) {
  const workerRef = useRef(null);
  const [model, setModel] = useState(() => ({ ...INITIAL, dtype: readVariant() }));
  const [storage, setStorage] = useState(null);
  const [answer, setAnswer] = useState(NO_ANSWER);
  const pendingRef = useRef("");
  const frameRef = useRef(0);
  const answerIdRef = useRef(0);
  const visibleIdRef = useRef(0);
  const dtypeRef = useRef(model.dtype);
  const stopRef = useRef(null);
  const waitingRef = useRef(new Map());
  const metaRef = useRef(new Map());
  const runtimeRef = useRef(null);
  const eventRef = useRef(onEvent);

  useEffect(() => {
    eventRef.current = onEvent;
  }, [onEvent]);

  const refreshStorage = useCallback(() => {
    estimateStorage().then(setStorage);
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("../worker/inference.worker.js", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;
    const flush = () => {
      frameRef.current = 0;
      const text = pendingRef.current;
      pendingRef.current = "";
      if (text) setAnswer((current) => ({ ...current, text: current.text + text }));
    };
    const finish = (id, result) => {
      cancelAnimationFrame(frameRef.current);
      frameRef.current = 0;
      pendingRef.current = "";
      waitingRef.current.get(id)?.(result);
      waitingRef.current.delete(id);
    };

    worker.addEventListener("message", ({ data }) => {
      if (data.type === "answer" || data.type === "answer-error") {
        const result =
          data.type === "answer"
            ? { text: data.text, reason: data.reason, stats: data.stats }
            : { error: data.message };
        const meta = metaRef.current.get(data.id) ?? {};
        metaRef.current.delete(data.id);
        eventRef.current?.({ type: "answer", runtime: runtimeRef.current, ...meta, ...result });
        finish(data.id, result);
        if (data.id !== visibleIdRef.current) return;
        setAnswer((current) => ({ ...current, ...result, streaming: false }));
      } else if (data.type === "chunk") {
        if (data.id !== visibleIdRef.current) return;
        pendingRef.current += data.text;
        frameRef.current ||= requestAnimationFrame(flush);
      } else if (data.type === "probe") {
        const dtype = readVariant();
        dtypeRef.current = dtype;
        if (data.cached[dtype]) {
          worker.postMessage({ type: "load", dtype });
          setModel({ ...INITIAL, dtype, state: "loading", fromCache: true, cached: data.cached });
        } else {
          setModel({ ...INITIAL, dtype, state: "download-required", cached: data.cached });
        }
      } else if (data.type === "status") {
        if (data.state === "idle") return;
        setModel((current) => ({
          ...current,
          state: data.state,
          dtype: data.dtype ?? current.dtype,
          device: data.device ?? current.device,
          fromCache: data.fromCache ?? current.fromCache,
          error: null,
        }));
      } else if (data.type === "progress") {
        setModel((current) => ({ ...current, progress: data }));
      } else if (data.type === "ready") {
        runtimeRef.current = data.runtime;
        dtypeRef.current = data.runtime.dtype;
        setModel({
          ...INITIAL,
          state: "ready",
          dtype: data.runtime.dtype,
          device: data.runtime.device,
          runtime: data.runtime,
          cached: data.cached,
        });
        eventRef.current?.({ type: "load", runtime: data.runtime });
        requestPersistence().then(refreshStorage);
      } else if (data.type === "removed") {
        setModel((current) => {
          const loaded = data.loaded && current.state === "ready";
          return loaded
            ? { ...current, cached: data.cached }
            : { ...INITIAL, dtype: current.dtype, state: "download-required", cached: data.cached };
        });
        if (!data.loaded) runtimeRef.current = null;
        refreshStorage();
      } else if (data.type === "error") {
        setModel((current) => ({ ...current, state: "error", error: data.message }));
        refreshStorage();
      }
    });
    stopRef.current = createStopSignal();
    worker.postMessage({ type: "init", stopBuffer: stopRef.current.buffer });
    worker.postMessage({ type: "probe" });
    refreshStorage();
    return () => {
      cancelAnimationFrame(frameRef.current);
      worker.terminate();
    };
  }, [refreshStorage]);

  const load = useCallback((preferred, dtype) => {
    const variant = dtype ?? dtypeRef.current;
    dtypeRef.current = variant;
    saveVariant(variant);
    setModel((current) => ({
      ...INITIAL,
      dtype: variant,
      state: "loading",
      cached: current.cached,
    }));
    workerRef.current?.postMessage({ type: "load", preferred, dtype: variant });
  }, []);

  const select = useCallback(
    (dtype, cached, preferred) => {
      if (cached) {
        load(preferred, dtype);
        return;
      }
      dtypeRef.current = dtype;
      saveVariant(dtype);
      runtimeRef.current = null;
      workerRef.current?.postMessage({ type: "unload" });
      setModel((current) => ({
        ...INITIAL,
        dtype,
        state: "download-required",
        cached: current.cached,
      }));
    },
    [load],
  );

  const remove = useCallback((dtype = null) => {
    setModel((current) =>
      dtype && !(current.state === "ready" && current.dtype === dtype)
        ? current
        : { ...current, state: "removing" },
    );
    workerRef.current?.postMessage({ type: "remove", dtype });
  }, []);

  const ask = useCallback((question, settings, { show = true, source = "ask" } = {}) => {
    answerIdRef.current += 1;
    const id = answerIdRef.current;
    metaRef.current.set(id, { question, source });
    pendingRef.current = "";
    if (show) {
      visibleIdRef.current = id;
      setAnswer({ ...NO_ANSWER, id, question, streaming: true });
    }
    const done = new Promise((resolve) => waitingRef.current.set(id, resolve));
    workerRef.current?.postMessage({ type: "generate", id, question, settings });
    return done;
  }, []);

  const stop = useCallback(() => {
    stopRef.current?.request();
    workerRef.current?.postMessage({ type: "stop" });
  }, []);

  return { model, storage, answer, load, select, remove, ask, stop };
}
