import { useCallback, useEffect, useRef, useState } from "react";
import { createStopSignal } from "../worker/stop-signal.js";

const INITIAL = { state: "checking", device: null, progress: null, runtime: null, error: null };
const NO_ANSWER = {
  id: 0,
  question: "",
  text: "",
  streaming: false,
  reason: null,
  stats: null,
  error: null,
};

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

export default function useModel() {
  const workerRef = useRef(null);
  const [model, setModel] = useState(INITIAL);
  const [storage, setStorage] = useState(null);
  const [answer, setAnswer] = useState(NO_ANSWER);
  const pendingRef = useRef("");
  const frameRef = useRef(0);
  const answerIdRef = useRef(0);
  const stopRef = useRef(null);

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
    worker.addEventListener("message", ({ data }) => {
      if (data.id != null && data.id !== answerIdRef.current) return;
      if (data.type === "chunk") {
        pendingRef.current += data.text;
        frameRef.current ||= requestAnimationFrame(flush);
      } else if (data.type === "answer") {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = 0;
        pendingRef.current = "";
        setAnswer((current) => ({
          ...current,
          text: data.text,
          streaming: false,
          reason: data.reason,
          stats: data.stats,
        }));
      } else if (data.type === "answer-error") {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = 0;
        pendingRef.current = "";
        setAnswer((current) => ({ ...current, streaming: false, error: data.message }));
      } else if (data.type === "probe") {
        if (data.fromCache) {
          worker.postMessage({ type: "load" });
          setModel({ ...INITIAL, state: "loading", fromCache: true });
        } else {
          setModel({ ...INITIAL, state: "download-required" });
        }
      } else if (data.type === "status") {
        setModel((current) => ({
          ...current,
          state: data.state,
          device: data.device ?? current.device,
          fromCache: data.fromCache ?? current.fromCache,
          error: null,
        }));
      } else if (data.type === "progress") {
        setModel((current) => ({ ...current, progress: data }));
      } else if (data.type === "ready") {
        setModel({
          ...INITIAL,
          state: "ready",
          device: data.runtime.device,
          runtime: data.runtime,
        });
        requestPersistence().then(refreshStorage);
      } else if (data.type === "removed") {
        setModel({ ...INITIAL, state: "download-required" });
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

  const load = useCallback((preferred) => {
    setModel({ ...INITIAL, state: "loading" });
    workerRef.current?.postMessage({ type: "load", preferred });
  }, []);

  const remove = useCallback(() => {
    setModel({ ...INITIAL, state: "removing" });
    workerRef.current?.postMessage({ type: "remove" });
  }, []);

  const ask = useCallback((question, settings) => {
    answerIdRef.current += 1;
    const id = answerIdRef.current;
    pendingRef.current = "";
    setAnswer({ ...NO_ANSWER, id, question, streaming: true });
    workerRef.current?.postMessage({ type: "generate", id, question, settings });
  }, []);

  const stop = useCallback(() => {
    stopRef.current?.request();
    workerRef.current?.postMessage({ type: "stop" });
  }, []);

  return { model, storage, answer, load, remove, ask, stop };
}
