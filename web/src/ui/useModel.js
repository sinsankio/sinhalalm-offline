import { useCallback, useEffect, useRef, useState } from "react";

const INITIAL = { state: "checking", device: null, progress: null, runtime: null, error: null };

async function estimateStorage() {
  try {
    const { usage, quota } = await navigator.storage.estimate();
    return { usage, quota };
  } catch {
    return null;
  }
}

export default function useModel() {
  const workerRef = useRef(null);
  const [model, setModel] = useState(INITIAL);
  const [storage, setStorage] = useState(null);

  const refreshStorage = useCallback(() => {
    estimateStorage().then(setStorage);
  }, []);

  useEffect(() => {
    const worker = new Worker(new URL("../worker/inference.worker.js", import.meta.url), {
      type: "module",
    });
    workerRef.current = worker;
    worker.addEventListener("message", ({ data }) => {
      if (data.type === "probe") {
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
        refreshStorage();
      } else if (data.type === "removed") {
        setModel({ ...INITIAL, state: "download-required" });
        refreshStorage();
      } else if (data.type === "error") {
        setModel((current) => ({ ...current, state: "error", error: data.message }));
        refreshStorage();
      }
    });
    worker.postMessage({ type: "probe" });
    refreshStorage();
    return () => worker.terminate();
  }, [refreshStorage]);

  const load = useCallback((preferred) => {
    setModel({ ...INITIAL, state: "loading" });
    workerRef.current?.postMessage({ type: "load", preferred });
  }, []);

  const remove = useCallback(() => {
    setModel({ ...INITIAL, state: "removing" });
    workerRef.current?.postMessage({ type: "remove" });
  }, []);

  return { model, storage, load, remove };
}
