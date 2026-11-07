import { useCallback, useEffect, useState } from "react";

const SAMPLE_MS = 2000;

function heap() {
  const memory = performance.memory;
  return memory ? { used: memory.usedJSHeapSize, limit: memory.jsHeapSizeLimit } : null;
}

export default function useMemory(active) {
  const [sample, setSample] = useState(heap);
  const [peak, setPeak] = useState(() => heap()?.used ?? null);
  const [page, setPage] = useState({ state: "idle", bytes: null, peak: null });

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(() => {
      const next = heap();
      setSample(next);
      if (next) setPeak((current) => Math.max(current ?? 0, next.used));
    }, SAMPLE_MS);
    return () => clearInterval(timer);
  }, [active]);

  const measure = useCallback(async () => {
    if (!performance.measureUserAgentSpecificMemory || !self.crossOriginIsolated) {
      setPage({ state: "unavailable", bytes: null, peak: null });
      return null;
    }
    setPage((current) => ({ ...current, state: "measuring" }));
    try {
      const { bytes } = await performance.measureUserAgentSpecificMemory();
      setPage((current) => ({ state: "done", bytes, peak: Math.max(current.peak ?? 0, bytes) }));
      return bytes;
    } catch {
      setPage((current) => ({ ...current, state: "unavailable" }));
      return null;
    }
  }, []);

  return { heap: sample, heapPeak: peak, page, measure };
}
