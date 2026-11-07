const canShare = () =>
  typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated === true;

export function createStopSignal(buffer = canShare() ? new SharedArrayBuffer(4) : null) {
  const flag = buffer ? new Int32Array(buffer) : null;
  let local = false;
  return {
    buffer,
    request() {
      local = true;
      if (flag) Atomics.store(flag, 0, 1);
    },
    reset() {
      local = false;
      if (flag) Atomics.store(flag, 0, 0);
    },
    get requested() {
      return local || (flag !== null && Atomics.load(flag, 0) === 1);
    },
  };
}
