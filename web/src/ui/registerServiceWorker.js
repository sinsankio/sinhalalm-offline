const RELOAD_KEY = "sinhalallm-isolation-reload";

export function registerServiceWorker() {
  if (!("serviceWorker" in navigator)) return;
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    if (self.crossOriginIsolated) return;
    try {
      if (sessionStorage.getItem(RELOAD_KEY)) return;
      sessionStorage.setItem(RELOAD_KEY, "1");
    } catch {
      return;
    }
    location.reload();
  });
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}
