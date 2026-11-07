const LOCAL_ID = "sinhalallm-1.7b/v2.1";
const HUB = "https://huggingface.co/";

export function modelSource({ repo, revision } = {}) {
  if (!repo) {
    return { remote: false, id: LOCAL_ID, revision: "main", base: `/models/${LOCAL_ID}/` };
  }
  const pinned = revision || "main";
  return {
    remote: true,
    id: repo,
    revision: pinned,
    base: `${HUB}${repo}/resolve/${encodeURIComponent(pinned)}/`,
  };
}

export const MODEL_SOURCE = modelSource({
  repo: import.meta.env?.VITE_MODEL_REPO,
  revision: import.meta.env?.VITE_MODEL_REVISION,
});

export const HUB_HOST = HUB;
