const VIRAMA = "්";
const ZWJ = "‍";

export const ZWJ_CLUSTERS = Object.freeze([
  "ක්ර",
  "ග්ර",
  "ඝ්ර",
  "ත්ර",
  "ද්ර",
  "ප්ර",
  "ශ්ර",
  "ඛ්ය",
  "ජ්ය",
  "ට්ය",
  "ද්ය",
  "ධ්ය",
  "භ්ය",
  "ශ්ය",
  "ෂ්ය",
]);

const JOINED = new Set(ZWJ_CLUSTERS.map((cluster) => cluster[0] + cluster[2]));
const CONJUNCT = /([඀-෿])්‍?(?=([යර]))/g;

export function normalizeZwj(text) {
  return text.replace(CONJUNCT, (_, base, next) =>
    JOINED.has(base + next) ? base + VIRAMA + ZWJ : base + VIRAMA,
  );
}

const UNJOINED = /([ක-ෆ])්(?=[යර])/g;

export function restoreZwj(text) {
  return text.replace(UNJOINED, (match, base) => (base === "ර" ? match : base + VIRAMA + ZWJ));
}
