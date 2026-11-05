const NAMED = Object.freeze({
  amp: "&",
  apos: "'",
  gt: ">",
  lt: "<",
  nbsp: " ",
  quot: '"',
});

const ENTITY = /&(?:#(\d{1,7})|#[xX]([0-9a-fA-F]{1,6})|([a-zA-Z]{2,8}));/g;
const PENDING = /&(?:#[xX]?[0-9a-fA-F]{0,7}|[a-zA-Z]{0,8})$/;

function fromCodePoint(codePoint, original) {
  const valid =
    codePoint > 0 && codePoint <= 0x10ffff && (codePoint < 0xd800 || codePoint > 0xdfff);
  return valid ? String.fromCodePoint(codePoint) : original;
}

export function decodeEntities(text) {
  return text.replace(ENTITY, (match, decimal, hex, name) => {
    if (name) return Object.hasOwn(NAMED, name) ? NAMED[name] : match;
    return fromCodePoint(decimal ? Number(decimal) : Number.parseInt(hex, 16), match);
  });
}

export function createEntityDecoder() {
  let pending = "";
  return {
    push(chunk) {
      const text = pending + chunk;
      const tail = PENDING.exec(text);
      pending = tail ? tail[0] : "";
      return decodeEntities(tail ? text.slice(0, tail.index) : text);
    },
    flush() {
      const text = decodeEntities(pending);
      pending = "";
      return text;
    },
  };
}
