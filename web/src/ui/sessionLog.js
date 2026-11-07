const DB_NAME = "sinhalallm";
const STORE = "sessions";
const LIMIT = 200;

function openDatabase() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () =>
      request.result.createObjectStore(STORE, { keyPath: "id", autoIncrement: true });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function withStore(mode, action) {
  const db = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = db.transaction(STORE, mode);
      const result = action(transaction.objectStore(STORE));
      transaction.oncomplete = () => resolve(result?.result);
      transaction.onerror = () => reject(transaction.error);
    });
  } finally {
    db.close();
  }
}

export async function addEntry(entry) {
  try {
    await withStore("readwrite", (store) =>
      store.add({ time: new Date().toISOString(), ...entry }),
    );
  } catch {
    return;
  }
}

export async function listEntries() {
  try {
    const entries = await withStore("readonly", (store) => store.getAll());
    return entries.slice(-LIMIT).reverse();
  } catch {
    return [];
  }
}

export async function clearEntries() {
  try {
    await withStore("readwrite", (store) => store.clear());
  } catch {
    return;
  }
}
