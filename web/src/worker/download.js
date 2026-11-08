export const DOWNLOAD_ATTEMPTS = 2;

export class DamagedFileError extends Error {
  constructor(name) {
    super(`The downloaded file ${name} is damaged. Please try again.`);
    this.name = "DamagedFileError";
  }
}

export class DownloadStatusError extends Error {
  constructor(name, status) {
    super(`Could not download ${name} (${status}).`);
    this.name = "DownloadStatusError";
    this.status = status;
  }
}

const gigabytes = (bytes) => `${(bytes / 1e9).toFixed(2)} GB`;

export function storageMessage(needed, available) {
  const room = Number.isFinite(available)
    ? ` The browser allows ${gigabytes(Math.max(available, 0))} more.`
    : "";
  return `The model (${gigabytes(needed)}) could not be stored on this device.${room} Free some disk space or delete the other model variant, then try again.`;
}

export function interruptedMessage(name) {
  return `The download of ${name} stopped before it finished. Check the connection and try again. If the disk is nearly full, free some space first.`;
}

export function isStorageError(error) {
  return error?.name === "QuotaExceededError";
}

export function shouldRetry(error) {
  if (error instanceof DownloadStatusError) return error.status >= 500;
  return !isStorageError(error) && !(error instanceof DamagedFileError);
}

export function downloadError(error, { name, needed }) {
  if (error instanceof DamagedFileError || error instanceof DownloadStatusError) return error;
  if (isStorageError(error)) return new Error(storageMessage(needed), { cause: error });
  return new Error(interruptedMessage(name), { cause: error });
}
