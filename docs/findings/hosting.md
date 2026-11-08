# Hosting on a Hugging Face static Space

The application must load in a cross-origin isolated page, register its service worker, download the model once and then answer offline. A static Space serves only files, so every one of these needs had to be checked on the real host rather than on the local preview server. In this document, we report how the release is deployed, what the live checks showed, and the two limits of the host that changed the deployment (S18 and S19), then the four problems the browser tests found on the live Space (S20–S23).

### 1. The deployment

`make deploy` runs the local CI pipeline and refuses to continue unless the current commit carries a version tag. It then does three things:

1. It uploads the model package (q8, q4, the tokenizer, `NOTICE` and `manifest.json` with the SHA-256 of every file) and a model card to the public model repository `sinsankio/sinhalallm-1.7b-onnx-web`, and reads the commit of that upload.
2. It builds the application with `VITE_MODEL_REPO` and `VITE_MODEL_REVISION` set to that repository and commit, so a release always loads exactly the files it was tested with.
3. It uploads the built application to the static Space `sinsankio/sinhalalm-offline` and removes the files of the previous build. A new Space is created private; Sineth made this one public after the second release candidate.

The worker downloads each model file from `https://huggingface.co/{repository}/resolve/{commit}/{file}` into Cache Storage, hashes it while it streams, and rejects a file whose SHA-256 does not match the manifest. When the pinned commit changes, the worker removes the cached files of the old commit. In a local test, a changed `tokenizer_config.json` was rejected as damaged and removed.

### 2. Live checks of the host

| Check | Result |
|---|---|
| `custom_headers` in the Space front matter (COOP `same-origin`, COEP `require-corp`, CORP `cross-origin`) | applied |
| `crossOriginIsolated` in the page | `true` |
| Service worker | controls the page after the first reload |
| Reload without a connection | the application loads from the service worker cache |

**S18 — A Space holds at most 1 GB.** The first deployment uploaded the application and the model package to the Space, and the upload failed with:

```
403 Forbidden: Storage limit reached for this space (Max: 1 GB)
```

The q4 package alone is 1.18 GB, so the models cannot live in the Space. They now live in a separate model repository, and the Space holds only the application.

**S19 — A private Space needs a signed address.** Opening a private static Space needs a short-lived token from `/api/spaces/{id}/jwt`, passed once as `?__sign=<token>`. The browser tests accept the token from a file (`SINHALALM_E2E_SIGN_FILE`) so that it never appears in a command line or a log. A public Space needs no token.

### 3. Problems found with the browser tests on the live Space

The browser tests ran against the public Space with `make e2e-live`, which points the same suite at the deployed address. On the local server, the model files come from the same origin and carry no revision in their address, so none of the following problems could show up there.

**S20 — Chrome reports a dropped connection like a full disk.** The first live run failed after 16 minutes with "The model (1.19 GB) could not be stored on this device", with 23 GB free on the disk and a 10.7 GB quota. Two later downloads of the same files on the live origin finished, so the cause was most likely a dropped connection. Chrome rejects `Cache.put` with a network error in that case, the same error it gives on a nearly full disk (S17), and the worker turned every `Cache.put` error into the storage message. Only `QuotaExceededError` gives the storage message now; any other failure names the file that stopped, keeps the original error as the cause, and the worker downloads that file once more first.

**S21 — The Hub sends `tokenizer.json` compressed and without a size.** From the live origin, a HEAD request for `tokenizer.json` returned no `Content-Length` in three of three rounds, while every other file had one, since the Hub serves it with `Content-Encoding: gzip`. With one size unknown, the total was unknown, and the progress bar never appeared during the 20-minute download. The worker now takes the sizes from `manifest.json` and sends no HEAD requests, which also removes 3 to 10 seconds before the download starts. The streamed bytes are counted after decompression, so they match the sizes in the manifest.

**S22 — transformers.js asks for one tokenizer file on `main`.** With the model cached, the application did not load offline: "The model could not be loaded. Failed to fetch". The only failed request was `GET …/resolve/main/tokenizer_config.json`. Before it loads the tokenizer, transformers.js 4.3.1 checks that this file exists, and the check leaves out the revision passed to `from_pretrained`. The cache holds the files of the pinned commit only, so the check went to the network; online, it quietly read the file from `main`. The worker now writes the pinned commit into `env.remotePathTemplate`, so every request goes to that commit.

**S23 — A new model card would have forced a new download.** `make deploy` pinned the application to the latest commit of the model repository. A commit that changes only the model card would give the application a new revision, and the worker removes the cached files of every other revision, so every reader would download 1.19 or 2.48 GB again. The application is now pinned to the last commit that changed `manifest.json`, which lists the SHA-256 of every model file. For the current package, that is still `fd97e545`.

### 4. Live results

| Release | Browser tests on the Space | Notes |
|---|---|---|
| v0.1.0-rc.2 | first test failed, the rest did not run | download failed with the storage message (S20) |
| v0.1.0-rc.3 | first test failed, the rest did not run | no progress bar (S21) |
| v0.1.0-rc.4 | 5 of 7 | answers offline failed (S22); download 17.4 min, greedy answers identical to ONNX Runtime in Python |
| v0.1.0-rc.5 | 7 of 7 (43.1 min) | download 35.7 min, greedy answers identical, answers offline in 31 s |

The Hub served the model at about 1.1 MB/s to this machine, so a first download of the 4-bit model took 17 to 36 minutes. The headless tests have no GPU, so they ran the 4-bit model on WebAssembly.
