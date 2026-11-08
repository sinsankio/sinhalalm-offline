# Changelog

All notable changes to this project are recorded here. Entries are grouped by the branch that introduced them, and each entry also records the attempts that were revised or rejected along the way, so that the reasoning behind the final state stays visible.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### fix/deploy-pin — A new model card would force a new download

#### Fixed

- `make deploy` pinned the application to the latest commit of the model repository. A commit that changed only the model card would therefore give the application a new revision, and the worker, which removes the cached files of other revisions, would make every reader download the model again (1.19 or 2.48 GB). The application is now pinned to the last commit that changed `manifest.json`. The manifest lists the SHA-256 of every model file, so that commit changes only when the model files change.

#### Results

- On `sinsankio/sinhalallm-1.7b-onnx-web`, the new lookup returns `fd97e545`, the commit the live application already uses.

### fix/pinned-revision — The tokenizer check ignored the pinned revision

#### Fixed

- On the live Space, the application did not load offline: "The model could not be loaded. Failed to fetch". Before it loads the tokenizer, transformers.js 4.3.1 checks that `tokenizer_config.json` exists, and that check drops the revision passed to `from_pretrained`. It asked for `resolve/main/tokenizer_config.json`, which is not in the cache (the cache holds the files of the pinned commit), so it went to the network. Online, this quietly read one file from `main` instead of the pinned commit. The worker now writes the pinned commit into `env.remotePathTemplate`, so every request goes to the pinned commit, even one that leaves the revision out.

#### Results

- With the cached profile of the live test opened offline, the only failed request was `GET …/resolve/main/tokenizer_config.json` (`net::ERR_INTERNET_DISCONNECTED`). The local tests never saw this, since a local model has no revision in its URLs.

### fix/download-progress — No progress bar for downloads from the Hub

#### Fixed

- On the live Space, the model panel said "Downloading the model…" without a progress bar for the whole download. The worker read each file's size from a HEAD request, and the Hub sends `tokenizer.json` compressed, without a `Content-Length`, so the total stayed unknown. The worker now takes the sizes from `manifest.json`, which already lists them, and no longer sends the HEAD requests, which took 3 to 10 seconds from the live origin before the download started.

#### Results

- On the local server, the progress bar appeared 119 ms after the download started, with "0 MB of 1189 MB".

### fix/download-errors — Interrupted downloads were reported as a full disk

#### Fixed

- The first browser test against the live Space failed with "The model (1.19 GB) could not be stored on this device", with 23 GB free on the disk and a 10.7 GB quota. The worker reported every `Cache.put` error as a storage problem, but Chrome raises the same error when the connection drops during a download. Only `QuotaExceededError` now gives the storage message. Any other failure says that the download of the named file stopped before it finished, and the original error is kept as the cause and logged in the console.
- The worker now downloads a file a second time after a network error or a server error (5xx). A damaged file, a missing file or a full quota is not retried.

#### Results

- Two later downloads of the 4-bit model from the Hub on the live origin both finished (about 20 minutes each at 1.1 MB/s), so the first failure could not be reproduced and was most likely a dropped connection.
- On the local server, with the connection reset once for `tokenizer.json`, the worker retried and the model loaded. With the connection reset every time, it stopped after two attempts with the new message and removed the partial files.

### feat/q8-default — The 8-bit model as default

#### Changed

- On hardware WebGPU, the application now offers the 8-bit model (2.48 GB) by default. Without hardware WebGPU, the 4-bit model (1.19 GB) is the default, since the 8-bit model is not practical on WebAssembly. A downloaded 4-bit model is kept instead of downloading the 8-bit one, and a variant chosen by the reader always wins.
- When the 8-bit model cannot be downloaded, stored or loaded, the worker removes its partial files, loads the 4-bit model instead, remembers that choice and says why in the model panel.
- The package manifest names q8 as the default and q4 as the fallback.

#### Results

- In headless Chrome on WebGPU, the first visit offered the 8-bit model and loaded it. Without WebGPU, it offered the 4-bit model. With one q8 data file blocked, the download failed, the partial files were removed and the 4-bit model was loaded with the reason shown.

#### Decisions

- Sineth made q8 the default with q4 only as a fallback. The download target of 1.2 GB therefore applies to the q4 fallback, while the default download is 2.48 GB.

### feat/phase4-defaults — Decisions on the acceptance targets

#### Added

- A warning in the model panel when the 8-bit model runs on WebAssembly, which recommends the 4-bit model in that browser.

#### Decisions

- Sineth kept the proposed defaults for the four open questions. The WebAssembly time-to-first-token target is dropped, since WebAssembly is the slow fallback. The 2 GB memory target counts the page memory measured by the browser, with the renderer process recorded alongside. The ZWJ rule for answers stays as it is until a native speaker names words that need a visible virama. q8 stays available on WebAssembly with a warning.

### docs/phase4-wrapup — Documentation at the end of Phase 4

#### Changed

- The README describes the graph rewrites for the browser, both model variants, the separate weight files and the streamed model cache, marks Phase 4 as done and lists the new folders and the `make quality` and `make package` targets for both variants.
- The proposal now describes the variant picker and the Diagnostics tab, the model cache filled by the worker, the result of the q8 quality gate, the end-to-end suite as built, and the Diagnostics benchmark as the first version of the RQ2 protocol.

#### Decisions

- Four questions from the acceptance targets are still open and are listed in [FINDINGS.md](FINDINGS.md): the WebAssembly time-to-first-token target, which memory figure counts for the 2 GB target, words that may need a visible virama before ය or ර, and how to offer q8 on WebAssembly.

### fix/diagnostics-ui — Review of the Diagnostics tab and variant choice

#### Fixed

- After the last benchmark question, the button read "Running 6 of 5…" while the page memory was measured. It now reads "Asking n of 13…" and then "Measuring memory…".
- The "About the model" card always described the 4-bit model. It now follows the chosen variant.

#### Added

- The benchmark asks all 13 sample questions instead of five, and opens a dialog with the results when it finishes: variant, backend, answered questions, median first token, median speed, tokens, total generation time and page memory.
- A variant picker in the model panel of the Ask tab, with a signal-strength icon for each variant (two of four bars for 4-bit, four for 8-bit) and a short description. A downloaded variant loads when it is chosen; a variant that is not downloaded is only selected, and the panel offers its download with the size. The Diagnostics tab keeps its download, load and delete buttons for each variant.

### feat/zwj-restoration — ZWJ in displayed answers (RQ3)

#### Added

- `restoreZwj` in `web/src/text/zwj.js`, which joins yansaya and rakaransaya written with a visible virama in the displayed answer, except after ර, where the ZWJ would form a repaya. It is on by default and can be turned off in the settings ("Join conjuncts with the zero-width joiner in answers"). Gate 4 turns it off, since it compares the raw answers with Python.

#### Results

- In 25 greedy answers of q4 (1,409 words), all 143 yansaya and rakaransaya already had the ZWJ, including clusters such as ව්‍ය that the model learned without it in the vocabulary, so the restoration changed nothing. The only unjoined case found so far is a fp32 reference answer that repeats ශ්රී from a question typed without the ZWJ.

#### Decisions

- The restoration stays on as a safety net, since it does not change correct text. A native-speaker check for words that intentionally keep a visible virama before ය or ර is still open.

### feat/diagnostics — Diagnostics tab, q8 variant and acceptance targets

#### Added

- A Diagnostics tab next to Ask, with the runtime details (variant, backend, GPU adapter, threads, isolation, load and warm-up times), memory (page memory measured by the browser, the JavaScript heap and their peaks, storage use, quota and persistence), switching between variants and backends, a benchmark of five sample questions with greedy decoding, and a session log.
- The session log is kept in IndexedDB in the browser. It records loads, answers and benchmark runs with their timings, but not the questions, and can be exported as JSON or cleared.
- The 8-bit variant (q8, 2.48 GB) as an optional download, after its own quality gate on all 1,012 FLORES+ sentences (perplexity 115.5, against 195.0 for Q4_K_M). Each variant can be downloaded, loaded and deleted separately, and the chosen variant is remembered.
- `make quality` now runs the quality gate for q4 and q8, and `make package` packages both.

#### Fixed

- Page memory was 2.67 GB with q4 on either backend, since ONNX Runtime keeps a copy of weights stored inside the graph file (S14). Both variants now store their weights in separate data files, which reduced page memory to 0.09 GB on WebGPU and 1.78 GB on WebAssembly.
- The q8 weights in one 2.48 GB file never finished loading, so they are split into two files of 1.24 GB (S15).
- Chrome did not cache the 1.18 GB weight file when transformers.js stored it from memory, so the model was downloaded on every visit (S16). The worker now streams the model files into Cache Storage itself.
- When the disk is nearly full, a download fails before the reported quota (S17). The application now removes the partial download and explains the cause instead of a network error.
- Probing the cache created an empty cache, so a deleted model left an empty cache behind.

#### Results

- On WebGPU (RTX 3060 Laptop GPU), q4 reached a median of 26.5 tokens per second with the first token after 69 ms, and q8 26.6 tokens per second after 76 ms. A 52-token question got its first token after 484 ms.
- Warm loads took 4.4 s on WebGPU and 6.9 s on WebAssembly, against a target of 10 s.
- On WebAssembly, the first token of the 52-token question took 5.2 s, above the 3 s target, and the renderer process used 2.35 GB, while the browser measured 1.78 GB of page memory. All values are in [browser-runtime.md](docs/findings/browser-runtime.md#13-acceptance-targets).

#### Attempts

- A first measurement of q8 in a single 2.48 GB file looked like a crash. Logging the progress showed that the data file never started to download in the browser, which led to S15.
- Two full RQ1 runs of q8 (about an hour each) were lost: the first failed on division by zero when no golden questions were requested, and the second on printing the missing agreement report, after which `make` deleted the finished report. A five-sentence run now checks the full path first.
- The first q8 download in the browser failed after 2.15 GB. Repeating it with the browser profile in memory showed that the disk, not the size, was the cause (S17).

### feat/browser-e2e — Browser tests and gate 4

#### Added

- A Playwright suite in `web/e2e/`, which builds the application, serves the production build and runs it in Chrome with a temporary browser profile. It covers the download with progress, gate 4, streaming and stopping an answer, a question longer than the context, settings and the model after a reload, answering offline, and deleting the model while offline.
- Gate 4 compares the greedy answers of the browser (WASM backend) with ONNX Runtime in Python on the ten golden questions, over the first 32 generated tokens.
- `make e2e`, which runs the suite only when the web code, the tests or the model package change, and is part of `make ci`. The temporary profile, with its copy of the model, is deleted after the run, also when a test fails.
- `@playwright/test` 1.56.1, using the installed Chrome instead of a downloaded browser.

#### Results

- All seven tests passed in 8.3 minutes, of which gate 4 took 7.2 minutes, and all ten answers matched.

#### Attempts

- In the first run, gate 4 failed on the fourth golden question. The question is typed without the ZWJ, which the application normalizes before encoding, while the Python reference used the question as typed. Gate 4 now turns the normalization off, since it compares the two runtimes on the same prompt, and the normalization itself is covered by the unit tests.

### docs/native-review — Quality gate review

#### Decisions

- Sineth Sankalpa completed the native-speaker review of the q4-v0-small answers, the second part of the quality gate. Together with the perplexity comparison against Q4_K_M, q4-v0-small has passed the full quality gate.
### feat/offline-pwa — Offline mode

#### Added

- A service worker, generated at build time, which stores the page, its scripts, styles and fonts, and the ONNX Runtime WebAssembly file in the browser on the first visit and serves them from there afterwards. Each build has its own cache, and older application caches are removed when a new version takes over. The model stays in the cache that transformers.js manages.
- The service worker adds the cross-origin isolation headers to the responses it serves, so that the WebAssembly backend keeps its threads on hosts that cannot set these headers. If a page is not isolated when the service worker first takes control, it reloads once.
- A web app manifest and an icon, so the application can be installed. The icon also replaces the missing favicon.
- After the model is loaded, the application asks the browser to keep its storage until the reader deletes it (`navigator.storage.persist()`), and the storage panel says whether the browser agreed.
- Without a connection and without a downloaded model, the download button is disabled and the model panel explains that one online visit is needed.
- `make preview` also serves the linked model, so that offline mode can be tested on the production build.

#### Results

- On the production build in headless Chrome, the model was downloaded once, the browser was switched offline and the page was reloaded. The page came from the service worker, the model loaded from the device in 3.2 s, the page stayed cross-origin isolated, and the greedy answer to the first golden question matched the Python answer exactly. With the model deleted while offline, the download button was disabled with the offline message.

#### Revised

- The ONNX Runtime WebAssembly files were copied into `web/public/ort/` before each build, while Vite already emitted the same 27 MB WebAssembly file from the ONNX Runtime package, so the build contained it twice. The worker now leaves the file location to ONNX Runtime, which loads the file Vite emits, and the copy script is removed. This also removes the warning that transformers.js could not cache the copied file.

#### Attempts

- The first run of the offline test was stopped by its 15-minute time limit without any output, since the output was piped through another command that held it back. The rerun, which wrote directly to a file, finished in 36 s.

### feat/settings — Saved generation settings

#### Added

- The generation settings are saved in the browser's local storage and restored on the next visit. Stored values are checked when they are read: values outside a slider's range are clamped, and values of the wrong type fall back to the defaults.
- A "Reset to defaults" button, which is disabled while the defaults are in use.
- `web/src/ui/settings.js` holds the slider ranges, and `web/tests/settings.test.js` checks how stored values are read.

#### Decisions

- The model variant selector of the plan moves to the Diagnostics tab, together with the q8 variant, since only one variant is packaged so far.

### docs/wasm-fallback — WebAssembly as a slow fallback

#### Decisions

- WebAssembly is kept as a slow fallback for browsers without hardware WebGPU, and its speed target of 5 tokens per second is dropped. It reached 0.6 tokens per second (S12), and a faster path would need a second, 8-bit model of about 1.8 GB. WebGPU met its targets in manual testing, and the time-to-first-token target for WebAssembly (3 s) stays.

### feat/qa-mode — Questions and streamed answers

#### Added

- The worker builds the ChatML prompt from the question, normalizes the ZWJ when the setting is on, and caps prompt and answer together at 1,024 tokens. A question that leaves fewer than 16 tokens for the answer is rejected with a message.
- Answers are streamed as they are generated, with HTML entities resolved, and the interface applies the streamed text at most once per frame. A temperature of 0 selects greedy decoding.
- A Stop button, Ctrl+Enter to ask, and a line under each answer with the number of tokens, the speed and the time to the first token. Backend switching and model deletion are locked while an answer is generated.
- `web/src/worker/generation.js` and `web/tests/generation.test.js` for the prompt, the generation options, the stop reason and the statistics.
- When the model runs on WASM because WebGPU is missing or only available in software, the model panel says so. On Linux, it also shows how to start Chrome with WebGPU on the GPU (S13). Selecting WebGPU explicitly now gives the same reason instead of a generic message.

#### Results

- In headless Chrome on an RTX 3060 Laptop GPU, the greedy answers to the first three golden questions matched the Python answers exactly. After the WebGPU fix (fix/webgpu-graph), a 64-token answer was generated at 27.6 tokens per second, with the first token after 0.1 s.
- Without a GPU, the same answers matched as well, but at 0.6 tokens per second (S12).

#### Attempts

- The first Stop button sent a message to the worker, which arrived only after the answer was complete, since the WebAssembly backend never yields to the worker's message queue while generating. The page and the worker now share a stop flag in shared memory, with the message kept as a fallback for pages that are not cross-origin isolated.

### fix/webgpu-graph — Operators that stalled WebGPU

#### Fixed

- WebGPU generated only 4 to 6 tokens per second on an RTX 3060 Laptop GPU, since `IsNaN` in every layer and the bitwise unpack of the embedding ran on the CPU and forced a copy from the GPU at every token. `convert/quantize.py` now rewrites `IsNaN(x)` as `Not(Equal(x, x))` and unpacks the embedding with integer division, and records both rewrites under `portable` in the variant report.
- `convert/package_web.py` refuses to package a model that contains `IsNaN`, `BitwiseAnd`, `BitShift` or `GatherBlockQuantized`.

#### Results

- A plain ONNX Runtime session in headless Chrome went from 245 ms to 53 to 57 ms per token. The rewritten model gives a maximum logit difference of 0.0 against the previous one, the same greedy answers and the same perplexity (158.014).

#### Attempts

- Casting the attention mask to integers around its per-layer `Slice` did not move the slice to the GPU and did not change the speed, so it was dropped.
- Telling ONNX Runtime to prefer the high-performance or the low-power GPU, and disabling WebGPU validation, did not change the speed.
- For WebAssembly, computing the 4-bit products with 8-bit activations (`accuracy_level` 4) was slower (3.7 s against 2.3 s per token), so it was not adopted. WebAssembly stays at 0.6 tokens per second (S12).

### feat/inference-worker — Model loading in a Web Worker

#### Added

- `web/src/worker/inference.worker.js`, which loads the model with transformers.js 4.3.1 in a Web Worker. It uses WebGPU when the browser offers a hardware adapter and rejects software adapters such as SwiftShader and llvmpipe, and otherwise falls back to WebAssembly with up to eight threads when the page is cross-origin isolated. A two-token warm-up run follows each load.
- The model is downloaded only after a click on "Download model (1.19 GB)". A model that is already in the browser's Cache Storage is loaded automatically when the page opens.
- The model panel shows the download progress, the backend, the load and warm-up times and whether the page is cross-origin isolated, and switches between Auto, WebGPU and WASM.
- A storage panel shows how much of the browser's storage quota the site uses, and deletes the downloaded model after a confirmation, since a 1.19 GB model can fill a nearly full disk.
- The ONNX Runtime WebAssembly files are copied from `node_modules` into `web/public/ort/` before `npm run dev` and `npm run build`, so the application does not load them from a CDN.

#### Results

- In headless Chrome without a hardware GPU, the model ran on WASM with seven threads. The first load from the local development server took 28.3 s plus a 6.7 s warm-up. After a reload, the model loaded from Cache Storage in 3.8 s plus a 5.4 s warm-up (9.2 s, within the 10 s target), and deleting it left the site with no cache and 0.00 GB of storage.

#### Attempts

- transformers.js loads its WebAssembly files from the jsDelivr CDN by default, which would break the offline mode, so the files are self-hosted.
- The first production build copied the 1.2 GB model into `web/dist/` through the link in `web/public/models/` and filled the internal disk. The build now copies only `public/ort/`, and the model is served separately.
- The first browser test failed to create a session, since the browser build of ONNX Runtime lacks `GatherBlockQuantized`. The fix was made on its own branch (fix/portable-embedding) and merged into this one.

### fix/portable-embedding — 4-bit embedding for the browser runtime

#### Fixed

- The packaged model could not be loaded in the browser. The asyncify build of ONNX Runtime Web has no implementation of `GatherBlockQuantized`, the operator the 4-bit quantizer uses for the embedding lookup. `convert/quantize.py` now replaces it with standard ONNX operators after quantization and records this as `portable_embedding` in the variant report.

#### Added

- A second package check, which creates a session in the asyncify WebAssembly build of ONNX Runtime Web and requires the first greedy token of each golden question.

#### Results

- The replacement gives a maximum logit difference of 0.0 and the same greedy answers on all ten golden questions. The quality gate passes unchanged (perplexity 158.014 against 194.965), and q4-v0-small stays at 1.184 GB.

#### Attempts

- The earlier package parity check passed, since it ran the model in the Node.js build of ONNX Runtime, which implements the operator. The failure appeared only in the first browser test of the inference worker, which is why the check now uses the browser build as well.

### feat/theme-toggle — Theme switch and sample questions

#### Added

- A theme switch in the header with Auto, Light and Dark. Auto follows the system setting, and the choice is kept in the browser's local storage.
- Thirteen sample questions instead of four: the ten golden questions and three general questions from the model author's evaluation set. Six are shown by default, and the rest appear behind a "More examples" toggle.

#### Revised

- The green accent was made lighter on request: `#287f53` in the light theme and `#6fcd98` in the dark theme. Text on green buttons and green links keeps a contrast of at least 4.5:1.

### feat/ui-shell — Application shell

#### Added

- A Vite and React application shell in `web/src/ui/`: a question box with four sample questions, an answer area, a model status panel, generation settings, a limitations notice and credits for the model and its base model. The interface follows the system's light or dark mode.
- Noto Sans Sinhala (SIL Open Font License) is bundled through `@fontsource/noto-sans-sinhala` 5.3.0, so that Sinhala text renders the same way without a network connection.
- React 19.1.10 with `@vitejs/plugin-react` 6.1.2 and the React lint rules, and `make dev` and `make preview` for local testing.

#### Revised

- The font package's default import also loaded the font's Latin glyphs, which gave English text very wide word spacing. The application now declares the font itself, limited to the Sinhala block, the zero-width joiners and the dandas, so that Sinhala uses the bundled font and everything else uses the system font.
- After the first review, the title changed from සිංහල ප්‍රශ්න to "Offline සිංහල", with "Offline Q&A in Sinhala Language" next to it, and the browser tab title changed to match.
- After the second review, the brown accent colors were replaced with green ones in both the light and the dark theme.

#### Decisions

- The interface labels stay in English for now, with Sinhala for the title, the question placeholder and the sample questions.
- The benchmark page of the plan becomes part of a Diagnostics tab, which also shows memory, speed, a session log kept in the browser (IndexedDB), and switching between backends and variants. The q8 variant is packaged for switching once WebGPU switching works.

### feat/web-model-package — Model package for the web application

#### Added

- `convert/package_web.py`, which copies the default variant (q4-v0-small) into a versioned web folder in the model store, `web/sinhalallm-1.7b/v2.1/`, with the configuration, the browser tokenizer files, `onnx/model_q4.onnx`, an attribution notice and a `manifest.json` with the size and SHA-256 hash of every file. It also records the greedy answers of the packaged file in ONNX Runtime for the ten golden questions.
- `web/tests/package-parity.test.js`, which loads the package with transformers.js in Node.js and requires the same greedy answers as ONNX Runtime in Python. It is skipped in the regular unit tests and runs through `make check-package`.
- `make package`, `make link-models` and `make check-package`. The package is linked into `web/public/models/` rather than copied, since the internal disk has too little space for a second copy, and `make ci` now runs the parity check when its inputs change.

#### Results

- The package is 1.189 GB, and transformers.js 4.3.1 reproduced the Python greedy answers on all ten golden questions.

### docs/phase4-plan — Browser application plan

#### Decisions

- The frontend is built with Vite and React. Next.js was considered, but a static Hugging Face Space cannot use its server-side features, so it would add weight and build complexity without a benefit for this application.
- Mojo is not used. Inference runs in the browser through ONNX Runtime Web (WebAssembly and WebGPU), and Python only runs the build-time pipeline, so neither language is on the inference path.
- The browser application applies runtime optimizations that need no change to the model files: WebGPU first with software adapters rejected, the key-value cache kept on the GPU, a warm-up run after loading, early loading with a persistent worker, WASM thread and session tuning, and streaming with frame-batched updates.
- Model-level optimizations (fp16 activations, logits for the last position only, fused attention) are left out of this version, since each would require the parity and quality gates to be passed again.

#### Changed

- Section 5.7 of the proposal now describes the Vite and React frontend and the runtime optimizations.

### feat/quality-gate — Shipped variant and quality gate

#### Decisions

- q4-v0-small (1.18 GB) is the default variant of the application. It fits the 1.2 GB download target, so the target stays unchanged.
- The quality gate no longer requires a perplexity within 2% of fp32, which no 4-bit variant of this model can meet. The shipped variant must instead have a perplexity no higher than the author's Q4_K_M release on the same FLORES+ sentences, and its answers are reviewed by a native speaker.
- q8 is kept as an optional higher-quality variant if the RQ2 benchmarks show that it is fast enough on WebGPU.

#### Added

- `eval_quality.py reference-gguf`, which measures the author's Q4_K_M release on FLORES+ through `transformers`, and `eval_quality.py quality-gate`, which compares the default variant with it on the same sentences.
- `make quality`, which runs the gate and is part of `make ci`, and `gguf` 0.19.0 in the export group.

#### Changed

- The proposal now defines the new quality gate in Section 5.6 and in the gate table, and names q4-v0-small as the shipped default.
- `docs/findings/rq1-quantization.md` records the decision and the full-set reference result.

#### Attempts

- The first run of the gate was interrupted when the session ended before it produced any output, and was run again.

#### Results

- The quality gate passes: q4-v0-small reaches a perplexity of 158.0 on all 1,012 sentences, against 195.0 for Q4_K_M (+72.2% against fp32).

### feat/rq1-quality — RQ1 quantization quality

#### Added

- `eval_quality.py rq1`, which measures perplexity and bits per character on the FLORES+ `sin_Sinh` devtest split (pinned revision and file hash), the top-1 agreement of the prompt logits and the greedy agreement of the ten golden answers against fp32. Per-sentence scores are stored, so that variants measured on a subset can be compared on the same sentences.
- Research variants in `quantize.py`: q4-head and q4-emb for the head and embedding ablation, q4-asym, q4-b16 and q4-hqq for the method screen, and the two candidates q4-v0 and q4-v0-small, which keep the value projection of layer 0 in fp32.
- `make rq1`, which evaluates every candidate variant. `make quantize` now builds fp16, q8, q4-v0 and q4-v0-small.
- `docs/findings/rq1-quantization.md` with the results and the investigation, and `docs/quantization-comparison.md` with the candidates' answers next to the fp32 answers.

#### Attempts

- The head and embedding ablation, asymmetric quantization, block size 16 and HQQ were each tried to close the 4-bit gap, and none brought the loss below +164%.
- Two possible pipeline faults were ruled out: forcing fp32 compute in the 4-bit kernel gave identical perplexity, and the dequantized 4-bit weights had a lower error than a reference round-to-nearest implementation.
- A sensitivity scan that kept one group of weight matrices in fp32 at a time found that the value projection of layer 0 alone causes most of the loss.
- The first evaluation of the two candidates did not start, since a line in `quantize.py` exceeded the length limit of the linter.
- A helper script named `gguf.py` shadowed the `gguf` package during the GGUF reference measurement and was renamed.

#### Results

- fp32 reached a perplexity of 113.2 (1.695 bits per character) on the 1,012 devtest sentences. fp16 (+0.0%) and q8 (+1.2%) stay within the 2% gate; the plain 4-bit variants lose +164% to +248%.
- q4-v0 (1.54 GB) loses +32.7% and q4-v0-small (1.18 GB) +39.6%, while the author's Q4_K_M GGUF loses +51.5% on the same 50 sentences.

#### Open

- The variant to ship by default, and the quality gate that replaces the 2% margin, which no 4-bit variant of this model can meet.

### feat/quantization — fp16, q8 and q4 variants

#### Added

- `convert/quantize.py`, which builds q8, q4 and q4-tied from the fp32 ONNX model with ONNX Runtime's block quantizer (MatMulNBits, block size 32, symmetric). q8 and q4 untie the output head into its own weight, and q4 also quantizes the embedding lookup to 4 bits.
- `export_onnx.py --dtype fp16`, which exports the fp16 variant directly from the PyTorch model loaded in fp16.
- Every variant is written as a complete transformers.js model folder, with the configuration and the browser tokenizer files next to `onnx/model_<suffix>.onnx`.
- `make quantize`, which builds all variants under a 14 GB memory cap (`QUANTIZE_MEMORY`) and is part of `make ci`.
- `docs/findings/quantization.md` with the sizes, methods and first quality signals.

#### Changed

- `OnnxCausalLM` in `eval_quality.py` now reads the key-value cache type from the model, so that it also runs the fp16 variant.
- The proposal now describes the actual quantization methods, the measured sizes and the quality risk of the q4 variant, and lists `quantize.py` instead of `quantize.sh`.

#### Attempts

- ONNX Runtime's fp16 converter and its dynamic 8-bit quantizer were both killed at a 12 GB memory cap, since each holds the fp32 model and a converted copy at the same time. fp16 is now exported directly from PyTorch, and q8 uses the 8-bit block quantizer.
- The first 8-bit block quantization was also killed at 12 GB (peak 12.6 GB) and succeeded under a 14 GB cap.
- Saving q8 failed with `EncodeError: Failed to serialize proto`, since the check for the 2 GB protobuf limit called `ByteSize()`, which itself fails above 2 GiB. The check now sums the initializer sizes instead.
- To free space on the external drive, the converted HelaAI models and checkpoints were deleted from the model store (about 600 MB); they can be regenerated from the git history.

#### Results

- Sizes: fp16 3.68 GB, q8 2.48 GB, q4 1.17 GB and q4-tied 1.53 GB. Only q4 fits the 1.2 GB target.
- On the first golden question, fp16 and q8 reproduced the fp32 answer, q4-tied kept the correct answer, and q4 fell into a loop. q8 generated at 0.1 tokens per second on the CPU.

### build/content-hash-stamps — Content-based pipeline rebuilds

#### Changed

- The pipeline targets in the Makefile now depend on SHA-256 stamps of their input files in `build/stamps/` instead of the files' modification times. A stamp is rewritten only when the content of its inputs changes, so `make ci` on an unchanged tree takes about 4 seconds instead of rerunning golden generation, gate 1, the export and gate 2 for about 10 minutes after every merge.
- The ONNX export now depends on gate 1 as an order-only prerequisite: gate 1 must pass first, but a rerun of gate 1 alone no longer triggers a new export.

### feat/onnx-export — ONNX export and parity gate 2

#### Added

- `convert/export_onnx.py`, which exports the model with the translated configuration through the Optimum exporter (`text-generation-with-past`, opset 18) in two steps: a memory-capped child traces the model, and the parent verifies every tensor file by size, ties the duplicated output head back to the embedding and writes one `model.onnx_data`.
- `convert/eval_quality.py gate2`, which compares the fp32 ONNX model in ONNX Runtime with the golden logits and greedy answers.
- `make onnx`, which runs the export and gate 2, and is now part of `make ci`. `EXPORT_MEMORY` (default 11G) sets the memory cap of the trace step.
- `docs/findings/onnx-export.md` with the export and gate 2 results.
- The export dependencies in the export group: `optimum` 2.1.0, `optimum-onnx` 0.1.0, `onnx` 1.23.2, `onnx-ir` 1.1.0, `onnxruntime` 1.31.0, `onnxslim` 0.1.98 and `accelerate` 1.15.0.

#### Changed

- Gate 2 now passes when the largest logit difference stays below 1e-4 of each position's largest absolute logit, the top-1 token is identical at every position and all greedy answers are identical. The proposal's gate table and Section 5.5 describe this criterion.
- `random_ids` moved into `convert/sinhalallm.py`, so that gate 1 and gate 2 build the same random input.
- The proposal lists `export_onnx.py` instead of `export_onnx.sh`, since the export needs the in-memory model with the translated configuration.

#### Attempts

- The single-process export, capped at 8.8 GB, was killed while Optimum merged the per-tensor files into one file, since that step loads the whole ONNX model while the PyTorch model is still in memory.
- With the merge disabled, tracing completed, but the process was still killed shortly after writing all tensor files, at caps of 8.8 GB and 11 GB. When the whole export ran inside one capped scope, the kill of the child also terminated the parent, so the finalize step never ran. The cap now applies only to the trace child, and the parent verifies the written files before continuing.
- One run failed because a second Claude session was running the same export at the same time, and both processes were killed for memory. The other session's export was stopped before the export was repeated.
- The first gate 2 run used the absolute tolerance of gate 1 and failed at 7.6e-4. The maximum was always at position 0 and depended on the sequence length, PyTorch's own SDPA-versus-eager difference was 7.1e-5, and disabling ONNX Runtime's graph optimizations did not remove the difference. The gate was therefore redefined with a relative tolerance and top-1 agreement instead of being loosened.

#### Results

- The finalized fp32 model has 219 initializers and a 6,961,897,472-byte data file, exactly 4 bytes per parameter.
- Gate 2 passes with a maximum relative logit difference of 3.46e-5, a maximum absolute difference of 7.6e-4 at position 0, identical top-1 tokens at every position and no greedy mismatches.

### feat/text-processing — ZWJ normalization and entity decoding

#### Added

- `web/src/text/zwj.js` with `normalizeZwj`, which adds the ZWJ in the 15 conjunct clusters that the model learned with it and removes it from all other `්ර` and `්ය` clusters.
- `web/src/text/entities.js` with `decodeEntities` and a streaming decoder that holds back a possible entity at the end of a chunk until the next chunk arrives.
- `web/tests/text.test.js`, which covers both modules and checks the 15 clusters against the tokenizer.
- `docs/findings/zwj-normalization.md`, which derives the 15 clusters from the vocabulary and compares three input rules on the tokenizer corpus.

#### Changed

- The proposal now describes `normalizeZwj` instead of `repairZwj`: S3, the text processing in Section 4.2 (D), its two examples and Figure 2, the request lifecycle, RQ3, the worker code, the project structure, the tests and the risks.
- S3 in `docs/findings/sinhalallm-inspection.md` now lists the 15 clusters instead of only the two cluster types.

#### Attempts

- The planned rule added the ZWJ after every virama that is followed by ර or ය. Measured on the 10,000-sentence tokenizer corpus, it increased the token count by 0.26%, since the vocabulary learned clusters such as ව්ය and ත්ය only without the ZWJ. The vocabulary contains exactly 15 clusters with the ZWJ, none of which also appears without it. A rule limited to these clusters reduced the tokens by 0.11%, and the final rule, which also removes the ZWJ from all other clusters, reduced them by 1.57% on the corpus and by 10.4% on text typed without the ZWJ.

#### Open

- The model most likely writes clusters such as සත්ය without the ZWJ, so restoring the ZWJ in the output is a candidate step for RQ3, pending native-speaker evaluation.

### feat/tokenizer-parity — Tokenizer parity gate 3

#### Added

- `make_golden.py tokenizer`, which samples 10,000 Sinhala sentences with a fixed seed from four public datasets at pinned revisions, encodes them with the Python tokenizer of the reference toolchain and writes them to `golden/generated/tokenizer_corpus.json` (not committed). It also writes the committed `golden/tokenizer_manifest.json`, with the source revisions, the seed and the corpus hash, and `golden/tokenizer_cases.json`, with 17 named edge cases.
- `web/tests/tokenizer-parity.test.js` (gate 3), which checks the corpus hash against the manifest and compares the transformers.js tokenizer with the Python tokenizer on the corpus, the edge cases and the chat prompts of the ten golden questions.
- `docs/findings/tokenizer-parity.md` with the results of gate 3.
- `@huggingface/transformers` 4.3.1 as the web application's runtime dependency, and `pyarrow` 26.0.0 in the reference environment for reading the corpus sources.

#### Changed

- `make_golden.py` now takes a target, `reference` or `tokenizer`, and `make test-web` regenerates the tokenizer corpus when its inputs change.
- The proposal lists the chat-template finding as S9 in Table 2.

#### Decisions

- transformers.js 4.3.1 is pinned rather than 4.3.2, since 4.3.2 was released less than two weeks before this branch.
- The corpus sentences are not committed, since the licenses of the four sources differ; the committed manifest makes the corpus reproducible and verifiable instead.

#### Attempts

- The first run of the chat-prompt checks failed with "Cannot use apply_chat_template() because tokenizer.chat_template is not set". The model stores its template in `chat_template.jinja`, which `AutoTokenizer` in transformers.js 4.3.1 does not read. `convert/sinhalallm.py` now writes a browser version of `tokenizer_config.json` with the template embedded, and the same function will package the tokenizer for the web application.

#### Results

- Gate 3 passes: 10,000 of 10,000 corpus sentences, 17 of 17 edge cases and 10 of 10 chat prompts match the Python tokenizer, and regeneration from scratch reproduced the committed manifest and cases.

### feat/sinhalallm-reference — Reference outputs and gate 1 for SinhalaLLM-1.7B

#### Added

- `convert/sinhalallm.py`, which downloads SinhalaLLM-1.7B at a pinned revision into the model store and verifies the SHA-256 hash of every file. A marker next to each verified file avoids hashing the 3.5 GB weights again until the file changes.
- `convert/make_golden.py` for SinhalaLLM-1.7B, which runs with `transformers` 5.19.0 and writes the prompt token IDs, prompt logits and 64-token greedy answers for ten golden questions, the logits of a random 1,024-token input and a regenerated `docs/model-outputs.md`.
- `convert/parity_check.py` for SinhalaLLM-1.7B (gate 1), which translates the configuration for `transformers` 4.57.6, compares logits and greedy answers with the golden outputs and runs the untranslated configuration as a negative control. The translated configuration is written to the model store for the export in Phase 3.
- `docs/findings/config-translation.md` with the results of gate 1.
- `make verify`, which reruns the golden generation and gate 1 from scratch.

#### Changed

- The Python environment is split into an export group (`transformers` 4.57.6) and a reference group (`transformers` 5.19.0), declared as conflicting `uv` dependency groups. The reference group runs in `.venv-reference`.
- `make ci` now rebuilds the golden outputs and the gate 1 report only when their inputs change, since each takes about three minutes on a CPU for a 1.74B model. `.DELETE_ON_ERROR` removes a failed report, so that a failed gate is never treated as up to date.
- `golden/prompts.json` now holds ten questions: the four golden questions and the six questions from the model comparison.

#### Removed

- The HelaAI pipeline: `convert/helaai.py`, the SinhalaGPT-to-Llama conversion and its golden outputs. The HelaAI findings stay in `docs/findings/` as a record of the previous model.

#### Results

- Gate 1 passes: a maximum absolute logit difference of 0.0 on all ten questions and the random input, no greedy mismatches and 1,740,474,368 parameters. The negative control differs by up to 11.66, and `make check-golden` reproduced the committed golden outputs exactly.
- With the untranslated RoPE base, short answers stayed plausible, while the longer answer to Q3 fell into repetition, so the error would most likely not be noticed by reading a few answers.
- The reference generation and gate 1 each peaked at about 8.9 GB of memory.

### docs/sinhalallm-proposal — Switch to SinhalaLLM-1.7B

#### Decisions

- SinhalaLLM-1.7B (`namindu94/SinhalaLLM-1.7B`, revision `b24d58b7929e1ed50f683b435a9a629a73e27af5`) replaces the HelaAI SinhalaLM models, after their question-answering quality was judged unsatisfactory and the comparison in `docs/model-comparison.md` was reviewed.
- The application provides question answering only, since SinhalaLLM-1.7B is published as a single instruction-tuned checkpoint. The text-continuation mode (former FR1) was dropped, and the functional requirements were renumbered.
- The targets were raised for a 1.74B model: at most 1.2 GB of download for the default variant, at most 2 GB of peak tab memory with a 1,024-token context cap, at most 10 s of warm load, a time to first token of at most 1 s on WebGPU and 3 s on WASM, and at least 20 tokens per second on WebGPU and 5 on multi-threaded WASM.
- The HelaAI pipeline code will be replaced with SinhalaLLM equivalents, while the HelaAI findings stay as a record of the previous model.

#### Changed

- The proposal was revised in place for SinhalaLLM-1.7B. Its structure and research intentions are unchanged, while the background, the inspection findings (S1–S8), RQ3, the design of the tokenizer and text processing, gate 1, gate 4, the project structure, the targets and the risks now describe the new model.
- RQ3 changed from recovering the ZWJ that the HelaAI tokenizer lost to repairing the ZWJ in user input and preserving it end to end.
- `FINDINGS.md` and the README now describe SinhalaLLM-1.7B, and mark the HelaAI findings as belonging to the previous model.

#### Added

- `docs/findings/model-selection.md`, which records why the HelaAI models were replaced and how SinhalaLLM-1.7B was selected.
- `docs/findings/sinhalallm-inspection.md`, which reports the inspection of the SinhalaLLM-1.7B repository. Its central finding is that `transformers` 4.57.6, which the ONNX exporter requires, loads the configuration with a RoPE base of 10000 instead of 130000, without any error.

### build/external-model-store — Model store on an external drive

#### Added

- A configurable model store for downloaded checkpoints, converted models and generated logits. It defaults to `build/`, and the uncommitted `local.mk` file can point `MODEL_STORE` to another location. The Makefile passes it to the pipeline as `SINHALALM_STORE`.
- The pipeline stops with "model store ... is not available; mount it or set MODEL_STORE" when an external store is not mounted, instead of downloading everything again to a new location.

#### Decisions

- The store lives on the external drive at `/media/sinsankio/QrioMatrix/sinhalalm-web`, since the internal disk has about 5 GB free and the SinhalaLLM-1.7B pipeline needs about 18 GB. The drive is formatted as exFAT, which does not support symbolic links, so checkpoints are downloaded as plain files into `hf/<repository>/` rather than into the Hugging Face cache.
- The drive had 21 GB free when the store was created, which leaves little headroom beyond the expected 18 GB.

### docs/model-comparison — Alternative model comparison

#### Added

- `docs/model-comparison.md`, which compares the current HelaAI instruct model with SinhalaLLM-1.7B, Qwen3.5-2B and Qwen3.5-0.8B on ten Sinhala questions, together with Sinhala tokenizer efficiency, ZWJ handling, license, browser download size and the changes each candidate would bring to the project plan.

#### Context

- The question-answering quality of both HelaAI SinhalaLM models was judged unsatisfactory after reviewing `docs/model-outputs.md`, so the project is evaluating alternative models before the Phase 2 work starts. The HelaAI pipeline and findings stay in place until a model is chosen.

#### Attempts

- The machine had about 5 GB of free disk space, so the candidates were screened one at a time from their 4-bit files (a 1.08 GB GGUF file for SinhalaLLM-1.7B and the `onnx-community` q4 builds for Qwen3.5), and each was deleted after its run. Gemma 4 E2B and SinLlama did not fit and were not screened.

### feat/golden-outputs-doc — Model outputs document

#### Added

- `docs/model-outputs.md`, which lists every golden prompt with the greedy output of both models, the stopping reason, the `<UNK>` count and, where `<UNK>` occurs, the same output with each `<UNK>` marked, so that the outputs can be validated by a native Sinhala speaker.
- `make_golden.py` generates the document from the same run as `golden/reference.json`, and `make check-golden` now fails if either file drifts.

#### Changed

- `golden/prompts.json` is now a single list of eight prompts, and every prompt runs through both models, instead of four prompts per model. The case keys changed from `base_0` and `it_0` to `p0_base` and `p0_it`.
- Gate 1 now runs on eight golden prompts per checkpoint and still passes with a logit difference of 0.0 and no greedy mismatches.
- The F6 and F8 evidence in `docs/findings/repository-audit.md` and the gate 1 table in `docs/findings/llama-equivalence.md` now refer to the eight-prompt set. With it, all 8 `<UNK>` tokens in the instruct generations follow a token ending in the virama.

### docs/html-entity-decision — HTML entity decoding

#### Decisions

- The application decodes HTML entities such as `&quot;` and `&amp;` in the outputs of both models, as part of the text post-processing in Phase 2. This resolves the open question raised by F12, and the proposal's description of text post-processing (Section 4.2, D) now includes this step.

### docs/fix-f12-claim — Narrow the F12 claim

#### Revised

- F12 in `docs/findings/repository-audit.md` previously stated that HTML entities most likely explain all ELI5-style artifacts noted in F8. The entity counts only support the `&quot;` and `&` artifacts, and `L;L2` did not appear in any of the 300 sampled rows, so the claim is now limited to the entities.

### docs/audit-findings — Repository audit and Llama equivalence

#### Added

- `docs/findings/repository-audit.md`, which reproduces the nine audit findings of the proposal against the HelaAI repositories at pinned revisions, corrects one detail of F4, adds three new findings (F10–F12) and lists suggested upstream fixes for the HelaAI authors.
- `docs/findings/llama-equivalence.md`, which reports the results of parity gate 1.
- Summaries of both documents in `FINDINGS.md`.

#### Changed

- The proposal now reflects the decisions taken since it was written: desktop and laptop computers as the only target devices (NFR5, RQ2), the benchmark device matrix (Section 6.4), the rater plan (Section 6.5), provisional performance targets (Section 6.6), a Hugging Face static Space as the host (Section 7.1), the local Makefile pipeline instead of GitHub Actions (Sections 5.8 and 7.2), and permission from the HelaAI authors (Section 5.1).
- The proposal's references to `tools/parity_check.py` now point to `convert/parity_check.py`, and its project structure lists `convert/helaai.py`.
- The proposal now plans quantization with ONNX Runtime's tools, since the transformers.js quantization script was removed in version 4 (Section 5.5), and computes the FLORES+ reference perplexity together with the quality evaluation (Section 5.2).
- F4, Figure 5 and the tokenizer regression tests in the proposal now state that the instruct template alone yields 24 tokens with two `<UNK>`, and that the 37 tokens with three `<UNK>` include the model card's example question.

#### Open

- F12 raises whether HTML entities such as `&quot;` should be decoded in instruct outputs. This is left as a question for the text post-processing work in Phase 2.

### feat/llama-conversion — Llama conversion and parity gate 1

#### Added

- `convert/parity_check.py`, which converts both checkpoints to `LlamaForCausalLM` and runs parity gate 1. The gate compares logits on the golden prompts and on a random 512-token input, compares 40-token greedy generations under key-value caching against `golden/reference.json`, checks that the converted model has 32,850,432 parameters, and reloads the saved model to confirm that the files on disk give the same logits.
- Before dropping them, the conversion checks that `lm_head.weight` is identical to `tok_emb.weight` and that every stored RoPE `cos` and `sin` table equals the recomputed table.
- `make convert`, which writes the converted models to `build/llama/<model>/` together with a `parity.json` report. `make ci` now runs gate 1.

#### Results

- Both checkpoints pass gate 1 with a maximum absolute logit difference of 0.0 on every case, no greedy mismatches and 32,850,432 parameters. The saved models reload with a difference of 0.0.
- Each converted fp32 `model.safetensors` is 131,409,848 bytes, which matches the 131.4 MB estimate in Section 5.5 of the proposal.

#### Decisions

- `generation_config.json` sets `eos_token_id` to 3 and uses the sampling defaults of the upstream `answer_question` function (temperature 0.7, top-k 50).

#### Attempts

- Loading the converted state with `strict=True` failed with `Missing key(s) in state_dict: "lm_head.weight"`, since `LlamaForCausalLM` expects the tied head as a key even though it shares storage with the embedding. The state now passes the embedding tensor under both names, and the conversion asserts that the two parameters share the same storage afterwards.

### feat/golden-reference — Golden reference outputs

#### Added

- `convert/helaai.py`, which downloads both HelaAI checkpoints at pinned commit revisions, verifies the SHA-256 hash of every file and loads the reference `SinhalaGPT` model and `FastBPETokenizer`.
- `convert/make_golden.py`, which writes `golden/reference.json` with the prompt token IDs, the top-5 next-token IDs and the 40-token greedy generations of both models, together with the full prompt logits in `golden/generated/reference_logits.npz`.
- `golden/prompts.json` with four base-model prompts and four instruct-model questions, including the question from the instruct model card.
- `make golden` and `make check-golden`. `make ci` now regenerates the golden outputs and fails if they differ from the committed file.
- Python dependencies for the reference model: `torch` 2.14.1 (CPU), `transformers` 4.57.6, `huggingface-hub` 0.36.2, `safetensors` 0.8.0 and `numpy` 2.5.4.

#### Decisions

- The instruct repository's `modeling_sinhalalm.py` cannot be imported as a module, since it is a training script that loads a local checkpoint with `torch.load` at import time and exits when the checkpoint is missing. Rather than importing it, `helaai.py` parses the pinned file and executes only the tokenizer, model and prompt-template definitions, so that the reference stays the exact upstream source without running any of its training code.
- `helaai.py` is a shared module inside `convert/` that is not listed in Section 5.8 of the proposal, since both `make_golden.py` and `parity_check.py` need the same reference loader.
- The `uv` configuration sets `exclude-newer` to 2026-10-22, so that transitive dependencies also follow the two-week pinning rule.
- The base and instruct repositories ship the same `bpe_vocabulary.json` (identical SHA-256), so a single tokenizer serves both models.
- The reference greedy generation recomputes the full sequence at every step and stops at `<EOS>` or after 40 tokens, which is the same stopping rule the Llama comparison in gate 1 uses.

### spike/toolchain-versions — Toolchain check

#### Added

- `docs/findings/toolchain.md`, which reports an end-to-end run of export, ONNX Runtime parity, transformers.js 4.3.2 generation and quantization on a tiny Llama model with the SinhalaLM configuration, together with the current Hugging Face Spaces constraints and a WebGPU probe.
- The first entry in `FINDINGS.md`.

#### Decisions

- The export toolchain is pinned to `transformers` 4.57.6, `optimum` 2.1.0 and `optimum-onnx` 0.1.0, together with `accelerate` 1.15.0 and `onnx-ir` 1.1.0. The pins are added to `pyproject.toml` in Phase 1, when the first conversion code needs them.
- Quantization uses ONNX Runtime's own tools, since transformers.js version 4 no longer ships its quantize script.
- The transformers.js version is pinned at the start of Phase 4, since 4.3.2 is not yet two weeks old.
- The hosting checks that need a live Space move from Phase 0 to Phase 5, since Hugging Face credentials are provided at deployment time.
- FLORES+ perplexity moves from the golden-reference step in Phase 1 to RQ1 in Phase 3, for the same reason, since FLORES+ is gated on Hugging Face.
- `pyproject.toml`, `uv.lock` and `.python-version` stay at the repository root rather than under `convert/`, since `uv` resolves the environment from the root and the Makefile runs all Python tools through it.

#### Revised

- The README opening previously stated that SinhalaLM is small enough to run entirely inside a web browser. It now follows the proposal's wording, which describes the models as a strong candidate for client-side inference, since that claim is only verified in Phase 4.

### chore/bootstrap — Project bootstrap

#### Added

- Folder structure from Section 5.8 of the proposal: `convert/`, `golden/`, `web/src/{tokenizer,text,worker,ui}`, `web/public/models/` and `web/tests/`, together with `docs/findings/` for per-topic findings.
- A Python environment managed by `uv`, pinned to Python 3.12, with `ruff` for linting and formatting.
- A Vite web project with Vitest, ESLint and Prettier. The development and preview servers send the `Cross-Origin-Opener-Policy` and `Cross-Origin-Embedder-Policy` headers, since multi-threaded WebAssembly requires cross-origin isolation.
- A root `Makefile` that serves as the local CI pipeline (`make setup`, `make lint`, `make test-web`, `make ci`), which replaces the GitHub Actions workflows planned in the proposal.
- `README.md`, `CHANGELOG.md` and `FINDINGS.md`.
- Permission status from the HelaAI authors recorded as given, which unblocks self-hosting the converted weights.

#### Decisions

- All dependency versions are pinned exactly, and each pinned version was released at least two weeks before it was adopted.
- Generated model files under `web/public/models/` and generated test fixtures under `golden/generated/` are excluded from version control, since they are reproduced by the pipeline.

#### Attempts

- The first install with the system npm 10.9.8 failed with `Cannot read properties of null (reading 'edgesOut')` while npm resolved the optional peer dependencies of Vitest 5.0.5. Downgrading Vitest to 4.1.12 failed in the same way, which pointed to npm rather than Vitest. npm 11.21.1 installed both versions without errors, so the Makefile now runs npm 11.21.1 through `npx` and Vitest stays at 5.0.5.
- After the first merge, the project folder was deleted by mistake, including its Git history. The repository was rebuilt from the session record with the same files, branches and commit messages, so the commit hashes and dates of this branch differ from the original ones.

#### Known issues

- `npm audit` reports a moderate denial-of-service advisory in `minimatch`, pulled in by ESLint 10. ESLint is a development-only tool and is not part of the browser bundle, so the advisory is accepted until a fixed ESLint release is available.
