# Changelog

All notable changes to this project are recorded here. Entries are grouped by the branch that introduced them, and each entry also records the attempts that were revised or rejected along the way, so that the reasoning behind the final state stays visible.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
