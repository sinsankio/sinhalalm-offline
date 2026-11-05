# Changelog

All notable changes to this project are recorded here. Entries are grouped by the branch that introduced them, and each entry also records the attempts that were revised or rejected along the way, so that the reasoning behind the final state stays visible.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
