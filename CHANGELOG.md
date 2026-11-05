# Changelog

All notable changes to this project are recorded here. Entries are grouped by the branch that introduced them, and each entry also records the attempts that were revised or rejected along the way, so that the reasoning behind the final state stays visible.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
