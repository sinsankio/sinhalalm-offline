import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "convert"))

from package_web import MODEL_NAME, MODEL_VERSION  # noqa: E402
from sinhalallm import REVISION, store_path  # noqa: E402

SPACE = "sinsankio/sinhalalm-offline"
MODEL_REPO = "sinsankio/sinhalallm-1.7b-onnx-web"
WEB = ROOT / "web"
NPM = ["npx", "-y", "npm@11.21.1"]
DIST = ROOT / "web" / "dist"
TEMPLATES = Path(__file__).resolve().parent


def release_tag() -> str:
    result = subprocess.run(
        ["git", "describe", "--exact-match", "--tags", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SystemExit("deploy needs a version tag on the current commit")
    return result.stdout.strip()


def upload_models(api: HfApi, repo: str, space: str) -> str:
    models = store_path("web", MODEL_NAME, MODEL_VERSION)
    if not (models / "manifest.json").exists():
        raise SystemExit("the model package is missing; run make package first")
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    card = (TEMPLATES / "model" / "README.md").read_text(encoding="utf-8")
    card = card.replace("{revision}", REVISION).replace("{space}", space)
    with tempfile.TemporaryDirectory() as staging:
        readme = Path(staging) / "README.md"
        readme.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo,
            commit_message="Update model card",
        )
    api.upload_folder(
        folder_path=models,
        repo_id=repo,
        commit_message=f"Upload model package {MODEL_NAME}/{MODEL_VERSION}",
    )
    return model_revision(api, repo)


def model_revision(api: HfApi, repo: str) -> str:
    [manifest] = api.get_paths_info(repo, ["manifest.json"], expand=True)
    return manifest.last_commit.oid


def build_app(repo: str, revision: str) -> None:
    env = {**os.environ, "VITE_MODEL_REPO": repo, "VITE_MODEL_REVISION": revision}
    subprocess.run([*NPM, "run", "build"], cwd=WEB, env=env, check=True)


def upload_app(api: HfApi, space: str, version: str) -> None:
    api.create_repo(space, repo_type="space", space_sdk="static", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        shutil.copytree(DIST, stage, dirs_exist_ok=True)
        readme = (TEMPLATES / "space" / "README.md").read_text(encoding="utf-8")
        (stage / "README.md").write_text(readme.replace("{version}", version), encoding="utf-8")
        api.upload_folder(
            folder_path=stage,
            repo_id=space,
            repo_type="space",
            commit_message=f"Deploy application {version}",
            delete_patterns=["assets/*", "*.js", "*.html", "*.webmanifest", "*.svg", "*.css"],
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Deploy the application and its models.")
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--model-repo", default=MODEL_REPO)
    args = parser.parse_args()

    version = release_tag()
    api = HfApi()
    revision = upload_models(api, args.model_repo, args.space)
    print(f"model package at {args.model_repo}@{revision}")
    build_app(args.model_repo, revision)
    upload_app(api, args.space, version)
    print(f"deployed application {version} to {args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
