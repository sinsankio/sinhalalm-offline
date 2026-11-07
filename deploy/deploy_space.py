import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "convert"))

from package_web import MODEL_NAME, MODEL_VERSION  # noqa: E402
from sinhalallm import store_path  # noqa: E402

SPACE = "sinsankio/sinhalalm-offline"
DIST = ROOT / "web" / "dist"
TEMPLATE = Path(__file__).resolve().parent / "space" / "README.md"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy the application to the Hugging Face Space."
    )
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--skip-models", action="store_true", help="upload the application only")
    args = parser.parse_args()

    version = release_tag()
    if not (DIST / "index.html").exists():
        raise SystemExit("web/dist is missing; build the application first")
    models = store_path("web", MODEL_NAME, MODEL_VERSION)
    if not (models / "manifest.json").exists():
        raise SystemExit("the model package is missing; run make package first")

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", private=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        shutil.copytree(DIST, stage, dirs_exist_ok=True)
        readme = TEMPLATE.read_text(encoding="utf-8").replace("{version}", version)
        (stage / "README.md").write_text(readme, encoding="utf-8")
        api.upload_folder(
            folder_path=stage,
            repo_id=args.space,
            repo_type="space",
            commit_message=f"Deploy application {version}",
            delete_patterns=["assets/*", "*.js", "*.html", "*.webmanifest", "*.svg"],
        )
    print(f"deployed application {version} to {args.space}")

    if not args.skip_models:
        api.upload_folder(
            folder_path=models,
            path_in_repo=f"models/{MODEL_NAME}/{MODEL_VERSION}",
            repo_id=args.space,
            repo_type="space",
            commit_message=f"Deploy model package {MODEL_NAME}/{MODEL_VERSION}",
        )
        print(f"deployed model package {MODEL_NAME}/{MODEL_VERSION} to {args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
