import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
SPACE = "sinsankio/sinhalalm-offline-article"
TEMPLATE = ROOT / "deploy" / "article"
COMMUNITY = ROOT / "docs" / "community"


def stage(target: Path) -> None:
    for name in ("index.html", "README.md"):
        shutil.copy(TEMPLATE / name, target / name)
    shutil.copy(COMMUNITY / "article.md", target / "article.md")
    diagrams = target / "diagrams"
    diagrams.mkdir()
    for svg in sorted((COMMUNITY / "diagrams").glob("*.svg")):
        shutil.copy(svg, diagrams / svg.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the community article as a static Space.")
    parser.add_argument("--space", default=SPACE)
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as staging:
        stage(Path(staging))
        api.upload_folder(
            folder_path=staging,
            repo_id=args.space,
            repo_type="space",
            commit_message="Publish the article",
            delete_patterns=["*.md", "*.html", "diagrams/*"],
        )
    print(f"published the article to {args.space}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
