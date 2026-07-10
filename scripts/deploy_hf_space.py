"""Create (or update) a Hugging Face Space and upload this project to it.

Prerequisite (one time): authenticate with a WRITE token so the token stays
on your machine and is never typed into a chat:

    hf auth login        # paste a token from https://huggingface.co/settings/tokens (role: write)

Then run:

    python scripts/deploy_hf_space.py                 # default space name "educational-chatbot"
    python scripts/deploy_hf_space.py --name my-space --private

Your live link will be printed as:  https://<user>-<space>.hf.space
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

IGNORE = [
    ".git/*", ".git*", "**/.git/*",
    ".venv/*", "venv/*", "**/__pycache__/*", "*.pyc",
    ".env", "*.db",
    "data/raw/*", "data/index/*", "data/web_cache/*",
    "models/*", "*.safetensors", "*.bin", "*.pt",
    "notebooks/*",  # training notebooks aren't needed to run the Space
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="educational-chatbot")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        print("Not logged in. Run:  hf auth login   (paste a WRITE token)")
        sys.exit(1)

    repo_id = f"{user}/{args.name}"
    print(f"Creating/updating Space: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=args.private,
        exist_ok=True,
    )

    print("Uploading project files ...")
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=str(ROOT),
        ignore_patterns=IGNORE,
        commit_message="Deploy Educational Chatbot",
    )

    live = f"https://{user.lower()}-{args.name}.hf.space"
    print("\nDone!")
    print(f"Space page: https://huggingface.co/spaces/{repo_id}")
    print(f"Live link:  {live}")
    print(
        "\nOptional: in the Space Settings -> Variables and secrets, add your "
        "trained model repos:\n"
        "  SRKI_INTENT_MODEL = <user>/srki-intent-roberta\n"
        "  SRKI_GENERATOR_MODEL = <user>/srki-generator   (and USE_GENERATOR = true)"
    )


if __name__ == "__main__":
    main()
