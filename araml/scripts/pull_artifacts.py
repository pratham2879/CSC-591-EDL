"""
pull_artifacts.py — Download model checkpoint and retrieval index from Hugging Face Hub.

Run on any new device after setting up the venv:
    python scripts/pull_artifacts.py --repo <your-hf-username>/araml-artifacts

Requirements:
    pip install huggingface_hub   (already in requirements.txt)
    huggingface-cli login         (only needed if repo is private)
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import hf_hub_download

ARTIFACTS = [
    ("results/best_model.pt",            "results/best_model.pt"),
    ("results/retrieval_index.faiss",    "results/retrieval_index.faiss"),
    ("results/retrieval_index_meta.npy", "results/retrieval_index_meta.npy"),
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pull(repo_id: str, token: str) -> None:
    print(f"Downloading artifacts from: https://huggingface.co/{repo_id}\n")

    for remote_path, local_rel in ARTIFACTS:
        local_path = os.path.join(_ROOT, local_rel)

        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / 1024 ** 2
            print(f"  EXISTS — skipping {local_rel}  ({size_mb:.1f} MB)")
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"  Downloading {remote_path} ...")

        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="model",
            local_dir=_ROOT,
            local_dir_use_symlinks=False,
            token=token,
        )
        size_mb = os.path.getsize(downloaded) / 1024 ** 2
        print(f"    Saved to {local_rel}  ({size_mb:.1f} MB)")

    print("\nAll artifacts ready. You can now run:")
    print("  python scripts/evaluate.py --checkpoint results/best_model.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",  required=True,
                        help="HuggingFace repo id, e.g. prathamP6969/araml-artifacts")
    parser.add_argument("--token", required=True,
                        help="HuggingFace token (hf_...)")
    args = parser.parse_args()
    pull(args.repo, args.token)
