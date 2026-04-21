"""
push_artifacts.py — Upload model checkpoint and retrieval index to Hugging Face Hub.

Run once from any device that has the trained artifacts:
    python scripts/push_artifacts.py --repo <your-hf-username>/araml-artifacts

Requirements:
    pip install huggingface_hub   (already in requirements.txt)
    huggingface-cli login         (run this once, stores your token)
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi

ARTIFACTS = [
    ("results/best_model.pt",             "results/best_model.pt"),
    ("results/retrieval_index.faiss",     "results/retrieval_index.faiss"),
    ("results/retrieval_index_meta.npy",  "results/retrieval_index_meta.npy"),
]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def push(repo_id: str, private: bool, token: str) -> None:
    api = HfApi(token=token)

    print(f"Using repo '{repo_id}' (create it manually at huggingface.co/new if needed)...")

    for local_rel, remote_path in ARTIFACTS:
        local_path = os.path.join(_ROOT, local_rel)
        if not os.path.exists(local_path):
            print(f"  SKIP — not found: {local_path}")
            continue
        size_mb = os.path.getsize(local_path) / 1024 ** 2
        print(f"  Uploading {local_rel}  ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        print(f"    Done.")

    print(f"\nAll artifacts uploaded to: https://huggingface.co/{repo_id}")
    print("On any other device, run:")
    print(f"  python scripts/pull_artifacts.py --repo {repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",    required=True,
                        help="HuggingFace repo id, e.g. yourname/araml-artifacts")
    parser.add_argument("--token",   required=True,
                        help="HuggingFace write token (hf_...)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make the repo private (default: True)")
    args = parser.parse_args()
    push(args.repo, args.private, args.token)
