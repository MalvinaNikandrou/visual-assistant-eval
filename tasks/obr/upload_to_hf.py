import argparse
import os

from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hf_repo_id",
        default="gpantaz/vae",
        help="Hugging Face repository ID containing model checkpoints",
    )

    parser.add_argument(
        "--local_dir",
        default="./",
        help="Local directory to download checkpoints",
    )

    parser.add_argument(
        "--token",
        help="Hugging Face API token",
    )

    parser.add_argument(
        "--local_prefix_folder_name",
        default="lora",
        choices=["lora", "squad"],
        help="Local prefix folder name",
    )

    parser.add_argument(
        "--upload_prefix_folder_name",
        default="wmt-",
        choices=["wmt-", "squad-"],
        help="Upload prefix folder name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    api = HfApi()
    api.create_repo(
        repo_id=args.hf_repo_id,
        repo_type="model",
        token=args.token,
        private=True,
        exist_ok=True,
    )

    pattern = ["*flores*", "*ntrex*"]
    local_dir = args.local_dir
    for path in os.listdir(local_dir):
        if not path.startswith(args.local_prefix_folder_name):
            continue

        local_folder_path = os.path.join(local_dir, path)
        if not os.path.isdir(local_folder_path):
            continue

        print(f"Uploading {local_folder_path} to {args.hf_repo_id}...")

        api.upload_folder(
            folder_path=local_folder_path,
            repo_id=args.hf_repo_id,
            path_in_repo=f"{args.upload_prefix_folder_name}{path}",
            repo_type="model",
            allow_patterns=pattern,
            token=args.token,
        )
