#!/usr/bin/env python3
"""Download a Hugging Face model file to a user-selected local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        default="ZechengLi19/Uni-Sign",
        help="Hugging Face repository ID (default: %(default)s)",
    )
    parser.add_argument(
        "--filename",
        default="wlasl_pose_only_islr.pth",
        help="File to download from the repository (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pretrained_weight/unisign"),
        help="Destination directory, relative or absolute (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision, tag, or commit (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token for gated/private repositories",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even when a cached copy is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=output_dir,
        token=args.token,
        force_download=args.force_download,
    )
    print(f"Downloaded: {Path(downloaded_path).resolve()}")


if __name__ == "__main__":
    main()
