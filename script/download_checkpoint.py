#!/usr/bin/env python3
"""Download the weights required to fine-tune Uni-Sign on any machine."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


MT5_FILES = [
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer_config.json",
]

CHECKPOINTS = {
    "pose-only": "wlasl_pose_only_islr.pth",
    "rgb-pose": "wlasl_rgb_pose_islr.pth",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=CHECKPOINTS,
        default="pose-only",
        help="Uni-Sign input mode to prepare (default: %(default)s)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("pretrained_weight"),
        help="Root directory for downloaded weights, relative or absolute (default: %(default)s)",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision, tag, or commit of Uni-Sign weights (default: %(default)s)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face access token for gated/private repositories",
    )
    parser.add_argument(
        "--skip-mt5",
        action="store_true",
        help="Skip mT5-base; use only when it is already available locally",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download again even if Hugging Face has a cached copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_mt5:
        mt5_dir = output_root / "mt5-base"
        print(f"Downloading mT5-base to {mt5_dir}")
        snapshot_download(
            repo_id="google/mt5-base",
            allow_patterns=MT5_FILES,
            local_dir=mt5_dir,
            token=args.token,
            force_download=args.force_download,
        )

    checkpoint_name = CHECKPOINTS[args.mode]
    checkpoint_dir = output_root / "unisign"
    print(f"Downloading {checkpoint_name} to {checkpoint_dir}")
    checkpoint_path = hf_hub_download(
        repo_id="ZechengLi19/Uni-Sign",
        filename=checkpoint_name,
        revision=args.revision,
        local_dir=checkpoint_dir,
        token=args.token,
        force_download=args.force_download,
    )
    print(f"Uni-Sign checkpoint: {Path(checkpoint_path).resolve()}")


if __name__ == "__main__":
    main()
