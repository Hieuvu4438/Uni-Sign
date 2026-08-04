#!/usr/bin/env python3
"""Create a checksummed immutable release manifest for the serving container."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serving.artifacts import MANIFEST_FILENAME, create_manifest  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", default="best_checkpoint.pth")
    parser.add_argument("--vocabulary", default="labels.json")
    parser.add_argument("--mt5", default="mt5-base")
    parser.add_argument("--pose-detector", default="pose-models/detector.onnx")
    parser.add_argument("--pose-estimator", default="pose-models/estimator.onnx")
    parser.add_argument(
        "--pose-mode",
        default="performance",
        choices=("performance", "balanced", "lightweight"),
    )
    parser.add_argument("--model-id", default="cosign-vi-islr")
    parser.add_argument("--version", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    manifest_path = bundle_dir / MANIFEST_FILENAME
    if manifest_path.exists() and not args.overwrite:
        parser.error(f"{manifest_path} already exists; use --overwrite to replace it")
    manifest = create_manifest(
        bundle_dir,
        checkpoint=args.checkpoint,
        vocabulary=args.vocabulary,
        mt5=args.mt5,
        pose_detector=args.pose_detector,
        pose_estimator=args.pose_estimator,
        model_id=args.model_id,
        version=args.version,
        pose_mode=args.pose_mode,
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote verified manifest: {manifest_path}")


if __name__ == "__main__":
    main()
