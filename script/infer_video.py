#!/usr/bin/env bash
#!/usr/bin/env python3
"""Uni-Sign Inference Script for Vietnamese Isolated Sign Language Recognition (ISLR)."""

import os
import sys
import json
import time
import pickle
import argparse
from pathlib import Path

# Limit thread count for consistent performance
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import cv2
import torch
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import utils
from models import Uni_Sign
from datasets import load_part_kp


def extract_poses_from_video(video_source, device="cuda", mode="performance", duration=None):
    """Extract RTMPose 133 whole-body keypoints from video or camera stream."""
    try:
        from rtmlib import Wholebody
    except ImportError:
        raise ImportError("rtmlib is required for video pose extraction.")

    wholebody = Wholebody(
        to_openpose=False,
        mode=mode,
        backend="onnxruntime",
        device=device if torch.cuda.is_available() else "cpu"
    )

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames_to_read = int(fps * duration) if duration else float("inf")

    frames = []
    start_time = time.time()

    while cap.isOpened() and len(frames) < total_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if duration and (time.time() - start_time) >= duration:
            break

    cap.release()

    if not frames:
        raise ValueError(f"No frames captured from {video_source}")

    keypoints_list = []
    scores_list = []

    for frame in frames:
        H, W, _ = frame.shape
        kps, scs = wholebody(frame)
        norm_kps = kps / np.array([W, H])[None, None]
        keypoints_list.append(norm_kps)
        scores_list.append(scs)

    return {"keypoints": keypoints_list, "scores": scores_list}


def load_pose_pkl(pkl_path):
    """Load pre-extracted pose file (.pkl)."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def preprocess_pose_data(pose, max_length=64):
    """Sample frames and format keypoints into body part tensors."""
    keypoints = pose["keypoints"]
    scores = pose["scores"]
    duration = len(scores)

    if duration == 0:
        raise ValueError("Pose data contains 0 frames")

    if duration <= max_length:
        indices = np.arange(duration)
    else:
        indices = np.linspace(0, duration - 1, num=max_length, dtype=np.int64)

    sampled_kps = [keypoints[i] for i in indices]
    sampled_scs = [scores[i] for i in indices]

    kps_with_scores = load_part_kp(sampled_kps, sampled_scs, force_ok=True)

    src_input = {}
    for part, tensor in kps_with_scores.items():
        src_input[part] = tensor.unsqueeze(0)

    seq_len = len(indices)
    src_input["attention_mask"] = torch.ones((1, seq_len), dtype=torch.long)
    src_input["name_batch"] = ["inference_sample"]
    src_input["src_length_batch"] = torch.tensor([seq_len], dtype=torch.long)

    return src_input


def load_model_and_vocab(ckpt_path, vocab_path, device="cuda"):
    """Load Uni-Sign model weights and canonical Vietnamese vocabulary."""
    parser = utils.get_args_parser()
    args = parser.parse_args([])
    args.dataset = "CoSign"
    args.task = "ISLR"
    args.language = "Vietnamese"
    args.closed_vocabulary = True
    args.label_vocab = str(vocab_path)
    args.max_length = 64
    args.rgb_support = False

    labels = utils.load_label_vocabulary(vocab_path)
    model = Uni_Sign(args=args)

    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    else:
        model = model.cpu()

    return model, labels


def run_inference(model, labels, src_input, device="cuda"):
    """Run model forward pass and return candidate predictions."""
    tgt_dummy = {"gt_sentence": [labels[0]]}

    target_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    for k, v in src_input.items():
        if isinstance(v, torch.Tensor):
            src_input[k] = v.to(dtype=torch.float32, device=target_device)

    with torch.no_grad():
        stack_out = model(src_input, tgt_dummy)
        scores = model.score_candidate_labels(stack_out, labels)
        probs = torch.softmax(scores, dim=-1)[0].cpu().tolist()
        raw_scores = scores[0].cpu().tolist()

    ranked = sorted(
        zip(labels, probs, raw_scores),
        key=lambda item: item[1],
        reverse=True
    )

    return {
        "prediction": ranked[0][0],
        "confidence": ranked[0][1],
        "top_5": [
            {"label": label, "probability": prob, "score": score}
            for label, prob, score in ranked[:5]
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Uni-Sign CoSign Inference Script")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", type=str, help="Input video file path (.mp4, .avi, .mov, .webm)")
    input_group.add_argument("--pose", type=str, help="Input pose keypoints file path (.pkl)")
    input_group.add_argument("--webcam", action="store_true", help="Record live video from webcam")

    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--duration", type=float, default=3.0, help="Webcam capture duration in seconds (default: 3.0)")
    parser.add_argument("--ckpt", type=str, default="out/cosign_pose_islr_seed42/best_checkpoint.pth", help="Checkpoint file path")
    parser.add_argument("--vocab", type=str, default="data/CoSign/metadata/labels.json", help="Label vocabulary JSON path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", choices=["cuda", "cpu"])
    parser.add_argument("--json", action="store_true", help="Output strictly in JSON format")
    args = parser.parse_args()

    if args.pose:
        pose_data = load_pose_pkl(args.pose)
    elif args.video:
        pose_data = extract_poses_from_video(args.video, device=args.device)
    elif args.webcam:
        pose_data = extract_poses_from_video(args.camera, device=args.device, duration=args.duration)

    src_input = preprocess_pose_data(pose_data, max_length=64)
    model, labels = load_model_and_vocab(args.ckpt, args.vocab, device=args.device)

    start_time = time.time()
    result = run_inference(model, labels, src_input, device=args.device)
    result["inference_time_ms"] = round((time.time() - start_time) * 1000, 2)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("-" * 50)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}% ({result['inference_time_ms']} ms)")
        print("-" * 50)
        print("Top 5 candidates:")
        for rank, item in enumerate(result["top_5"], 1):
            print(f"  {rank}. {item['label']:<15} Proba: {item['probability'] * 100:>6.2f}%  (Log-Likelihood: {item['score']:.4f})")
        print("-" * 50)


if __name__ == "__main__":
    main()
