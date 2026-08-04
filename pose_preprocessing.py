"""Shared, dependency-light preprocessing for Uni-Sign whole-body poses.

The training loader and the inference service must use exactly the same
133-keypoint partitioning and normalisation.  Keeping it in this module avoids
making the serving process import ``decord`` and dataset-loading code.
"""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch


POSE_KEYPOINT_COUNT = 133
POSE_CONFIDENCE_THRESHOLD = 0.3


def uniform_frame_indices(length: int, max_length: int) -> np.ndarray:
    """Return deterministic, uniformly distributed indices for inference."""
    if length <= 0:
        raise ValueError("Pose data contains no frames")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if length <= max_length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, num=max_length, dtype=np.int64)


def crop_scale(motion: np.ndarray, thr: float = POSE_CONFIDENCE_THRESHOLD):
    """Normalise a ``T x N x 3`` coordinate/confidence pose to ``[-1, 1]``.

    This is the same operation previously implemented in ``datasets.py``.  A
    zero result is intentional when there are too few confident keypoints.
    """
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None

    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    scale = max(xmax - xmin, ymax - ymin)
    if scale == 0:
        return np.zeros(motion.shape), 0, None

    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def _validate_frame_arrays(skeletons: Iterable[np.ndarray], confs: Iterable[np.ndarray]) -> None:
    skeletons = list(skeletons)
    confs = list(confs)
    if not skeletons or len(skeletons) != len(confs):
        raise ValueError("Pose keypoints and scores must contain the same non-zero frame count")
    first_skeleton = np.asarray(skeletons[0])
    first_conf = np.asarray(confs[0])
    if first_skeleton.shape != (1, POSE_KEYPOINT_COUNT, 2):
        raise ValueError(
            "Expected keypoints with shape (1, 133, 2), "
            f"got {first_skeleton.shape}")
    if first_conf.shape != (1, POSE_KEYPOINT_COUNT):
        raise ValueError(
            "Expected scores with shape (1, 133), "
            f"got {first_conf.shape}")


def load_part_kp(skeletons, confs, force_ok: bool = False):
    """Build Uni-Sign body/hand/face tensors from RTMPose whole-body frames.

    ``force_ok`` is retained for compatibility with the training loader.  It
    does not alter the original transform but makes call sites explicit about
    accepting zeroed low-confidence poses.
    """
    del force_ok  # Kept to preserve the established function contract.
    _validate_frame_arrays(skeletons, confs)
    threshold = POSE_CONFIDENCE_THRESHOLD
    kps_with_scores = {}
    scale = None

    for part in ("body", "left", "right", "face_all"):
        kps = []
        confidences = []
        for skeleton, conf in zip(skeletons, confs):
            skeleton = np.asarray(skeleton)[0]
            conf = np.asarray(conf)[0]
            if part == "body":
                selected_kps = skeleton[[0] + list(range(3, 11)), :]
                selected_conf = conf[[0] + list(range(3, 11))]
            elif part == "left":
                selected_kps = skeleton[91:112, :]
                selected_kps = selected_kps - selected_kps[0, :]
                selected_conf = conf[91:112]
            elif part == "right":
                selected_kps = skeleton[112:133, :]
                selected_kps = selected_kps - selected_kps[0, :]
                selected_conf = conf[112:133]
            else:  # face_all
                face_indices = list(range(23, 40, 2)) + list(range(83, 91)) + [53]
                selected_kps = skeleton[face_indices, :]
                selected_kps = selected_kps - selected_kps[-1, :]
                selected_conf = conf[face_indices]
            kps.append(selected_kps)
            confidences.append(selected_conf)

        kps_array = np.stack(kps, axis=0)
        confidence_array = np.stack(confidences, axis=0)
        if part == "body":
            result, scale, _ = crop_scale(
                np.concatenate([kps_array, confidence_array[..., None]], axis=-1),
                threshold,
            )
        else:
            if scale is None:
                raise RuntimeError("Body scale must be computed before other pose parts")
            result = np.concatenate([kps_array, confidence_array[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = result[..., :2] / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= threshold] = 0
        # Keep inference directly compatible with ``nn.Linear`` in CPU/FP32
        # smoke tests and avoid an unnecessary float64 host tensor before a
        # training/inference device cast.
        kps_with_scores[part] = torch.tensor(result, dtype=torch.float32)

    return kps_with_scores


def preprocess_pose_sequence(pose: dict, max_length: int = 64) -> dict:
    """Validate, uniformly sample, and format one trusted in-memory pose sequence."""
    if not isinstance(pose, dict) or {"keypoints", "scores"}.difference(pose):
        raise ValueError("Pose sequence must contain keypoints and scores")
    keypoints = pose["keypoints"]
    scores = pose["scores"]
    _validate_frame_arrays(keypoints, scores)
    indices = uniform_frame_indices(len(scores), max_length)
    sampled_keypoints = [keypoints[index] for index in indices]
    sampled_scores = [scores[index] for index in indices]
    parts = load_part_kp(sampled_keypoints, sampled_scores, force_ok=True)
    src_input = {part: tensor.unsqueeze(0) for part, tensor in parts.items()}
    src_input["attention_mask"] = torch.ones((1, len(indices)), dtype=torch.long)
    src_input["src_length_batch"] = torch.tensor([len(indices)], dtype=torch.long)
    src_input["name_batch"] = ["inference_sample"]
    return src_input
