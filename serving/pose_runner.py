"""RTMPose extraction with deterministic primary-signer selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np

from pose_preprocessing import POSE_CONFIDENCE_THRESHOLD, POSE_KEYPOINT_COUNT
from serving.artifacts import ModelBundle
from serving.errors import ServiceError


class WholeBodyEstimator(Protocol):
    def __call__(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class PoseQuality:
    decoded_frames: int
    detected_frames: int
    multi_person_frames: int
    mean_hand_keypoint_score: float

    @property
    def primary_person_coverage(self) -> float:
        return self.detected_frames / self.decoded_frames if self.decoded_frames else 0.0


@dataclass(frozen=True)
class PoseSequence:
    pose: dict
    quality: PoseQuality


class PrimaryPersonTracker:
    """Choose one signer deterministically across a sequence of detections."""

    def __init__(self, confidence_threshold: float = POSE_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = confidence_threshold
        self._last_center: np.ndarray | None = None

    def select(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        if keypoints is None or scores is None:
            return None, None, 0
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.size == 0 and scores.size == 0:
            return None, None, 0
        if keypoints.ndim != 3 or keypoints.shape[1:] != (POSE_KEYPOINT_COUNT, 2):
            raise ServiceError("POSE_FORMAT_INVALID", "Pose estimator returned unexpected keypoint shape")
        if scores.shape != keypoints.shape[:2]:
            raise ServiceError("POSE_FORMAT_INVALID", "Pose estimator returned unexpected score shape")
        if len(keypoints) == 0:
            return None, None, 0

        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for candidate_kps, candidate_scores in zip(keypoints, scores):
            confident = candidate_scores >= self.confidence_threshold
            if confident.sum() < 4:
                continue
            valid_points = candidate_kps[confident]
            minimum = valid_points.min(axis=0)
            maximum = valid_points.max(axis=0)
            area = float(np.prod(np.maximum(maximum - minimum, 1.0)) / max(width * height, 1))
            center = (minimum + maximum) / 2
            body_score = float(candidate_scores[:17].mean())
            centrality = 1.0 - min(
                float(np.linalg.norm(center / np.array([width, height]) - 0.5)),
                1.0,
            )
            continuity = 0.0
            if self._last_center is not None:
                continuity = 1.0 - min(
                    float(np.linalg.norm((center - self._last_center) / np.array([width, height]))),
                    1.0,
                )
            rank = body_score * 2.0 + area + centrality * 0.25 + continuity * 0.75
            candidates.append((rank, candidate_kps, candidate_scores))

        if not candidates:
            return None, None, 0
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, selected_kps, selected_scores = candidates[0]
        confident = selected_scores >= self.confidence_threshold
        self._last_center = selected_kps[confident].mean(axis=0)
        return selected_kps, selected_scores, len(candidates)


class PoseRunner:
    """Long-lived RTMPose runner for trusted, already-decoded video frames."""

    def __init__(self, estimator: WholeBodyEstimator) -> None:
        self.estimator = estimator

    @classmethod
    def load(cls, bundle: ModelBundle, *, backend: str, mode: str, device: str) -> "PoseRunner":
        try:
            from rtmlib import Wholebody
        except ImportError as exc:
            raise ServiceError(
                "POSE_RUNTIME_UNAVAILABLE",
                "RTMLib is not installed in the serving environment",
                503,
            ) from exc
        try:
            mode_config = Wholebody.MODE[mode]
            estimator = Wholebody(
                det=str(bundle.pose_detector_path),
                pose=str(bundle.pose_estimator_path),
                det_input_size=mode_config["det_input_size"],
                pose_input_size=mode_config["pose_input_size"],
                to_openpose=False,
                mode=mode,
                backend=backend,
                device=device,
            )
        except Exception as exc:  # rtmlib has backend-specific exception classes
            raise ServiceError("POSE_RUNTIME_UNAVAILABLE", "Unable to initialise RTMPose", 503) from exc
        return cls(estimator)

    def extract(self, frames: Iterable[np.ndarray]) -> PoseSequence:
        tracker = PrimaryPersonTracker()
        keypoints_list: list[np.ndarray] = []
        scores_list: list[np.ndarray] = []
        hand_scores: list[float] = []
        decoded_frames = 0
        multi_person_frames = 0

        for frame in frames:
            if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
                raise ServiceError("VIDEO_DECODE_FAILED", "Decoded frame must be an HxWx3 image")
            decoded_frames += 1
            height, width = frame.shape[:2]
            try:
                keypoints, scores = self.estimator(frame)
            except Exception as exc:
                raise ServiceError("POSE_EXTRACTION_FAILED", "RTMPose failed for an input frame", 503, retryable=True) from exc
            selected_kps, selected_scores, people = tracker.select(keypoints, scores, width, height)
            if people > 1:
                multi_person_frames += 1
            if selected_kps is None or selected_scores is None:
                continue
            normalised = selected_kps.astype(np.float32) / np.array([width, height], dtype=np.float32)
            keypoints_list.append(normalised[None, :, :])
            scores_list.append(selected_scores.astype(np.float32)[None, :])
            hand_scores.append(float(selected_scores[91:133].mean()))

        if not decoded_frames:
            raise ServiceError("NO_DECODABLE_FRAMES", "Video contained no decodable frames")
        quality = PoseQuality(
            decoded_frames=decoded_frames,
            detected_frames=len(keypoints_list),
            multi_person_frames=multi_person_frames,
            mean_hand_keypoint_score=float(np.mean(hand_scores)) if hand_scores else 0.0,
        )
        if not keypoints_list:
            raise ServiceError("NO_PERSON_DETECTED", "No signer was detected in the clip")
        return PoseSequence({"keypoints": keypoints_list, "scores": scores_list}, quality)
