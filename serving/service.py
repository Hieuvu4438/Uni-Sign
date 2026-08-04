"""Synchronous orchestration of safe video, pose, quality, and model stages."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from pose_preprocessing import preprocess_pose_sequence
from serving.artifacts import ModelBundle
from serving.errors import ModelUnavailableError
from serving.model_runner import ModelRunner
from serving.pose_runner import PoseRunner
from serving.quality import assess_pose_quality, assess_prediction
from serving.settings import ServiceSettings
from serving.video import VideoProcessor


class InferenceService:
    """Ready-to-use singleton service; callers provide a safe local temp file."""

    def __init__(
        self,
        settings: ServiceSettings,
        model_runner: ModelRunner,
        pose_runner: PoseRunner,
        video_processor: VideoProcessor | None = None,
    ) -> None:
        self.settings = settings
        self.model_runner = model_runner
        self.pose_runner = pose_runner
        self.video_processor = video_processor or VideoProcessor(settings)

    @classmethod
    def load(cls, settings: ServiceSettings) -> "InferenceService":
        settings.validate()
        if settings.model_bundle_dir is None:
            raise ModelUnavailableError("MODEL_BUNDLE_DIR is required")
        bundle = ModelBundle.load(settings.model_bundle_dir)
        if settings.pose_mode != bundle.pose_mode:
            raise ModelUnavailableError(
                f"POSE_MODE={settings.pose_mode} does not match bundle pose_mode={bundle.pose_mode}"
            )
        model_runner = ModelRunner.load(bundle, device_name=settings.device, dtype_name=settings.model_dtype)
        pose_runner = PoseRunner.load(
            bundle,
            backend=settings.pose_backend,
            mode=settings.pose_mode,
            device=settings.device,
        )
        model_runner.warmup()
        return cls(settings, model_runner, pose_runner)

    @property
    def ready(self) -> bool:
        return True

    def model_metadata(self) -> dict[str, Any]:
        return self.model_runner.metadata()

    def labels(self) -> list[str]:
        return list(self.model_runner.labels)

    def predict_file(self, path: Path, top_k: int) -> dict[str, Any]:
        started = time.perf_counter()
        info = self.video_processor.probe(path)
        probed_at = time.perf_counter()
        # Decode is a generator consumed by pose extraction, keeping at most
        # one full-resolution frame in host memory at a time.
        pose_sequence = self.pose_runner.extract(self.video_processor.decode(path, info))
        posed_at = time.perf_counter()
        quality = pose_sequence.quality
        pose_decision = assess_pose_quality(quality, self.settings)
        quality_payload = {
            "duration_ms": round(info.duration_seconds * 1000),
            "decoded_frames": quality.decoded_frames,
            "detected_pose_frames": quality.detected_frames,
            "pose_frames": min(quality.detected_frames, 64),
            "primary_person_coverage": round(quality.primary_person_coverage, 4),
            "mean_hand_keypoint_score": round(quality.mean_hand_keypoint_score, 4),
        }
        timing = {
            # FFmpeg decode and RTMPose overlap in this streaming pipeline, so
            # record probe separately and expose their combined processing time.
            "probe": round((probed_at - started) * 1000, 2),
            "decode_and_pose": round((posed_at - probed_at) * 1000, 2),
        }
        if pose_decision.status != "ok":
            timing["total"] = round((time.perf_counter() - started) * 1000, 2)
            return {
                "status": pose_decision.status,
                "reason_codes": list(pose_decision.reason_codes),
                "retryable": pose_decision.retryable,
                "model": self.model_metadata(),
                "input": quality_payload,
                "timing_ms": timing,
            }

        src_input = preprocess_pose_sequence(pose_sequence.pose, max_length=64)
        prepared_at = time.perf_counter()
        prediction = self.model_runner.predict(src_input, top_k=top_k)
        inferred_at = time.perf_counter()
        confidence_decision = assess_prediction(
            prediction.best.relative_score,
            prediction.score_margin_to_second,
            self.settings,
        )
        timing.update(
            {
                "preprocess": round((prepared_at - posed_at) * 1000, 2),
                "model": round((inferred_at - prepared_at) * 1000, 2),
                "total": round((inferred_at - started) * 1000, 2),
            }
        )
        candidates = [asdict(candidate) for candidate in prediction.candidates]
        result = {
            "status": confidence_decision.status,
            "retryable": confidence_decision.retryable,
            "model": self.model_metadata(),
            "prediction": {
                **asdict(prediction.best),
                "score_margin_to_second": prediction.score_margin_to_second,
            },
            "top_k": candidates,
            "input": quality_payload,
            "timing_ms": timing,
        }
        if confidence_decision.reason_codes:
            result["reason_codes"] = list(confidence_decision.reason_codes)
        return result
