"""Validated runtime configuration for the inference service."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    value = int(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


@dataclass(frozen=True)
class ServiceSettings:
    """Settings intentionally limited to deployment-relevant runtime values."""

    model_bundle_dir: Path | None = None
    device: str = "cuda"
    model_dtype: str = "fp32"
    pose_backend: str = "onnxruntime"
    pose_mode: str = "performance"
    max_upload_bytes: int = 20 * 1024 * 1024
    max_duration_seconds: float = 8.0
    max_decoded_frames: int = 192
    max_pixels: int = 1920 * 1080
    max_queue_depth: int = 1
    top_k_default: int = 5
    temp_dir: Path = Path("/tmp/unisign")
    service_api_key: str = ""
    demo_mode: bool = False
    min_person_coverage: float = 0.70
    min_mean_hand_score: float = 0.20
    min_relative_score: float | None = None
    min_score_margin: float | None = None

    @classmethod
    def from_env(cls) -> "ServiceSettings":
        raw_bundle = os.getenv("MODEL_BUNDLE_DIR", "").strip()
        bundle = Path(raw_bundle).resolve() if raw_bundle else None
        dtype = os.getenv("MODEL_DTYPE", "fp32").lower()
        if dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("MODEL_DTYPE must be fp32, fp16, or bf16")
        pose_mode = os.getenv("POSE_MODE", "performance")
        if pose_mode not in {"performance", "balanced", "lightweight"}:
            raise ValueError("POSE_MODE must be performance, balanced, or lightweight")
        raw_relative_score = os.getenv("MIN_RELATIVE_SCORE", "").strip()
        raw_margin = os.getenv("MIN_SCORE_MARGIN", "").strip()
        return cls(
            model_bundle_dir=bundle,
            device=os.getenv("DEVICE", "cuda").lower(),
            model_dtype=dtype,
            pose_backend=os.getenv("POSE_BACKEND", "onnxruntime"),
            pose_mode=pose_mode,
            max_upload_bytes=_env_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024, 1),
            max_duration_seconds=_env_float("MAX_DURATION_SECONDS", 8.0, 0.1),
            max_decoded_frames=_env_int("MAX_DECODED_FRAMES", 192, 1),
            max_pixels=_env_int("MAX_PIXELS", 1920 * 1080, 1),
            max_queue_depth=_env_int("MAX_QUEUE_DEPTH", 1, 1),
            top_k_default=_env_int("TOP_K_DEFAULT", 5, 1),
            temp_dir=Path(os.getenv("TEMP_DIR", "/tmp/unisign")).resolve(),
            service_api_key=os.getenv("SERVICE_API_KEY", ""),
            demo_mode=_env_bool("DEMO_MODE", False),
            min_person_coverage=_env_float("MIN_PERSON_COVERAGE", 0.70, 0.0),
            min_mean_hand_score=_env_float("MIN_MEAN_HAND_SCORE", 0.20, 0.0),
            min_relative_score=float(raw_relative_score) if raw_relative_score else None,
            min_score_margin=float(raw_margin) if raw_margin else None,
        )

    def validate(self) -> None:
        if self.device not in {"cuda", "cpu"}:
            raise ValueError("DEVICE must be cuda or cpu")
        if not 1 <= self.top_k_default <= 5:
            raise ValueError("TOP_K_DEFAULT must be in 1..5")
        if not 0 <= self.min_person_coverage <= 1:
            raise ValueError("MIN_PERSON_COVERAGE must be in 0..1")
        if not 0 <= self.min_mean_hand_score <= 1:
            raise ValueError("MIN_MEAN_HAND_SCORE must be in 0..1")
        if self.min_relative_score is not None and not 0 <= self.min_relative_score <= 1:
            raise ValueError("MIN_RELATIVE_SCORE must be in 0..1")
        if self.min_score_margin is not None and self.min_score_margin < 0:
            raise ValueError("MIN_SCORE_MARGIN must be non-negative")
