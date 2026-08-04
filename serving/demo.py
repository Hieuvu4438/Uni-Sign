"""Clearly marked no-model service used only for API contract testing."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any


# Keep the frontend contract useful without copying training media or model
# weights into a Swagger/demo container. Production labels always come from the
# checksummed model bundle instead.
DEMO_LABELS = (
    "Ban ngày", "Ban đêm", "Bàn tay", "Bạn thân", "Bệnh viện", "Chiều", "Chào", "Chân",
    "Chúng ta", "Chậm lại", "Con gấu", "Cá", "Có thể", "Cơ thể", "Cứu", "Dễ",
    "Hôm nay", "Họ", "Học sinh", "Khóc", "Mua", "Mời vào", "Nghe", "Ngón tay",
    "Nhà", "Nhìn", "Nhầm", "Nói", "Nặng", "Ăn",
)
DEMO_VOCABULARY_SHA256 = sha256("\n".join(DEMO_LABELS).encode("utf-8")).hexdigest()


class DemoInferenceService:
    """Returns deterministic shaped responses without decoding or inferring.

    It exists exclusively for Swagger/frontend integration.  Every response is
    labelled ``demo`` and must never be used as a recognition result.
    """

    ready = True

    def model_metadata(self) -> dict[str, Any]:
        return {
            "id": "cosign-vi-islr-demo",
            "version": "api-contract-demo",
            "checkpoint_sha256": "0" * 64,
            "vocabulary_sha256": DEMO_VOCABULARY_SHA256,
            "label_count": len(DEMO_LABELS),
            "max_frames": 64,
            "pose_mode": "not-applicable",
            "dtype": "not-applicable",
            "is_demo": True,
        }

    def labels(self) -> list[str]:
        return list(DEMO_LABELS)

    def predict_file(self, path: Path, top_k: int) -> dict[str, Any]:
        payload_hash = sha256(path.read_bytes()).digest()
        offset = payload_hash[0] % len(DEMO_LABELS)
        ranked_labels = [DEMO_LABELS[(offset + index) % len(DEMO_LABELS)] for index in range(top_k)]
        raw_weights = [1 / (index + 1) for index in range(top_k)]
        normalizer = sum(raw_weights)
        candidates = [
            {
                "label": label,
                "rank": index + 1,
                "relative_score": round(raw_weights[index] / normalizer, 6),
                "log_likelihood": round(-0.25 * index, 6),
            }
            for index, label in enumerate(ranked_labels)
        ]
        return {
            "status": "ok",
            "retryable": False,
            "demo": True,
            "model": self.model_metadata(),
            "prediction": {**candidates[0], "score_margin_to_second": 0.25 if top_k > 1 else 0.0},
            "top_k": candidates,
            "input": {
                "duration_ms": 0,
                "decoded_frames": 0,
                "detected_pose_frames": 0,
                "pose_frames": 0,
                "primary_person_coverage": 0.0,
                "mean_hand_keypoint_score": 0.0,
                "validation": "skipped_in_demo_mode",
            },
            "timing_ms": {"demo": 0.0, "total": 0.0},
        }
