"""Versioned quality and uncertainty decisions for service responses."""

from __future__ import annotations

from dataclasses import dataclass

from serving.pose_runner import PoseQuality
from serving.settings import ServiceSettings


@dataclass(frozen=True)
class QualityDecision:
    status: str
    reason_codes: tuple[str, ...]
    retryable: bool


def assess_pose_quality(quality: PoseQuality, settings: ServiceSettings) -> QualityDecision:
    reasons: list[str] = []
    if quality.primary_person_coverage < settings.min_person_coverage:
        reasons.append("LOW_PERSON_COVERAGE")
    if quality.mean_hand_keypoint_score < settings.min_mean_hand_score:
        reasons.append("LOW_HAND_VISIBILITY")
    if quality.decoded_frames and quality.multi_person_frames / quality.decoded_frames > 0.25:
        reasons.append("MULTIPLE_PEOPLE")
    if reasons:
        return QualityDecision("low_quality", tuple(reasons), True)
    return QualityDecision("ok", (), False)


def assess_prediction(
    relative_score: float,
    score_margin: float,
    settings: ServiceSettings,
) -> QualityDecision:
    reasons: list[str] = []
    if settings.min_relative_score is not None and relative_score < settings.min_relative_score:
        reasons.append("LOW_RELATIVE_SCORE")
    if settings.min_score_margin is not None and score_margin < settings.min_score_margin:
        reasons.append("LOW_SCORE_MARGIN")
    if reasons:
        return QualityDecision("low_confidence", tuple(reasons), True)
    return QualityDecision("ok", (), False)
