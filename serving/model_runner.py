"""Long-lived Uni-Sign closed-vocabulary model runner."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from models import Uni_Sign
from serving.artifacts import ModelBundle
from serving.errors import ServiceError


@dataclass(frozen=True)
class CandidatePrediction:
    label: str
    rank: int
    relative_score: float
    log_likelihood: float


@dataclass(frozen=True)
class ModelPrediction:
    candidates: tuple[CandidatePrediction, ...]
    score_margin_to_second: float

    @property
    def best(self) -> CandidatePrediction:
        return self.candidates[0]


class ModelRunner:
    """Loads checkpoint/mT5 once and scores the immutable 30-label vocabulary."""

    def __init__(self, model: Uni_Sign, bundle: ModelBundle, device: torch.device, dtype: torch.dtype) -> None:
        self.model = model
        self.bundle = bundle
        self.device = device
        self.dtype = dtype
        self.labels = bundle.labels
        self.candidate_token_ids = self.model.mt5_tokenizer(
            list(self.labels),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
        ).input_ids.cpu()

    @classmethod
    def load(cls, bundle: ModelBundle, *, device_name: str, dtype_name: str) -> "ModelRunner":
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ServiceError("MODEL_UNAVAILABLE", "CUDA was requested but is not available", 503)
        device = torch.device(device_name)
        if dtype_name == "fp16":
            dtype = torch.float16
        elif dtype_name == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        if device.type == "cpu" and dtype != torch.float32:
            raise ServiceError("MODEL_UNAVAILABLE", "CPU serving requires MODEL_DTYPE=fp32", 503)

        args = SimpleNamespace(
            hidden_dim=256,
            dataset="CoSign",
            task="ISLR",
            language="Vietnamese",
            label_smoothing=0.0,
            rgb_support=False,
            max_length=64,
            mt5_path=str(bundle.mt5_dir),
        )
        try:
            model = Uni_Sign(args=args)
            checkpoint = torch.load(bundle.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=True)
            model.to(device=device, dtype=dtype)
            model.eval()
        except Exception as exc:
            raise ServiceError("MODEL_LOAD_FAILED", "Unable to load Uni-Sign checkpoint strictly", 503) from exc
        return cls(model, bundle, device, dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.bundle.model_id,
            "version": self.bundle.version,
            "checkpoint_sha256": self.bundle.checkpoint_sha256,
            "vocabulary_sha256": self.bundle.vocabulary_sha256,
            "label_count": len(self.labels),
            "max_frames": 64,
            "pose_mode": self.bundle.pose_mode,
            "dtype": str(self.dtype).removeprefix("torch."),
        }

    def warmup(self) -> None:
        """Exercise the real encode/scoring path before readiness is enabled."""
        source = {
            "body": torch.zeros((1, 1, 9, 3), dtype=torch.float32),
            "left": torch.zeros((1, 1, 21, 3), dtype=torch.float32),
            "right": torch.zeros((1, 1, 21, 3), dtype=torch.float32),
            "face_all": torch.zeros((1, 1, 18, 3), dtype=torch.float32),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
            "src_length_batch": torch.ones((1,), dtype=torch.long),
            "name_batch": ["startup_warmup"],
        }
        prediction = self.predict(source, top_k=1)
        if not prediction.candidates:
            raise ServiceError("MODEL_LOAD_FAILED", "Model warm-up returned no candidates", 503)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, src_input: dict, top_k: int) -> ModelPrediction:
        if not 1 <= top_k <= min(5, len(self.labels)):
            raise ValueError("top_k must be in 1..5")
        model_input = {}
        for key, value in src_input.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    model_input[key] = value.to(device=self.device, dtype=self.dtype)
                else:
                    model_input[key] = value.to(device=self.device)
            else:
                model_input[key] = value
        stack_out = self.model.encode(model_input)
        scores = self.model.score_candidate_labels(
            stack_out,
            list(self.labels),
            candidate_token_ids=self.candidate_token_ids,
        )[0]
        relative_scores = torch.softmax(scores.float(), dim=0)
        ranked_indices = torch.argsort(scores, descending=True)[:top_k].cpu().tolist()
        candidates = tuple(
            CandidatePrediction(
                label=self.labels[index],
                rank=rank + 1,
                relative_score=float(relative_scores[index].cpu()),
                log_likelihood=float(scores[index].float().cpu()),
            )
            for rank, index in enumerate(ranked_indices)
        )
        sorted_scores = torch.sort(scores, descending=True).values
        margin = float((sorted_scores[0] - sorted_scores[1]).float().cpu()) if len(sorted_scores) > 1 else 0.0
        return ModelPrediction(candidates=candidates, score_margin_to_second=margin)
