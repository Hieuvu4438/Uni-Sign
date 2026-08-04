"""Immutable model-bundle manifest creation and verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
import unicodedata

from serving.errors import ServiceError


MANIFEST_FILENAME = "manifest.json"
MANIFEST_SCHEMA_VERSION = 1


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree deterministically."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _relative_path(bundle_dir: Path, value: str, artifact_name: str) -> Path:
    candidate = (bundle_dir / value).resolve()
    try:
        candidate.relative_to(bundle_dir.resolve())
    except ValueError as exc:
        raise ServiceError(
            "MODEL_MANIFEST_INVALID",
            f"Artifact {artifact_name} must stay inside the model bundle",
            503,
        ) from exc
    return candidate


def _load_labels(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    entries = raw.get("labels", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ServiceError("MODEL_MANIFEST_INVALID", "Vocabulary must contain a labels list", 503)
    labels: list[str] = []
    for entry in entries:
        text = entry.get("text") if isinstance(entry, dict) else entry
        if not isinstance(text, str) or not text.strip():
            raise ServiceError("MODEL_MANIFEST_INVALID", "Vocabulary contains an empty label", 503)
        labels.append(unicodedata.normalize("NFC", " ".join(text.strip().split())))
    if len(labels) != len(set(labels)):
        raise ServiceError("MODEL_MANIFEST_INVALID", "Vocabulary contains duplicate labels", 503)
    return labels


@dataclass(frozen=True)
class ModelBundle:
    root: Path
    manifest: dict[str, Any]
    checkpoint_path: Path
    vocabulary_path: Path
    mt5_dir: Path
    pose_detector_path: Path
    pose_estimator_path: Path
    labels: tuple[str, ...]

    @property
    def model_id(self) -> str:
        return self.manifest["model"]["id"]

    @property
    def version(self) -> str:
        return self.manifest["model"]["version"]

    @property
    def checkpoint_sha256(self) -> str:
        return self.manifest["artifacts"]["checkpoint"]["sha256"]

    @property
    def vocabulary_sha256(self) -> str:
        return self.manifest["artifacts"]["vocabulary"]["sha256"]

    @property
    def pose_mode(self) -> str:
        return self.manifest["runtime"]["pose_mode"]

    @classmethod
    def load(cls, root: Path) -> "ModelBundle":
        root = Path(root).resolve()
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ServiceError("MODEL_MANIFEST_MISSING", f"Missing {manifest_path}", 503)
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
                raise ValueError("unsupported schema_version")
            model = manifest["model"]
            runtime = manifest["runtime"]
            artifacts = manifest["artifacts"]
            if not isinstance(model, dict) or not isinstance(runtime, dict) or not isinstance(artifacts, dict):
                raise ValueError("model, runtime, and artifacts must be objects")
            if runtime["dataset"] != "CoSign" or runtime["task"] != "ISLR":
                raise ValueError("bundle must target CoSign ISLR")
            if runtime["language"] != "Vietnamese" or runtime["rgb_support"] is not False:
                raise ValueError("bundle must use Vietnamese pose-only configuration")
            if runtime["max_length"] != 64:
                raise ValueError("bundle must use max_length=64")
            if runtime["pose_mode"] not in {"performance", "balanced", "lightweight"}:
                raise ValueError("bundle has an unsupported pose_mode")
            if not isinstance(model["id"], str) or not isinstance(model["version"], str):
                raise ValueError("model id/version must be strings")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ServiceError("MODEL_MANIFEST_INVALID", f"Invalid model manifest: {exc}", 503) from exc

        resolved: dict[str, Path] = {}
        for name in ("checkpoint", "vocabulary", "mt5", "pose_detector", "pose_estimator"):
            entry = artifacts.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ServiceError("MODEL_MANIFEST_INVALID", f"Missing artifact {name}", 503)
            path = _relative_path(root, entry["path"], name)
            if not path.exists():
                raise ServiceError("MODEL_ARTIFACT_MISSING", f"Missing artifact {name}", 503)
            expected_hash = entry.get("sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ServiceError("MODEL_MANIFEST_INVALID", f"Invalid SHA-256 for {name}", 503)
            actual_hash = sha256_path(path)
            if actual_hash != expected_hash:
                raise ServiceError("MODEL_ARTIFACT_HASH_MISMATCH", f"Hash mismatch for {name}", 503)
            resolved[name] = path

        labels = _load_labels(resolved["vocabulary"])
        expected_count = runtime.get("label_count")
        if expected_count != len(labels) or expected_count != 30:
            raise ServiceError(
                "MODEL_MANIFEST_INVALID",
                "CoSign serving requires exactly 30 canonical labels",
                503,
            )
        return cls(
            root=root,
            manifest=manifest,
            checkpoint_path=resolved["checkpoint"],
            vocabulary_path=resolved["vocabulary"],
            mt5_dir=resolved["mt5"],
            pose_detector_path=resolved["pose_detector"],
            pose_estimator_path=resolved["pose_estimator"],
            labels=tuple(labels),
        )


def create_manifest(
    bundle_dir: Path,
    *,
    checkpoint: str,
    vocabulary: str,
    mt5: str,
    pose_detector: str,
    pose_estimator: str,
    model_id: str,
    version: str,
    pose_mode: str = "performance",
) -> dict[str, Any]:
    """Create a manifest for files already placed under ``bundle_dir``."""
    bundle_dir = Path(bundle_dir).resolve()
    items = {
        "checkpoint": checkpoint,
        "vocabulary": vocabulary,
        "mt5": mt5,
        "pose_detector": pose_detector,
        "pose_estimator": pose_estimator,
    }
    artifacts = {}
    for name, relative in items.items():
        path = _relative_path(bundle_dir, relative, name)
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
        artifacts[name] = {"path": relative, "sha256": sha256_path(path)}
    labels = _load_labels(_relative_path(bundle_dir, vocabulary, "vocabulary"))
    if len(labels) != 30:
        raise ValueError(f"Expected exactly 30 labels, found {len(labels)}")
    if pose_mode not in {"performance", "balanced", "lightweight"}:
        raise ValueError("pose_mode must be performance, balanced, or lightweight")
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": {"id": model_id, "version": version},
        "runtime": {
            "dataset": "CoSign",
            "task": "ISLR",
            "language": "Vietnamese",
            "rgb_support": False,
            "max_length": 64,
            "label_count": 30,
            "pose_mode": pose_mode,
        },
        "artifacts": artifacts,
    }
