from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from pose_preprocessing import preprocess_pose_sequence, uniform_frame_indices
from serving.artifacts import MANIFEST_FILENAME, ModelBundle, create_manifest
from serving.model_runner import CandidatePrediction, ModelPrediction, ModelRunner
from serving.pose_runner import PoseQuality, PoseSequence, PrimaryPersonTracker
from serving.quality import assess_pose_quality, assess_prediction
from serving.service import InferenceService
from serving.settings import ServiceSettings
from serving.video import VideoInfo, VideoProcessor, _rotation_degrees


def make_pose(frame_count: int = 70) -> dict:
    keypoints = []
    scores = []
    for frame in range(frame_count):
        points = np.zeros((1, 133, 2), dtype=np.float32)
        points[0, :, 0] = np.linspace(0.1, 0.9, 133)
        points[0, :, 1] = np.linspace(0.2, 0.8, 133) + frame * 0.0001
        keypoints.append(points)
        scores.append(np.ones((1, 133), dtype=np.float32))
    return {"keypoints": keypoints, "scores": scores}


class PreprocessingTests(unittest.TestCase):
    def test_uniform_preprocessing_matches_model_shape_contract(self):
        src_input = preprocess_pose_sequence(make_pose(), max_length=64)
        self.assertEqual(src_input["body"].shape, (1, 64, 9, 3))
        self.assertEqual(src_input["left"].shape, (1, 64, 21, 3))
        self.assertEqual(src_input["right"].shape, (1, 64, 21, 3))
        self.assertEqual(src_input["face_all"].shape, (1, 64, 18, 3))
        self.assertEqual(src_input["attention_mask"].shape, (1, 64))
        self.assertEqual(src_input["body"].dtype, torch.float32)
        self.assertEqual(uniform_frame_indices(70, 64)[[0, -1]].tolist(), [0, 69])


class ArtifactTests(unittest.TestCase):
    def test_manifest_verifies_all_release_artifacts_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mt5-base").mkdir()
            (root / "mt5-base" / "config.json").write_text("{}", encoding="utf-8")
            (root / "pose-models").mkdir()
            (root / "best_checkpoint.pth").write_bytes(b"trusted checkpoint")
            (root / "pose-models" / "detector.onnx").write_bytes(b"detector")
            (root / "pose-models" / "estimator.onnx").write_bytes(b"estimator")
            labels = ["Ban ngày"] + [f"label-{index}" for index in range(1, 30)]
            (root / "labels.json").write_text(json.dumps({"labels": labels}, ensure_ascii=False), encoding="utf-8")
            manifest = create_manifest(
                root,
                checkpoint="best_checkpoint.pth",
                vocabulary="labels.json",
                mt5="mt5-base",
                pose_detector="pose-models/detector.onnx",
                pose_estimator="pose-models/estimator.onnx",
                model_id="cosign-vi-islr",
                version="test-r1",
            )
            (root / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
            bundle = ModelBundle.load(root)
            self.assertEqual(bundle.version, "test-r1")
            self.assertEqual(len(bundle.labels), 30)
            (root / "best_checkpoint.pth").write_bytes(b"tampered")
            with self.assertRaisesRegex(Exception, "Hash mismatch"):
                ModelBundle.load(root)

    def test_manifest_cli_writes_a_loadable_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "mt5-base").mkdir()
            (root / "mt5-base" / "config.json").write_text("{}", encoding="utf-8")
            (root / "pose-models").mkdir()
            (root / "best_checkpoint.pth").write_bytes(b"trusted checkpoint")
            (root / "pose-models" / "detector.onnx").write_bytes(b"detector")
            (root / "pose-models" / "estimator.onnx").write_bytes(b"estimator")
            (root / "labels.json").write_text(
                json.dumps({"labels": [f"label-{index}" for index in range(30)]}),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    sys.executable,
                    "script/create_model_manifest.py",
                    "--bundle-dir", str(root),
                    "--version", "test-r2",
                ],
                check=True,
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True,
                text=True,
            )
            self.assertTrue((root / MANIFEST_FILENAME).is_file())
            self.assertEqual(ModelBundle.load(root).version, "test-r2")


class PoseSelectionTests(unittest.TestCase):
    def test_tracker_chooses_and_tracks_primary_person(self):
        tracker = PrimaryPersonTracker()
        keypoints = np.zeros((2, 133, 2), dtype=np.float32)
        scores = np.ones((2, 133), dtype=np.float32)
        keypoints[0, :, :] = [80, 50]  # smaller candidate near left edge
        keypoints[0, :10, 0] = np.linspace(60, 100, 10)
        keypoints[0, :10, 1] = np.linspace(30, 70, 10)
        keypoints[1, :, :] = [500, 300]  # larger, central candidate
        keypoints[1, :20, 0] = np.linspace(350, 650, 20)
        keypoints[1, :20, 1] = np.linspace(100, 500, 20)
        selected, selected_scores, people = tracker.select(keypoints, scores, 1000, 600)
        self.assertEqual(people, 2)
        self.assertIsNotNone(selected)
        self.assertIsNotNone(selected_scores)
        self.assertGreater(float(selected[:, 0].mean()), 300)

    def test_tracker_handles_an_empty_detector_result(self):
        selected, selected_scores, people = PrimaryPersonTracker().select(
            np.empty((0, 133, 2), dtype=np.float32),
            np.empty((0, 133), dtype=np.float32),
            640,
            480,
        )
        self.assertIsNone(selected)
        self.assertIsNone(selected_scores)
        self.assertEqual(people, 0)


class QualityTests(unittest.TestCase):
    def test_quality_and_confidence_policies_are_explicit(self):
        settings = ServiceSettings(min_person_coverage=0.8, min_mean_hand_score=0.5)
        quality = PoseQuality(10, 6, 4, 0.3)
        decision = assess_pose_quality(quality, settings)
        self.assertEqual(decision.status, "low_quality")
        self.assertIn("LOW_PERSON_COVERAGE", decision.reason_codes)
        self.assertIn("LOW_HAND_VISIBILITY", decision.reason_codes)
        self.assertIn("MULTIPLE_PEOPLE", decision.reason_codes)
        confident = assess_prediction(0.2, 0.1, ServiceSettings(min_relative_score=0.4, min_score_margin=0.2))
        self.assertEqual(confident.status, "low_confidence")


class _TokenResult:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class _FakeTokenizer:
    def __call__(self, labels, **_kwargs):
        return _TokenResult(torch.ones((len(labels), 2), dtype=torch.long))


class _FakeModel:
    mt5_tokenizer = _FakeTokenizer()

    def encode(self, source):
        return {"source": source}

    def score_candidate_labels(self, _encoded, labels, candidate_token_ids=None):
        assert candidate_token_ids is not None
        return torch.tensor([[0.1, 0.8, 0.3] + [-1.0] * (len(labels) - 3)])


class ModelRunnerTests(unittest.TestCase):
    def test_model_runner_uses_pre_tokenised_closed_vocabulary(self):
        bundle = type(
            "Bundle",
            (),
            {
                "labels": tuple(["a", "b", "c"] + [f"x{index}" for index in range(27)]),
                "model_id": "test",
                "version": "v1",
                "checkpoint_sha256": "a" * 64,
                "vocabulary_sha256": "b" * 64,
            },
        )()
        runner = ModelRunner(_FakeModel(), bundle, torch.device("cpu"), torch.float32)
        source = {"body": torch.zeros((1, 1, 9, 3)), "name_batch": ["test"]}
        prediction = runner.predict(source, top_k=3)
        self.assertEqual(prediction.best.label, "b")
        self.assertGreater(prediction.score_margin_to_second, 0)


class _StreamingVideoProcessor:
    def probe(self, _path):
        return VideoInfo(1.0, 64, 48, 2.0)

    def decode(self, _path, _info):
        yield np.zeros((48, 64, 3), dtype=np.uint8)
        yield np.zeros((48, 64, 3), dtype=np.uint8)


class _StreamingPoseRunner:
    def extract(self, frames):
        self.consumed_frames = sum(1 for _ in frames)
        return PoseSequence(make_pose(2), PoseQuality(2, 2, 0, 1.0))


class _ServiceModelRunner:
    labels = tuple(["Ban ngày"] + [f"label-{index}" for index in range(1, 30)])

    def metadata(self):
        return {"id": "test", "version": "v1", "label_count": 30, "max_frames": 64}

    def predict(self, _source, top_k):
        candidates = tuple(
            CandidatePrediction(label, index + 1, 0.8 / (index + 1), -1.0 - index)
            for index, label in enumerate(self.labels[:top_k])
        )
        return ModelPrediction(candidates, 0.5)


class ServiceOrchestrationTests(unittest.TestCase):
    def test_service_streams_decoded_frames_into_pose_runner(self):
        pose_runner = _StreamingPoseRunner()
        service = InferenceService(
            ServiceSettings(),
            _ServiceModelRunner(),
            pose_runner,
            _StreamingVideoProcessor(),
        )
        result = service.predict_file(Path("unused-by-fake.mp4"), top_k=2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(pose_runner.consumed_frames, 2)
        self.assertEqual(result["input"]["pose_frames"], 2)
        self.assertEqual(len(result["top_k"]), 2)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class VideoProcessorTests(unittest.TestCase):
    def test_ffmpeg_probe_and_bounded_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "sample.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "lavfi", "-i", "color=c=blue:s=64x48:r=8:d=1",
                    "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
            )
            processor = VideoProcessor(ServiceSettings(max_duration_seconds=2, max_decoded_frames=3, max_pixels=64 * 48))
            info, frames = processor.decode_all(video)
            self.assertEqual((info.width, info.height), (64, 48))
            self.assertGreaterEqual(len(frames), 1)
            self.assertLessEqual(len(frames), 3)
            self.assertEqual(frames[0].shape, (48, 64, 3))

    def test_rotation_metadata_is_normalised(self):
        self.assertEqual(_rotation_degrees({"tags": {"rotate": "90"}}), 90)
        self.assertEqual(_rotation_degrees({"side_data_list": [{"rotation": -90}]}), 270)


if __name__ == "__main__":
    unittest.main()
