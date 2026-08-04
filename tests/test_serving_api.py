from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from serving.api import create_app
from serving.errors import ServiceError
from serving.settings import ServiceSettings


class FakeInferenceService:
    ready = True

    def __init__(self) -> None:
        self.seen_path: Path | None = None

    def model_metadata(self):
        return {
            "id": "cosign-vi-islr",
            "version": "test-r1",
            "checkpoint_sha256": "a" * 64,
            "vocabulary_sha256": "b" * 64,
            "label_count": 30,
            "max_frames": 64,
        }

    def labels(self):
        return ["Ban ngày"] + [f"label-{index}" for index in range(1, 30)]

    def predict_file(self, path: Path, top_k: int):
        self.seen_path = path
        assert path.exists()
        if path.read_bytes() == b"raise":
            raise ServiceError("NO_PERSON_DETECTED", "No signer was detected", 422, retryable=True)
        return {
            "status": "ok",
            "retryable": False,
            "model": self.model_metadata(),
            "prediction": {
                "label": "Ban ngày",
                "rank": 1,
                "relative_score": 0.8,
                "log_likelihood": -1.2,
                "score_margin_to_second": 0.4,
            },
            "top_k": [
                {
                    "label": "Ban ngày",
                    "rank": 1,
                    "relative_score": 0.8,
                    "log_likelihood": -1.2,
                }
            ][:top_k],
            "input": {"duration_ms": 1000, "decoded_frames": 10, "pose_frames": 10},
            "timing_ms": {"decode": 1, "pose": 2, "preprocess": 1, "model": 3, "total": 7},
        }


class ApiTests(unittest.TestCase):
    def make_client(self, service_api_key: str = ""):
        self.temporary = tempfile.TemporaryDirectory()
        service = FakeInferenceService()
        settings = ServiceSettings(
            temp_dir=Path(self.temporary.name),
            service_api_key=service_api_key,
            max_upload_bytes=32,
        )
        return TestClient(create_app(settings, service)), service

    def tearDown(self):
        temporary = getattr(self, "temporary", None)
        if temporary:
            temporary.cleanup()

    def test_health_model_and_labels_contract(self):
        client, _ = self.make_client()
        with client:
            self.assertEqual(client.get("/livez").json(), {"status": "live"})
            ready = client.get("/readyz")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["model"]["label_count"], 30)
            self.assertEqual(len(client.get("/v1/labels").json()["labels"]), 30)

    def test_prediction_upload_has_stable_response_and_cleans_temp_file(self):
        client, service = self.make_client()
        with client:
            response = client.post(
                "/v1/predictions",
                headers={"X-Request-ID": "request-123"},
                files={"video": ("capture.webm", b"video-bytes", "video/webm")},
                data={"top_k": "1", "client_capture_id": "browser-1"},
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["request_id"], "request-123")
        self.assertEqual(payload["prediction"]["label"], "Ban ngày")
        self.assertEqual(payload["client_capture_id"], "browser-1")
        self.assertEqual(response.headers["X-Request-ID"], "request-123")
        self.assertIsNotNone(service.seen_path)
        self.assertFalse(service.seen_path.exists())

    def test_api_rejects_invalid_size_top_k_and_service_error(self):
        client, _ = self.make_client()
        with client:
            oversize = client.post(
                "/v1/predictions",
                files={"video": ("capture.webm", b"x" * 33, "video/webm")},
            )
            self.assertEqual(oversize.status_code, 413)
            self.assertEqual(oversize.json()["code"], "FILE_TOO_LARGE")
            bad_top_k = client.post(
                "/v1/predictions",
                files={"video": ("capture.webm", b"ok", "video/webm")},
                data={"top_k": "6"},
            )
            self.assertEqual(bad_top_k.status_code, 400)
            self.assertEqual(bad_top_k.json()["code"], "INVALID_TOP_K")
            no_person = client.post(
                "/v1/predictions",
                files={"video": ("capture.webm", b"raise", "video/webm")},
            )
            self.assertEqual(no_person.status_code, 422)
            self.assertEqual(no_person.json()["code"], "NO_PERSON_DETECTED")
            self.assertTrue(no_person.json()["retryable"])
            missing_video = client.post("/v1/predictions", data={"top_k": "5"})
            self.assertEqual(missing_video.status_code, 400)
            self.assertEqual(missing_video.json()["code"], "INVALID_MULTIPART")

    def test_private_api_key_is_enforced(self):
        client, _ = self.make_client(service_api_key="secret")
        with client:
            denied = client.get("/v1/model")
            self.assertEqual(denied.status_code, 401)
            allowed = client.get("/v1/model", headers={"Authorization": "Bearer secret"})
            self.assertEqual(allowed.status_code, 200)

    def test_not_ready_service_keeps_liveness_but_returns_503_readiness(self):
        settings = ServiceSettings(model_bundle_dir=None, temp_dir=Path(tempfile.gettempdir()))
        with TestClient(create_app(settings)) as client:
            self.assertEqual(client.get("/livez").status_code, 200)
            response = client.get("/readyz")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["code"], "MODEL_NOT_READY")


if __name__ == "__main__":
    unittest.main()
