# Running the CoSign Inference Service

This is the implementation guide for the deployment design in [the service strategy](COSIGN_INFERENCE_SERVICE_DEPLOYMENT_STRATEGY.md). The API recognises one isolated Vietnamese sign per uploaded clip from the fixed 30-label CoSign vocabulary. It is not a continuous-sign or sentence-translation API.

## What was implemented

- `serving/`: FastAPI application, strict immutable model-bundle validation, FFmpeg decoding, RTMPose primary-person tracking, training-compatible preprocessing, closed-set model scoring, quality decisions, request IDs, bounded admission, and metrics.
- `pose_preprocessing.py`: shared 133-keypoint preprocessing used by both `datasets.py` training data loading and the service.
- `docker/`: CUDA/PyTorch serving image, non-root entrypoint, private Compose deployment reference, and example environment file.
- `script/create_model_manifest.py`: checksums a complete model release before the service is allowed to load it.

The service loads its model, mT5 assets, and RTMPose models once at startup. It never accesses a user's webcam directly: the browser captures a clip and the application backend sends it to `POST /v1/predictions` over a private network.

## Swagger UI and frontend contract testing

OpenAPI and Swagger UI are available at `/openapi.json` and `/docs`. Protected endpoints use the **ServiceBearer** scheme. In Swagger's **Authorize** dialog, paste the API key itself (for example `swagger-dev-key`); Swagger UI adds `Bearer ` automatically.

The real service correctly returns `503 MODEL_NOT_READY` until a verified GPU model release is mounted. For frontend work before those artefacts are available, use the explicit demo mode:

```bash
cd /home/haipd/Uni-Sign
SERVICE_API_KEY=swagger-dev-key \
  docker compose -f docker/docker-compose.swagger.yml up --build
```

Open `http://127.0.0.1:56568/docs`, click **Authorize**, enter `swagger-dev-key`, then select any non-empty file for `POST /v1/predictions`. Demo responses contain both `demo: true` and `model.is_demo: true`; they skip video, pose, and model validation and must never be treated as recognition results. Their purpose is frontend testing of upload handling, authorization, response parsing, and errors.

For a local Python-only Swagger session:

```bash
cd /home/haipd/Uni-Sign
DEMO_MODE=true SERVICE_API_KEY=swagger-dev-key \
  uvicorn serving.api:app --host 127.0.0.1 --port 8080
```

The Swagger demo compose file does not require a GPU, model bundle, mT5 assets, or RTMPose ONNX files.

## 1. Prepare an immutable model release

Do not copy model data into Git or the Docker build context. On the deployment server, create one release directory with this exact layout:

```text
/srv/unisign/models/
└── cosign-vi-islr/
    └── 2026-08-04-r1/
        ├── best_checkpoint.pth
        ├── labels.json
        ├── mt5-base/
        │   └── ... complete local pretrained_weight/mt5-base contents ...
        └── pose-models/
            ├── detector.onnx
            └── estimator.onnx
```

Copy these trusted assets into it:

- `best_checkpoint.pth`: the selected CoSign fine-tuning checkpoint. The current training report references `out/cosign_pose_islr_seed42/best_checkpoint.pth`; it is not present in this source checkout, so it must be supplied as a release artefact.
- `labels.json`: `data/CoSign/metadata/labels.json` from the same model/data release.
- `mt5-base/`: all files from `pretrained_weight/mt5-base/`.
- `detector.onnx` and `estimator.onnx`: the exact local RTMPose Wholebody detector and pose ONNX files to use in production. They must be compatible with RTMLib's `Wholebody(..., to_openpose=False)` and produce a 133-keypoint layout. Do not rely on RTMLib's automatic download in a deployed container.

Create and verify the manifest after all five artefacts are in place:

```bash
cd /home/haipd/Uni-Sign
python script/create_model_manifest.py \
  --bundle-dir /srv/unisign/models/cosign-vi-islr/2026-08-04-r1 \
  --pose-mode performance \
  --version 2026-08-04-r1
```

The command requires exactly 30 unique labels and writes `manifest.json` with a SHA-256 for every artefact. `POSE_MODE` at runtime must match the mode recorded in this manifest because it controls RTMPose detector/estimator input sizes. Any later modification causes startup readiness to fail. To replace a manifest deliberately after a complete release update, use `--overwrite` and assign a new versioned directory rather than mutating a running release.

## 2. Build and start the container

Docker daemon/GPU execution is not authorised in this development environment, so these commands are for the deployment server after Docker Engine and NVIDIA Container Toolkit are available.

```bash
cd /home/haipd/Uni-Sign
cp docker/.env.example docker/.env
# Edit docker/.env: set SERVICE_API_KEY and MODEL_RELEASES_DIR.

docker compose --env-file docker/.env -f docker/docker-compose.yml build
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
```

The Compose file deliberately binds the service only to `127.0.0.1:56568`, mounts `/models` read-only, uses a tmpfs for transient uploads, requires a service API key, and starts one ASGI worker. Put the application backend on the same private Docker network/host or expose the API through a private TLS/mTLS gateway; do not publish the service port directly to the internet.

Before sending traffic, verify liveness and readiness from the server:

```bash
curl --fail-with-body http://127.0.0.1:56568/livez
curl --fail-with-body http://127.0.0.1:56568/readyz
curl --fail-with-body \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  http://127.0.0.1:56568/v1/model
```

`/livez` means the HTTP process exists. `/readyz` returns `503` until the manifest hashes, model checkpoint, local mT5 assets, RTMPose assets, CUDA runtime, and model initialisation have succeeded. Route traffic only after readiness returns `200`.

## 3. Backend integration

The application frontend should use `getUserMedia`/`MediaRecorder` to capture a 3–6 second MP4 or WebM clip containing one signer and one complete sign. The frontend sends it to the **application backend**, and only the backend calls the private inference service.

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $SERVICE_API_KEY" \
  -H "X-Request-ID: 9bfcbf42-8243-4a62-8b87-8329d5ae0aa7" \
  -F "video=@capture.webm;type=video/webm" \
  -F "top_k=5" \
  -F "client_capture_id=web-7a4e" \
  http://127.0.0.1:56568/v1/predictions
```

The backend must keep the returned `X-Request-ID`, map stable error codes to user-facing Vietnamese copy, and avoid retry loops on `422` input errors. The API has these routes:

| Route | Authentication | Purpose |
| --- | --- | --- |
| `GET /livez` | none | Process liveness for container orchestration. |
| `GET /readyz` | none | Model readiness; returns `503` before a verified model is loaded. |
| `GET /v1/model` | service bearer token | Active release metadata and hashes. |
| `GET /v1/labels` | service bearer token | Canonical 30-label vocabulary. |
| `POST /v1/predictions` | service bearer token | Multipart clip prediction. |
| `GET /metrics` | service bearer token | Prometheus metrics for the private monitoring plane. |

For a successful result, `prediction.relative_score` is a softmax over only the 30 candidate labels. It is a relative ranking score, **not a calibrated probability**. Display Top-5 and retry guidance. The existing signer-independent test report is 34.48% Top-1 / 64.50% Top-5, so the product must not make high-stakes automatic decisions from Top-1 alone.

## 4. Runtime policy and environment variables

| Variable | Default | Notes |
| --- | --- | --- |
| `MODEL_BUNDLE_DIR` | required | One immutable release directory with `manifest.json`. |
| `DEVICE` | `cuda` | The production Compose configuration requires a CUDA GPU. CPU use is only for development/contract tests. |
| `MODEL_DTYPE` | `fp32` | `fp16` and `bf16` exist but must be promoted only after numerical regression testing on the target GPU. |
| `POSE_BACKEND` / `POSE_MODE` | `onnxruntime` / `performance` | Must match the local pose assets included in the release. |
| `MAX_UPLOAD_BYTES` | `20971520` | Multipart upload limit. |
| `MAX_DURATION_SECONDS` | `8` | Hard maximum decoded video duration. |
| `MAX_DECODED_FRAMES` | `192` | Upper bound before uniform FFmpeg frame sampling. Model preprocessing then uses at most 64 frames. |
| `MAX_PIXELS` | `2073600` | Maximum source image pixels (1920×1080 by default). |
| `MAX_QUEUE_DEPTH` | `1` | One in-flight GPU request initially; tune only after load testing. |
| `SERVICE_API_KEY` | required by Compose | Private backend-to-service credential. Use a secret manager in production. |
| `DEMO_MODE` | `false` | Enables deterministic Swagger/frontend contract responses without loading model artefacts. Never enable in production. |
| `MIN_PERSON_COVERAGE` / `MIN_MEAN_HAND_SCORE` | `0.70` / `0.20` | Pose quality gates. Validate and version final values on deployment-like data. |
| `MIN_RELATIVE_SCORE` / `MIN_SCORE_MARGIN` | disabled | Do not enable until calibrated on a labelled validation set. |

The service refuses client paths, remote URLs, and pickles. It spools clips to a restricted temp file, probes/decode them using FFmpeg, deletes the temporary file in all normal/error paths, and returns stable errors such as `FILE_TOO_LARGE`, `VIDEO_TOO_LONG`, `NO_PERSON_DETECTED`, `GPU_QUEUE_FULL`, and `MODEL_NOT_READY`. A malformed multipart request now consistently returns `INVALID_MULTIPART`; clients must let their HTTP library set the multipart boundary rather than manually composing the `Content-Type` header.

## 5. Mandatory release checks

Run these checks for each service/model release:

```bash
cd /home/haipd/Uni-Sign
python -m unittest tests.test_serving_core tests.test_serving_api -v
python -m py_compile pose_preprocessing.py models.py datasets.py serving/*.py script/create_model_manifest.py

# On a Docker/NVIDIA host:
docker compose --env-file docker/.env -f docker/docker-compose.yml build
docker compose --env-file docker/.env -f docker/docker-compose.yml up -d
curl --fail-with-body http://127.0.0.1:8080/readyz
# Then submit a trusted golden capture and compare Top-1, Top-5 order, and
# tolerance-bounded scores against the model release baseline.
```

The unit/API tests do not replace a full GPU end-to-end check. The model binary, final fine-tuned checkpoint, and local RTMPose ONNX artefacts are deliberately absent from the repository/Docker context, so final readiness and golden-clip tests must run on the deployment server with the mounted release bundle.

## 6. Operational constraints

- Do not configure multiple Uvicorn workers for one GPU: each worker would load another 587M-parameter model copy. Start with one worker/one GPU and bounded queue admission.
- Do not make one inference call per webcam frame. The endpoint expects an isolated-sign clip and does not detect sign boundaries.
- Keep the service private. User authentication, user-facing rate limits, and retention/consent policy belong to the application backend.
- Do not persist video, frames, or keypoints by default. A feedback dataset must be a separate explicit opt-in workflow with controlled storage and deletion.
- Treat model/image release as a pair. Retain the previous verified image digest and read-only model bundle for rollback.
