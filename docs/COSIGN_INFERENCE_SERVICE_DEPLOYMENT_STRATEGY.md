# CoSign Vietnamese ISLR Inference Service: API and Docker Deployment Strategy

> Implementation status: the clip-based API, inference core, model-bundle validation, tests, and Docker/Compose files described by this plan are implemented. See [INFERENCE_SERVICE.md](INFERENCE_SERVICE.md) for the executable deployment and backend integration guide.

## 1. Purpose, scope, and recommended first release

This document defines the production integration plan for the fine-tuned **Uni-Sign CoSign 30-class Vietnamese isolated-sign recognition (ISLR)** model.  It covers the service boundary, API contract, webcam capture flow, inference pipeline, Docker deployment, operations, and the implementation order.

The initial release must recognise **one trimmed, isolated Vietnamese sign per request**.  It must not be presented as continuous sign-language translation or sentence recognition: the training data and the current checkpoint were trained and evaluated on one word/sign per clip.

The recommended first production release is a synchronous, private HTTP inference service:

1. The web/mobile client captures a short clip from its own webcam.
2. The application backend authenticates the user and forwards the clip to the private model service.
3. The model service extracts a single person's 133 whole-body RTMPose sequence, samples it to the trained 64-frame representation, scores all 30 allowed labels, and returns Top-1 and Top-5 results.
4. The application backend returns the result to the client and applies product policy (display, retry advice, audit, feedback collection).

This deliberately avoids a WebSocket/continuous-stream implementation in the first release.  A clip-based API matches the data, creates a clear sign boundary, is much simpler to operate, and is the right baseline against which a later real-time stream must be measured.

## 2. Evidence from this repository

| Item | Deployment implication |
| --- | --- |
| The CoSign task is closed-vocabulary ISLR with 30 Unicode NFC-normalised Vietnamese labels. | Every prediction must be chosen from `data/CoSign/metadata/labels.json`; do not use free-text generation in the service. |
| The reported held-out signer-independent test result is 34.48% Top-1 and 64.50% Top-5. | Return Top-5, uncertainty signals, and a retry recommendation. A high relative score must not be described as a calibrated probability or a guarantee of correctness. |
| The reported fine-tuned checkpoint is `out/cosign_pose_islr_seed42/best_checkpoint.pth` and is approximately 1.17 GB. | Keep model artefacts out of Git and mount/download them as an immutable, versioned release bundle. The service needs a CUDA-capable GPU for practical latency. |
| `models.Uni_Sign` uses pose-only ST-GCN encoders followed by mT5. The CoSign prompt language is Vietnamese. | Instantiate with `dataset=CoSign`, `task=ISLR`, `language=Vietnamese`, `rgb_support=False`, `max_length=64`, and strict checkpoint loading. |
| `score_candidate_labels` scores the 30 labels by length-normalised mT5 conditional log-likelihood. | This is the production decoder. Softmaxing those 30 scores produces a **relative closed-set score**, not a calibrated confidence. |
| Training poses are RTMPose whole-body arrays: one person per frame, `keypoints: (1,133,2)` and `scores: (1,133)`. | The server must select and track exactly one signer before calling the existing preprocessing logic. Passing zero or multiple people into the current path would break the expected shape or silently use an arbitrary person. |
| `script/infer_video.py` already demonstrates video/pose inference. | Treat it as a developer CLI and regression oracle, not the production server. It loads model/pose components per invocation, retains all frames in memory, has no request validation, and imports training-oriented dependencies. |
| `demo/online_inference.py` is an older generic online demo using free generation. | Do not wrap this file in an API. It does not enforce the CoSign closed vocabulary and is unsuitable as the service core. |
| `rtmlib.Wholebody` obtains detector/pose model files automatically when not already cached. | Production startup must never rely on external downloads. Pin, checksum, and package/mount the exact ONNX pose assets used by the serving image. |

The dataset reports contain two count views: 3,054 canonical videos in the class table and 3,028 videos in the split summary.  The service does not need training videos, but the model release manifest must record the exact final data/split revision used for a checkpoint so this discrepancy is traceable.

## 3. Architecture and trust boundary

An end user's camera is attached to their browser/device, not to the remote Docker container.  Therefore the browser is responsible for requesting camera permission and producing a short video clip; the server is responsible for safe decoding, pose extraction, and model inference.

```text
Browser or mobile client
  getUserMedia + MediaRecorder (one sign, start/stop)
            |
            | HTTPS; authenticated application request
            v
Application backend / API gateway
  authentication, authorisation, product audit, rate limits
            |
            | private network + service credential + request ID
            v
Uni-Sign inference container (one GPU worker per loaded model)
  validate -> transcode/decode -> person tracking -> pose -> preprocess
  -> closed-set scoring -> quality checks -> JSON result
            |
            v
Application backend -> client result / retry guidance
```

The model service is an internal compute service, not the public authentication authority.  The recommended network rule is that only the application backend/API gateway can call it.  A browser must not receive a long-lived model-service credential.

For a small deployment, the backend and service can run on the same Docker host and communicate through an internal Docker network.  For a larger deployment, place the service behind an internal load balancer and use mTLS or a short-lived service token.  The API schema below remains unchanged in both arrangements.

## 4. Product and capture contract

### 4.1 Client capture requirements

The client UX should guide the signer before recording:

- One person only, facing the camera; keep torso, arms, and both hands in view.
- A plain enough background, stable device, and adequate front lighting.
- Record one complete sign plus a short still interval before and after it.
- Start/stop manually for the first release. The model has no learned sign-boundary detector.
- Capture a 3–6 second clip at a practical resolution such as 480p/720p and a controlled frame rate. Browser output will often be WebM; Safari clients may produce MP4. Both must be covered by acceptance tests.

The client should send the original clip as binary multipart data. It must not send a webcam device ID, local filesystem path, a base64 JSON field, a public URL, or a Python pickle file.

### 4.2 Prediction semantics

The service must distinguish these outcomes:

| Outcome | Meaning | Client action |
| --- | --- | --- |
| `ok` | A closed-set label was selected and basic video/pose quality checks passed. | Show Top-1 plus optionally Top-5. |
| `low_quality` | There was a usable clip but the signer, hands, or pose coverage did not meet service policy. | Ask the user to re-record; do not treat as a recognised word. |
| `low_confidence` | The input was usable but the score/margin falls below thresholds calibrated on validation data. | Show a retry and optionally the candidates. |
| Request error | Unsupported/corrupt/oversize input or invalid request. | Correct the client request; use the returned problem code. |
| Service unavailable | Model is not ready or the GPU queue is saturated. | Retry according to `Retry-After`; do not retry indefinitely. |

No `unknown sign` threshold should be hard-coded from intuition.  The model was trained only on 30 known labels, so an out-of-vocabulary sign still receives one of those labels.  Select low-confidence and low-quality thresholds from a labelled validation set that includes difficult recordings and, if possible, out-of-vocabulary examples. Version those thresholds with the model.

## 5. API contract (version 1)

Use JSON responses, RFC 7807-style error bodies, ISO 8601 UTC timestamps, and an `X-Request-ID` header.  The backend may provide a request ID; otherwise the service creates one and returns it.  The service should publish its OpenAPI schema at `/openapi.json` and documentation only on the private/admin network.

### 5.1 Required endpoints

| Method and path | Caller | Purpose |
| --- | --- | --- |
| `GET /livez` | Orchestrator/load balancer | Process is alive. It must not require a GPU inference. |
| `GET /readyz` | Orchestrator/load balancer | Model, vocabulary, pose models, and GPU are loaded; return `503` until then. |
| `GET /v1/model` | Application backend/admin | Return the active model and contract metadata, without exposing filesystem paths. |
| `GET /v1/labels` | Application backend | Return the canonical 30-label vocabulary and its version/hash. |
| `POST /v1/predictions` | Application backend | Submit one video clip and synchronously receive the prediction. |
| `GET /metrics` | Private Prometheus scraper | Prometheus metrics. Never expose this endpoint publicly. |

There is intentionally no arbitrary `video_url`, `path`, `pose_path`, or `checkpoint_path` parameter. These would permit SSRF, local-file access, or unsafe pickle loading.

### 5.2 `POST /v1/predictions`

The initial transport is `multipart/form-data`:

| Field | Type | Required | Rule |
| --- | --- | --- | --- |
| `video` | binary file | Yes | Accept only explicitly supported media containers/codecs after content sniffing and decode validation. Do not trust a filename or MIME type alone. |
| `top_k` | integer | No | Default `5`; constrain to `1..5`. |
| `client_capture_id` | string | No | An opaque, non-sensitive ID for client/backend correlation; validate length and characters. |
| `client_duration_ms` | integer | No | Observability only; server-decoded duration is authoritative. |

The gateway and service both enforce a request-size limit.  Start with a conservative configurable limit such as 20 MiB, plus configurable duration, decoded-frame, dimensions, and pixel-count limits.  Choose final limits from the capture UX and load testing; reject before expensive GPU work where possible.

Example request from the application backend:

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $MODEL_SERVICE_TOKEN" \
  -H "X-Request-ID: 9bfcbf42-8243-4a62-8b87-8329d5ae0aa7" \
  -F "video=@capture.webm;type=video/webm" \
  -F "top_k=5" \
  -F "client_capture_id=web-7a4e" \
  http://unisign-inference:8080/v1/predictions
```

Example successful response:

```json
{
  "request_id": "9bfcbf42-8243-4a62-8b87-8329d5ae0aa7",
  "status": "ok",
  "model": {
    "id": "cosign-vi-islr",
    "version": "2026-08-04-r1",
    "checkpoint_sha256": "<immutable-checkpoint-sha256>",
    "vocabulary_sha256": "<immutable-vocabulary-sha256>",
    "label_count": 30,
    "max_frames": 64
  },
  "prediction": {
    "label": "Ban ngày",
    "rank": 1,
    "relative_score": 0.62,
    "log_likelihood": -1.847,
    "score_margin_to_second": 0.31
  },
  "top_k": [
    {
      "label": "Ban ngày",
      "rank": 1,
      "relative_score": 0.62,
      "log_likelihood": -1.847
    },
    {
      "label": "Hôm nay",
      "rank": 2,
      "relative_score": 0.31,
      "log_likelihood": -2.157
    }
  ],
  "input": {
    "duration_ms": 3810,
    "decoded_frames": 114,
    "pose_frames": 64,
    "primary_person_coverage": 1.0,
    "mean_hand_keypoint_score": 0.79
  },
  "timing_ms": {
    "decode": 42,
    "pose": 302,
    "preprocess": 3,
    "model": 118,
    "total": 482
  },
  "created_at": "2026-08-04T08:00:00Z"
}
```

`relative_score` is the softmax over the 30 candidate log-likelihoods. It is included only as a ranking aid. It must be named and documented this way until calibration (for example, temperature scaling and validation reliability measurements) supports exposing a probability-like confidence.

Example low-quality response is still HTTP `200` because the media was processed successfully:

```json
{
  "request_id": "9bfcbf42-8243-4a62-8b87-8329d5ae0aa7",
  "status": "low_quality",
  "reason_codes": ["MULTIPLE_PEOPLE", "LOW_HAND_VISIBILITY"],
  "retryable": true,
  "guidance": "Record one signer with both hands and upper body visible.",
  "model": {"id": "cosign-vi-islr", "version": "2026-08-04-r1"}
}
```

### 5.3 Error contract

Return a stable machine-readable `code` in all errors. The backend should map the code to its own localised user message rather than displaying internal detail.

| HTTP | Code examples | Meaning |
| --- | --- | --- |
| `400` | `INVALID_MULTIPART`, `INVALID_TOP_K` | Request syntax or fields are invalid. |
| `413` | `FILE_TOO_LARGE`, `VIDEO_TOO_LONG`, `PIXEL_LIMIT_EXCEEDED` | Configured safety limit exceeded. |
| `415` | `UNSUPPORTED_MEDIA_TYPE`, `UNSUPPORTED_CODEC` | The uploaded file is not a supported decodable video. |
| `422` | `NO_DECODABLE_FRAMES`, `NO_PERSON_DETECTED`, `POSE_FORMAT_INVALID` | The request was valid but cannot yield a model input. |
| `429` | `RATE_LIMITED`, `GPU_QUEUE_FULL` | Back off; include `Retry-After` where a retry is useful. |
| `503` | `MODEL_NOT_READY`, `MODEL_UNAVAILABLE` | Model bundle/GPU is not usable. |
| `500` | `INFERENCE_FAILED` | Unexpected server error; log correlation ID, not media or model internals. |

Example error body:

```json
{
  "type": "https://api.example.internal/problems/no-person-detected",
  "title": "No signer was detected",
  "status": 422,
  "code": "NO_PERSON_DETECTED",
  "request_id": "9bfcbf42-8243-4a62-8b87-8329d5ae0aa7",
  "retryable": true
}
```

### 5.4 Future streaming API, not part of the initial release

Do not call the clip endpoint once per webcam frame. It would repeatedly run detection and the 30-label decoder, cause GPU contention, and produce unstable labels before a sign has completed.

Only after the clip endpoint meets its accuracy, latency, and load targets should a versioned WebSocket or WebRTC-data-channel session API be introduced. Such a session must buffer frames, track the primary signer, perform explicit start/end-of-sign detection, emit provisional results with `is_final: false`, and make the final result semantically equivalent to `POST /v1/predictions`. It needs its own session limits, backpressure, cancel operation, and end-to-end evaluation; it is not a transport-only change.

## 6. Inference pipeline design

### 6.1 Request lifecycle

1. **Authenticate and admit.** The gateway authenticates the user; the model service verifies the internal service credential, request size, rate/queue capacity, and request ID.
2. **Spool safely.** Stream upload to a unique, permissions-restricted temporary file. Do not load the whole multipart body into memory. Delete it in `finally`, including error paths.
3. **Probe and decode.** Use `ffprobe`/`ffmpeg` or an equivalently robust pinned decoder to verify container/codec, duration, resolution, and actual frames. Normalise decode orientation. OpenCV alone is not reliable for all browser-produced WebM/H.264 variants.
4. **Bound work while retaining time coverage.** Enforce configured decoded-frame/pixel limits. For clips above the pose-frame work budget, select source frames uniformly over the entire clip rather than taking only the first N frames. Record any resampling in metrics.
5. **Detect, choose, and track one signer.** Run the pinned RTMPose whole-body detector/pose estimator in `to_openpose=False` mode. Select a primary person deterministically (initial detection: largest plausible centred person; later frames: highest overlap/pose similarity to the previous primary person). Do not allow detector output order to choose the signer. Reject or return `low_quality` if no stable primary person exists, if another person is consistently competing, or coverage falls below policy.
6. **Validate and normalise pose.** Convert the tracked person on each retained frame to exactly `keypoints (1,133,2)` normalised by that frame's width/height and `scores (1,133)`. Reject NaN/Inf and impossible shapes. This is identical to the dataset contract documented in `datasets.load_part_kp`.
7. **Apply training-compatible preprocessing.** Uniformly select at most 64 valid pose frames, then use the repository's body/left/right/face partitioning and normalisation (`load_part_kp` / `crop_scale`, including the 0.3 keypoint threshold). Pad only using the same semantics as the training collator. Record pose-quality figures before zeroing weak joints.
8. **Encode and score the closed set.** Run the Uni-Sign pose encoder once in `eval()` and `torch.inference_mode()`. Score the full canonical vocabulary with `score_candidate_labels`; sort descendent score and retain the requested Top-k.
9. **Apply calibrated decision policy.** Combine video/pose validation, Top-1 relative score, and Top-1 minus Top-2 margin. Mark a result `low_confidence` only with versioned thresholds calibrated offline. Always return model/vocabulary versions.
10. **Respond and clean up.** Emit structured timings and quality metadata, no raw video/frames/keypoints. Remove temporary media immediately unless the authenticated user explicitly opted into a separately governed feedback workflow.

### 6.2 Required refactor before exposing an API

The production API should not import or execute the CLI script. Create a small serving package and move reusable inference code into testable components:

```text
serving/
├── api.py                 # FastAPI application, routes, exception mapping
├── settings.py            # validated environment configuration
├── schemas.py             # Pydantic request/response/problem schemas
├── service.py             # orchestration, admission control, response assembly
├── model_runner.py        # one-time model/vocabulary load and closed-set scoring
├── pose_runner.py         # one-time RTMPose setup, primary-person tracking
├── video.py               # safe probe/decode/frame sampling/temp-file cleanup
├── quality.py             # versioned quality and confidence decision policy
└── telemetry.py           # logs, metrics, trace/request correlation

tests/
├── unit/                  # preprocessing, person choice, schemas, decision policy
├── api/                   # HTTP contract and failure cases with test doubles
├── integration/           # checkpoint + pose + model regression clips
└── docker/                # image health/readiness smoke tests

docker/
├── Dockerfile
├── docker-compose.yml     # local/server reference deployment
├── entrypoint.sh
└── .dockerignore

requirements-serving.txt   # only runtime dependencies, fully pinned
```

Specific code changes required during that refactor:

- Extract a model method that constructs `inputs_embeds` and `attention_mask` without calculating the dummy-label cross-entropy loss. `script/infer_video.py` currently calls the full training `forward` once and then calls `score_candidate_labels`; the service should encode once and avoid the unnecessary loss decoder pass.
- Pre-tokenise the immutable 30-label vocabulary once at service startup. Retain the exact current scoring semantics while avoiding tokenisation on every request. Any further batching/decoder optimisation must be validated against the baseline scores.
- Make lightweight shared utilities independent of `utils.py`, which imports DeepSpeed at module import time. Serving should not need training-only DeepSpeed. Likewise, avoid importing `datasets.py` solely for pose preprocessing if that forces `decord` to be installed.
- Load `Uni_Sign`, the checkpoint, the mT5 local directory, vocabulary, and RTMPose objects once in the FastAPI lifespan startup hook. Use strict checkpoint loading and set the models to `eval()` before `/readyz` becomes healthy.
- Preserve `rgb_support=False`. The CoSign checkpoint and training report describe a pose-only model; enabling the RGB branch would change architecture/checkpoint compatibility and require a separate validated release.
- Never accept `.pkl` from clients. Python pickle is unsafe for untrusted input. A private, authenticated diagnostic pose endpoint could exist later only with a non-pickle, schema-validated tensor format; it is not needed for normal integration.

### 6.3 Concurrency and performance strategy

One 587.75M-parameter model and pose engine should be loaded **once per GPU**, not once per request. Start with one ASGI worker per GPU because multiple Uvicorn/Gunicorn workers duplicate GPU memory. Bound concurrent GPU work with a semaphore/queue; fail fast with `429 GPU_QUEUE_FULL` rather than allowing uncontrolled CUDA out-of-memory failures.

The first version should measure, not promise, latency. Capture these stage durations separately: queue wait, upload, probe/decode, pose, preprocess, model scoring, and total. GPU batching can be added later by a bounded micro-batcher that batches only compatible preprocessed pose tensors; it must preserve request ordering, per-request timeouts, and the numerical prediction regression test.

Run correctness and performance trials for FP32, BF16, and FP16 on the actual serving GPU. Adopt reduced precision only after comparing labels, margins, and quality outcomes to the validated FP32 baseline. The L40S configuration reported for training is evidence the model trains on that GPU, not an inference latency/service-capacity benchmark.

## 7. Model artefact and release management

The service should require a read-only release bundle rather than repository-relative implicit paths:

```text
/models/cosign-vi-islr/2026-08-04-r1/
├── manifest.json
├── best_checkpoint.pth
├── labels.json
├── mt5-base/               # complete local Hugging Face mT5 assets
└── pose-models/            # exact RTMPose/detector ONNX assets used at serving
```

`manifest.json` should include at least:

- `model_id`, semantic `version`, release timestamp, repository Git commit, and service API version;
- SHA-256 and byte size for every artefact;
- `checkpoint_format` (`model` key versus raw state dict) and `strict_load: true`;
- `dataset_name: CoSign`, task `ISLR`, language `Vietnamese`, `rgb_support: false`, `max_length: 64`, pose layout/version, and vocabulary hash;
- closed-set score/quality threshold policy version and calibration dataset revision;
- supported input format/limits and build image digest.

At startup, the service validates the manifest, verifies hashes, verifies there are exactly 30 non-empty canonical labels, checks the checkpoint can load strictly, and records the release version in every response and metric label. `/readyz` remains `503` if any check fails.

The current source tree's report references `best_checkpoint.pth`, but the model binary should be distributed through a secured model store or server volume, not committed to Git or baked into a normal source image. The same is true of the mT5 and RTMPose weights. This makes source builds small, supports a rollback by changing one immutable mounted release directory, and avoids a network dependency at container startup.

## 8. Docker strategy

### 8.1 Image design

Build a dedicated serving image rather than reusing the training environment. Training-only packages such as DeepSpeed, TensorBoard, and TensorFlow increase image size and attack surface without helping HTTP inference.

The implementation should use a CUDA/PyTorch base compatible with the repository's `torch==2.1.1+cu121` / `torchvision==0.16.1+cu121` model environment, for example a pinned PyTorch CUDA 12.1 runtime image. Verify the exact tag and NVIDIA driver compatibility during the image build; record both the base-image digest and `nvidia-smi` driver minimum in the release manifest.

The eventual `docker/Dockerfile` should follow this sequence:

1. Start from a digest-pinned CUDA/PyTorch runtime base.
2. Install only system runtime packages needed for safe media processing, principally `ffmpeg`, `libglib2.0-0`, and the required OpenCV graphics/runtime libraries. Do not install a compiler toolchain in the final image unless a pinned wheel genuinely needs it.
3. Copy and install a fully pinned `requirements-serving.txt`: FastAPI, Uvicorn, Pydantic, PyTorch/Torchvision compatibility pins, Transformers/SentencePiece, NumPy, OpenCV headless, ONNX Runtime GPU, Prometheus client, and the vendored/installed RTMLib package. Resolve and test wheel compatibility under the target Python/CUDA combination.
4. Copy the source, including the local RTMLib package and new `serving/` code. Do not copy `data/`, `out/`, `.git/`, local environments, test media, or checkpoint artefacts.
5. Create and run as a non-root `app` user. Create a writable temporary directory owned by that user; mount `/models` read-only.
6. Make pose-model assets available from `/models/.../pose-models` or package verified assets in the image. Disable automatic internet downloads in normal startup.
7. Set deterministic resource limits: one Uvicorn worker per GPU, conservative thread counts, a writable temp directory, unbuffered logs, and `TOKENIZERS_PARALLELISM=false`.
8. Expose only the internal service port and launch the ASGI app. The actual production command must not use `--reload`.

Conceptual final command (the precise module name is created during implementation):

```dockerfile
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
```

Use an OCI image tag containing the API and model-service release, then deploy it by immutable image digest. Scan it for vulnerabilities and create an SBOM in CI. Rebuild when PyTorch/CUDA, FFmpeg, ONNX Runtime, or base-image security updates require it; validate numerical output after every update.

### 8.2 Runtime configuration

All deployment settings should be environment variables or a mounted read-only configuration file, never hard-coded into routes:

| Setting | Example | Purpose |
| --- | --- | --- |
| `MODEL_BUNDLE_DIR` | `/models/cosign-vi-islr/2026-08-04-r1` | Immutable active artefact bundle. |
| `DEVICE` | `cuda` | Explicit startup contract; fail readiness if CUDA is required but absent. |
| `POSE_BACKEND` / `POSE_MODE` | `onnxruntime` / `performance` | Pinned pose inference behaviour. |
| `MAX_UPLOAD_BYTES` | `20971520` | Defence before decode. |
| `MAX_DURATION_SECONDS` | `8` | Bound request work to the capture contract. |
| `MAX_DECODED_FRAMES` | release-configured | Bound CPU/GPU pose work while preserving uniform temporal coverage. |
| `MAX_QUEUE_DEPTH` | release-configured | Prevent GPU overload. |
| `TOP_K_DEFAULT` | `5` | API response default. |
| `SERVICE_AUTH_*` | secret reference | Internal authentication/mTLS configuration; do not log values. |
| `TEMP_DIR` | `/tmp/unisign` | Per-request media staging path. |
| `LOG_LEVEL` | `INFO` | Structured operational logs; production must not log clips. |

The production server needs the NVIDIA Container Toolkit and an NVIDIA driver compatible with the selected CUDA image. Validate GPU availability with a Docker GPU smoke test before deploying this image. A CPU-only developer image may exist for contract tests, but should not be promoted as the production performance tier.

### 8.3 Reference Compose topology

The following is a topology target for `docker/docker-compose.yml`, not a substitute for secret management or an external TLS gateway:

```yaml
services:
  unisign-inference:
    image: registry.example.com/unisign-inference@sha256:<image-digest>
    restart: unless-stopped
    gpus: all
    environment:
      MODEL_BUNDLE_DIR: /models/cosign-vi-islr/2026-08-04-r1
      DEVICE: cuda
      MAX_UPLOAD_BYTES: "20971520"
      MAX_DURATION_SECONDS: "8"
      TOKENIZERS_PARALLELISM: "false"
    volumes:
      - /srv/unisign/models:/models:ro
      - type: tmpfs
        target: /tmp/unisign
        tmpfs:
          size: 256m
    ports:
      - "127.0.0.1:8080:8080"
    networks:
      - model-internal
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez')"]
      interval: 30s
      timeout: 5s
      retries: 3

networks:
  model-internal:
    internal: true
```

If the application backend is on a different host, do not expose port 8080 openly as in a public `0.0.0.0:8080` mapping. Use a private network/load balancer, TLS/mTLS, firewall rules, and a separate ingress/gateway. Terminate public TLS and authenticate users at the application gateway.

## 9. Security, privacy, and reliability controls

### 9.1 Security controls

- Authenticate at the application backend and require a separate internal service credential or mTLS identity at the inference API.
- Enforce request byte, duration, resolution, frame count, queue-depth, connection, and processing-time limits. Rate-limit before decoding.
- Content-sniff and decode video in a constrained path; never execute a user-provided filename, shell fragment, URL, archive, or pickle. Use argument arrays rather than shell interpolation for FFmpeg subprocesses.
- Run as non-root with a read-only root filesystem where practical, read-only model mounts, a small writable temp filesystem, and no Docker socket or host devices other than the assigned GPU.
- Restrict CORS to the application origin only if a browser is deliberately allowed to call the model service. The preferred design has no browser CORS access at all.
- Keep `/docs`, `/openapi.json`, `/metrics`, debug traces, and model metadata on private/admin networks as appropriate. Do not return stack traces to callers.
- Use dependency scanning, image signing/attestation, pinned hashes, and regular rebuilds. Patch media-decoder and ONNX Runtime vulnerabilities promptly.

### 9.2 Privacy controls

Webcam video can be biometric/personal data. Establish a product policy before public deployment:

- Process media ephemerally and delete temporary input, decoded frames, and extracted keypoints at request completion by default.
- Do not log raw media, keypoints, full request bodies, client IP addresses beyond the gateway's policy, or user identifiers in application logs.
- Return/use a generated request ID that is not personally identifying. Map it to a user only in the application system under that system's access controls.
- If feedback clips are needed for retraining, make upload storage opt-in, disclose purpose/retention/access, encrypt it, isolate it from serving, and support deletion. Never quietly retain all inference clips.

### 9.3 Reliability controls

- Liveness means the HTTP process runs; readiness means all checks described in Section 7 have completed and a warm-up has succeeded. Do not route traffic merely because the process started.
- Use graceful shutdown: stop accepting work, drain the bounded queue within a deadline, finish/cancel safely, then release CUDA resources.
- Keep one known-good previous model bundle mounted for immediate rollback. Roll back the image and the model bundle as a tested pair.
- Treat CUDA OOM, ONNX Runtime failures, corrupt media, and crashed decoder processes as controlled errors. Restart only the affected worker/container according to policy; never report a stale model as ready.

## 10. Observability and acceptance metrics

Every request produces one structured log record with: request ID, model/version/hash, status/code, request byte bucket, decoded/pose frame count, person/hand quality aggregates, queue/decode/pose/model/total durations, GPU error category, and retryable flag. Exclude raw video, raw frames, pose tensors, credential values, and labels of identifiable users unless business policy explicitly permits them.

Expose private Prometheus-compatible metrics such as:

- `unisign_http_requests_total{route,status,code}`;
- `unisign_request_duration_seconds{stage,model_version}`;
- `unisign_gpu_queue_depth` and `unisign_gpu_queue_rejections_total`;
- `unisign_pose_frames_total`, `unisign_pose_no_person_total`, `unisign_pose_multiple_person_total`, and `unisign_low_quality_total{reason}`;
- `unisign_predictions_total{status,model_version}` and score/margin histograms with carefully bounded cardinality;
- `unisign_model_ready{model_version}`, `unisign_model_load_failures_total`, and `unisign_artifact_hash_mismatch_total`.

Do not label metrics with request IDs, client IDs, or user IDs; that creates unbounded cardinality and can leak data.

Set latency, throughput, availability, and queue SLO values only after benchmarking with a representative production GPU and browser clips. Establish baseline distributions first; training throughput is not a serving SLO. Monitor input domain shift using anonymised aggregate pose-quality, duration, resolution, score/margin, and low-quality rates, then investigate changes with consented evaluation data.

## 11. Validation plan

### 11.1 Correctness tests

1. **Checkpoint and artefact validation:** strict load succeeds; manifest hashes, 30-label vocabulary, mT5 assets, and pinned pose assets are present.
2. **Preprocessing regression:** for a fixed set of saved trusted pose files, the serving preprocessor creates the same tensors and candidate ranking as the current `script/infer_video.py --pose` baseline.
3. **Video-to-pose regression:** use representative MP4, AVI, browser WebM, portrait-orientation, short, long, poor-light, and multi-person clips. Verify deterministic frame/person selection and expected quality status.
4. **Closed-set contract:** every success prediction and every Top-k label belongs to `labels.json`; Top-k is sorted and no duplicate appears.
5. **API contract tests:** test valid upload, missing field, invalid `top_k`, oversized/corrupt/unsupported media, no person, multi-person, queue full, unavailable model, request ID propagation, and cleanup after error.
6. **Security tests:** verify arbitrary paths/URLs/pickles are rejected, FFmpeg parameters are not shell-injected, model mount is read-only, and raw media is absent from logs/temp storage after completion.

### 11.2 Docker and deployment tests

1. Build the image without any dataset or checkpoint binary in the build context.
2. Run a CPU contract-test image and a GPU smoke-test image with `docker run --gpus all` on the target class of server.
3. Mount a release bundle read-only and verify `/livez` then `/readyz` transitions only after strict load/warm-up.
4. Submit a golden test clip and compare Top-1, Top-5 ordering, and tolerance-bounded scores to the release baseline.
5. Run bounded-concurrency/load tests to find queue/rejection behaviour before public traffic.
6. Exercise corrupted model artefacts, unavailable GPU, forced worker restart, disk-full temporary storage, and rollback to the previous release bundle.

### 11.3 Model-quality gate

Before a model version is promoted, evaluate it on the untouched signer-independent test set and a separately collected deployment-like validation set. Report per-class Top-1, Top-5, confusion matrix, capture/pose failure rate, and calibration/low-confidence behaviour. Because the current reported test Top-1 is 34.48%, release UX must be designed around assistance and retry, not automatic high-stakes decisions.

## 12. Implementation milestones and deliverables

| Phase | Deliverables | Exit criteria |
| --- | --- | --- |
| 0. Freeze the release inputs | Release bundle, hashes, model manifest, representative golden clips, final compatibility matrix. | A checkpoint can be strict-loaded with exactly the intended 30-label vocabulary and local mT5/pose assets. |
| 1. Refactor serving core | `serving/model_runner.py`, preprocessing/pose modules, tests that match the current CLI baseline. | Model/pose objects load once; closed-set prediction works without importing a CLI or requiring a client pickle. |
| 2. Build HTTP API | FastAPI schemas/routes, problem responses, lifecycle/readiness, bounded queue, structured logging/metrics, API tests. | Backend can submit one valid clip and receive stable versioned Top-5 JSON; invalid inputs fail safely. |
| 3. Containerise | `requirements-serving.txt`, Dockerfile, Compose reference, `.dockerignore`, non-root image, model-volume contract. | GPU container starts with no external model download and passes health/readiness/golden-clip tests. |
| 4. Integrate application | Backend client, timeouts/retries, authentication, browser capture UI, user-facing quality/uncertainty messages. | An authenticated user completes capture → prediction → display through the real backend, with no direct public model-service access. |
| 5. Operate and improve | Dashboards, alerts, load tests, feedback/consent flow, threshold calibration, release/rollback runbook. | SLOs and capacity limits are evidence-based; rollback and incident handling are rehearsed. |
| 6. Optional real-time mode | Session protocol, signer/sign-boundary logic, incremental buffering, separate evaluation. | It beats or matches clip mode on a defined real-time dataset without destabilising the clip API. |

## 13. Decisions to make before implementation

The following default choices are recommended so implementation can start safely; confirm or change them when the application backend and server details are known.

| Decision | Recommended default | Why it matters |
| --- | --- | --- |
| First client protocol | Browser records a 3–6 second clip; backend calls synchronous multipart endpoint. | Matches isolated-sign data and avoids premature streaming complexity. |
| Service exposure | Private service reachable only by the application backend. | Centralises user auth and limits attack surface. |
| Model artefacts | Immutable versioned read-only bundle outside Git/image. | A 1.17 GB checkpoint and dependent model weights need secure, reproducible release management. |
| Serving GPU | Start with one validated CUDA GPU and one ASGI worker; use the training L40S only as a known-compatible starting point. | Avoids duplicated model memory and makes capacity measurable. |
| Input policy | MP4/WebM initially; fixed request/duration/pixel caps; no URLs or pickles. | Covers common browser capture while protecting decoder/GPU resources. |
| Result UX | Top-1 + Top-5, score margin, low-quality/low-confidence retry guidance. | The held-out accuracy supports assistive output rather than unqualified certainty. |
| Video retention | Delete by default; opt-in, governed feedback only. | Webcam recordings are sensitive personal data. |

## 14. Definition of done for the first deployable service

The initial deployment is complete when all of the following are true:

- A clean server can build/pull the pinned image, mount the verified release bundle, start with `--gpus all`, and become ready without internet access.
- The service contains no training dataset and has no dependency on a host webcam device.
- The application backend calls documented, versioned APIs over a private authenticated channel, propagates request IDs, and handles all specified statuses.
- A browser user can capture one sign, upload via the application, and receive a closed-set Vietnamese Top-5 result or precise retry guidance.
- Serving predictions for trusted golden inputs agree with the locked baseline, and every returned label is in the 30-label vocabulary.
- Request bounds, temporary-file deletion, non-root execution, model checksums, health/readiness, metrics, logs, and rollback procedures are tested.
- Production documentation states clearly that this is Vietnamese isolated-word recognition, records the evaluated accuracy and limitations, and does not promise continuous sentence translation or calibrated certainty.
