"""Private FastAPI interface for CoSign isolated-sign inference."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hmac
import logging
from pathlib import Path
import tempfile
import threading
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.exceptions import HTTPException as StarletteHTTPException

from serving.demo import DemoInferenceService
from serving.errors import ModelUnavailableError, QueueFullError, ServiceError
from serving.service import InferenceService
from serving.settings import ServiceSettings
from serving.telemetry import Metrics

security_bearer = HTTPBearer(
    auto_error=False,
    scheme_name="ServiceBearer",
    description="Enter the service API key only; Swagger UI sends the Bearer prefix.",
)
logger = logging.getLogger(__name__)


class UnavailableInferenceService:
    """Allows liveness/error routes to work when startup cannot load a model."""

    def __init__(self, error: ServiceError) -> None:
        self.error = error

    @property
    def ready(self) -> bool:
        return False

    def _raise(self):
        raise self.error

    def model_metadata(self):
        self._raise()

    def labels(self):
        self._raise()

    def predict_file(self, _path, _top_k):
        self._raise()


def _problem(error: ServiceError, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "type": f"https://unisign.internal/problems/{error.code.lower()}",
            "title": error.message,
            "status": error.status_code,
            "code": error.code,
            "request_id": request_id,
            "retryable": error.retryable,
        },
    )


def _request_id(request: Request) -> str:
    supplied = request.headers.get("X-Request-ID", "").strip()
    if supplied and len(supplied) <= 128 and all(char.isalnum() or char in "-_." for char in supplied):
        return supplied
    return str(uuid4())


async def _require_service_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_bearer),
) -> None:
    settings: ServiceSettings = request.app.state.settings
    if not settings.service_api_key:
        return
    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not hmac.compare_digest(credentials.credentials, settings.service_api_key)
    ):
        raise ServiceError("UNAUTHORIZED", "Invalid service credential", 401)


async def _save_upload(upload: UploadFile, settings: ServiceSettings) -> Path:
    settings.temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = {
        "video/mp4": ".mp4",
        "video/webm": ".webm",
        "video/quicktime": ".mov",
        "video/x-msvideo": ".avi",
    }.get((upload.content_type or "").lower(), ".video")
    uploaded_bytes = 0
    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="request-",
            dir=settings.temp_dir,
            delete=False,
        ) as handle:
            path = Path(handle.name)
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                uploaded_bytes += len(chunk)
                if uploaded_bytes > settings.max_upload_bytes:
                    raise ServiceError("FILE_TOO_LARGE", "Upload exceeds the configured size limit", 413)
                handle.write(chunk)
        if uploaded_bytes == 0:
            raise ServiceError("INVALID_MULTIPART", "Video upload is empty", 400)
        return path
    except Exception:
        if path is not None:
            path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def create_app(
    settings: ServiceSettings | None = None,
    service: InferenceService | UnavailableInferenceService | Any | None = None,
) -> FastAPI:
    """Create an app factory so unit tests can inject a no-GPU fake service."""
    settings = settings or ServiceSettings.from_env()
    metrics = Metrics()
    admitted = threading.BoundedSemaphore(settings.max_queue_depth)
    in_flight = 0
    in_flight_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            try:
                settings.validate()
                if settings.demo_mode:
                    logger.warning("DEMO_MODE is enabled: no model inference will run")
                    app.state.service = DemoInferenceService()
                else:
                    app.state.service = await run_in_threadpool(InferenceService.load, settings)
            except ServiceError as error:
                app.state.service = UnavailableInferenceService(error)
            except Exception:
                logger.exception("Model service startup failed")
                app.state.service = UnavailableInferenceService(
                    ModelUnavailableError("Model startup failed; inspect server logs")
                )
        else:
            app.state.service = service
        yield

    app = FastAPI(
        title="Uni-Sign CoSign Vietnamese ISLR Inference API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.metrics = metrics

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError):
        logger.info("Service request failed: status=%s code=%s", error.status_code, error.code)
        request_id = getattr(request.state, "request_id", _request_id(request))
        metrics.observe_request(request.url.path, error.status_code, error.code)
        response = _problem(error, request_id)
        response.headers["X-Request-ID"] = request_id
        if error.status_code == 429:
            response.headers["Retry-After"] = "1"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, _error: RequestValidationError):
        error = ServiceError("INVALID_MULTIPART", "Request fields are invalid or missing", 400)
        request_id = getattr(request.state, "request_id", _request_id(request))
        metrics.observe_request(request.url.path, error.status_code, error.code)
        response = _problem(error, request_id)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, error: StarletteHTTPException):
        detail = str(error.detail)
        if error.status_code == 400 and ("multipart" in detail.lower() or "boundary" in detail.lower()):
            return await service_error_handler(
                request,
                ServiceError("INVALID_MULTIPART", "Multipart video upload is malformed", 400),
            )
        return JSONResponse(status_code=error.status_code, content={"detail": error.detail})

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request.state.request_id = _request_id(request)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.get("/livez")
    async def livez():
        return {"status": "live"}

    @app.get("/readyz")
    async def readyz(request: Request):
        active_service = request.app.state.service
        if not active_service.ready:
            raise ModelUnavailableError("Model bundle or runtime is not ready")
        return {"status": "ready", "model": active_service.model_metadata()}

    @app.get("/v1/model", dependencies=[Depends(_require_service_auth)])
    async def model_metadata(request: Request):
        active_service = request.app.state.service
        if not active_service.ready:
            raise ModelUnavailableError()
        return active_service.model_metadata()

    @app.get("/v1/labels", dependencies=[Depends(_require_service_auth)])
    async def labels(request: Request):
        active_service = request.app.state.service
        if not active_service.ready:
            raise ModelUnavailableError()
        return {"labels": active_service.labels(), "model": active_service.model_metadata()}

    @app.post("/v1/predictions", dependencies=[Depends(_require_service_auth)])
    async def predict(
        request: Request,
        video: UploadFile = File(...),
        top_k: str | int | None = Form(None),
        client_capture_id: str | None = Form(None),
        client_duration_ms: str | int | None = Form(None),
    ):
        del client_duration_ms  # Server-decoded duration is authoritative.
        if client_capture_id is not None:
            capture_id = str(client_capture_id).strip()
            if not capture_id or capture_id.lower() == "string":
                client_capture_id = None
            elif len(capture_id) > 128 or not all(char.isalnum() or char in "-_." for char in capture_id):
                raise ServiceError("INVALID_MULTIPART", "client_capture_id is invalid", 400)
            else:
                client_capture_id = capture_id

        parsed_top_k: int | None = None
        if top_k is not None:
            top_k_str = str(top_k).strip()
            # Swagger UI may submit its optional form placeholder literally as
            # "string" or "null" when a developer does not edit the field.
            if top_k_str.lower() not in {"", "string", "null", "none"}:
                try:
                    parsed_top_k = int(top_k_str)
                except ValueError:
                    raise ServiceError("INVALID_TOP_K", "top_k must be in 1..5", 400)

        requested_top_k = settings.top_k_default if parsed_top_k is None else parsed_top_k
        if not 1 <= requested_top_k <= 5:
            raise ServiceError("INVALID_TOP_K", "top_k must be in 1..5", 400)
        active_service = request.app.state.service
        if not active_service.ready:
            raise ModelUnavailableError()
        if not admitted.acquire(blocking=False):
            raise QueueFullError()
        nonlocal in_flight
        with in_flight_lock:
            in_flight += 1
            metrics.set_queue_depth(in_flight)
        path: Path | None = None
        try:
            path = await _save_upload(video, settings)
            result = await run_in_threadpool(active_service.predict_file, path, requested_top_k)
            result["request_id"] = request.state.request_id
            result["created_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            if client_capture_id is not None:
                result["client_capture_id"] = client_capture_id
            metrics.observe_request("/v1/predictions", 200)
            metrics.observe_result_status(result["status"])
            for stage, duration_ms in result.get("timing_ms", {}).items():
                metrics.observe_duration_ms(stage, duration_ms)
            return result
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            admitted.release()
            with in_flight_lock:
                in_flight -= 1
                metrics.set_queue_depth(in_flight)

    @app.get("/metrics", dependencies=[Depends(_require_service_auth)])
    async def metrics_endpoint(request: Request):
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    return app


app = create_app()
