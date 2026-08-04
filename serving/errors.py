"""Stable service errors mapped to API problem responses."""

from __future__ import annotations


class ServiceError(Exception):
    """An expected serving failure with an API-safe status and code."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 422,
        *,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


class ModelUnavailableError(ServiceError):
    def __init__(self, message: str = "The model is not ready") -> None:
        super().__init__("MODEL_NOT_READY", message, 503, retryable=True)


class QueueFullError(ServiceError):
    def __init__(self) -> None:
        super().__init__("GPU_QUEUE_FULL", "The inference queue is full", 429, retryable=True)
