"""Low-cardinality Prometheus metrics for the private inference service."""

from __future__ import annotations

try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
except ImportError:  # Keeps source-level/API tests usable before runtime deps are installed.
    CollectorRegistry = Counter = Gauge = Histogram = generate_latest = None


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry() if CollectorRegistry else None
        if not self.registry:
            return
        self.requests = Counter(
            "unisign_http_requests_total",
            "HTTP requests handled by the Uni-Sign service",
            ["route", "status", "code"],
            registry=self.registry,
        )
        self.duration = Histogram(
            "unisign_request_duration_seconds",
            "Request stage duration in seconds",
            ["stage"],
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "unisign_gpu_queue_in_flight",
            "Inference requests currently admitted to the bounded GPU queue",
            registry=self.registry,
        )
        self.low_quality = Counter(
            "unisign_low_quality_total",
            "Low-quality or low-confidence result count",
            ["status"],
            registry=self.registry,
        )

    def observe_request(self, route: str, status: int, code: str = "OK") -> None:
        if self.registry:
            self.requests.labels(route=route, status=str(status), code=code).inc()

    def observe_duration_ms(self, stage: str, duration_ms: float) -> None:
        if self.registry:
            self.duration.labels(stage=stage).observe(max(0, duration_ms) / 1000)

    def set_queue_depth(self, value: int) -> None:
        if self.registry:
            self.queue_depth.set(value)

    def observe_result_status(self, status: str) -> None:
        if self.registry and status != "ok":
            self.low_quality.labels(status=status).inc()

    def render(self) -> tuple[bytes, str]:
        if not self.registry:
            return b"# prometheus_client is not installed\n", "text/plain; version=0.0.4"
        return generate_latest(self.registry), "text/plain; version=0.0.4; charset=utf-8"
