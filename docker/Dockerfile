# syntax=docker/dockerfile:1
# Torch/Torchvision 2.1.1 with CUDA 12.1 matches the original Uni-Sign runtime.
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    TEMP_DIR=/tmp/unisign

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serving.txt /tmp/requirements-serving.txt
RUN pip install --no-cache-dir -r /tmp/requirements-serving.txt

COPY . /app
# RTMLib is vendored in this repository. Its dependencies are pinned above so
# its setup metadata cannot replace the CUDA-specific ONNX Runtime package.
RUN pip install --no-cache-dir --no-deps /app/demo/rtmlib-main
RUN python -c "import fastapi, onnxruntime, torch, torchvision, transformers; assert torch.__version__.startswith('2.1.1')"

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /tmp/unisign \
    && chown -R app:app /app /tmp/unisign \
    && chmod +x /app/docker/entrypoint.sh

USER app
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/livez', timeout=3)"

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--proxy-headers"]
