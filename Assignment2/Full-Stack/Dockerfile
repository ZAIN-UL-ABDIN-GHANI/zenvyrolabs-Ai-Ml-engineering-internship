# syntax=docker/dockerfile:1.7

FROM python:3.10-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=3600 \
    PIP_RETRIES=5 \
    PIP_CACHE_DIR=/root/.cache/pip \
    VOICE_STUDIO_ROOT=/app \
    VOICE_STUDIO_BIND_HOST=0.0.0.0 \
    HF_HOME=/app/hf_cache \
    TEMP=/app/temp \
    TMP=/app/temp \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN printf '%s\n' \
        '#!/bin/sh' \
        'set -eu' \
    'retries=0' \
        'while true; do' \
        '  "$@" && exit 0' \
    '  retries=$((retries + 1))' \
    '  if [ "$retries" -gt 5 ]; then exit 1; fi' \
    '  echo "pip command failed; retrying in 120 seconds (retry ${retries}/5)" >&2' \
        '  sleep 120' \
        'done' \
    > /usr/local/bin/pip-retry \
    && chmod +x /usr/local/bin/pip-retry

COPY requirements.txt ./
COPY requirements-rvc.txt ./

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m venv venv \
    && /usr/local/bin/pip-retry venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /usr/local/bin/pip-retry venv/bin/pip install --index-url "${TORCH_INDEX_URL}" \
        torch==2.13.0+cpu \
        torchvision==0.18.0+cpu \
        torchaudio==2.13.0+cpu \
    && /usr/local/bin/pip-retry venv/bin/pip install -r requirements.txt

RUN --mount=type=cache,id=pip-cache,target=/root/.cache/pip,sharing=locked \
    python -m venv rvc_venv \
    && /usr/local/bin/pip-retry rvc_venv/bin/python -m pip install --upgrade pip==24.0 setuptools wheel \
    && /usr/local/bin/pip-retry rvc_venv/bin/pip install --index-url "${TORCH_INDEX_URL}" torch==2.13.0+cpu torchaudio==2.13.0+cpu \
    && /usr/local/bin/pip-retry rvc_venv/bin/pip install -r requirements-rvc.txt

COPY . .

RUN mkdir -p saved_voices rvc_models training_data temp logs hf_cache \
    outputs \
    && useradd --create-home --uid 1000 --shell /bin/bash voicestudio \
    && chown -R voicestudio:voicestudio /app

USER voicestudio

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/venv/bin/python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860', timeout=3).status == 200 else 1)"]

CMD ["/app/venv/bin/python", "/app/app.py"]
