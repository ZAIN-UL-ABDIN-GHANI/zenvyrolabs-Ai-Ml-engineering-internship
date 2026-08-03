FROM python:3.10-slim-bookworm

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
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
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
COPY requirements-rvc.txt ./

RUN python -m venv venv \
    && venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}" \
    && venv/bin/pip install --no-cache-dir -r requirements.txt

RUN python -m venv rvc_venv \
    # rvc-python pulls omegaconf==2.0.6, which pip 24.1+ rejects during metadata parsing.
    && rvc_venv/bin/python -m pip install --upgrade pip==24.0 setuptools wheel \
    && rvc_venv/bin/pip install --no-cache-dir torch torchaudio --index-url "${TORCH_INDEX_URL}" \
    && rvc_venv/bin/pip install --no-cache-dir -r requirements-rvc.txt

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
