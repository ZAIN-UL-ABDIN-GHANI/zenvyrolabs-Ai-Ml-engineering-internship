# Zenvyrolabs Advanced Voice Studio — production image
# Realizes SDS §13 (Deployment Overview): one Application Runtime Boundary
# plus two AI Processing Runtime Boundaries (primary + isolated RVC),
# matching the project's existing dual-venv subprocess architecture
# (SDS Decision D-3) rather than inventing a new one.

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        ffmpeg \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Primary runtime: Presentation + Application + Domain + most AI engines ---
COPY requirements.txt ./
RUN python3.10 -m venv venv \
    && venv/bin/pip install --no-cache-dir --upgrade pip \
    && venv/bin/pip install --no-cache-dir torch torchvision torchaudio --index-url ${TORCH_INDEX_URL} \
    && venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Isolated runtime: RVC voice conversion only (SDS Decision D-3) ---
COPY requirements-rvc.txt ./
RUN python3.10 -m venv rvc_venv \
    && rvc_venv/bin/pip install --no-cache-dir --upgrade pip \
    && rvc_venv/bin/pip install --no-cache-dir torch torchaudio --index-url ${TORCH_INDEX_URL} \
    && rvc_venv/bin/pip install --no-cache-dir -r requirements-rvc.txt

# --- Application source ---
COPY . .

# --- Persistent-store directories (mounted as volumes in docker-compose.yml) ---
RUN mkdir -p saved_voices rvc_models training_data temp logs hf_cache

# --- Non-root execution (SDS §18 security posture) ---
RUN useradd --create-home --uid 1000 voicestudio \
    && chown -R voicestudio:voicestudio /app
USER voicestudio

ENV VOICE_STUDIO_ROOT=/app \
    VOICE_STUDIO_BIND_HOST=0.0.0.0 \
    HF_HOME=/app/hf_cache

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860', timeout=3).status == 200 else 1)" || exit 1

CMD ["/app/venv/bin/python", "/app/app.py"]
