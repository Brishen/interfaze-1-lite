# Interfaze 1 Lite on llama.cpp, in a container: llamacpp/run.sh on top of llama.cpp's
# own CUDA server image. Quantized GGUFs for the reasoning core and the document reader;
# fits two 24 GB cards or one 40 GB+ card, with no FP8 requirement.
#
# Build:  docker build -f llamacpp/Dockerfile -t interfaze-1-lite:llamacpp .
# Run:    docker run --gpus all --ipc=host -p 8000:8000 -v "$PWD/models:/models" interfaze-1-lite:llamacpp
#
# HF_TOKEN is optional here: only speaker attribution needs it (the diarization model is
# gated). On a host with a llama.cpp build of its own, llamacpp/run.sh runs the same
# stack without this image.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /opt/interfaze
ENV UV_PYTHON=3.12 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python

# Dependencies first, so a code change does not reinstall torch.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --extra llamacpp --no-install-project
COPY src ./src
COPY components.env.example ./
COPY llamacpp ./llamacpp
RUN uv sync --frozen --extra llamacpp && chmod +x llamacpp/run.sh

# The base image finds llama-server's libraries only from its own working directory, /app.
ENV LD_LIBRARY_PATH=/app:/usr/local/cuda/lib64 \
    LLAMA_SERVER=/app/llama-server \
    MODELS_DIR=/models/gguf \
    HF_HOME=/models/hf \
    PADDLE_PDX_CACHE_HOME=/models/paddlex \
    LOG_DIR=/models/logs

# Weights are a mounted volume, not a layer.
VOLUME ["/models"]

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=3600s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/opt/interfaze/llamacpp/run.sh"]
