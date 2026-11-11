# Studio tool service: runs the heavy background jobs of WeBuddhist Studio tools.
# Build: docker build -t studio-tool-service:0.0.1 .   (or `tutor images build studio-tool-service`)
FROM python:3.12-slim-bookworm

# ffmpeg reads and writes the audio in every pipeline step.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only PyTorch: the server has no GPU, and the CPU build is much smaller.
ARG TORCH_VERSION=2.14.1
ARG TORCHAUDIO_VERSION=2.11.0
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as a normal user, with the same uid as Tutor's own containers.
RUN useradd --uid 1000 --create-home app && mkdir /app/models && chown app /app/models
USER app

# Download the speaker model into the image, so jobs never wait for (or depend on) a download.
ENV MODEL_DIR=/app/models \
    HF_HOME=/app/models/huggingface
RUN python -c "from speechbrain.inference.speaker import EncoderClassifier; \
EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/app/models/spkrec-ecapa-voxceleb')"
ENV HF_HUB_OFFLINE=1

# The code is used from /app/src (not pip-installed), so `tutor dev` can mount a live checkout over it.
COPY src ./src
ENV PYTHONPATH=/app/src

CMD ["python", "-m", "studio_tool_service.worker"]
