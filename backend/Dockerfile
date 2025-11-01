# backend/Dockerfile
FROM python:3.11-slim

# ---- Basics & faster installs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=10

# Build deps (psycopg2 etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir is the parent of your package dir "backend"
WORKDIR /workspace

# Toolchain, then CPU-only torch (prevents CUDA pulls)
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1

# Dependencies first (better layer cache)
COPY requirements.txt /workspace/backend/requirements.txt
RUN pip install --no-cache-dir -r /workspace/backend/requirements.txt

# (Optional) Warm the HF cache to avoid cold starts
# ENV HF_HOME=/workspace/.cache/huggingface
# RUN python - <<'PY'
# from sentence_transformers import SentenceTransformer
# SentenceTransformer('intfloat/e5-small-v2')
# PY

# Your source
COPY . /workspace/backend/

# Make 'backend' importable (same as local)
ENV PYTHONPATH=/workspace

# Uvicorn entrypoint
CMD ["uvicorn", "backend.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
