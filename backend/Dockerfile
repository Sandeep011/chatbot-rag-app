# backend/Dockerfile
FROM python:3.11-slim

# System deps (optional but handy for PDFs, etc.)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first to leverage Docker layer caching
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend source
COPY backend /app/backend
ENV PYTHONPATH=/app

# Expose the app on 0.0.0.0:8000
CMD ["uvicorn", "backend.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
