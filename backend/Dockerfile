# ============================================================
# PhishGuard Backend – Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder)  – installs Python deps in a virtualenv
#   Stage 2 (runtime)  – lean production image
#
# Build:
#   docker build -t phishguard-backend ./backend
#
# Run:
#   docker run -p 8000:8000 -v $(pwd)/backend/models:/app/models phishguard-backend
# ============================================================

# ---------- Stage 1: builder ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# System-level build tools required for some Python packages (e.g. xgboost)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  g++ \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Create virtualenv and install dependencies
RUN python -m venv /opt/venv && \
  /opt/venv/bin/pip install --upgrade pip && \
  /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

LABEL maintainer="Aayush Paudel, Chhandak Mukherjee, Arnav Makhija"
LABEL version="1.0.0"
LABEL description="PhishGuard – phishing email classification API"

# Runtime dependencies for xgboost / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv

# Put virtualenv on PATH
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create non-root user for security
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Copy application code
COPY app.py train.py ./
COPY utils/ ./utils/

# Create models directory (will be mounted or populated by train step)
RUN mkdir -p models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start with Gunicorn + Uvicorn workers for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]
