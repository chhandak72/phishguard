#!/usr/bin/env bash
# render_startup.sh
# -----------------
# Called by Render as the startCommand.
# Trains the model on first deploy (or if the model file is missing),
# then starts the API server.

set -e

echo "==> Python: $(python --version)"
echo "==> Working dir: $(pwd)"
echo "==> Files: $(ls)"

MODEL_PATH="models/stacking_pipeline.joblib"

if [ ! -f "$MODEL_PATH" ]; then
  echo "==> Model not found. Starting training with 10k sample rows..."
  python train.py --sample 10000
  echo "==> Training complete."
else
  echo "==> Model found at $MODEL_PATH, skipping training."
fi

echo "==> Starting API server on port ${PORT:-8000}..."
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --log-level info
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --log-level info
