#!/usr/bin/env bash
# Train the model locally and commit the model artifact to the repo
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Use venv python if present
if [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="python3"
fi

echo "Using python: $($PYTHON --version 2>&1)"

echo "Training model (sample 2000 for speed)..."
$PYTHON backend/train.py --sample 2000

MODEL_PATH=backend/models/stacking_pipeline.joblib
if [ -f "$MODEL_PATH" ]; then
  echo "Model created: $MODEL_PATH"
  git add "$MODEL_PATH" backend/models/model_metadata.json || true
  git commit -m "chore: add trained model artifact" || echo "No changes to commit"
  echo "Committed model to git. Push to remote with: git push origin main"
else
  echo "Model not found after training. Check backend/train.py output for errors." >&2
  exit 1
fi
