#!/usr/bin/env bash
# PhishGuard – start backend + frontend dev servers
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "▶ Starting FastAPI backend on http://localhost:8000 …"
"$REPO/.venv/bin/uvicorn" backend.app:app --app-dir "$REPO" --port 8000 &
BACKEND_PID=$!

echo "▶ Starting React frontend on http://localhost:3000 …"
cd "$REPO/frontend"
./node_modules/.bin/vite &
FRONTEND_PID=$!

echo ""
echo "✅ PhishGuard is running!"
echo "   Backend  → http://localhost:8000/health"
echo "   Frontend → http://localhost:3000"
echo ""
echo "Press Ctrl-C to stop both servers."

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" INT TERM
wait
