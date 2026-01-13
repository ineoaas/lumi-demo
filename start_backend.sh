#!/usr/bin/env bash
# Start backend API (uvicorn)
. .venv/bin/activate 2>/dev/null || true
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
echo "Backend starting on http://127.0.0.1:8000"
