#!/usr/bin/env bash
# Start backend and web demo (simple one-liner)
. .venv/bin/activate 2>/dev/null || true
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 > /tmp/lumi-backend.log 2>&1 &
python -m http.server 8080 --directory web > /tmp/lumi-web.log 2>&1 &
echo "Started backend (http://127.0.0.1:8000) and web demo (http://127.0.0.1:8080/index.html)"
