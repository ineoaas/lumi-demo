#!/usr/bin/env bash
# Serve the polished web demo on port 8080
python -m http.server 8080 --directory web &
echo "Web demo available at http://127.0.0.1:8080/index.html"