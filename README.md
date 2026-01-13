# Lumi AI — Emotion-to-Color Service

Overview

Lumi AI converts short text inputs into human-readable emotion labels and color hues for UI use. It is intended to be embedded in web demos or client apps as a stateless API that returns an interpretable emotion and associated color.

How it works

- Ensemble approach:
  - A dedicated emotion classifier predicts probabilities across common emotion labels (joy, sadness, anger, fear, disgust, surprise, love, neutral).
  - A zero-shot classifier is used as a fallback to score how well the text matches each of the conceptual emotion categories defined in `EMOTION_MAP`.
- Multi-label aggregation:
  - If the emotion classifier indicates multiple emotions, the service aggregates them, applying a small boost to less-common emotions to reduce a happy/sad dominance.
- Decision rules:
  - Use the emotion model when confident; otherwise fall back to zero-shot; if neither is confident, return `Neutral/Mixed`.
- API responses include debug fields (`method`, `raw_emotion`, `candidates`, and `version`) to help inspect model behavior and tune thresholds.

Run the demo (one-minute) ✅

A short, tested set of steps to run the backend API and web demo locally on Linux/macOS.

Prerequisites

- Python 3.10+ and `python -m venv` available
- A shell (bash, zsh, or POSIX sh)

Quick start (recommended)

```bash
# From project root — one command that starts backend + web in background
. .venv/bin/activate      # activate virtualenv (use `source .venv/bin/activate` if your shell supports it)
chmod +x run_demo.sh      # (only needed once; safe to run)
./run_demo.sh
# Open your browser: http://127.0.0.1:8080/index.html
```

What the script does

- Starts the backend API on http://127.0.0.1:8000 (uvicorn running `backend.main:app`).
- Serves the static web demo on http://127.0.0.1:8080 (Python `http.server`).
- Logs are written to `/tmp/lumi-backend.log` and `/tmp/lumi-web.log`.

Start components separately

- Backend only:
```bash
. .venv/bin/activate
.venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
```
- Web only:
```bash
python -m http.server 8080 --directory web &
```

Quick smoke tests (curl)

- Free-text prediction:
```bash
curl -s -X POST http://127.0.0.1:8000/predict_text \
  -H "Content-Type: application/json" \
  -d '{"text":"My dog died"}'
```
- Lines prediction:
```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"lines":["Morning","My dog died","Evening"]}'
```

Stopping the demo

- Stop backend:
```bash
pkill -f "uvicorn backend.main" || true
```
- Stop web server:
```bash
pkill -f "http.server" || true
```

Troubleshooting

- First request may be slow: the model downloads on first use — check `/tmp/lumi-backend.log` to watch progress (`tail -f /tmp/lumi-backend.log`).
- If `pip install` fails with `OSError: [Errno 28] No space left on device` when installing large wheels (torch): try
```bash
TMPDIR=/var/tmp python -m pip install --no-cache-dir -r requirements.txt
```
- If your shell reports `source: command not found`, use the POSIX dot form:
```bash
. .venv/bin/activate
```

Tests

- Quick, dependency-light test:
```bash
PYTHONPATH=. python tests/run_quick.py
```
- Full pytest suite (requires dependencies):
```bash
. .venv/bin/activate
PYTHONPATH=. .venv/bin/pytest -q
```

Notes

- You can also set `EMOTION_MODEL` to point to a fine-tuned local model before starting the server, e.g.:
```bash
export EMOTION_MODEL=models/emotion-finetuned
./run_demo.sh
```
- For production, serve over HTTPS and restrict CORS to trusted domains.

Flutter integration (basic example)

- Ensure the backend is reachable from the mobile app (use your server URL and enable CORS or provide a proxy).
- Example Dart call using `http` package:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictText(String text) async {
  final res = await http.post(
    Uri.parse('http://YOUR_SERVER/predict_text'),
    headers: { 'Content-Type': 'application/json' },
    body: jsonEncode({ 'text': text }),
  );
  if (res.statusCode != 200) throw Exception('Request failed');
  return jsonDecode(res.body) as Map<String, dynamic>;
}

// Convert received hue to Flutter Color (simple helper)
import 'dart:ui';
Color hslToColor(double h, double s, double l) {
  // Convert HSL (0..360, 0..1, 0..1) to ARGB Color.
  // This is a compact implementation; you can use a library for more features.
  final c = (1 - (2 * l - 1).abs()) * s;
  final x = c * (1 - ((h / 60) % 2 - 1).abs());
  final m = l - c / 2;
  double r = 0, g = 0, b = 0;
  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }
  return Color.fromARGB(255, ((r + m) * 255).round(), ((g + m) * 255).round(), ((b + m) * 255).round());
}
```

Swapping to a fine-tuned model

- If you fine-tune a model locally and save it (example: `models/emotion-finetuned`), start the server with `EMOTION_MODEL` set to that path:

```bash
export EMOTION_MODEL=models/emotion-finetuned
python main.py
```

Production notes and expectations

- For mobile/web production, serve the API over HTTPS and restrict CORS to the domains you use.
- Collect a representative labeled dataset (200–500 examples) for fine-tuning and calibration to reach high reliability; even then, no model is 100% accurate for all inputs.

If you want, I can:
- Run a small fine-tune on your machine and report results, or
- Seed a labeled dataset with more examples so you can start fine-tuning immediately.
