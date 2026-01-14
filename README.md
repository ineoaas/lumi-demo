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

Flutter integration (complete step-by-step guide)

### New Feature: One-Sentence Day Summary

After the user submits their input, the API now returns a `summary` field with a human-readable, second-person recap of their day in a single sentence. For example:

**Input:**
- Morning: Woke up early and had coffee
- A key moment: Got promoted at work
- Interaction: Caught up with an old friend
- A challenge: Had an argument with my sibling
- Evening: Relaxed with a good book

**Response includes:**
```json
{
  "emotion": "Calm",
  "hue": 120,
  "confidence": "31.2%",
  "summary": "You woke up early and had coffee, got promoted at work, caught up with an old friend."
}
```

### Step-by-Step Flutter Setup

1. **Ensure your backend is running and accessible:**
   - If running locally: `http://127.0.0.1:8000`
   - If on a remote server or emulator: Replace with your server IP address (e.g., `http://192.168.x.x:8000` for a LAN device)

2. **Add the `http` package to your `pubspec.yaml`:**
   ```yaml
   dependencies:
     flutter:
       sdk: flutter
     http: ^0.13.6
   ```

3. **Create a service class in your Flutter app:**

   ```dart
   import 'dart:convert';
   import 'package:http/http.dart' as http;
   
   class LumiService {
     final String baseUrl = 'http://YOUR_SERVER:8000';  // Replace with your backend URL
   
     // Predict from a single text string
     Future<Map<String, dynamic>> predictText(String text) async {
       final response = await http.post(
         Uri.parse('$baseUrl/predict_text'),
         headers: {'Content-Type': 'application/json'},
         body: jsonEncode({'text': text}),
       );
   
       if (response.statusCode == 200) {
         return jsonDecode(response.body) as Map<String, dynamic>;
       } else {
         throw Exception('Failed to predict: ${response.statusCode}');
       }
     }
   
     // Predict from multiple lines (like the demo)
     Future<Map<String, dynamic>> predictLines(List<String> lines) async {
       final response = await http.post(
         Uri.parse('$baseUrl/predict'),
         headers: {'Content-Type': 'application/json'},
         body: jsonEncode({'lines': lines}),
       );
   
       if (response.statusCode == 200) {
         return jsonDecode(response.body) as Map<String, dynamic>;
       } else {
         throw Exception('Failed to predict: ${response.statusCode}');
       }
     }
   
     // Helper: Convert HSL hue (0-360) to Flutter Color
     Color hueToColor(int? hue) {
       if (hue == null) return Colors.grey;
       // Convert hue to HSL Color
       return HSLColor.fromAHSL(1.0, hue.toDouble(), 0.85, 0.65).toColor();
     }
   }
   ```

4. **Use the service in your Flutter widget:**

   ```dart
   import 'package:flutter/material.dart';
   
   class LumiDemoScreen extends StatefulWidget {
     @override
     State<LumiDemoScreen> createState() => _LumiDemoScreenState();
   }
   
   class _LumiDemoScreenState extends State<LumiDemoScreen> {
     final lumiService = LumiService();
     final controllers = List.generate(5, (_) => TextEditingController());
     Map<String, dynamic>? prediction;
     bool isLoading = false;
   
     @override
     void dispose() {
       for (var c in controllers) c.dispose();
       super.dispose();
     }
   
     void _analyzeDay() async {
       setState(() => isLoading = true);
       try {
         final lines = controllers.map((c) => c.text).toList();
         final result = await lumiService.predictLines(lines);
         setState(() {
           prediction = result;
           isLoading = false;
         });
       } catch (e) {
         ScaffoldMessenger.of(context).showSnackBar(
           SnackBar(content: Text('Error: $e')),
         );
         setState(() => isLoading = false);
       }
     }
   
     @override
     Widget build(BuildContext context) {
       return Scaffold(
         appBar: AppBar(title: Text('Lumi — Your Day in Color')),
         body: SingleChildScrollView(
           padding: EdgeInsets.all(16),
           child: Column(
             children: [
               ...List.generate(
                 5,
                 (i) => Padding(
                   padding: EdgeInsets.only(bottom: 12),
                   child: TextField(
                     controller: controllers[i],
                     decoration: InputDecoration(
                       hintText: [
                         'Morning...',
                         'A key moment...',
                         'Interaction...',
                         'A challenge...',
                         'Evening thought...',
                       ][i],
                       border: OutlineInputBorder(),
                     ),
                   ),
                 ),
               ),
               SizedBox(height: 16),
               ElevatedButton(
                 onPressed: isLoading ? null : _analyzeDay,
                 child: Text(isLoading ? 'Processing...' : 'Reveal My Color'),
               ),
               if (prediction != null) ...[
                 SizedBox(height: 24),
                 Container(
                   width: 120,
                   height: 120,
                   decoration: BoxDecoration(
                     color: lumiService.hueToColor(prediction!['hue'] as int?),
                     shape: BoxShape.circle,
                     boxShadow: [
                       BoxShadow(
                         color: lumiService.hueToColor(prediction!['hue'] as int?).withOpacity(0.3),
                         blurRadius: 20,
                         spreadRadius: 5,
                       ),
                     ],
                   ),
                 ),
                 SizedBox(height: 16),
                 Text(
                   prediction!['emotion'] as String? ?? 'Neutral',
                   style: Theme.of(context).textTheme.headlineSmall,
                 ),
                 Text(
                   'Confidence: ${prediction!['confidence']}',
                   style: Theme.of(context).textTheme.bodySmall,
                 ),
                 SizedBox(height: 12),
                 // Display the summary sentence
                 Container(
                   padding: EdgeInsets.all(12),
                   decoration: BoxDecoration(
                     color: Colors.grey[100],
                     borderRadius: BorderRadius.circular(8),
                   ),
                   child: Text(
                     prediction!['summary'] as String? ?? 'No summary available',
                     style: Theme.of(context).textTheme.bodyMedium,
                     textAlign: TextAlign.center,
                   ),
                 ),
               ],
             ],
           ),
         ),
       );
     }
   }
   ```

5. **Testing with your device:**
   - **Local testing (emulator):** Use `http://10.0.2.2:8000` to access your host machine's localhost from an Android emulator.
   - **Physical device on LAN:** Use your machine's LAN IP (e.g., `http://192.168.1.100:8000`).
   - **Remote server:** Use your server's URL.

6. **Production deployment:**
   - Serve the backend over HTTPS.
   - Update your Flutter app to use the HTTPS URL.
   - Implement error handling and retry logic for network failures.

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
