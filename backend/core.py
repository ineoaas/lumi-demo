import os
from typing import List
from transformers import pipeline
import torch

zero_shot = None
emotion_classifier = None
summarizer_pipeline = None
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")

# Thresholds
EMOTION_CONFIDENCE_THRESHOLD = 0.45
NEUTRAL_FALLBACK_THRESHOLD = 0.22
MULTI_LABEL_THRESHOLD = 0.20
PROMOTE_LABEL_BOOST = 0.04
MULTI_LABEL_ACCEPT_FACTOR = 0.75

EMOTION_MAP = {
    "Joy/Happy": {"hue": 60, "label": "Joyful"},
    "Calm/Relaxed": {"hue": 120, "label": "Calm"},
    "Sad/Depressed": {"hue": 240, "label": "Sad"},
    "Angry/Irritated": {"hue": 0, "label": "Angry"},
    "Fearful/Anxious": {"hue": 180, "label": "Anxious"},
    "Disgusted/Envious": {"hue": 300, "label": "Disgusted"},
    "Surprised/Inspired": {"hue": 30, "label": "Inspired"},
    "Anticipation/Optimistic": {"hue": 90, "label": "Optimistic"},
    "Neutral/Mixed": {"hue": None, "label": "Neutral"}
}

EMOTION_MODEL_MAP = {
    "joy": "Joy/Happy",
    "sadness": "Sad/Depressed",
    "anger": "Angry/Irritated",
    "fear": "Fearful/Anxious",
    "disgust": "Disgusted/Envious",
    "surprise": "Surprised/Inspired",
    "love": "Joy/Happy",
    "neutral": "Neutral/Mixed"
}


def load_pipelines():
    global zero_shot, emotion_classifier, summarizer_pipeline
    if zero_shot is not None and emotion_classifier is not None and summarizer_pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline
        if zero_shot is None:
            zero_shot = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        if emotion_classifier is None:
            emotion_classifier = hf_pipeline("text-classification", model=EMOTION_MODEL)
        if summarizer_pipeline is None:
            print("Loading summarization model...")
            # Use a reliable, GPU-friendly summarization model
            summarizer_pipeline = hf_pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            print("Summarization model loaded!")
    except Exception as e:
        print("Warning: could not initialize transformers pipelines:", e)


def analyze_text(text_to_analyze: str) -> dict:
    """Analyze a single text string and return the prediction dict.
    This function does not depend on FastAPI/pydantic so tests can import it without heavy deps.
    """
    text = text_to_analyze.strip()
    if not text:
        return {"emotion": "Neutral", "hue": None, "confidence": "0%", "method": "none", "candidates": [], "version": {"emotion_model": EMOTION_MODEL, "zero_shot": "facebook/bart-large-mnli"}, "summary": "No text provided, so emotion is Neutral."}

    def make_summary(user_text):
        load_pipelines()
        import re
        lines = [l.strip() for l in re.split(r'[\n\r]+', user_text) if l.strip()]
        if not lines:
            return "You had a quiet day."
        
        # Extract main verbs/actions from each line and combine them
        # This creates a condensed summary like "You did X, then Y, then Z"
        actions = []
        for line in lines:
            # Remove common label prefixes
            line_clean = re.sub(r'^(Morning|Afternoon|Evening|Night|Day|Morning:|Afternoon:|Evening:|Night:|A key moment|Interaction|A challenge|Event|Thing|Activity|Note)[\s:]*', '', line, flags=re.IGNORECASE)
            if line_clean:
                # Convert to second person if needed
                line_clean = re.sub(r'\bI\b', 'you', line_clean, flags=re.IGNORECASE)
                line_clean = re.sub(r'\bmy\b', 'your', line_clean, flags=re.IGNORECASE)
                line_clean = re.sub(r'\bme\b', 'you', line_clean, flags=re.IGNORECASE)
                line_clean = line_clean.strip()
                # Lowercase first letter
                if line_clean:
                    line_clean = line_clean[0].lower() + line_clean[1:]
                    actions.append(line_clean)
        
        if not actions:
            return "You had a quiet day."
        
        # Create a natural summary by joining actions with commas
        if len(actions) == 1:
            action = actions[0]
            # Avoid "You you..." by checking if it already starts with "you"
            if action.lower().startswith('you '):
                summary = action[0].upper() + action[1:]  # Capitalize first letter
            else:
                summary = f"You {action}"
        elif len(actions) <= 3:
            summary = "You " + ", ".join(actions)
        else:
            # For 4+ actions, combine them more concisely
            summary = "You " + ", ".join(actions[:3]) + "."
        
        # Ensure ends with period
        if not summary.endswith('.'):
            summary += '.'
        
        return summary

    candidate_labels = list(EMOTION_MAP.keys())

    # Try emotion classifier (multi-label aware)
    label_scores = {}
    try:
        if callable(emotion_classifier):
            emotion_scores = emotion_classifier(text, return_all_scores=True)
            if isinstance(emotion_scores, list) and len(emotion_scores) and isinstance(emotion_scores[0], list):
                emotion_scores = emotion_scores[0]
            label_scores = {e["label"].lower(): float(e["score"]) for e in emotion_scores}
    except Exception:
        label_scores = {}

    model_label = None
    model_score = 0.0

    selected = {label: score for label, score in label_scores.items() if score >= MULTI_LABEL_THRESHOLD}
    if selected:
        mapped_scores = {}
        for label, score in selected.items():
            boost = PROMOTE_LABEL_BOOST if label not in ("joy", "sadness", "neutral") else 0.0
            adj = score + boost
            mapped_key = EMOTION_MODEL_MAP.get(label, "Neutral/Mixed")
            mapped_scores[mapped_key] = max(mapped_scores.get(mapped_key, 0.0), adj)
        mapped_key, mapped_score = max(mapped_scores.items(), key=lambda kv: kv[1])
        if mapped_score >= EMOTION_CONFIDENCE_THRESHOLD * MULTI_LABEL_ACCEPT_FACTOR:
            color_data = EMOTION_MAP.get(mapped_key, EMOTION_MAP["Neutral/Mixed"])
            candidates = []
            for k, v in sorted(mapped_scores.items(), key=lambda kv: kv[1], reverse=True):
                c_hue = EMOTION_MAP.get(k, {}).get("hue")
                candidates.append({"source": "emotion-model", "label": k, "hue": c_hue, "score": v})
            result = {"emotion": color_data["label"], "hue": color_data["hue"], "confidence": f"{mapped_score:.1%}", "raw_emotion": mapped_key, "method": "emotion-model-multi", "candidates": candidates, "version": {"emotion_model": EMOTION_MODEL, "zero_shot": "facebook/bart-large-mnli"}}
            result["summary"] = make_summary(text)
            return result

    if label_scores:
        top_emotion_pred = max(label_scores.items(), key=lambda kv: kv[1])
        model_label = top_emotion_pred[0]
        model_score = top_emotion_pred[1]
        if model_label and model_score >= EMOTION_CONFIDENCE_THRESHOLD:
            mapped_key = EMOTION_MODEL_MAP.get(model_label, "Neutral/Mixed")
            color_data = EMOTION_MAP[mapped_key]
            top_labels = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            candidates = []
            for lab, sc in top_labels:
                mapped = EMOTION_MODEL_MAP.get(lab, "Neutral/Mixed")
                hue = EMOTION_MAP.get(mapped, {}).get("hue")
                candidates.append({"source": "emotion-model", "label": mapped, "hue": hue, "score": sc})
            result = {"emotion": color_data["label"], "hue": color_data["hue"], "confidence": f"{model_score:.1%}", "raw_emotion": mapped_key, "method": "emotion-model", "candidates": candidates, "version": {"emotion_model": EMOTION_MODEL, "zero_shot": "facebook/bart-large-mnli"}}
            result["summary"] = make_summary(text)
            return result

    # Fallback to zero-shot
    load_pipelines()
    try:
        if callable(zero_shot):
            output = zero_shot(text, candidate_labels, multi_label=False)
        else:
            output = {"labels": [], "scores": []}
    except Exception:
        output = {"labels": [], "scores": []}

    if output.get("labels"):
        top_emotion = output["labels"][0]
        top_score = float(output["scores"][0])
    else:
        top_emotion = "Neutral/Mixed"
        top_score = 0.0

    if max(top_score, model_score) < NEUTRAL_FALLBACK_THRESHOLD:
        top_emotion = "Neutral/Mixed"
        color_data = EMOTION_MAP[top_emotion]
        zs_candidates = []
        for lbl, sc in zip(output.get("labels", [])[:3], output.get("scores", [])[:3]):
            hue = EMOTION_MAP.get(lbl, {}).get("hue")
            zs_candidates.append({"source": "zero-shot", "label": lbl, "hue": hue, "score": float(sc)})
        result = {"emotion": color_data["label"], "hue": color_data["hue"], "confidence": f"{max(top_score, model_score):.1%}", "raw_emotion": top_emotion, "method": "fallback-neutral", "candidates": zs_candidates, "version": {"emotion_model": EMOTION_MODEL, "zero_shot": "facebook/bart-large-mnli"}}
        result["summary"] = make_summary(text)
        return result

    if model_score > top_score and model_label:
        mapped_key = EMOTION_MODEL_MAP.get(model_label, "Neutral/Mixed")
        color_data = EMOTION_MAP[mapped_key]
        chosen_score = model_score
        method = "emotion-model"
        raw = mapped_key
    else:
        color_data = EMOTION_MAP[top_emotion]
        chosen_score = top_score
        method = "zero-shot"
        raw = top_emotion

    candidates = []
    if label_scores:
        for lab, sc in sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            mapped = EMOTION_MODEL_MAP.get(lab, "Neutral/Mixed")
            hue = EMOTION_MAP.get(mapped, {}).get("hue")
            candidates.append({"source": "emotion-model", "label": mapped, "hue": hue, "score": float(sc)})
    for lbl, sc in zip(output.get("labels", [])[:3], output.get("scores", [])[:3]):
        hue = EMOTION_MAP.get(lbl, {}).get("hue")
        candidates.append({"source": "zero-shot", "label": lbl, "hue": hue, "score": float(sc)})

    result = {"emotion": color_data["label"], "hue": color_data["hue"], "confidence": f"{chosen_score:.1%}", "raw_emotion": raw, "method": method, "candidates": candidates, "version": {"emotion_model": EMOTION_MODEL, "zero_shot": "facebook/bart-large-mnli"}}
    result["summary"] = make_summary(text)
    return result


def analyze_lines(lines: List[str]) -> dict:
    return analyze_text(" ".join([l for l in lines if l.strip()]))
