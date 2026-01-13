import asyncio
import pytest
import backend.main as main
from backend.main import Entry

# Stubs to simulate pipeline outputs without downloading models
def stub_emotion_high_sad(text, return_all_scores=True):
    return [{"label": "sadness", "score": 0.90}, {"label": "joy", "score": 0.05}]

def stub_emotion_low_scores(text, return_all_scores=True):
    return [{"label": "joy", "score": 0.10}, {"label": "sadness", "score": 0.09}]

def stub_zero_shot_sad(text, candidate_labels, multi_label=False):
    return {"labels": ["Sad/Depressed"], "scores": [0.95]}


def test_emotion_model_detects_sad(monkeypatch):
    monkeypatch.setattr(main, "emotion_classifier", stub_emotion_high_sad)
    res = asyncio.run(main.predict_color(Entry(lines=["My dog died"])))
    assert res["emotion"] == "Sad"
    assert res["method"].startswith("emotion-model")


def test_zero_shot_fallback(monkeypatch):
    monkeypatch.setattr(main, "emotion_classifier", stub_emotion_low_scores)
    monkeypatch.setattr(main, "zero_shot", stub_zero_shot_sad)
    res = asyncio.run(main.predict_color(Entry(lines=["It was such a loss"])))
    assert res["emotion"] == "Sad"
    assert res["method"] == "zero-shot"


def test_multi_label_aggregation(monkeypatch):
    # Two labels above MULTI_LABEL_THRESHOLD -> multi aggregation path
    monkeypatch.setattr(main, "emotion_classifier", lambda text, return_all_scores=True: [{"label": "fear", "score": 0.22}, {"label": "surprise", "score": 0.25}, {"label": "joy", "score": 0.08}])
    res = asyncio.run(main.predict_color(Entry(lines=["I was terrified but also excited"])))
    assert res["method"].startswith("emotion-model")
    assert res["emotion"] in [v["label"] for v in main.EMOTION_MAP.values()]