import asyncio
import backend.core as core

# Simple assertions without pytest

def stub_emotion_high_sad(text, return_all_scores=True):
    return [{"label": "sadness", "score": 0.90}, {"label": "joy", "score": 0.05}]


def stub_emotion_low_scores(text, return_all_scores=True):
    return [{"label": "joy", "score": 0.10}, {"label": "sadness", "score": 0.09}]


def stub_zero_shot_sad(text, candidate_labels, multi_label=False):
    return {"labels": ["Sad/Depressed"], "scores": [0.95]}


async def test_emotion_model_detects_sad():
    core.emotion_classifier = stub_emotion_high_sad
    res = core.analyze_text("My dog died")
    assert res["emotion"] == "Sad", f"expected Sad got {res}"
    assert res["method"].startswith("emotion-model"), res
    print("test_emotion_model_detects_sad passed")


async def test_zero_shot_fallback():
    core.emotion_classifier = stub_emotion_low_scores
    core.zero_shot = stub_zero_shot_sad
    res = core.analyze_text("It was such a loss")
    assert res["emotion"] == "Sad", f"expected Sad got {res}"
    assert res["method"] == "zero-shot", res
    print("test_zero_shot_fallback passed")


async def test_multi_label_aggregation():
    core.emotion_classifier = lambda text, return_all_scores=True: [{"label": "fear", "score": 0.22}, {"label": "surprise", "score": 0.25}, {"label": "joy", "score": 0.08}]
    res = core.analyze_text("I was terrified but also excited")
    assert (res["method"].startswith("emotion-model") or res["method"] == "zero-shot"), res
    assert res["emotion"] in [v["label"] for v in core.EMOTION_MAP.values()], res
    print("test_multi_label_aggregation passed")


async def run_all():
    await test_emotion_model_detects_sad()
    await test_zero_shot_fallback()
    await test_multi_label_aggregation()

if __name__ == "__main__":
    asyncio.run(run_all())
    print("All quick tests passed")
