import json
import asyncio
import itertools
from backend.main import predict_color, Entry
import backend.main as main

# Simple calibration script: grid-search a few thresholds on the sample dataset.
# WARNING: Running this will call the real models and may download them if not present.

DATA_PATH = "../data/calibration_samples.json"

DEFAULT_GRID = {
    "EMOTION_CONFIDENCE_THRESHOLD": [0.30, 0.35, 0.40, 0.45, 0.50],
    "MULTI_LABEL_THRESHOLD": [0.15, 0.20, 0.25],
    "PROMOTE_LABEL_BOOST": [0.00, 0.02, 0.04]
}


def load_samples(path=DATA_PATH):
    with open(path, "r") as f:
        return json.load(f)


async def score_config(samples):
    correct = 0
    total = len(samples)
    for s in samples:
        pred = await predict_color(Entry(lines=[s["text"]]))
        # Compare raw_emotion (mapped key) to label
        if pred.get("raw_emotion") == s["label"]:
            correct += 1
    return correct, total


def run_grid(grid=DEFAULT_GRID):
    samples = load_samples()
    best = None
    for t_conf, multi_t, boost in itertools.product(grid["EMOTION_CONFIDENCE_THRESHOLD"], grid["MULTI_LABEL_THRESHOLD"], grid["PROMOTE_LABEL_BOOST"]):
        # Set globals
        main.EMOTION_CONFIDENCE_THRESHOLD = t_conf
        main.MULTI_LABEL_THRESHOLD = multi_t
        main.PROMOTE_LABEL_BOOST = boost
        print(f"Testing config: EMOTION_CONFIDENCE_THRESHOLD={t_conf}, MULTI_LABEL_THRESHOLD={multi_t}, PROMOTE_LABEL_BOOST={boost}")
        correct, total = asyncio.run(score_config(samples))
        acc = correct / total
        print(f"  Accuracy: {correct}/{total} = {acc:.2%}\n")
        if best is None or acc > best[0]:
            best = (acc, t_conf, multi_t, boost)
    print("Best config:")
    print(best)
    return best


if __name__ == "__main__":
    print("WARNING: This will use the real models and may download them if not present.")
    run_grid()