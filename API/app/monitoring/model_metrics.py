from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"

PREDICTIONS_LOG = METRICS_DIR / "predictions_log.csv"
MODEL_METRICS_FILE = METRICS_DIR / "model_metrics.json"


def calculate_model_metrics() -> dict:

    if not PREDICTIONS_LOG.exists():
        return {
            "total_predictions": 0,
            "emotion_distribution": {},
            "most_frequent_emotion": None,
            "confidence_available": False,
        }

    df = pd.read_csv(PREDICTIONS_LOG)

    if df.empty:
        return {
            "total_predictions": 0,
            "emotion_distribution": {},
            "most_frequent_emotion": None,
            "confidence_available": False,
        }

    total_predictions = len(df)

    emotion_counts = (
        df["predicted_emotion"]
        .value_counts()
        .to_dict()
    )

    emotion_distribution = {
        emotion: round(count / total_predictions, 4)
        for emotion, count in emotion_counts.items()
    }

    most_frequent_emotion = (
        df["predicted_emotion"]
        .value_counts()
        .idxmax()
    )

    metrics = {
        "total_predictions": total_predictions,
        "emotion_distribution": emotion_distribution,
        "most_frequent_emotion": most_frequent_emotion,
        "confidence_available": False,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with MODEL_METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    return metrics