from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"

FEEDBACK_LOG = METRICS_DIR / "feedback_log.csv"
FEEDBACK_METRICS_FILE = METRICS_DIR / "feedback_metrics.json"


def calculate_feedback_metrics() -> dict:
    if not FEEDBACK_LOG.exists():
        return {
            "total_feedback": 0,
            "total_corrections": 0,
            "agreement_rate": 0.0,
            "correction_rate": 0.0,
            "corrected_emotion_distribution": {},
            "most_corrected_emotion": None,
        }

    df = pd.read_csv(FEEDBACK_LOG)

    if df.empty:
        return {
            "total_feedback": 0,
            "total_corrections": 0,
            "agreement_rate": 0.0,
            "correction_rate": 0.0,
            "corrected_emotion_distribution": {},
            "most_corrected_emotion": None,
        }

    total_feedback = len(df)

    total_corrections = int(
        (df["predicted_emotion"] != df["corrected_emotion"]).sum()
    )

    total_agreements = int(
        (df["predicted_emotion"] == df["corrected_emotion"]).sum()
    )

    correction_rate = round(total_corrections / total_feedback, 4)
    agreement_rate = round(total_agreements / total_feedback, 4)

    corrected_counts = (
        df["corrected_emotion"]
        .value_counts()
        .to_dict()
    )

    corrected_emotion_distribution = {
        emotion: round(count / total_feedback, 4)
        for emotion, count in corrected_counts.items()
    }

    most_corrected_emotion = (
        df["corrected_emotion"]
        .value_counts()
        .idxmax()
    )

    metrics = {
        "total_feedback": total_feedback,
        "total_corrections": total_corrections,
        "agreement_rate": agreement_rate,
        "correction_rate": correction_rate,
        "corrected_emotion_distribution": corrected_emotion_distribution,
        "most_corrected_emotion": most_corrected_emotion,
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with FEEDBACK_METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=4)

    return metrics