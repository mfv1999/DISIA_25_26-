from pathlib import Path
from datetime import datetime, timezone
import csv

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"
FEEDBACK_LOG = METRICS_DIR / "feedback_log.csv"


def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def log_feedback(
    original_text: str,
    predicted_emotion: str,
    corrected_emotion: str,
    source: str = "manual",
) -> None:
    _ensure_metrics_dir()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_text": original_text,
        "predicted_emotion": predicted_emotion,
        "corrected_emotion": corrected_emotion,
        "source": source,
    }

    file_exists = FEEDBACK_LOG.exists()

    with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)