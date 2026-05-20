from pathlib import Path
from datetime import datetime, timezone
import csv
import json

from app.alerts.telegram_alerts import send_drift_alert_if_needed


BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"
PREDICTIONS_LOG = METRICS_DIR / "predictions_log.csv"


def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def log_prediction(result: dict) -> None:
    _ensure_metrics_dir()

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_text": result.get("original_text", ""),
        "predicted_emotion": result.get("prediction", "unknown"),
        "confidence": None,
        "features": json.dumps(result.get("features", {}), ensure_ascii=False),
    }

    file_exists = PREDICTIONS_LOG.exists()

    with PREDICTIONS_LOG.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
    send_drift_alert_if_needed()
        