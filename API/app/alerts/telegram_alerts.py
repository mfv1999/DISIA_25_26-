from pathlib import Path
from datetime import datetime, timezone
import json

import requests
import os

from app.monitoring.drift_metrics import calculate_drift_metrics

BASE_DIR = Path(__file__).resolve().parent.parent.parent
METRICS_DIR = BASE_DIR / "metrics"
ALERT_STATE_FILE = METRICS_DIR / "alert_state.json"


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def _ensure_metrics_dir() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _load_alert_state() -> dict:
    _ensure_metrics_dir()

    if not ALERT_STATE_FILE.exists():
        return {
            "drift_alert_active": False,
            "last_alert_at": None,
        }

    with ALERT_STATE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_alert_state(state: dict) -> None:
    _ensure_metrics_dir()

    with ALERT_STATE_FILE.open("w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=4)


def send_telegram_message(message: str) -> bool:
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
        },
        timeout=10,
    )

    return response.status_code == 200


def send_drift_alert_if_needed() -> bool:
    metrics = calculate_drift_metrics()
    state = _load_alert_state()

    if not metrics["drift_detected"]:
        if state.get("drift_alert_active"):
            state["drift_alert_active"] = False
            _save_alert_state(state)
        return False

    if state.get("drift_alert_active"):
        return False

    message = (
        "Alerta: se ha detectado deriva en el modelo.\n"
        f"Drift máximo: {metrics['max_drift_score']}\n"
        f"Umbral: {metrics['drift_threshold']}\n"
        f"Distribución actual: {metrics['current_distribution']}"
    )

    sent = send_telegram_message(message)

    if sent:
        state["drift_alert_active"] = True
        state["last_alert_at"] = datetime.now(timezone.utc).isoformat()
        _save_alert_state(state)

    return sent