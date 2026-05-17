from app.monitoring.model_metrics import calculate_model_metrics
## MODIFICAR CON METRICAS DE DISTRUBUCION REALES O QUE QUERAMOS COMO UMBRALES

REFERENCE_DISTRIBUTION = {
    "joy": 0.35,
    "anger": 0.25,
    "sadness": 0.20,
    "fear": 0.05,
    "surprise": 0.05,
    "disgust": 0.05,
    "anticipation": 0.05,
}

DRIFT_THRESHOLD = 0.20


def calculate_drift_metrics() -> dict:
    metrics = calculate_model_metrics()
    current_distribution = metrics.get("emotion_distribution", {})

    all_emotions = set(REFERENCE_DISTRIBUTION.keys()) | set(current_distribution.keys())

    drift_score_by_emotion = {}

    for emotion in all_emotions:
        reference_value = REFERENCE_DISTRIBUTION.get(emotion, 0.0)
        current_value = current_distribution.get(emotion, 0.0)
        drift_score_by_emotion[emotion] = round(abs(current_value - reference_value), 4)

    max_drift_score = max(drift_score_by_emotion.values()) if drift_score_by_emotion else 0.0

    return {
        "reference_distribution": REFERENCE_DISTRIBUTION,
        "current_distribution": current_distribution,
        "drift_score_by_emotion": drift_score_by_emotion,
        "max_drift_score": round(max_drift_score, 4),
        "drift_detected": max_drift_score >= DRIFT_THRESHOLD,
        "drift_threshold": DRIFT_THRESHOLD,
    }