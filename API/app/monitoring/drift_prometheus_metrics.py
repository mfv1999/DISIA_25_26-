from prometheus_client import Gauge

from app.monitoring.drift_metrics import calculate_drift_metrics


MODEL_DRIFT_SCORE = Gauge(
    "model_drift_score",
    "Desviación absoluta entre la distribución actual y la distribución de referencia",
    ["emotion"]
)

MODEL_MAX_DRIFT_SCORE = Gauge(
    "model_max_drift_score",
    "Máxima desviación detectada entre todas las emociones"
)

MODEL_DRIFT_DETECTED = Gauge(
    "model_drift_detected",
    "Indica si se ha detectado deriva: 1 sí, 0 no"
)

MODEL_DRIFT_THRESHOLD = Gauge(
    "model_drift_threshold",
    "Umbral configurado para detectar deriva"
)


def collect_drift_metrics() -> None:
    metrics = calculate_drift_metrics()

    for emotion, drift_score in metrics["drift_score_by_emotion"].items():
        MODEL_DRIFT_SCORE.labels(emotion=emotion).set(drift_score)

    MODEL_MAX_DRIFT_SCORE.set(metrics["max_drift_score"])
    MODEL_DRIFT_DETECTED.set(1 if metrics["drift_detected"] else 0)
    MODEL_DRIFT_THRESHOLD.set(metrics["drift_threshold"])