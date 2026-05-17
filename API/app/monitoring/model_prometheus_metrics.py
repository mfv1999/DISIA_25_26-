from prometheus_client import Gauge

from app.monitoring.model_metrics import calculate_model_metrics


MODEL_TOTAL_PREDICTIONS = Gauge(
    "model_total_predictions",
    "Número total de predicciones registradas por el modelo"
)

MODEL_CONFIDENCE_AVAILABLE = Gauge(
    "model_confidence_available",
    "Indica si existe métrica de confianza del modelo: 1 sí, 0 no"
)

MODEL_EMOTION_DISTRIBUTION = Gauge(
    "model_emotion_distribution",
    "Distribución de emociones predichas por el modelo",
    ["emotion"]
)


def collect_model_metrics() -> None:
    metrics = calculate_model_metrics()

    MODEL_TOTAL_PREDICTIONS.set(metrics["total_predictions"])

    MODEL_CONFIDENCE_AVAILABLE.set(
        1 if metrics["confidence_available"] else 0
    )

    for emotion, value in metrics["emotion_distribution"].items():
        MODEL_EMOTION_DISTRIBUTION.labels(emotion=emotion).set(value)


def current_model_snapshot() -> dict:
    metrics = calculate_model_metrics()

    return {
        "model": {
            "total_predictions": metrics["total_predictions"],
            "emotion_distribution": metrics["emotion_distribution"],
            "most_frequent_emotion": metrics["most_frequent_emotion"],
            "confidence_available": metrics["confidence_available"],
        }
    }