from prometheus_client import Gauge

from app.monitoring.feedback_metrics import calculate_feedback_metrics

MODEL_FEEDBACK_TOTAL = Gauge(
    "model_feedback_total",
    "Número total de feedbacks registrados"
)

MODEL_FEEDBACK_CORRECTIONS_TOTAL = Gauge(
    "model_feedback_corrections_total",
    "Número total de correcciones realizadas sobre predicciones del modelo"
)

MODEL_FEEDBACK_AGREEMENT_RATE = Gauge(
    "model_feedback_agreement_rate",
    "Porcentaje de feedback que confirma la predicción del modelo"
)

MODEL_FEEDBACK_CORRECTION_RATE = Gauge(
    "model_feedback_correction_rate",
    "Porcentaje de feedback que corrige la predicción del modelo"
)

MODEL_FEEDBACK_CORRECTED_EMOTION_DISTRIBUTION = Gauge(
    "model_feedback_corrected_emotion_distribution",
    "Distribución de emociones corregidas en el feedback",
    ["emotion"]
)


def collect_feedback_metrics() -> None:
    metrics = calculate_feedback_metrics()

    MODEL_FEEDBACK_TOTAL.set(metrics["total_feedback"])
    MODEL_FEEDBACK_CORRECTIONS_TOTAL.set(metrics["total_corrections"])
    MODEL_FEEDBACK_AGREEMENT_RATE.set(metrics["agreement_rate"])
    MODEL_FEEDBACK_CORRECTION_RATE.set(metrics["correction_rate"])

    for emotion, value in metrics["corrected_emotion_distribution"].items():
        MODEL_FEEDBACK_CORRECTED_EMOTION_DISTRIBUTION.labels(
            emotion=emotion
        ).set(value)