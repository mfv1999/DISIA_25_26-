from app.ml.inference import predict_emotion
from fastapi import APIRouter, HTTPException, Query, Request

from app.ml.retrain_model import retrain_model_from_feedback
from app.models.schemas import AnalyzeResponse
from app.monitoring.feedback_prometheus_metrics import collect_feedback_metrics
from app.monitoring.feedback_retraining import build_feedback_retraining_dataset
from app.preprocessing.feature_extraction import extract
from app.monitoring.prediction_logger import log_prediction
from app.monitoring.model_metrics import calculate_model_metrics
from fastapi import APIRouter, HTTPException, Query, Request, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from app.monitoring.drift_prometheus_metrics import collect_drift_metrics

from app.monitoring.model_prometheus_metrics import collect_model_metrics
from app.monitoring.feedback_metrics import calculate_feedback_metrics
router = APIRouter()

@router.get("/extract_data")
def extract_data(
    request: Request,
    text: str = Query(..., min_length=1, max_length=10000)
):
    try:
        features = extract(
            text=text,
            lexicons=request.app.state.lexicons,
            negation_window=request.app.state.negation_window,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "original_text": text,
        "features": features,
    }

@router.get("/predict", response_model=AnalyzeResponse)
def analyze(request: Request, text: str = Query(..., min_length=1, max_length=10000)):
    try:
        result = predict_emotion(
            text=text,
            lexicons=request.app.state.lexicons,
            negation_window=request.app.state.negation_window,
        )

        log_prediction(result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(
        original_text=result["original_text"],
        features=result["features"],
        prediction=result["prediction"],
    )
@router.get("/model_metrics")
def model_metrics():
    return calculate_model_metrics()

@router.get("/metrics")
def prometheus_metrics():
    collect_model_metrics()
    collect_drift_metrics()
    collect_feedback_metrics()

    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

from app.models.schemas import FeedbackRequest
from app.monitoring.feedback_logger import log_feedback

@router.post("/feedback")
def register_feedback(feedback: FeedbackRequest):
    log_feedback(
        original_text=feedback.original_text,
        predicted_emotion=feedback.predicted_emotion,
        corrected_emotion=feedback.corrected_emotion,
        source=feedback.source,
    )

    return {
        "message": "Feedback registrado correctamente",
        "data": feedback.model_dump(),
    }


@router.get("/feedback/metrics")
def get_feedback_metrics():
    return calculate_feedback_metrics()

@router.post("/feedback/retraining-dataset")
def generate_feedback_retraining_dataset():
    return build_feedback_retraining_dataset()


@router.post("/feedback/retrain")
def retrain_model_with_feedback():
    return retrain_model_from_feedback()