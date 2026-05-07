from app.ml.inference import predict_emotion
from fastapi import APIRouter, HTTPException, Query, Request

from app.models.schemas import AnalyzeResponse
from app.preprocessing.feature_extraction import extract

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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    print(result)

    return AnalyzeResponse(
        original_text=result["original_text"],
        features=result["features"],
        prediction=result["prediction"],
    )