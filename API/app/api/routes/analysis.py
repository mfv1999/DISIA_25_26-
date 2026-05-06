from fastapi import APIRouter, HTTPException, Query, Request

from app.models.schemas import AnalyzeResponse
from app.preprocessing.feature_extraction import extract

router = APIRouter()

@router.get("/extract_data", response_model=AnalyzeResponse)
def analyze(request: Request, text: str = Query(..., min_length=1, max_length=10000)):
    try:
        features = extract(
            text,
            request.app.state.lexicons,
            request.app.state.negation_window,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(original_text=text, features=features)