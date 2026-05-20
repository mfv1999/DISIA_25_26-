from typing import Optional
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    original_text: str
    features: dict
    prediction: str


class FeedbackRequest(BaseModel):
    original_text: str
    predicted_emotion: str
    corrected_emotion: str
    source: str = "manual"
