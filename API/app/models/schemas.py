from typing import Optional
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    original_text: str
    features: dict
    prediction: str
