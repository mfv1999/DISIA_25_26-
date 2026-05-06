from typing import Optional
from pydantic import BaseModel


class AnalyzeResponse(BaseModel):
    original_text: str
    features: dict
    sentiment: Optional[str] = None        # disponible en fases posteriores
    sentiment_score: Optional[float] = None
