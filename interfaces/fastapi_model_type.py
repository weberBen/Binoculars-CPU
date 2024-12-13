from pydantic import BaseModel
from typing import Dict

class ModelInfo(BaseModel):
    threshold: float
    observer_model: str
    performer_model: str

class PredictionResponse(BaseModel):
    score: float
    class_: str  # Note: using class_ since 'class' is a Python keyword
    label: str
    total_elapsed_time: float
    total_token_count: int
    content_length: int
    chunk_count: int
    model: ModelInfo