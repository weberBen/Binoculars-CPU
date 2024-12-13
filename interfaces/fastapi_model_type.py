from pydantic import BaseModel
from typing import Dict, List
import datetime

class ModelInfo(BaseModel):
    threshold: float
    observer_model: str
    performer_model: str

class SinglePredictionResponse(BaseModel):
    score: float
    label_class: int  # Note: using class_ since 'class' is a Python keyword
    label: str
    content_length: int
    chunk_count: int

class PredictionResponse(BaseModel):
    total_token_count: int
    total_gpu_time: float
    model: ModelInfo
    request_response_at: datetime.datetime
    request_received_at: datetime.datetime
    results: List[SinglePredictionResponse]