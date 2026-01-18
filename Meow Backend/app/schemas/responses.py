from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class PredictionResponse(BaseModel):
    model_name: str
    emotion_category: str  # e.g., "Happily Surprised"
    confidence: float
    # Specific to Vision-LLM: The generated textual explanation
    explanation: Optional[str] = None 
    # Optional: Latency in seconds
    processing_time: float
    # Optional: Heatmap or extra metadata
    metadata: Optional[Dict[str, Any]] = None

class ModelInfoResponse(BaseModel):
    active_model: Optional[str]
    available_models: List[str]
    status: str

class ErrorResponse(BaseModel):
    detail: str