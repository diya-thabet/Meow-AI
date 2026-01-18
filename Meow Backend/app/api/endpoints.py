from fastapi import APIRouter, UploadFile, File, HTTPException, status
from app.schemas.responses import PredictionResponse, ModelInfoResponse
from app.models.manager import model_manager
from app.services.image_utils import read_image_file
import time

router = APIRouter()

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/info", response_model=ModelInfoResponse)
def get_model_info():
    """Returns the currently active model and available options."""
    return {
        "active_model": model_manager.active_model_name,
        "available_models": list(model_manager.MODEL_REGISTRY.keys()),
        "status": "ready" if model_manager.get_model() else "no_model_loaded"
    }

@router.post("/config/switch-model")
def switch_model(model_name: str):
    """Hot-swap the AI model backend."""
    try:
        model_manager.load_model(model_name)
        return {"message": f"Successfully switched to {model_name}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Main Inference Endpoint.
    1. Checks if model exists.
    2. Processes image.
    3. Returns FER-CE result.
    """
    model = model_manager.get_model()
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="No AI model is currently loaded. Please load a model via /config/switch-model."
        )

    # 1. Read Image
    start_time = time.time()
    try:
        contents = await file.read()
        image = read_image_file(contents)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # 2. Inference
    try:
        result = model.predict(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    
    duration = time.time() - start_time

    # 3. Format Response
    return {
        "model_name": model_manager.active_model_name,
        "emotion_category": result.get("label", "Unknown"),
        "confidence": result.get("confidence", 0.0),
        "explanation": result.get("explanation", None),
        "processing_time": duration,
        "metadata": result.get("metadata", {})
    }