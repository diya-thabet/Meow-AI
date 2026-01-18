from typing import Optional
from app.core.config import settings
from app.models.base import FERModel
from app.models.vision_only import ResNetBaseline, MockModel
from app.models.vision_llm import VisionLLM_BLIP2

class ModelManager:
    """
    Singleton class to manage the active model.
    Allows hot-swapping models at runtime.
    """
    _instance = None
    
    # Registry of available classes
    MODEL_REGISTRY = {
        "mock": MockModel,
        "resnet": ResNetBaseline,
        "blip2": VisionLLM_BLIP2,
        # Add "llava": LLaVAModel, etc. here later
    }

    def __init__(self):
        self.active_model: Optional[FERModel] = None
        self.active_model_name: str = "None"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name: str):
        """Unloads current model and loads the new one."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not found in registry.")

        print(f"--- Switching to model: {model_name} ---")
        
        # Clean up old model (Python GC usually handles this, but specific GPU cleanup can go here)
        self.active_model = None
        
        # Instantiate and Load new model
        model_class = self.MODEL_REGISTRY[model_name]
        new_model = model_class()
        try:
            new_model.load(settings.DEVICE)
            self.active_model = new_model
            self.active_model_name = model_name
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            self.active_model = None
            self.active_model_name = "Error"
            raise e

    def get_model(self) -> Optional[FERModel]:
        return self.active_model

# Global instance
model_manager = ModelManager.get_instance()