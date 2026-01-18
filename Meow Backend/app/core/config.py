import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "FER-CE Vision-LLM API"
    API_V1_STR: str = "/api/v1"
    
    # Default model to load on startup (e.g., "mock", "resnet", "blip2")
    DEFAULT_MODEL: str = "mock" 
    
    # Device configuration
    DEVICE: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

    class Config:
        env_file = ".env"

settings = Settings()