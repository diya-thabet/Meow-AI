from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import endpoints
from app.models.manager import model_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the default model defined in .env
    print("--- Starting FER-CE Backend ---")
    try:
        model_manager.load_model(settings.DEFAULT_MODEL)
    except Exception as e:
        print(f"Warning: Could not load default model: {e}")
        # We don't crash, we just start with no model active
    yield
    # Shutdown logic (if any)
    print("--- Shutting down ---")

app = FastAPI(
    title=settings.PROJECT_NAME,
    lifespan=lifespan
)

# CORS (Allow your frontend to talk to this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(endpoints.router, prefix=settings.API_V1_STR)

@app.get("/")
def root():
    return {"message": "Welcome to FER-CE Vision-LLM API. Go to /docs for Swagger UI."}