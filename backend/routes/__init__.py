"""Backend API Routers Package."""

from backend.routes.health import router as health_router
from backend.routes.inference import router as inference_router
from backend.routes.dialogue import router as dialogue_router

__all__ = ["health_router", "inference_router", "dialogue_router"]
