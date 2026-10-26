"""FastAPI Backend Application Factory for H.E.A.R.T. AI."""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.routes.health import router as health_router
from backend.routes.inference import router as inference_router
from backend.routes.dialogue import router as dialogue_router
from backend.routes.demo import router as demo_router
from backend.routes.elderreact import router as elderreact_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    """Load ElderReact service at startup; release at shutdown."""
    from backend.services.elderreact_service import get_elderreact_service
    svc = get_elderreact_service()
    if svc.is_ready:
        log.info("ElderReact service ready (%s).", svc.model_variant)
    else:
        log.warning("ElderReact service NOT ready: %s", svc._load_error)
    yield
    log.info("H.E.A.R.T. AI backend shutting down.")

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    """Instantiate and configure FastAPI application."""
    app = FastAPI(
        title="H.E.A.R.T. AI Backend Service",
        description="Human-centered Emotion-Aware Reliability-aware Technology for AI Companionship API",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS for web / desktop / local frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include route modules
    app.include_router(health_router)
    app.include_router(inference_router)
    app.include_router(dialogue_router)
    app.include_router(demo_router)
    app.include_router(elderreact_router)  # ElderReact multi-label emotion + chat

    # Mount static assets directory if it exists
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/", include_in_schema=False)
        @app.get("/demo", include_in_schema=False)
        async def serve_demo():
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                return FileResponse(str(index_path))
            return {"message": "H.E.A.R.T. AI Demonstrator UI building..."}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
