"""Health & System Status API Routes."""

from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> Dict[str, Any]:
    """System health check and component status."""
    return {
        "status": "healthy",
        "service": "H.E.A.R.T. AI Backend",
        "version": "0.1.0",
        "taxonomy": "MELD-7",
        "models": {
            "text": "RoBERTa",
            "audio": "Wav2Vec2.0",
            "visual": "ViT",
            "fusion": ["Model A (Baseline)", "Model B (Disagreement)", "Model C (Reliability)"],
        },
    }
