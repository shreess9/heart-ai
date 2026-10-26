"""Multimodal Emotion & Reliability Inference API Routes."""

from fastapi import APIRouter, HTTPException
from backend.schemas.api import MultimodalInferenceRequest
from backend.services.pipeline_service import BackendPipelineService
from dialogue.schemas import StructuredAffectiveEvidence

router = APIRouter(prefix="/api/v1/inference", tags=["Inference"])


@router.post("/multimodal", response_model=StructuredAffectiveEvidence)
def run_multimodal_inference(req: MultimodalInferenceRequest) -> StructuredAffectiveEvidence:
    """Run multimodal inference, cross-modal disagreement, predictive uncertainty, and fusion."""
    try:
        service = BackendPipelineService.get_instance()
        evidence = service.run_inference(
            text=req.text,
            fusion_type=req.fusion_type,
            mock_mode=req.mock_mode,
        )
        return evidence
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
