"""Affective Dialogue API Routes."""

from fastapi import APIRouter, HTTPException
from backend.schemas.api import DialogueRespondRequest, DialogueRespondResponse
from backend.services.pipeline_service import BackendPipelineService

router = APIRouter(prefix="/api/v1/dialogue", tags=["Dialogue"])


@router.post("/respond", response_model=DialogueRespondResponse)
def respond_dialogue(req: DialogueRespondRequest) -> DialogueRespondResponse:
    """Generate reliability-aware companion response."""
    try:
        service = BackendPipelineService.get_instance()
        resp = service.generate_dialogue_reply(
            user_message=req.user_message,
            evidence=req.evidence,
        )
        return DialogueRespondResponse(**resp)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
