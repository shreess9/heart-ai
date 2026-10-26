"""H.E.A.R.T. AI: Research Demonstrator API Endpoints."""

import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel, Field

from backend.services.demo_service import ResearchDemonstratorService, DEMO_SCENARIOS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["Research Demonstrator"])


class DemonstratorAnalyzeRequest(BaseModel):
    """Request payload for the demonstrator analysis endpoint (JSON mode)."""
    text: Optional[str] = Field("How are you feeling today?", description="User conversational text input")
    audio_base64: Optional[str] = Field(None, description="Optional base64 encoded audio clip")
    visual_base64: Optional[str] = Field(None, description="Optional base64 encoded visual image keyframe")
    mode: str = Field("custom", description="Operation mode: 'custom' (Mode 1), 'upload' (Mode 2), 'meld' (Mode 3), or 'demo' (Mode 4)")
    sample_id: Optional[str] = Field(None, description="MELD sample ID when mode is 'meld'")
    scenario_id: Optional[str] = Field(None, description="Demonstration scenario ID when mode is 'demo'")
    scenario_key: Optional[str] = Field(None, description="Alias for scenario_id")
    history: Optional[List[Dict[str, str]]] = Field(None, description="Recent conversation turns for dynamic companion context")


@router.post("/analyze")
async def analyze_conversation(request: Request) -> Dict[str, Any]:
    """Execute end-to-end multimodal perception, disagreement, reliability gating, and response generation.
    
    Accepts both application/json payloads and multipart/form-data file uploads.
    """
    try:
        service = ResearchDemonstratorService.get_instance()
        content_type = request.headers.get("content-type", "")

        if "multipart/form-data" in content_type:
            form = await request.form()
            mode = str(form.get("mode", "upload"))
            text = str(form.get("text", "")).strip()
            sample_id = form.get("sample_id")
            scenario_id = form.get("scenario_id") or form.get("scenario_key")
            history_raw = form.get("history")
            history = None
            if history_raw:
                try:
                    import json
                    history = json.loads(str(history_raw)) if isinstance(history_raw, str) else history_raw
                except Exception as he:
                    logger.warning(f"Could not parse history from form data: {he}")

            audio_bytes = None
            audio_filename = None
            audio_field = form.get("audio")
            if audio_field and hasattr(audio_field, "read"):
                audio_bytes = await audio_field.read()
                audio_filename = getattr(audio_field, "filename", "audio_upload.wav")

            video_bytes = None
            video_filename = None
            video_field = form.get("video")
            if video_field and hasattr(video_field, "read"):
                video_bytes = await video_field.read()
                video_filename = getattr(video_field, "filename", "video_upload.mp4")

            result = service.analyze_conversation(
                text=text,
                mode=mode,
                sample_id=str(sample_id) if sample_id else None,
                scenario_id=str(scenario_id) if scenario_id else None,
                audio_bytes=audio_bytes if (audio_bytes and len(audio_bytes) > 0) else None,
                audio_filename=audio_filename,
                video_bytes=video_bytes if (video_bytes and len(video_bytes) > 0) else None,
                video_filename=video_filename,
                history=history,
            )
            return result

        else:
            # application/json body
            body = await request.json()
            text = body.get("text", "")
            audio_base64 = body.get("audio_base64")
            visual_base64 = body.get("visual_base64")
            mode = body.get("mode", "custom")
            sample_id = body.get("sample_id")
            scenario_id = body.get("scenario_id") or body.get("scenario_key")
            history = body.get("history")

            logger.info(f"POST /api/demo/analyze (json) | mode={mode} | text='{str(text)[:50]}' | sample_id={sample_id} | scenario={scenario_id} | history_len={len(history) if history else 0}")

            result = service.analyze_conversation(
                text=text,
                audio_base64=audio_base64,
                visual_base64=visual_base64,
                mode=mode,
                sample_id=sample_id,
                scenario_id=scenario_id,
                history=history,
            )
            return result

    except HTTPException:
        raise
    except ValueError as ve:
        logger.warning(f"Validation error in analyze endpoint: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error analyzing conversation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/scenarios")
def get_demo_scenarios() -> Dict[str, Any]:
    """Return all predefined demonstration scenarios."""
    return DEMO_SCENARIOS


@router.get("/meld_samples")
def get_meld_samples(limit: int = 0) -> List[Dict[str, Any]]:
    """Return curated list of authentic MELD test set samples."""
    service = ResearchDemonstratorService.get_instance()
    return service.get_meld_samples(limit=limit)


@router.get("/research_summary")
def get_research_summary() -> Dict[str, Any]:
    """Return canonical MELD test set evaluation results for the Research Evidence tab."""
    service = ResearchDemonstratorService.get_instance()
    return service.get_research_summary()


@router.get("/provider_status")
def get_provider_status() -> Dict[str, Any]:
    """Return LLM provider status without exposing secrets."""
    service = ResearchDemonstratorService.get_instance()
    telemetry = service.llm_provider.get_telemetry()
    is_avail = service.llm_provider.is_available()
    return {
        "provider": telemetry.get("provider", "Rule-Based Fallback"),
        "model": telemetry.get("model", "default"),
        "is_available": is_avail,
        "is_llm_generated": telemetry.get("is_llm_generated", False),
    }
