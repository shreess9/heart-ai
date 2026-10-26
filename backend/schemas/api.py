"""FastAPI Request and Response Models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from dialogue.schemas import StructuredAffectiveEvidence


class MultimodalInferenceRequest(BaseModel):
    """Request payload for multimodal inference endpoint."""
    text: str = Field(..., description="User text utterance")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio waveform (.wav)")
    visual_base64: Optional[str] = Field(None, description="Base64 encoded visual keyframe image")
    fusion_type: str = Field("reliability_aware", description="Fusion model type: baseline, disagreement_aware, reliability_aware")
    mock_mode: bool = Field(True, description="Execute via test harness")


class DialogueRespondRequest(BaseModel):
    """Request payload for dialogue conversational response."""
    user_message: str = Field(..., description="User message string")
    evidence: Optional[StructuredAffectiveEvidence] = Field(None, description="Pre-computed affective evidence")
    session_id: Optional[str] = Field(None, description="Conversation session identifier")


class DialogueRespondResponse(BaseModel):
    """Response payload for companion reply."""
    reply: str
    reliability_level: str
    fused_emotion: str
    safety_guidance: str
    is_mock: bool = True
