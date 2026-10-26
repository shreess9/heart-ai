"""
backend/schemas/elderreact_api.py
===================================
Typed request/response schemas for ElderReact emotion inference endpoints.

These are SEPARATE from the existing MELD api.py schemas and do not modify them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Shared label ordering (must match ElderReact annotation order) ─────────────
EMOTION_LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


# ══════════════════════════════════════════════════════════════════════════════
# Response schemas
# ══════════════════════════════════════════════════════════════════════════════

class EmotionProbabilities(BaseModel):
    """
    Six independent sigmoid probabilities — one per emotion.
    Values are in [0, 1] but do NOT sum to 1 (multi-label, not softmax).
    Do not interpret the maximum as a confident single-label prediction.
    """
    Anger:     float = Field(..., ge=0.0, le=1.0)
    Disgust:   float = Field(..., ge=0.0, le=1.0)
    Fear:      float = Field(..., ge=0.0, le=1.0)
    Happiness: float = Field(..., ge=0.0, le=1.0)
    Sadness:   float = Field(..., ge=0.0, le=1.0)
    Surprise:  float = Field(..., ge=0.0, le=1.0)


class EmotionPredictionResponse(BaseModel):
    """
    Response from POST /emotion/predict.

    Fields
    ------
    is_mock
        True when the server returned random values (e.g., model not loaded).
        Mock responses MUST NOT be used for emotional adaptation.
    model_variant
        Which checkpoint was used: "ER-A", "ER-B", or "ER-C".
        ER-A is the default and has the best calibration (ECE=0.212 on test).
    emotion_labels
        Canonical label order: ["Anger","Disgust","Fear","Happiness","Sadness","Surprise"].
    probabilities
        Six independent probabilities (NOT a probability simplex).
    active_emotions
        Emotions whose probability >= their dev-tuned threshold.
        Empty list means no emotion exceeded its threshold — prediction is inconclusive.
    thresholds
        Dev-tuned per-emotion thresholds used to determine active_emotions.
        Do NOT change these at inference time.
    reliability_level
        "high" | "moderate" | "low".
        "high" is currently unreachable because ER-A ECE (0.212) exceeds the
        high-reliability ECE threshold (0.20). All real predictions will be
        "moderate" or "low" until better-calibrated models are trained.
    reliability_note
        Human-readable explanation for the reliability level.
    audio_feature_source
        Describes how audio features were produced.
        "covarep-148dim-subset-72dim" — exact trained features (from .npy).
        "mel-proxy-72dim"             — Mel-spectrogram proxy (distribution mismatch!).
        "mock"                        — random, not valid.
    video_feature_source
        Describes how video features were produced.
        "openface-1418dim-subset-178dim" — exact trained features (from .npy).
        "cv2-proxy-178dim"               — cv2 statistics proxy (distribution mismatch!).
        "mock"                           — random, not valid.
    warnings
        Processing warnings. MUST be surfaced to the user or developer.
        Proxy features always generate a warning.
    error
        Non-null when inference failed. probabilities will be empty.
    """
    is_mock:               bool
    model_variant:         str
    emotion_labels:        List[str]         = Field(default=EMOTION_LABELS)
    probabilities:         Dict[str, float]
    active_emotions:       List[str]
    thresholds:            Dict[str, float]
    reliability_level:     str               = Field(..., pattern="^(high|moderate|low)$")
    reliability_note:      str
    audio_feature_source:  str
    video_feature_source:  str
    warnings:              List[str]         = Field(default_factory=list)
    error:                 Optional[str]     = None


class HealthDetail(BaseModel):
    """Sub-component health status."""
    ready:   bool
    detail:  str


class HealthResponse(BaseModel):
    """Response from GET /health."""
    status:          str   = Field(..., description="'healthy' or 'degraded'")
    service:         str
    version:         str
    elderreact_model: HealthDetail
    checkpoints:     Dict[str, bool]
    raw_media_inference: HealthDetail
    meld_pipeline:   HealthDetail


# ══════════════════════════════════════════════════════════════════════════════
# Chat schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    """Request for POST /chat."""
    message:    str            = Field(..., min_length=1, max_length=2000,
                                       description="User message text.")
    session_id: Optional[str] = Field(None, description="Conversation session identifier.")
    context:    Optional[EmotionPredictionResponse] = Field(
        None,
        description=(
            "Optional emotion prediction to condition the response. "
            "Only used if reliability_level is 'moderate' or 'high' "
            "and is_mock is False."
        ),
    )


class ChatResponse(BaseModel):
    """Response from POST /chat."""
    reply:             str
    session_id:        Optional[str]
    llm_available:     bool  = Field(
        False,
        description="True when an LLM backend is configured and produced this reply.",
    )
    affective_context: Optional[str] = Field(
        None,
        description="Emotion context used to condition the reply, if any.",
    )
    note:              str = Field(
        "",
        description="Human-readable note about the chat service state.",
    )
