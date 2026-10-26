"""
backend/routes/elderreact.py
==============================
ElderReact emotion inference and chat API routes.

These routes are ADDITIVE — they do not modify or replace existing routes
(health.py, inference.py, dialogue.py, demo.py) or the MELD pipeline.

Endpoints
---------
  GET  /health           — updated in backend/main.py lifespan; served here too
  POST /emotion/predict  — multipart upload of audio+video or .npy features
  POST /chat             — text chat, optionally conditioned on emotion context

Startup behaviour
-----------------
The ElderReact service singleton is loaded once via FastAPI lifespan.
Subsequent requests reuse the already-loaded model, scalers, and thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.schemas.elderreact_api import (
    ChatRequest,
    ChatResponse,
    EmotionPredictionResponse,
    HealthDetail,
    HealthResponse,
)
from backend.services.elderreact_service import (
    ELDERREACT_EMOTIONS,
    SUPPORTED_AUDIO_EXT,
    SUPPORTED_VIDEO_EXT,
    get_elderreact_service,
)

log = logging.getLogger(__name__)

router = APIRouter(tags=["ElderReact Emotion"])

_CKPT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "elderreact" / "official" / "checkpoints"


# ══════════════════════════════════════════════════════════════════════════════
# GET /health
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/health/elderreact",
    response_model=HealthResponse,
    summary="ElderReact model and service readiness",
    description=(
        "Returns readiness of the ElderReact model, checkpoints, scalers, "
        "and whether raw-media (proxy) inference is available. "
        "The original MELD /health endpoint is preserved at GET /health."
    ),
)
def health_check() -> HealthResponse:
    svc = get_elderreact_service()

    ckpt_status = {
        "er_a.pkl":    (_CKPT_DIR / "er_a.pkl").exists(),
        "er_b.pkl":    (_CKPT_DIR / "er_b.pkl").exists(),
        "er_c.pkl":    (_CKPT_DIR / "er_c.pkl").exists(),
        "scalers.pkl": (_CKPT_DIR / "scalers.pkl").exists(),
    }

    # Raw-media inference requires librosa + cv2
    raw_ok, raw_detail = True, "librosa and cv2 available"
    try:
        import librosa  # noqa: F401
    except ImportError:
        raw_ok, raw_detail = False, "librosa not installed (pip install librosa)"
    try:
        import cv2  # noqa: F401
    except ImportError:
        raw_ok = False
        raw_detail = "opencv not installed (pip install opencv-python-headless)"

    overall = "healthy" if svc.is_ready else "degraded"

    return HealthResponse(
        status=overall,
        service="H.E.A.R.T. AI — ElderReact Inference Backend",
        version="1.0.0",
        elderreact_model=HealthDetail(
            ready=svc.is_ready,
            detail=(
                f"Loaded: {svc.model_variant} checkpoint, 6-emotion multi-label MLP. "
                "Thresholds tuned on dev split."
                if svc.is_ready
                else f"NOT READY: {svc._load_error}"
            ),
        ),
        checkpoints=ckpt_status,
        raw_media_inference=HealthDetail(ready=raw_ok, detail=raw_detail),
        meld_pipeline=HealthDetail(
            ready=True,
            detail="Existing MELD pipeline unaffected (inference_pipeline.py).",
        ),
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /emotion/predict
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/emotion/predict",
    response_model=EmotionPredictionResponse,
    summary="Multi-label emotion prediction from audio+video",
    description="""
Accept an audio file and a video file (or pre-extracted .npy feature arrays)
and return six independent emotion probabilities.

**Input modes (exactly one must be used):**

| Mode | Fields | Notes |
|------|--------|-------|
| Raw media | `audio_file` + `video_file` | Proxy features — documented distribution mismatch |
| Pre-extracted features | `audio_features` + `video_features` | Exact COVAREP+OpenFace features |
| Mock | `mock=true` | Random output, clearly labelled |

**Output label order:** Anger, Disgust, Fear, Happiness, Sadness, Surprise

Probabilities are INDEPENDENT (multi-label) — they do NOT sum to 1.
Do not collapse to a single argmax label.
""",
    responses={
        200: {"description": "Successful prediction"},
        422: {"description": "Validation error (wrong format, missing inputs, etc.)"},
        503: {"description": "Model not loaded"},
    },
)
async def predict_emotion(
    audio_file:       Optional[UploadFile] = File(None,
        description="Audio clip (.wav, .mp3, .m4a, .flac, .ogg, .aac). Max 30MB."),
    video_file:       Optional[UploadFile] = File(None,
        description="Video clip (.mp4, .mov, .avi, .webm, .mkv, .m4v). Max 50MB."),
    audio_features:   Optional[UploadFile] = File(None,
        description="Pre-extracted audio features as .npy (148-dim COVAREP or 72-dim subset). Max 2MB."),
    video_features:   Optional[UploadFile] = File(None,
        description="Pre-extracted video features as .npy (1418-dim OpenFace or 178-dim subset). Max 2MB."),
    mock:             bool = Form(False,
        description="If true, return random mock output (is_mock=true in response)."),
) -> EmotionPredictionResponse:
    svc = get_elderreact_service()

    # ── Mock mode ────────────────────────────────────────────────────────
    if mock:
        result = svc.predict_mock()
        return _build_response(result)

    # ── Read uploaded bytes ──────────────────────────────────────────────
    audio_bytes      = await audio_file.read()    if audio_file      else None
    video_bytes      = await video_file.read()    if video_file      else None
    audio_npy_bytes  = await audio_features.read() if audio_features else None
    video_npy_bytes  = await video_features.read() if video_features else None

    audio_filename = audio_file.filename     if audio_file      else None
    video_filename = video_file.filename     if video_file      else None

    # Validate .npy extensions
    if audio_features and not Path(audio_features.filename or "").suffix.lower() == ".npy":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="audio_features must be a .npy file.",
        )
    if video_features and not Path(video_features.filename or "").suffix.lower() == ".npy":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="video_features must be a .npy file.",
        )

    # ── Run inference ────────────────────────────────────────────────────
    result = svc.predict(
        audio_bytes=audio_bytes,
        audio_filename=audio_filename,
        video_bytes=video_bytes,
        video_filename=video_filename,
        audio_npy_bytes=audio_npy_bytes,
        video_npy_bytes=video_npy_bytes,
    )

    # Surface service-level errors as proper HTTP responses
    if result.get("error"):
        code = result.pop("_status_code", 500)
        raise HTTPException(status_code=code, detail=result["error"])

    result.pop("_status_code", None)
    return _build_response(result)


def _build_response(result: dict) -> EmotionPredictionResponse:
    return EmotionPredictionResponse(
        is_mock=result["is_mock"],
        model_variant=result["model_variant"],
        emotion_labels=result["emotion_labels"],
        probabilities=result["probabilities"],
        active_emotions=result["active_emotions"],
        thresholds=result["thresholds"],
        reliability_level=result["reliability_level"],
        reliability_note=result["reliability_note"],
        audio_feature_source=result["audio_feature_source"],
        video_feature_source=result["video_feature_source"],
        warnings=result.get("warnings", []),
        error=result.get("error"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# POST /chat
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Affective text chat",
    description="""
Text chat interface. Optionally accepts emotion context from /emotion/predict.

**Current status:** LLM not yet configured.
The endpoint returns a clearly documented placeholder response.
Emotion context is accepted and validated but not yet used to condition generation.

When an LLM is integrated, it will only use emotion context if:
- `context.is_mock` is False
- `context.reliability_level` is "moderate" or "high"
""",
)
async def chat(req: ChatRequest) -> ChatResponse:
    # Validate emotion context if provided
    affective_context: Optional[str] = None
    if req.context is not None:
        ctx = req.context
        if ctx.is_mock:
            affective_context = None   # never use mock context
        elif ctx.reliability_level in ("moderate", "high") and ctx.active_emotions:
            emo_str = ", ".join(
                f"{e} ({ctx.probabilities.get(e, 0):.2f})"
                for e in ctx.active_emotions
            )
            affective_context = (
                f"[{ctx.reliability_level.upper()} RELIABILITY — ElderReact] "
                f"Active emotions: {emo_str}. "
                f"Note: {ctx.reliability_note}"
            )
        # low reliability → affective_context stays None (neutral response)

    return ChatResponse(
        reply=(
            "Thank you for sharing that with me. "
            "I am here to listen and support you. "
            "(LLM not yet configured — this is a placeholder response.)"
        ),
        session_id=req.session_id,
        llm_available=False,
        affective_context=affective_context,
        note=(
            "The LLM integration is not yet configured. "
            "Emotion context is received and validated but not used for generation. "
            "See docs/api-contract.md for integration roadmap."
        ),
    )
