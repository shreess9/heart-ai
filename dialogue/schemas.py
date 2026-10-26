"""Structured Affective Evidence Schemas using Pydantic.

Defines the contract between multimodal perception, reliability estimation,
and conversational response adaptation.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class ModalityPredictionSchema(BaseModel):
    """Prediction output for an individual modality."""
    modality: Literal["text", "audio", "visual"]
    probabilities: List[float] = Field(..., description="7-class softmax probabilities")
    predicted_emotion: str = Field(..., description="Argmax emotion class label")
    predicted_id: int = Field(..., description="Argmax emotion class index")
    confidence: float = Field(..., description="Maximum class probability in [0, 1]")
    is_mock: bool = Field(False, description="Flag indicating if generated via test harness")


class DisagreementSchema(BaseModel):
    """Cross-modal disagreement metrics."""
    jsd: float = Field(..., description="Generalized 3-distribution JSD (D1) normalized in [0, 1]")
    jsd_pairwise_mean: float = Field(..., description="Mean pairwise JSD (D2) in [0, 1]")
    categorical: int = Field(..., description="D3 categorical conflict: 0 (all agree), 1 (one differs), 2 (all differ)")


class UncertaintySchema(BaseModel):
    """Predictive uncertainty and entropy metrics."""
    predictive_entropy: float = Field(..., description="Shannon entropy in bits [0, log2(7)]")
    normalized_entropy: float = Field(..., description="Normalized entropy bounded in [0, 1]")


class FusedPredictionSchema(BaseModel):
    """Output from multimodal fusion layer."""
    model_name: str = Field(..., description="Fusion model identifier (e.g. baseline, disagreement_aware, reliability_aware)")
    probabilities: List[float] = Field(..., description="Fused 7-class softmax probabilities")
    predicted_emotion: str = Field(..., description="Fused argmax emotion class")
    predicted_id: int = Field(..., description="Fused argmax emotion class index")
    confidence: float = Field(..., description="Raw maximum probability")
    calibrated_confidence: float = Field(..., description="Temperature-scaled confidence")


class StructuredAffectiveEvidence(BaseModel):
    """Complete structured affective evidence package passed to dialogue engine."""
    utterance_text: Optional[str] = Field(None, description="User spoken or typed utterance")
    text: ModalityPredictionSchema
    audio: ModalityPredictionSchema
    visual: ModalityPredictionSchema
    disagreement: DisagreementSchema
    uncertainty: UncertaintySchema
    fused: FusedPredictionSchema
    reliability_level: Literal["high", "moderate", "low"] = Field(
        ..., description="Assessed reliability level governing response policy"
    )
    safety_guidance: str = Field(
        ..., description="Direct instructions for conversational assistant behavior"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
