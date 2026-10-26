"""Privacy-Conscious Caregiver Dashboard Schemas.

Provides high-level, longitudinal engagement patterns and ambiguous interaction
indicators without pathologizing internal emotional states or clinical diagnosis.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class InteractionSummaryItem(BaseModel):
    """Aggregated turn summary for caregiver overview."""
    timestamp_iso: str
    engagement_duration_minutes: float
    predominant_expressive_tendency: str = Field(
        ..., description="Broad communicative mood (e.g. calm, expressive, quiet)"
    )
    reliability_indicator: str = Field(
        ..., description="Signal clarity: 'clear_concordant', 'mildly_divergent', 'ambiguous'"
    )
    supportive_note: str = Field(
        ..., description="Non-clinical, privacy-respecting summary note"
    )


class CaregiverDashboardSummary(BaseModel):
    """High-level longitudinal summary for family and caregiver support."""
    elder_pseudonym: str
    reporting_window: str = Field("Past 7 Days", description="Summary timeframe")
    total_interactions_count: int
    total_conversation_minutes: float
    expression_distribution_overview: Dict[str, float] = Field(
        ..., description="Broad percentage distribution over expressive patterns"
    )
    ambiguity_index: float = Field(
        ..., description="Average proportion of interactions with cross-modal divergence [0, 1]"
    )
    suggested_engagement_prompts: List[str] = Field(
        ..., description="Gentle conversation starter suggestions for family visits"
    )
    safety_disclaimer: str = Field(
        default="This summary provides communication patterns for companionship support and does not constitute medical, psychological, or clinical assessment.",
        description="Mandatory Responsible AI disclaimer"
    )
