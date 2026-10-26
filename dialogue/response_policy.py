"""Response Policy Engine for Reliability-Aware Elder Companionship.

Evaluates structured affective evidence and assigns appropriate conversational
adaptation directives based on reliability level and safety guardrails.
"""

from typing import Dict, Literal, Tuple
from dialogue.schemas import StructuredAffectiveEvidence, DisagreementSchema, UncertaintySchema, FusedPredictionSchema


class AffectiveResponsePolicy:
    """Evaluates affective evidence and applies Responsible AI conversational rules."""

    def __init__(
        self,
        high_conf_thresh: float = 0.70,
        high_max_jsd: float = 0.12,
        high_max_entropy: float = 0.40,
        mod_conf_thresh: float = 0.45,
        mod_max_jsd: float = 0.22,
        mod_max_entropy: float = 0.70,
    ):
        self.high_conf_thresh = high_conf_thresh
        self.high_max_jsd = high_max_jsd
        self.high_max_entropy = high_max_entropy
        self.mod_conf_thresh = mod_conf_thresh
        self.mod_max_jsd = mod_max_jsd
        self.mod_max_entropy = mod_max_entropy

    def evaluate_reliability_level(
        self,
        confidence: float,
        jsd: float,
        normalized_entropy: float,
        d3_categorical: int,
    ) -> Literal["high", "moderate", "low"]:
        """Determine reliability category from multimodal evidence."""
        # HIGH RELIABILITY: high confidence, low conflict, low entropy, all agree (D3=0)
        if (
            confidence >= self.high_conf_thresh
            and jsd <= self.high_max_jsd
            and normalized_entropy <= self.high_max_entropy
            and d3_categorical == 0
        ):
            return "high"

        # MODERATE RELIABILITY: moderate confidence, moderate conflict, D3 <= 1
        if (
            confidence >= self.mod_conf_thresh
            and jsd <= self.mod_max_jsd
            and normalized_entropy <= self.mod_max_entropy
            and d3_categorical <= 1
        ):
            return "moderate"

        # LOW RELIABILITY (Fallback): High conflict, high uncertainty, or weak confidence
        return "low"

    def generate_safety_guidance(
        self,
        reliability_level: Literal["high", "moderate", "low"],
        fused_emotion: str,
    ) -> str:
        """Generate Responsible AI directives for LLM prompt generation."""
        if reliability_level == "high":
            return (
                f"Modality signals are concordant and reliable ({fused_emotion}). "
                "You may acknowledge the user's emotional tone warmly and empathetically, "
                "while maintaining respectful companionship without claiming medical or clinical certainty."
            )
        elif reliability_level == "moderate":
            return (
                f"Modality signals show mild divergence or moderate uncertainty (suggesting {fused_emotion}). "
                "Use gentle, open-ended supportive language. Avoid strong assertions about how the user feels "
                "and allow them to naturally steer the topic."
            )
        else:
            return (
                "Modality signals show substantial cross-modal disagreement or high uncertainty. "
                "Do NOT assume or state any emotional label. Maintain a warm, supportive, neutral presence "
                "and offer open clarification if appropriate."
            )

    def formulate_adaptation_prompt(
        self,
        evidence: StructuredAffectiveEvidence,
    ) -> str:
        """Generate prompt addendum for LLM dialogue generation."""
        guidance = evidence.safety_guidance
        level = evidence.reliability_level
        emotion = evidence.fused.predicted_emotion

        directive = (
            f"[Affective Evidence State: {level.upper()} RELIABILITY]\n"
            f"[Observed Expression Tendency: {emotion}]\n"
            f"[Safety Directive: {guidance}]\n"
            "Respond naturally to the user following these directives."
        )
        return directive
