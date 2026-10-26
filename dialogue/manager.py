"""Dialogue Manager for Affective Conversational Companion.

Orchestrates affective evidence, reliability-aware response strategy,
multi-turn conversation history, and dynamic LLM generation.
"""

from typing import Any, Dict, List, Optional
from dialogue.schemas import StructuredAffectiveEvidence
from dialogue.response_policy import AffectiveResponsePolicy
from dialogue.llm_provider import BaseLLMProvider, get_llm_provider


class DialogueManager:
    """Conversational dialogue manager for older adult companionship."""

    def __init__(
        self,
        policy: Optional[AffectiveResponsePolicy] = None,
        provider: Optional[BaseLLMProvider] = None,
        max_history_turns: int = 10,
    ):
        self.policy = policy or AffectiveResponsePolicy()
        self.provider = provider or get_llm_provider()
        self.max_history_turns = max_history_turns
        self.conversation_history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        """Add user message to conversation history."""
        if message and message.strip():
            self.conversation_history.append({"role": "user", "content": message.strip()})
            self._trim_history()

    def add_assistant_message(self, message: str) -> None:
        """Add assistant reply to conversation history."""
        if message and message.strip():
            self.conversation_history.append({"role": "assistant", "content": message.strip()})
            self._trim_history()

    def _trim_history(self) -> None:
        """Keep history bounded to max_history_turns."""
        if len(self.conversation_history) > self.max_history_turns * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_turns * 2):]

    def clear_history(self) -> None:
        """Reset conversation history."""
        self.conversation_history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        """Return shallow copy of current dialogue history."""
        return list(self.conversation_history)

    def generate_response(
        self,
        user_message: str,
        evidence: StructuredAffectiveEvidence,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate conversational response conditioned on affective evidence and reliability level."""
        level = evidence.reliability_level.upper()
        emotion = evidence.fused.predicted_emotion
        confidence = evidence.fused.calibrated_confidence
        d1 = evidence.disagreement.jsd
        d3 = evidence.disagreement.categorical
        disagree_status = f"D1={d1:.3f}, D3={d3}"

        # Determine strategy from policy
        if d3 == 2 and d1 >= 0.35:
            response_strategy = "CLARIFICATION"
        elif level == "HIGH":
            response_strategy = "STRONG_ADAPTATION"
        elif level in ["MODERATE", "MEDIUM"]:
            response_strategy = "CAUTIOUS_ADAPTATION"
        else:
            response_strategy = "NEUTRAL_RESPONSE"

        hist = history if history is not None else self.conversation_history

        result = self.provider.generate_response(
            user_utterance=user_message,
            emotion=emotion,
            confidence=confidence,
            d1=d1,
            d3=d3,
            disagreement_status=disagree_status,
            reliability_level=level,
            response_strategy=response_strategy,
            conversation_history=hist,
        )

        reply_text = result.get("reply", "I am listening closely. What would you like to talk about today?")
        
        # Update internal history
        self.add_user_message(user_message)
        self.add_assistant_message(reply_text)

        return {
            "reply": reply_text,
            "reliability_level": evidence.reliability_level,
            "fused_emotion": emotion,
            "response_strategy": response_strategy,
            "safety_guidance": evidence.safety_guidance,
            "provider": result.get("provider", "Rule-Based Fallback"),
            "model": result.get("model", "default"),
            "is_llm_generated": result.get("is_llm_generated", False),
        }
