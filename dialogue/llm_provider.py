"""H.E.A.R.T. AI: LLM Provider & Dynamic Companion Response Engine.

Integrates:
1. Google Gemini via OpenAI-Compatible REST Endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`)
2. OpenAI / OpenAI-Compatible Endpoints (OpenAI, OpenRouter, Groq, Ollama, LocalAI)
3. Rule-Based Dynamic Fallback Generator (Context-aware offline engine with explicit labeling)

Enforces Reliability-Gated Response Policy:
- The Multimodal Perception + Reliability Layer determines the Allowed Response Strategy BEFORE calling the LLM.
- The LLM Provider is strictly constrained by this strategy directive and Responsible AI elder-care guidelines.
- The LLM does NOT independently classify emotions or override the reliability gate.
"""

import json
import logging
import os
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default Gemini Endpoint
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GEMINI_MODEL = "gemini-flash-lite-latest"


class BaseLLMProvider(ABC):
    """Abstract Base Class for Companion LLM Providers."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider has an active API key/model connection configured."""
        pass

    @abstractmethod
    def get_telemetry(self) -> Dict[str, Any]:
        """Return metadata describing provider status and configuration."""
        pass

    @abstractmethod
    def generate_response(
        self,
        user_utterance: str,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Generate a natural conversational response strictly governed by the reliability strategy."""
        pass


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider via official OpenAI-compatible endpoint.
    
    Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
    Environment Variables: GEMINI_API_KEY, GEMINI_MODEL, GEMINI_BASE_URL
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 15,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("GEMINI_BASE_URL")
            or DEFAULT_GEMINI_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "provider": "Gemini LLM",
            "model": self.model,
            "is_llm_generated": True,
            "base_url": self.base_url,
        }

    def generate_response(
        self,
        user_utterance: str,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not configured in environment.")

        system_prompt = self._build_system_prompt(
            emotion=emotion,
            confidence=confidence,
            d1=d1,
            d3=d3,
            disagreement_status=disagreement_status,
            reliability_level=reliability_level,
            response_strategy=response_strategy,
        )

        messages = [{"role": "system", "content": system_prompt}]

        # Append bounded recent conversation history (up to last 10 turns)
        if conversation_history:
            recent_history = conversation_history[-10:]
            for turn in recent_history:
                role = turn.get("role", "user")
                # Map role names: 'companion'/'assistant' -> 'assistant', 'user' -> 'user'
                mapped_role = "assistant" if role in ["assistant", "companion", "bot"] else "user"
                content = turn.get("content") or turn.get("text") or ""
                if content and content.strip():
                    messages.append({"role": mapped_role, "content": content.strip()})

        # Append current user utterance
        messages.append({"role": "user", "content": user_utterance.strip()})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 300,
        }

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data["choices"][0]
                reply_text = choice["message"]["content"].strip()
                return {
                    "reply": reply_text,
                    "provider": "Gemini LLM",
                    "model": self.model,
                    "is_llm_generated": True,
                    "strategy_enforced": response_strategy,
                    "reliability_level": reliability_level,
                    "inferred_emotion": emotion,
                }
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            logger.error(f"Gemini API HTTPError {http_err.code}: {err_body}")
            raise RuntimeError(f"Gemini API Error ({http_err.code}): {err_body}")
        except urllib.error.URLError as url_err:
            logger.error(f"Gemini API Connection/URL Error: {url_err}")
            raise RuntimeError(f"Gemini Network Connection Error: {url_err.reason}")
        except Exception as e:
            logger.error(f"Gemini API Unexpected Error: {e}")
            raise RuntimeError(f"Gemini API Error: {str(e)}")

    def _build_system_prompt(
        self,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
    ) -> str:
        strategy_upper = response_strategy.upper()
        if "STRONG" in strategy_upper:
            strategy_instruction = (
                "STRONG EMOTION-AWARE ADAPTATION DIRECTIVE:\n"
                f"- High reliability concordant evidence detected '{emotion}'.\n"
                "- You ARE authorized to warmly acknowledge the emotional tone, offer attuned comfort or share joy, "
                "and ask a supportive, empathetic follow-up question."
            )
        elif "CAUTIOUS" in strategy_upper:
            strategy_instruction = (
                "CAUTIOUS EMOTIONAL ADAPTATION DIRECTIVE:\n"
                f"- Moderate reliability with mild divergence detected '{emotion}'.\n"
                "- Use gentle, tentative phrasing. Do NOT assert the inferred emotion as certainty or fact. "
                "Keep the topic open-ended and let the user guide the emotional depth."
            )
        elif "CLARIFICATION" in strategy_upper:
            strategy_instruction = (
                "CLARIFICATION DIRECTIVE (CROSS-MODAL CONFLICT DETECTED):\n"
                "- High incongruity between verbal and non-verbal cues. Do NOT assume or assert the user's emotion.\n"
                "- Respond naturally to their spoken words and gently ask an open clarification question to understand how they are feeling."
            )
        else:
            strategy_instruction = (
                "NEUTRAL SUPPORTIVE DIRECTIVE (HIGH UNCERTAINTY):\n"
                "- Emotion evidence is diffuse or uncertain. Provide a warm, respectful, neutral conversational presence "
                "without making assumptions about their emotional state."
            )

        return (
            "You are H.E.A.R.T. AI, an emotion-aware conversational companion designed for respectful interaction with older adults.\n\n"
            "CRITICAL SAFETY & RESPONSIBLE AI CONSTRAINTS:\n"
            "1. NEVER diagnose depression, anxiety, dementia, cognitive decline, or any medical/psychiatric condition.\n"
            "2. NEVER claim to be human, a doctor, or a therapist. Never foster unhealthy emotional dependency.\n"
            "3. Be warm, concise, patient, and respectful (1 to 3 sentences max). Never be condescending or patronizing.\n"
            "4. NEVER expose internal model metrics (do NOT mention D1, D2, D3, entropy, or probability values in your reply).\n"
            "5. Strictly follow the response adaptation directive below determined by the H.E.A.R.T. AI Reliability Gate.\n\n"
            f"RELIABILITY GATE STATUS:\n"
            f"- Reliability Level: {reliability_level.upper()}\n"
            f"- Allowed Strategy: {response_strategy}\n\n"
            f"{strategy_instruction}"
        )


class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI / OpenAI-compatible endpoint provider (e.g. OpenAI, OpenRouter, Groq, Ollama, LocalAI)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 15,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("LLM_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL") or "gpt-4o-mini"
        self.timeout = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "provider": "OpenAI-Compatible",
            "model": self.model,
            "is_llm_generated": True,
            "base_url": self.base_url,
        }

    def generate_response(
        self,
        user_utterance: str,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("LLM API key not configured.")

        system_prompt = self._build_system_prompt(
            emotion=emotion,
            confidence=confidence,
            d1=d1,
            d3=d3,
            disagreement_status=disagreement_status,
            reliability_level=reliability_level,
            response_strategy=response_strategy,
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            recent_history = conversation_history[-10:]
            for turn in recent_history:
                role = turn.get("role", "user")
                mapped_role = "assistant" if role in ["assistant", "companion", "bot"] else "user"
                content = turn.get("content") or turn.get("text") or ""
                if content and content.strip():
                    messages.append({"role": mapped_role, "content": content.strip()})

        messages.append({"role": "user", "content": user_utterance.strip()})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,
        }

        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                choice = resp_data["choices"][0]
                reply_text = choice["message"]["content"].strip()
                return {
                    "reply": reply_text,
                    "provider": "OpenAI-Compatible",
                    "model": self.model,
                    "is_llm_generated": True,
                    "strategy_enforced": response_strategy,
                    "reliability_level": reliability_level,
                    "inferred_emotion": emotion,
                }
        except urllib.error.HTTPError as http_err:
            err_body = http_err.read().decode("utf-8", errors="ignore")
            logger.error(f"OpenAI API HTTPError {http_err.code}: {err_body}")
            raise RuntimeError(f"OpenAI API Error ({http_err.code}): {err_body}")
        except Exception as e:
            logger.error(f"OpenAI API Connection Error: {e}")
            raise RuntimeError(f"OpenAI Connection Error: {e}")

    def _build_system_prompt(
        self,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
    ) -> str:
        return (
            "You are H.E.A.R.T. AI, an empathetic, respectful conversational companion for older adults.\n\n"
            "CRITICAL SAFETY & RESPONSIBLE AI CONSTRAINTS:\n"
            "- NEVER diagnose depression, cognitive decline, anxiety, or any medical/psychiatric condition.\n"
            "- NEVER claim to be human, a doctor, or a therapist. Never foster emotional dependency.\n"
            "- Respond with warmth, patience, clarity, and grounded conversational support (1-3 sentences).\n\n"
            "RELIABILITY-GATED RESPONSE STRATEGY DIRECTIVE:\n"
            f"- Inferred Affect: '{emotion}' (Confidence: {confidence*100:.1f}%, Disagreement: {disagreement_status}, D1={d1:.3f}, D3={d3})\n"
            f"- Reliability Gate Level: {reliability_level.upper()}\n"
            f"- Allowed Strategy: {response_strategy}\n\n"
            "MANDATORY STRATEGY BEHAVIOR:\n"
            "- If STRONG_ADAPTATION (High Reliability): You are authorized to acknowledge the user's emotional state warmly and empathetically. Offer attuned comfort or share joy, followed by a gentle supportive question.\n"
            "- If CAUTIOUS_ADAPTATION (Medium Reliability): Use gentle, tentative phrasing. Do NOT state the emotion as certainty. Keep the topic open and let the user guide the tone.\n"
            "- If CLARIFICATION (Low Reliability / Cross-Modal Conflict): Do NOT assume any emotional state. Acknowledge the user's spoken words and gently ask an open-ended question to let them clarify how they are feeling.\n"
            "- If NEUTRAL_RESPONSE (Low Reliability / High Uncertainty): Provide a warm, supportive, neutral presence without assuming an emotion."
        )


class RuleBasedDynamicFallbackProvider(BaseLLMProvider):
    """Context-aware fallback generator when no external LLM API key is configured.
    
    Dynamically crafts personalized older-adult companion responses conditioned on:
    1. Exact user utterance lexical content
    2. Modality emotion evidence & calibrated confidence
    3. Cross-modal disagreement (D1, D3)
    4. Reliability gate response policy tier (Strong, Cautious, Clarification, Neutral)
    """

    def is_available(self) -> bool:
        return False

    def get_telemetry(self) -> Dict[str, Any]:
        return {
            "provider": "Rule-Based Dynamic Fallback",
            "model": "rule-based-affective-policy-v2",
            "is_llm_generated": False,
        }

    def generate_response(
        self,
        user_utterance: str,
        emotion: str,
        confidence: float,
        d1: float,
        d3: int,
        disagreement_status: str,
        reliability_level: str,
        response_strategy: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        u_clean = user_utterance.strip().lower()
        strategy = response_strategy.upper()

        # Extract contextual topics if present
        topic = "what you shared"
        if any(w in u_clean for w in ["grandchild", "grandchildren", "grandson", "granddaughter"]):
            topic = "your time with your grandchildren"
        elif any(w in u_clean for w in ["garden", "flowers", "plants"]):
            topic = "your garden"
        elif any(w in u_clean for w in ["family", "gathering", "dinner", "visit"]):
            topic = "your family gathering"
        elif any(w in u_clean for w in ["daughter", "son", "kids", "children"]):
            topic = "your family"
        elif any(w in u_clean for w in ["house", "home", "quiet"]):
            topic = "the house being quiet"
        elif any(w in u_clean for w in ["walk", "park", "outside"]):
            topic = "your walk outside"

        # Tier 1: STRONG EMOTION-AWARE ADAPTATION (High Reliability)
        if "STRONG" in strategy:
            if emotion == "joy":
                reply = f"I'm so glad to hear that! It sounds like {topic} brought you real happiness today. What made it feel so special?"
            elif emotion in ["sadness", "grief"]:
                reply = f"I hear you, and I'm right here with you. It sounds like {topic} is feeling heavy today. Please take all the time you need—would you like to talk more about it?"
            elif emotion == "fear":
                reply = f"I understand why that might feel unsettling. You are in a safe space right now. How can I best support you with {topic}?"
            elif emotion == "anger":
                reply = f"I can hear the frustration in what you shared about {topic}, and that is completely understandable. Would it help to talk through it together?"
            elif emotion == "surprise":
                reply = f"That really sounds like quite an unexpected moment with {topic}! How are you feeling as you take it in?"
            else:
                reply = f"Thank you for sharing that with me. I always appreciate our conversations about {topic}—what else is on your mind today?"

        # Tier 2: CAUTIOUS EMOTIONAL ADAPTATION (Medium Reliability)
        elif "CAUTIOUS" in strategy:
            if emotion in ["sadness", "fear"]:
                reply = f"It sounds like you might have something on your mind regarding {topic}. How are you feeling about everything right now?"
            elif emotion == "joy":
                reply = f"It seems like things might be going well with {topic}. Would you like to tell me a little more about how your day is going?"
            else:
                reply = f"Thank you for sharing your thoughts about {topic}. How is the rest of your day shaping up?"

        # Tier 3: CLARIFICATION (Low Reliability due to Cross-Modal Conflict)
        elif "CLARIFICATION" in strategy:
            reply = f"I want to make sure I understand you properly regarding {topic}. How are you feeling about things right now?"

        # Tier 4: NEUTRAL SUPPORTIVE (Low Reliability due to High Uncertainty)
        else:
            reply = f"I'm listening closely to your thoughts about {topic}. What would you like to explore next?"

        return {
            "reply": reply,
            "provider": "Rule-Based Dynamic Fallback",
            "model": "rule-based-affective-policy-v2",
            "is_llm_generated": False,
            "strategy_enforced": response_strategy,
            "reliability_level": reliability_level,
            "inferred_emotion": emotion,
        }


def get_llm_provider() -> BaseLLMProvider:
    """Factory function with explicit provider priority:
    1. GEMINI_API_KEY -> GeminiLLMProvider
    2. OPENAI_API_KEY / LLM_API_KEY -> OpenAILLMProvider
    3. No key -> RuleBasedDynamicFallbackProvider
    """
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key and gemini_key.strip():
        gemini_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        gemini_base_url = os.environ.get("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL)
        return GeminiLLMProvider(api_key=gemini_key.strip(), base_url=gemini_base_url, model=gemini_model)

    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if openai_key and openai_key.strip():
        return OpenAILLMProvider(api_key=openai_key.strip())

    return RuleBasedDynamicFallbackProvider()
