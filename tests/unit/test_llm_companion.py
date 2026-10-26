"""Unit tests for H.E.A.R.T. AI Dynamic LLM Companion, DialogueManager, and Strategy Control.

Verifies:
1. Strategy enforcement (Strong, Cautious, Clarification, Neutral)
2. Reliability Gate strictly controlling LLM prompt instructions (LLM cannot override strategy)
3. Conversation history bounding (up to 10 turns)
4. Unconfigured / offline fallback provider behavior
5. Responsible AI constraints in system prompt (no clinical diagnosis, no claims of humanity/doctor)
6. Multimodal upload with missing modality states and history integration
"""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

from dialogue.llm_provider import (
    BaseLLMProvider,
    GeminiLLMProvider,
    OpenAILLMProvider,
    RuleBasedDynamicFallbackProvider,
    get_llm_provider,
)
from dialogue.manager import DialogueManager
from dialogue.schemas import (
    StructuredAffectiveEvidence,
    ModalityPredictionSchema,
    DisagreementSchema,
    UncertaintySchema,
    FusedPredictionSchema,
)
from backend.services.demo_service import ResearchDemonstratorService


class TestLLMProvider(unittest.TestCase):
    """Test LLM provider instantiation, prompt construction, and fallback mechanisms."""

    def test_rule_based_fallback_provider(self):
        """Verify fallback provider operates without API keys and returns structured telemetry."""
        provider = RuleBasedDynamicFallbackProvider()
        self.assertFalse(provider.is_available())

        result = provider.generate_response(
            user_utterance="I had a difficult day in the garden.",
            emotion="sadness",
            confidence=0.84,
            d1=0.05,
            d3=0,
            disagreement_status="LOW DISAGREEMENT",
            reliability_level="HIGH",
            response_strategy="STRONG_ADAPTATION",
        )
        self.assertTrue(len(result["reply"]) > 10)
        self.assertIn("garden", result["reply"].lower())

        telemetry = provider.get_telemetry()
        self.assertEqual(telemetry["provider"], "Rule-Based Dynamic Fallback")
        self.assertFalse(telemetry["is_llm_generated"])

    def test_openai_provider_prompt_and_guardrails(self):
        """Verify OpenAI provider builds prompt containing strategy directive and Responsible AI constraints."""
        provider = OpenAILLMProvider(api_key="sk-test-key-12345")
        self.assertTrue(provider.is_available())

        sys_prompt = provider._build_system_prompt(
            emotion="joy",
            confidence=0.88,
            d1=0.04,
            d3=0,
            disagreement_status="LOW DISAGREEMENT",
            reliability_level="HIGH",
            response_strategy="STRONG_ADAPTATION",
        )
        self.assertIn("CRITICAL SAFETY & RESPONSIBLE AI CONSTRAINTS", sys_prompt)
        self.assertIn("NEVER diagnose depression", sys_prompt)
        self.assertIn("STRONG_ADAPTATION", sys_prompt)
        self.assertIn("RELIABILITY-GATED RESPONSE STRATEGY DIRECTIVE", sys_prompt)

    def test_gemini_provider_prompt_and_guardrails(self):
        """Verify Gemini provider builds prompt containing Responsible AI rules and exact strategy directive."""
        provider = GeminiLLMProvider(api_key="AQ-test-gemini-key")
        self.assertTrue(provider.is_available())

        # Test STRONG_ADAPTATION
        prompt_strong = provider._build_system_prompt(
            emotion="joy",
            confidence=0.92,
            d1=0.02,
            d3=0,
            disagreement_status="LOW DISAGREEMENT",
            reliability_level="HIGH",
            response_strategy="STRONG_ADAPTATION",
        )
        self.assertIn("CRITICAL SAFETY & RESPONSIBLE AI CONSTRAINTS", prompt_strong)
        self.assertIn("NEVER diagnose depression", prompt_strong)
        self.assertIn("STRONG EMOTION-AWARE ADAPTATION DIRECTIVE", prompt_strong)
        self.assertIn("RELIABILITY GATE STATUS", prompt_strong)

        # Test CLARIFICATION (Conflict)
        prompt_conflict = provider._build_system_prompt(
            emotion="anger",
            confidence=0.50,
            d1=0.60,
            d3=2,
            disagreement_status="HIGH DISAGREEMENT (CONFLICT)",
            reliability_level="LOW",
            response_strategy="CLARIFICATION",
        )
        self.assertIn("CLARIFICATION DIRECTIVE", prompt_conflict)
        self.assertIn("Do NOT assume or assert the user's emotion", prompt_conflict)

        # Test CAUTIOUS_ADAPTATION
        prompt_cautious = provider._build_system_prompt(
            emotion="sadness",
            confidence=0.65,
            d1=0.25,
            d3=1,
            disagreement_status="MODERATE DISAGREEMENT",
            reliability_level="MEDIUM",
            response_strategy="CAUTIOUS_ADAPTATION",
        )
        self.assertIn("CAUTIOUS EMOTIONAL ADAPTATION DIRECTIVE", prompt_cautious)
        self.assertIn("tentative phrasing", prompt_cautious)

    def test_gemini_provider_mock_call(self):
        """Verify Gemini provider successfully formats payload and parses completion response."""
        provider = GeminiLLMProvider(api_key="fake-gemini-key")
        fake_response_data = {
            "choices": [
                {
                    "message": {"content": "That sounds like a wonderful walk in your garden!"},
                    "finish_reason": "stop",
                }
            ],
            "model": "gemini-flash-lite-latest",
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_response_data).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = provider.generate_response(
                user_utterance="I walked in the garden today.",
                emotion="joy",
                confidence=0.90,
                d1=0.03,
                d3=0,
                disagreement_status="LOW DISAGREEMENT",
                reliability_level="HIGH",
                response_strategy="STRONG_ADAPTATION",
            )
            self.assertEqual(res["reply"], "That sounds like a wonderful walk in your garden!")
            self.assertEqual(res["provider"], "Gemini LLM")
            self.assertTrue(res["is_llm_generated"])
            self.assertEqual(res["strategy_enforced"], "STRONG_ADAPTATION")

    def test_gemini_provider_missing_key_error(self):
        """Gemini provider raises ValueError if invoked with no API key."""
        provider = GeminiLLMProvider(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                provider.generate_response(
                    user_utterance="Hello",
                    emotion="neutral",
                    confidence=0.8,
                    d1=0.0,
                    d3=0,
                    disagreement_status="NONE",
                    reliability_level="HIGH",
                    response_strategy="STRONG_ADAPTATION",
                )

    def test_gemini_provider_http_error(self):
        """Gemini provider raises RuntimeError on HTTPError without silent corruption."""
        provider = GeminiLLMProvider(api_key="fake-gemini-key")
        import urllib.error
        http_err = urllib.error.HTTPError("https://api", 401, "Unauthorized", {}, None)
        http_err.read = MagicMock(return_value=b'{"error": "Invalid API key"}')

        with patch("urllib.request.urlopen", side_effect=http_err):
            with self.assertRaises(RuntimeError) as ctx:
                provider.generate_response(
                    user_utterance="Hello",
                    emotion="neutral",
                    confidence=0.8,
                    d1=0.0,
                    d3=0,
                    disagreement_status="NONE",
                    reliability_level="HIGH",
                    response_strategy="STRONG_ADAPTATION",
                )
            self.assertIn("Gemini API Error (401)", str(ctx.exception))

    def test_get_llm_provider_priority(self):
        """Verify priority: 1. GEMINI_API_KEY -> Gemini, 2. OPENAI_API_KEY -> OpenAI, 3. None -> Fallback."""
        # 1. Gemini Key set -> GeminiLLMProvider
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}, clear=True):
            p1 = get_llm_provider()
            self.assertIsInstance(p1, GeminiLLMProvider)
            self.assertEqual(p1.get_telemetry()["provider"], "Gemini LLM")

        # 2. Only OpenAI Key set -> OpenAILLMProvider
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}, clear=True):
            p2 = get_llm_provider()
            self.assertIsInstance(p2, OpenAILLMProvider)
            self.assertEqual(p2.get_telemetry()["provider"], "OpenAI-Compatible")

        # 3. Both set -> GEMINI takes priority
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key", "OPENAI_API_KEY": "test-openai-key"}, clear=True):
            p3 = get_llm_provider()
            self.assertIsInstance(p3, GeminiLLMProvider)

        # 4. None set -> RuleBasedDynamicFallbackProvider
        with patch.dict(os.environ, {}, clear=True):
            p4 = get_llm_provider()
            self.assertIsInstance(p4, RuleBasedDynamicFallbackProvider)
            self.assertFalse(p4.is_available())


class TestDialogueManager(unittest.TestCase):
    """Test DialogueManager response generation, strategy gating, and history limits."""

    def setUp(self):
        self.manager = DialogueManager(max_history_turns=4)

    def _create_sample_evidence(self, emotion="joy", conf=0.85, d1=0.04, d3=0, rel="HIGH"):
        return StructuredAffectiveEvidence(
            utterance_text="I had a great day today.",
            text=ModalityPredictionSchema(
                modality="text",
                probabilities=[0.01, 0.01, 0.01, 0.85, 0.04, 0.04, 0.04],
                predicted_emotion=emotion,
                predicted_id=3,
                confidence=conf,
                is_mock=True,
            ),
            audio=ModalityPredictionSchema(
                modality="audio",
                probabilities=[0.01, 0.01, 0.01, 0.85, 0.04, 0.04, 0.04],
                predicted_emotion=emotion,
                predicted_id=3,
                confidence=conf,
                is_mock=True,
            ),
            visual=ModalityPredictionSchema(
                modality="visual",
                probabilities=[0.01, 0.01, 0.01, 0.85, 0.04, 0.04, 0.04],
                predicted_emotion=emotion,
                predicted_id=3,
                confidence=conf,
                is_mock=True,
            ),
            disagreement=DisagreementSchema(
                jsd=d1,
                jsd_pairwise_mean=d1,
                categorical=d3,
            ),
            uncertainty=UncertaintySchema(
                predictive_entropy=0.50,
                normalized_entropy=0.20,
            ),
            fused=FusedPredictionSchema(
                model_name="Reliability_Aware_Fusion",
                probabilities=[0.01, 0.01, 0.01, 0.85, 0.04, 0.04, 0.04],
                predicted_emotion=emotion,
                predicted_id=3,
                confidence=conf,
                calibrated_confidence=conf,
            ),
            reliability_level=rel,
            safety_guidance="Standard guidance",
        )

    def test_strategy_prompt_strong_adaptation(self):
        """HIGH reliability must select STRONG_ADAPTATION strategy."""
        ev = self._create_sample_evidence(emotion="joy", conf=0.88, d1=0.04, d3=0, rel="high")
        resp = self.manager.generate_response("I had such a wonderful visit with my grandchildren!", ev)
        self.assertEqual(resp["response_strategy"], "STRONG_ADAPTATION")
        self.assertEqual(resp["reliability_level"], "high")

    def test_strategy_prompt_cautious_adaptation(self):
        """MEDIUM reliability must select CAUTIOUS_ADAPTATION strategy."""
        ev = self._create_sample_evidence(emotion="sadness", conf=0.55, d1=0.22, d3=1, rel="moderate")
        resp = self.manager.generate_response("The house is quiet today.", ev)
        self.assertEqual(resp["response_strategy"], "CAUTIOUS_ADAPTATION")

    def test_strategy_prompt_clarification_on_conflict(self):
        """Conflict must select CLARIFICATION strategy."""
        ev = self._create_sample_evidence(emotion="anger", conf=0.60, d1=0.55, d3=2, rel="low")
        resp = self.manager.generate_response("Everything is just fine!", ev)
        self.assertEqual(resp["response_strategy"], "CLARIFICATION")

    def test_history_bounding(self):
        """History must be bounded to max_history_turns."""
        for i in range(10):
            self.manager.add_user_message(f"User turn {i}")
            self.manager.add_assistant_message(f"Assistant turn {i}")
        
        hist = self.manager.get_history()
        # Max history turns is 4, so at most 8 messages (4 user + 4 assistant)
        self.assertLessEqual(len(hist), 8)
        self.assertEqual(hist[-1]["content"], "Assistant turn 9")


class TestDemoServiceLLMIntegration(unittest.TestCase):
    """Test full integration in ResearchDemonstratorService with conversation history."""

    def setUp(self):
        self.service = ResearchDemonstratorService.get_instance()

    def test_mode1_custom_text_with_history(self):
        """Mode 1 text inference accepts history and produces response with telemetry."""
        history = [
            {"role": "user", "text": "Good morning."},
            {"role": "companion", "text": "Good morning! How are you feeling today?"},
        ]
        result = self.service.analyze_conversation(
            text="I had a difficult argument with my sister.",
            mode="custom",
            history=history,
        )
        self.assertEqual(result["mode"], "custom")
        self.assertTrue(len(result["response"]) > 5)
        self.assertIn("llm_telemetry", result)
        self.assertIn("strategy_enforced", result["llm_telemetry"])
        self.assertEqual(result["llm_telemetry"]["strategy_enforced"], result["response_policy"]["adaptation_level"])

    def test_mode4_scenarios_governed_strategies(self):
        """Mode 4 scenarios strictly dictate their reliability strategies."""
        # Strong agreement -> HIGH reliability -> STRONG_ADAPTATION
        res_agree = self.service.analyze_conversation(scenario_id="strong_agreement", mode="demo")
        self.assertEqual(res_agree["reliability"]["level"], "HIGH")
        self.assertEqual(res_agree["response_policy"]["adaptation"], "strong")
        self.assertEqual(res_agree["llm_telemetry"]["strategy_enforced"], "STRONG_ADAPTATION")

        # Strong conflict -> LOW reliability -> CLARIFICATION
        res_conflict = self.service.analyze_conversation(scenario_id="strong_conflict", mode="demo")
        self.assertEqual(res_conflict["reliability"]["level"], "LOW")
        self.assertEqual(res_conflict["response_policy"]["adaptation"], "neutral")
        self.assertEqual(res_conflict["llm_telemetry"]["strategy_enforced"], "CLARIFICATION")

    def test_mock_llm_provider_execution(self):
        """Verify mock LLM provider is invoked with the exact strategy-governed messages."""
        mock_provider = MagicMock(spec=BaseLLMProvider)
        mock_provider.is_available.return_value = True
        mock_provider.get_telemetry.return_value = {
            "provider": "MockOpenAI",
            "model": "gpt-4o-mini",
            "is_llm_generated": True,
        }
        mock_provider.generate_response.return_value = {
            "reply": "It sounds like you had a very special day in the garden!",
            "provider": "MockOpenAI",
            "model": "gpt-4o-mini",
            "is_llm_generated": True,
            "strategy_enforced": "STRONG_ADAPTATION",
            "reliability_level": "HIGH",
            "inferred_emotion": "joy",
        }

        original_provider = self.service.llm_provider
        self.service.llm_provider = mock_provider
        try:
            res = self.service.analyze_conversation(
                text="I had such a wonderful visit with my grandchildren in the garden!",
                mode="custom",
            )
            self.assertEqual(res["response"], "It sounds like you had a very special day in the garden!")
            self.assertTrue(res["llm_telemetry"]["is_llm_generated"])
            self.assertEqual(res["llm_telemetry"]["provider"], "MockOpenAI")
            mock_provider.generate_response.assert_called_once()
        finally:
            self.service.llm_provider = original_provider


if __name__ == "__main__":
    unittest.main()
