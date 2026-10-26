"""Unit tests for Affective Response Policy and Dialogue Manager."""

import unittest
from dialogue.response_policy import AffectiveResponsePolicy
from dialogue.schemas import (
    StructuredAffectiveEvidence,
    ModalityPredictionSchema,
    DisagreementSchema,
    UncertaintySchema,
    FusedPredictionSchema,
)
from dialogue.manager import DialogueManager


class TestDialoguePolicy(unittest.TestCase):

    def setUp(self):
        self.policy = AffectiveResponsePolicy()
        self.manager = DialogueManager(policy=self.policy)

    def test_high_reliability_transition(self):
        """High confidence, low JSD, low entropy, and D3=0 yields HIGH reliability."""
        level = self.policy.evaluate_reliability_level(
            confidence=0.85,
            jsd=0.05,
            normalized_entropy=0.20,
            d3_categorical=0,
        )
        self.assertEqual(level, "high")
        guidance = self.policy.generate_safety_guidance(level, "joy")
        self.assertIn("concordant", guidance)

    def test_moderate_reliability_transition(self):
        """Moderate confidence or mild divergence (D3=1) yields MODERATE reliability."""
        level = self.policy.evaluate_reliability_level(
            confidence=0.55,
            jsd=0.15,
            normalized_entropy=0.50,
            d3_categorical=1,
        )
        self.assertEqual(level, "moderate")
        guidance = self.policy.generate_safety_guidance(level, "sadness")
        self.assertIn("open-ended", guidance)

    def test_low_reliability_transition(self):
        """High conflict (D3=2 or high JSD) yields LOW reliability fallback."""
        level = self.policy.evaluate_reliability_level(
            confidence=0.40,
            jsd=0.35,
            normalized_entropy=0.85,
            d3_categorical=2,
        )
        self.assertEqual(level, "low")
        guidance = self.policy.generate_safety_guidance(level, "neutral")
        self.assertIn("Do NOT assume", guidance)

    def test_dialogue_manager_generation(self):
        """DialogueManager generates conversational response conditioned on evidence."""
        evidence = StructuredAffectiveEvidence(
            utterance_text="I've been feeling a bit lonely today.",
            text=ModalityPredictionSchema(
                modality="text",
                probabilities=[0.05, 0.02, 0.01, 0.02, 0.10, 0.75, 0.05],
                predicted_emotion="sadness",
                predicted_id=5,
                confidence=0.75,
                is_mock=True,
            ),
            audio=ModalityPredictionSchema(
                modality="audio",
                probabilities=[0.05, 0.05, 0.05, 0.05, 0.15, 0.60, 0.05],
                predicted_emotion="sadness",
                predicted_id=5,
                confidence=0.60,
                is_mock=True,
            ),
            visual=ModalityPredictionSchema(
                modality="visual",
                probabilities=[0.05, 0.05, 0.05, 0.05, 0.10, 0.65, 0.05],
                predicted_emotion="sadness",
                predicted_id=5,
                confidence=0.65,
                is_mock=True,
            ),
            disagreement=DisagreementSchema(
                jsd=0.04,
                jsd_pairwise_mean=0.045,
                categorical=0,
            ),
            uncertainty=UncertaintySchema(
                predictive_entropy=0.80,
                normalized_entropy=0.28,
            ),
            fused=FusedPredictionSchema(
                model_name="Model_C_Reliability_Aware_Attention",
                probabilities=[0.02, 0.01, 0.01, 0.02, 0.08, 0.84, 0.02],
                predicted_emotion="sadness",
                predicted_id=5,
                confidence=0.84,
                calibrated_confidence=0.78,
            ),
            reliability_level="high",
            safety_guidance=self.policy.generate_safety_guidance("high", "sadness"),
        )

        resp = self.manager.generate_response("I've been feeling a bit lonely today.", evidence)
        self.assertIn("reply", resp)
        self.assertEqual(resp["reliability_level"], "high")
        self.assertEqual(resp["fused_emotion"], "sadness")


if __name__ == "__main__":
    unittest.main()
