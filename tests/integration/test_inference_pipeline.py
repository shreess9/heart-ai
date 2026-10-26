"""Integration tests for Multimodal Inference Pipeline."""

import unittest
from pipeline.inference_pipeline import MultimodalInferencePipeline
from dialogue.schemas import StructuredAffectiveEvidence


class TestInferencePipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.pipeline_rel = MultimodalInferencePipeline(
            mock_mode=True, fusion_model_type="reliability_aware"
        )
        self.pipeline_base = MultimodalInferencePipeline(
            mock_mode=True, fusion_model_type="baseline"
        )
        self.pipeline_disag = MultimodalInferencePipeline(
            mock_mode=True, fusion_model_type="disagreement_aware"
        )

    def test_end_to_end_reliability_aware_pipeline(self):
        """Pipeline executes end-to-end and returns validated StructuredAffectiveEvidence."""
        utterance = "It's a quiet afternoon, I was just reading a book."
        evidence = self.pipeline_rel.process_utterance(text_input=utterance)

        self.assertIsInstance(evidence, StructuredAffectiveEvidence)
        self.assertEqual(evidence.utterance_text, utterance)

        # Verify text, audio, visual components
        self.assertEqual(len(evidence.text.probabilities), 7)
        self.assertEqual(len(evidence.audio.probabilities), 7)
        self.assertEqual(len(evidence.visual.probabilities), 7)

        # Verify disagreement bounds
        self.assertGreaterEqual(evidence.disagreement.jsd, 0.0)
        self.assertLessEqual(evidence.disagreement.jsd, 1.0)
        self.assertIn(evidence.disagreement.categorical, [0, 1, 2])

        # Verify entropy
        self.assertGreaterEqual(evidence.uncertainty.predictive_entropy, 0.0)
        self.assertGreaterEqual(evidence.uncertainty.normalized_entropy, 0.0)
        self.assertLessEqual(evidence.uncertainty.normalized_entropy, 1.0)

        # Verify fused output
        self.assertEqual(len(evidence.fused.probabilities), 7)
        self.assertIn(evidence.reliability_level, ["high", "moderate", "low"])
        self.assertTrue(len(evidence.safety_guidance) > 10)

    def test_baseline_and_disagreement_pipelines(self):
        """Baseline and Disagreement pipelines execute successfully."""
        ev_base = self.pipeline_base.process_utterance("Hello!")
        self.assertEqual(ev_base.fused.model_name, "Model_A_Baseline_MLP")

        ev_dis = self.pipeline_disag.process_utterance("Hello!")
        self.assertEqual(ev_dis.fused.model_name, "Model_B_Disagreement_Aware_MLP")


if __name__ == "__main__":
    unittest.main()
