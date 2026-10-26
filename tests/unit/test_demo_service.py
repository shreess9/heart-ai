"""Unit tests for ResearchDemonstratorService."""

import unittest
from backend.services.demo_service import ResearchDemonstratorService, DEMO_SCENARIOS


class TestResearchDemonstratorService(unittest.TestCase):

    def setUp(self):
        self.service = ResearchDemonstratorService.get_instance()

    def test_scenarios_integrity(self):
        """Verify all 4 demo scenarios contain valid distributions and metadata."""
        scenarios = DEMO_SCENARIOS
        self.assertEqual(len(scenarios), 4)
        for sc_id in ["strong_agreement", "moderate_disagreement", "strong_conflict", "low_confidence"]:
            self.assertIn(sc_id, scenarios)
            sc = scenarios[sc_id]
            self.assertEqual(len(sc["text_dist"]), 7)
            self.assertEqual(len(sc["audio_dist"]), 7)
            self.assertEqual(len(sc["visual_dist"]), 7)
            self.assertAlmostEqual(sum(sc["text_dist"]), 1.0, places=2)
            self.assertAlmostEqual(sum(sc["audio_dist"]), 1.0, places=2)
            self.assertAlmostEqual(sum(sc["visual_dist"]), 1.0, places=2)
            self.assertTrue(sc["is_demo"])

    def test_get_meld_samples_curated(self):
        """Verify MELD test samples are loaded with required schema."""
        samples = self.service.get_meld_samples(limit=25)
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("sample_id", s)
            self.assertIn("dialogue_id", s)
            self.assertIn("speaker", s)
            self.assertIn("utterance", s)
            self.assertIn("ground_truth_emotion", s)

    def test_custom_user_utterance_text_only(self):
        """Verify custom utterance analysis reports authentic text perception and absent audio/visual sensors."""
        text = "I feel so grateful for the help from my neighbors today."
        res = self.service.analyze_conversation(text=text, mode="custom")

        self.assertEqual(res["mode"], "custom")
        self.assertEqual(res["analyzed_utterance"], text)
        self.assertTrue(res["modalities"]["text"]["is_available"])
        self.assertFalse(res["modalities"]["audio"]["is_available"])
        self.assertFalse(res["modalities"]["visual"]["is_available"])
        self.assertEqual(res["disagreement"]["d1"], 0.0)
        self.assertIn("response", res)
        self.assertIn("why_response", res)

    def test_meld_utterance_analysis(self):
        """Verify MELD utterance analysis uses authentic sample metadata."""
        res = self.service.analyze_conversation(
            text="Why do all you’re coffee mugs have numbers on the bottom?",
            mode="meld",
            sample_id="dia0_utt0"
        )
        self.assertEqual(res["mode"], "meld")
        self.assertEqual(res["sample_id"], "dia0_utt0")
        self.assertIn("modalities", res)
        self.assertIn("d1", res["disagreement"])
        self.assertIn("d3", res["disagreement"])

    def test_demo_scenario_strong_agreement(self):
        """Scenario 1: Concordant signals produce High Reliability and Strong Adaptation."""
        res = self.service.analyze_conversation(
            text="I had such a wonderful visit with my grandchildren in the garden today!",
            mode="demo",
            scenario_id="strong_agreement"
        )
        self.assertEqual(res["mode"], "demo")
        self.assertEqual(res["reliability"]["level"], "HIGH")
        self.assertEqual(res["response_policy"]["adaptation"], "strong")
        self.assertEqual(res["disagreement"]["level"], "LOW")
        self.assertEqual(res["disagreement"]["d3"], 0)

    def test_demo_scenario_moderate_disagreement(self):
        """Scenario 2: Divergent signals produce Medium Reliability and Cautious Adaptation."""
        res = self.service.analyze_conversation(
            text="The house feels quiet today.",
            mode="demo",
            scenario_id="moderate_disagreement"
        )
        self.assertEqual(res["mode"], "demo")
        self.assertEqual(res["reliability"]["level"], "MEDIUM")
        self.assertEqual(res["response_policy"]["adaptation"], "cautious")
        self.assertEqual(res["disagreement"]["level"], "MODERATE")

    def test_demo_scenario_strong_conflict(self):
        """Scenario 3: Conflicting modalities produce Low Reliability and Clarification."""
        res = self.service.analyze_conversation(
            text="No, everything is fine!",
            mode="demo",
            scenario_id="strong_conflict"
        )
        self.assertEqual(res["mode"], "demo")
        self.assertEqual(res["reliability"]["level"], "LOW")
        self.assertEqual(res["disagreement"]["level"], "HIGH")
        self.assertEqual(res["disagreement"]["d3"], 2)

    def test_demo_scenario_low_confidence(self):
        """Scenario 4: Diffuse distribution produces Low Reliability and Neutral Response."""
        res = self.service.analyze_conversation(
            text="Just another ordinary afternoon.",
            mode="demo",
            scenario_id="low_confidence"
        )
        self.assertEqual(res["mode"], "demo")
        self.assertEqual(res["reliability"]["level"], "LOW")
        self.assertLess(res["fusion"]["confidence"], 0.40)

    def test_safety_and_non_clinical_language(self):
        """Verify generated responses avoid medical or diagnostic language."""
        forbidden_clinical_words = ["depressed", "bipolar", "schizophrenic", "diagnosed", "disorder", "clinical", "dementia"]
        for sc_id in ["strong_agreement", "moderate_disagreement", "strong_conflict", "low_confidence"]:
            res = self.service.analyze_conversation(text="Test input", mode="demo", scenario_id=sc_id)
            reply = res["response"].lower()
            for word in forbidden_clinical_words:
                self.assertNotIn(word, reply, f"Forbidden clinical term '{word}' found in response.")


if __name__ == "__main__":
    unittest.main()
