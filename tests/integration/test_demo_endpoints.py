"""Integration tests for H.E.A.R.T. AI Research Demonstrator Endpoints."""

import unittest
from fastapi.testclient import TestClient
from backend.main import create_app


class TestDemoEndpoints(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_demo_index_page(self):
        """GET / and GET /demo return HTML 200."""
        r1 = self.client.get("/")
        self.assertEqual(r1.status_code, 200)
        self.assertIn("H.E.A.R.T. AI", r1.text)

        r2 = self.client.get("/demo")
        self.assertEqual(r2.status_code, 200)
        self.assertIn("Reliability-Aware Multimodal Emotion Demonstrator", r2.text)

    def test_demo_scenarios_endpoint(self):
        """GET /api/demo/scenarios returns predefined demonstration scenarios."""
        resp = self.client.get("/api/demo/scenarios")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("strong_agreement", data)
        self.assertIn("moderate_disagreement", data)
        self.assertIn("strong_conflict", data)
        self.assertIn("low_confidence", data)

        for key, sc in data.items():
            self.assertIn("title", sc)
            self.assertIn("expected_reliability", sc)
            self.assertIn("expected_adaptation", sc)
            self.assertIn("is_demo", sc)
            self.assertTrue(sc["is_demo"])

    def test_demo_meld_samples_endpoint(self):
        """GET /api/demo/meld_samples returns real MELD test set items."""
        resp = self.client.get("/api/demo/meld_samples?limit=20")
        self.assertEqual(resp.status_code, 200)
        samples = resp.json()
        self.assertGreaterEqual(len(samples), 1)

        first = samples[0]
        self.assertIn("sample_id", first)
        self.assertIn("dialogue_id", first)
        self.assertIn("speaker", first)
        self.assertIn("utterance", first)
        self.assertIn("ground_truth_emotion", first)

    def test_demo_research_summary_endpoint(self):
        """GET /api/demo/research_summary returns canonical MELD test set benchmarks."""
        resp = self.client.get("/api/demo/research_summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["dataset"], "MELD Test Set (N=2610)")
        self.assertIn("benchmarks", data)
        self.assertIn("policy_simulation", data)

        benchmarks = {b["model"]: b for b in data["benchmarks"]}
        self.assertIn("Model A (Early Concat)", benchmarks)
        self.assertAlmostEqual(benchmarks["Model A (Early Concat)"]["accuracy"], 0.6801, places=4)

        policies = {p["policy"]: p for p in data["policy_simulation"]}
        self.assertIn("Full Reliability + Entropy", policies)
        self.assertAlmostEqual(policies["Full Reliability + Entropy"]["strong_adaptation_acc"], 0.8702, places=4)

    def test_analyze_custom_user_utterance(self):
        """Mode 1: POST /api/demo/analyze with custom text returns exact text and honest modality flags."""
        custom_sentence = "I had a wonderful day in the park with my old friends."
        payload = {
            "text": custom_sentence,
            "mode": "custom"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "custom")
        self.assertEqual(data["analyzed_utterance"], custom_sentence)
        self.assertEqual(data["input_text"], custom_sentence)

        # Modality honesty
        self.assertTrue(data["modalities"]["text"]["is_available"])
        self.assertFalse(data["modalities"]["audio"]["is_available"])
        self.assertFalse(data["modalities"]["visual"]["is_available"])

        self.assertIn("response", data)
        self.assertIn("why_response", data)

    def test_analyze_unique_utterances_not_stale(self):
        """Verify different custom utterances produce different analyses and replies."""
        p1 = {"text": "I am so happy and thrilled today!", "mode": "custom"}
        r1 = self.client.post("/api/demo/analyze", json=p1).json()

        p2 = {"text": "I feel lonely and sad this evening.", "mode": "custom"}
        r2 = self.client.post("/api/demo/analyze", json=p2).json()

        self.assertEqual(r1["analyzed_utterance"], p1["text"])
        self.assertEqual(r2["analyzed_utterance"], p2["text"])
        self.assertNotEqual(r1["response"], r2["response"])

    def test_analyze_meld_dataset_sample(self):
        """Mode 2: POST /api/demo/analyze with MELD sample_id executes canonical analysis."""
        payload = {
            "mode": "meld",
            "sample_id": "dia0_utt0",
            "text": "Why do all you’re coffee mugs have numbers on the bottom?"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "meld")
        self.assertIn("dia0_utt0", data.get("sample_id", ""))
        self.assertTrue(data["modalities"]["text"]["is_available"])
        self.assertTrue(data["modalities"]["audio"]["is_available"])
        self.assertTrue(data["modalities"]["visual"]["is_available"])
        self.assertIn("d1", data["disagreement"])
        self.assertIn("d3", data["disagreement"])
        self.assertIn("reliability", data)

    def test_demo_analyze_scenario_1_strong_agreement(self):
        """Mode 3: Scenario 1 yields High Reliability & Strong Adaptation."""
        payload = {
            "mode": "demo",
            "scenario_key": "strong_agreement"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["fusion"]["emotion"], "joy")
        self.assertGreaterEqual(data["fusion"]["confidence"], 0.75)
        self.assertEqual(data["disagreement"]["level"], "LOW")
        self.assertEqual(data["reliability"]["level"], "HIGH")
        self.assertEqual(data["response_policy"]["adaptation"], "strong")
        self.assertTrue(any(w in data["response"].lower() for w in ["glad", "wonderful", "positive", "uplifting"]))
        self.assertIn("reliability is HIGH", data["why_response"])

    def test_demo_analyze_scenario_2_moderate_disagreement(self):
        """Mode 3: Scenario 2 yields Medium Reliability & Cautious Adaptation."""
        payload = {
            "mode": "demo",
            "scenario_key": "moderate_disagreement"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["disagreement"]["level"], "MODERATE")
        self.assertEqual(data["reliability"]["level"], "MEDIUM")
        self.assertEqual(data["response_policy"]["adaptation"], "cautious")

    def test_demo_analyze_scenario_3_strong_conflict(self):
        """Mode 3: Scenario 3 yields Low Reliability & Neutral/Clarification Adaptation."""
        payload = {
            "mode": "demo",
            "scenario_key": "strong_conflict"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["disagreement"]["level"], "HIGH")
        self.assertEqual(data["reliability"]["level"], "LOW")
        self.assertEqual(data["response_policy"]["adaptation"], "neutral")

    def test_demo_analyze_scenario_4_low_confidence(self):
        """Mode 3: Scenario 4 yields Low Confidence & Neutral/Clarification Adaptation."""
        payload = {
            "mode": "demo",
            "scenario_key": "low_confidence"
        }
        resp = self.client.post("/api/demo/analyze", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertEqual(data["mode"], "demo")
        self.assertLess(data["fusion"]["confidence"], 0.40)
        self.assertEqual(data["reliability"]["level"], "LOW")
        self.assertEqual(data["response_policy"]["adaptation"], "neutral")


if __name__ == "__main__":
    unittest.main()
