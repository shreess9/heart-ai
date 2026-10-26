"""Integration tests for FastAPI Backend Routes."""

import unittest
from fastapi.testclient import TestClient
from backend.main import create_app


class TestAPIEndpointsIntegration(unittest.TestCase):

    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_endpoint(self):
        """GET /health returns 200 OK with system status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["taxonomy"], "MELD-7")

    def test_multimodal_inference_endpoint(self):
        """POST /api/v1/inference/multimodal executes pipeline and returns structured JSON."""
        payload = {
            "text": "I was thinking about my old garden today.",
            "fusion_type": "reliability_aware",
            "mock_mode": True,
        }
        response = self.client.post("/api/v1/inference/multimodal", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("text", data)
        self.assertIn("audio", data)
        self.assertIn("visual", data)
        self.assertIn("disagreement", data)
        self.assertIn("uncertainty", data)
        self.assertIn("fused", data)
        self.assertIn("reliability_level", data)
        self.assertIn("safety_guidance", data)

    def test_dialogue_respond_endpoint(self):
        """POST /api/v1/dialogue/respond returns companion reply conditioned on affective state."""
        payload = {
            "user_message": "Good morning, how are you today?",
        }
        response = self.client.post("/api/v1/dialogue/respond", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("reply", data)
        self.assertIn("reliability_level", data)
        self.assertIn("safety_guidance", data)


if __name__ == "__main__":
    unittest.main()
