"""Integration tests for Demonstrator API supporting JSON and Multipart Audio/Video Uploads."""

import io
import os
import tempfile
import unittest
import numpy as np
import soundfile as sf
import cv2
from fastapi.testclient import TestClient

from backend.main import app


class TestMultimodalApiIntegration(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Generate synthetic WAV
        sr = 16000
        t = np.linspace(0, 1.0, sr, dtype=np.float32)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        wav_buf = io.BytesIO()
        sf.write(wav_buf, sine_wave, sr, format='WAV')
        self.audio_bytes = wav_buf.getvalue()

        # Generate synthetic MP4
        temp_mp4 = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mp4, fourcc, 10.0, (160, 120))
        for i in range(10):
            frame = np.full((120, 160, 3), (i * 15, 100, 200), dtype=np.uint8)
            out.write(frame)
        out.release()
        with open(temp_mp4, 'rb') as f:
            self.video_bytes = f.read()
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)

    def test_json_endpoint_mode_custom(self):
        """Test Mode 1: JSON payload for manual text."""
        resp = self.client.post("/api/demo/analyze", json={
            "text": "I feel so grateful today.",
            "mode": "custom"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "custom")
        self.assertEqual(data["modalities"]["text"]["is_available"], True)
        self.assertEqual(data["modalities"]["audio"]["is_available"], False)
        self.assertEqual(data["modalities"]["visual"]["is_available"], False)
        self.assertIn("response", data)

    def test_json_endpoint_mode_meld(self):
        """Test Mode 3: JSON payload for MELD utterance."""
        resp = self.client.post("/api/demo/analyze", json={
            "text": "Why do all your coffee mugs have numbers?",
            "sample_id": "dia0_utt0",
            "mode": "meld"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "meld")
        self.assertEqual(data["sample_id"], "dia0_utt0")

    def test_json_endpoint_mode_demo(self):
        """Test Mode 4: JSON payload for demo scenario."""
        resp = self.client.post("/api/demo/analyze", json={
            "scenario_id": "strong_agreement",
            "mode": "demo"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["mode"], "demo")
        self.assertEqual(data["reliability"]["level"], "HIGH")

    def test_multipart_upload_audio_only(self):
        """Test Mode 2: Multipart form-data with Audio file."""
        files = {
            "audio": ("my_audio.wav", self.audio_bytes, "audio/wav")
        }
        data = {
            "mode": "upload"
        }
        resp = self.client.post("/api/demo/analyze", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertEqual(res_data["mode"], "upload")
        self.assertIn("audio", res_data["uploaded_modalities"])
        self.assertIn("audio_metadata", res_data["modalities"]["audio"])

    def test_multipart_upload_video_only(self):
        """Test Mode 2: Multipart form-data with Video file."""
        files = {
            "video": ("my_video.mp4", self.video_bytes, "video/mp4")
        }
        data = {
            "mode": "upload"
        }
        resp = self.client.post("/api/demo/analyze", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertEqual(res_data["mode"], "upload")
        self.assertIn("video", res_data["uploaded_modalities"])
        self.assertIn("video_metadata", res_data["modalities"]["visual"])

    def test_multipart_upload_text_and_audio(self):
        """Test Mode 2: Multipart form-data with Text + Audio."""
        files = {
            "audio": ("recording.wav", self.audio_bytes, "audio/wav")
        }
        data = {
            "mode": "upload",
            "text": "I had a wonderful conversation this morning."
        }
        resp = self.client.post("/api/demo/analyze", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertEqual(res_data["mode"], "upload")
        self.assertEqual(res_data["transcript_status"], "User provided")
        self.assertIn("text", res_data["uploaded_modalities"])
        self.assertIn("audio", res_data["uploaded_modalities"])

    def test_multipart_upload_full_multimodal(self):
        """Test Mode 2: Multipart form-data with Text + Audio + Video."""
        files = {
            "audio": ("recording.wav", self.audio_bytes, "audio/wav"),
            "video": ("clip.mp4", self.video_bytes, "video/mp4"),
        }
        data = {
            "mode": "upload",
            "text": "Family celebration video and audio recording."
        }
        resp = self.client.post("/api/demo/analyze", data=data, files=files)
        self.assertEqual(resp.status_code, 200)
        res_data = resp.json()
        self.assertEqual(res_data["mode"], "upload")
        self.assertEqual(len(res_data["uploaded_modalities"]), 3)
        self.assertIn("disagreement", res_data)
        self.assertIn("reliability", res_data)
        self.assertIn("companion_response", res_data)

    def test_multipart_upload_invalid_extension(self):
        """Test Mode 2: Invalid file extension triggers HTTP 400 error."""
        files = {
            "audio": ("virus.exe", b"bad content", "application/octet-stream")
        }
        data = {
            "mode": "upload"
        }
        resp = self.client.post("/api/demo/analyze", data=data, files=files)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Unsupported audio format", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
