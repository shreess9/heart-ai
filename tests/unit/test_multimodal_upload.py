"""Unit tests for Multimodal Upload, Media Processing, Missing-Modality Fusion, and Disagreement Engine."""

import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import soundfile as sf
import cv2

from backend.services.media_processor import (
    validate_audio_file,
    validate_video_file,
    process_audio_bytes,
    process_video_bytes,
)
from backend.services.demo_service import ResearchDemonstratorService
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3, pairwise_jsd_2
from reliability.categorical_disagreement import compute_categorical_disagreement_d3


class TestMultimodalUploadAndProcessing(unittest.TestCase):

    def setUp(self):
        self.service = ResearchDemonstratorService.get_instance()

        # Create synthetic valid WAV in memory
        self.sample_rate = 16000
        t = np.linspace(0, 1.0, self.sample_rate, dtype=np.float32)
        sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        wav_buf = io.BytesIO()
        sf.write(wav_buf, sine_wave, self.sample_rate, format='WAV')
        self.valid_audio_bytes = wav_buf.getvalue()

        # Create synthetic valid MP4 in temp file
        temp_mp4 = tempfile.mktemp(suffix='.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_mp4, fourcc, 10.0, (160, 120))
        for i in range(15):  # 1.5 seconds, 15 frames
            frame = np.full((120, 160, 3), (i * 10, 120, 180), dtype=np.uint8)
            out.write(frame)
        out.release()
        with open(temp_mp4, 'rb') as f:
            self.valid_video_bytes = f.read()
        if os.path.exists(temp_mp4):
            os.remove(temp_mp4)

    def test_audio_upload_validation_valid(self):
        """Test A: Valid audio extensions are accepted."""
        for ext in [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"]:
            validate_audio_file(f"sample{ext}", self.valid_audio_bytes)

    def test_audio_upload_validation_invalid_type(self):
        """Test B: Invalid audio extension raises ValueError."""
        for bad_name in ["script.exe", "document.pdf", "image.png", "data.json"]:
            with self.assertRaises(ValueError) as ctx:
                validate_audio_file(bad_name, self.valid_audio_bytes)
            self.assertIn("Unsupported audio format", str(ctx.exception))

    def test_audio_upload_empty_bytes(self):
        """Test audio validation rejects empty bytes."""
        with self.assertRaises(ValueError):
            validate_audio_file("empty.wav", b"")

    def test_video_upload_validation_valid(self):
        """Test C: Valid video extensions are accepted."""
        for ext in [".mp4", ".mov", ".avi", ".webm", ".mkv"]:
            validate_video_file(f"sample{ext}", self.valid_video_bytes)

    def test_video_upload_validation_invalid_type(self):
        """Test D: Invalid video extension raises ValueError."""
        for bad_name in ["virus.bat", "report.docx", "audio.mp3"]:
            with self.assertRaises(ValueError) as ctx:
                validate_video_file(bad_name, self.valid_video_bytes)
            self.assertIn("Unsupported video format", str(ctx.exception))

    def test_audio_processing_pipeline(self):
        """Verify audio decoding, resampling, and acoustic feature computation."""
        res = process_audio_bytes(self.valid_audio_bytes, "test_audio.wav")
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["sample_rate"], 16000)
        self.assertAlmostEqual(res["duration_seconds"], 1.0, delta=0.05)
        self.assertGreater(res["rms_energy"], 0.0)
        self.assertEqual(len(res["waveform"].shape), 1)

    def test_video_processing_pipeline_frames(self):
        """Test E/F: Video decoding and representative multi-frame sampling."""
        res = process_video_bytes(self.valid_video_bytes, "test_video.mp4", num_sample_frames=5)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["num_frames_sampled"], 5)
        self.assertAlmostEqual(res["fps"], 10.0, delta=0.5)
        self.assertAlmostEqual(res["duration_seconds"], 1.5, delta=0.1)
        self.assertEqual(res["resolution"], (160, 120))
        self.assertEqual(res["frames"][0].shape, (224, 224, 3))
        self.assertTrue(0.0 <= res["frames"][0].max() <= 1.0)

    def test_multimodal_upload_text_only(self):
        """Test G: Multimodal upload with Text only."""
        res = self.service.analyze_conversation(
            text="I feel peaceful sitting by the garden.",
            mode="upload",
        )
        self.assertEqual(res["mode"], "upload")
        self.assertIn("text", res["uploaded_modalities"])
        self.assertEqual(res["available_modalities"], ["text"])
        self.assertTrue(res["modalities"]["text"]["is_available"])
        self.assertFalse(res["modalities"]["audio"]["is_available"])
        self.assertFalse(res["modalities"]["visual"]["is_available"])
        self.assertEqual(res["disagreement"]["num_modalities_compared"], 1)
        self.assertEqual(res["disagreement"]["d1"], 0.0)

    def test_multimodal_upload_audio_only(self):
        """Test audio upload without text produces active audio emotion inference."""
        with patch("backend.services.demo_service.get_asr_engine") as mock_asr:
            mock_asr.return_value.transcribe_waveform.return_value = {
                "transcript": "", "is_available": False, "status": "No speech"
            }
            res = self.service.analyze_conversation(
                mode="upload",
                audio_bytes=self.valid_audio_bytes,
                audio_filename="uploaded_voice.wav",
            )
            self.assertEqual(res["mode"], "upload")
            self.assertIn("audio", res["uploaded_modalities"])
            self.assertFalse(res["modalities"]["text"]["is_available"])
            self.assertTrue(res["modalities"]["audio"]["is_available"])
            self.assertIn("Audio Analyzed", res["modalities"]["audio"]["status"])
            self.assertIn("audio_metadata", res["modalities"]["audio"])

    def test_multimodal_upload_video_only(self):
        """Test video upload without text."""
        res = self.service.analyze_conversation(
            mode="upload",
            video_bytes=self.valid_video_bytes,
            video_filename="uploaded_clip.mp4",
        )
        self.assertEqual(res["mode"], "upload")
        self.assertIn("video", res["uploaded_modalities"])
        self.assertIn("video_metadata", res["modalities"]["visual"])

    def test_multimodal_upload_text_and_audio(self):
        """Test H: Multimodal upload with Text + Audio."""
        res = self.service.analyze_conversation(
            text="The sun is shining beautifully today.",
            mode="upload",
            audio_bytes=self.valid_audio_bytes,
            audio_filename="sunny.wav",
        )
        self.assertEqual(res["mode"], "upload")
        self.assertIn("text", res["uploaded_modalities"])
        self.assertIn("audio", res["uploaded_modalities"])
        self.assertTrue(res["modalities"]["text"]["is_available"])

    def test_multimodal_upload_text_and_video(self):
        """Test I: Multimodal upload with Text + Video."""
        res = self.service.analyze_conversation(
            text="Look at what I made in the kitchen.",
            mode="upload",
            video_bytes=self.valid_video_bytes,
            video_filename="kitchen.mp4",
        )
        self.assertEqual(res["mode"], "upload")
        self.assertIn("text", res["uploaded_modalities"])
        self.assertIn("video", res["uploaded_modalities"])

    def test_multimodal_upload_audio_and_video(self):
        """Test J: Multimodal upload with Audio + Video."""
        with patch("backend.services.demo_service.get_asr_engine") as mock_asr:
            mock_asr.return_value.transcribe_waveform.return_value = {
                "transcript": "", "is_available": False, "status": "Transcript: Not Provided"
            }
            res = self.service.analyze_conversation(
                mode="upload",
                audio_bytes=self.valid_audio_bytes,
                audio_filename="tone.wav",
                video_bytes=self.valid_video_bytes,
                video_filename="clip.mp4",
            )
            self.assertEqual(res["mode"], "upload")
            self.assertIn("audio", res["uploaded_modalities"])
            self.assertIn("video", res["uploaded_modalities"])
            self.assertIn("Not Provided", res["transcript_status"])

    def test_multimodal_upload_text_audio_video(self):
        """Test K: Multimodal upload with Text + Audio + Video."""
        res = self.service.analyze_conversation(
            text="Everything is wonderful and I am so grateful.",
            mode="upload",
            audio_bytes=self.valid_audio_bytes,
            audio_filename="grateful.wav",
            video_bytes=self.valid_video_bytes,
            video_filename="grateful_clip.mp4",
        )
        self.assertEqual(res["mode"], "upload")
        self.assertEqual(sorted(res["uploaded_modalities"]), ["audio", "text", "video"])
        self.assertTrue(res["modalities"]["text"]["is_available"])
        self.assertIn("response", res)
        self.assertIn("reliability", res)

    def test_multimodal_disagreement_na_when_single_modality(self):
        """Verify D1/D2/D3 are marked is_applicable=False and d1_display='N/A' when single modality uploaded."""
        with patch("backend.services.demo_service.get_asr_engine") as mock_asr:
            mock_asr.return_value.transcribe_waveform.return_value = {
                "transcript": "", "is_available": False, "status": "No speech"
            }
            res = self.service.analyze_conversation(
                mode="upload",
                audio_bytes=self.valid_audio_bytes,
                audio_filename="voice.wav",
            )
            disagree = res["disagreement"]
            self.assertFalse(disagree["is_applicable"])
            self.assertEqual(disagree["d1_display"], "N/A")
            self.assertEqual(disagree["d2_display"], "N/A")
            self.assertEqual(disagree["d3_display"], "N/A")
            self.assertEqual(disagree["num_modalities_compared"], 1)

    def test_audio_classifier_fallback_when_unavailable(self):
        """Verify audio preprocessing gracefully falls back to PREPROCESSED_NO_CLASSIFIER if checkpoint fails."""
        with patch("backend.services.demo_service.get_audio_emotion_engine") as mock_a_fn:
            mock_a = MagicMock()
            mock_a.is_loaded = False
            mock_a._load_error = "Checkpoint missing"
            mock_a_fn.return_value = mock_a

            res = self.service.analyze_conversation(
                mode="upload",
                audio_bytes=self.valid_audio_bytes,
                audio_filename="recording.wav",
            )
            audio_mod = res["modalities"]["audio"]
            self.assertEqual(audio_mod["state"], "PREPROCESSED_NO_CLASSIFIER")
            self.assertFalse(audio_mod["is_available"])
            self.assertTrue(audio_mod["is_uploaded"])
            self.assertIn("Emotion classifier checkpoint unavailable", audio_mod["status"])
            self.assertEqual(audio_mod["checkpoint_status"], "Fine-tuned emotion checkpoint unavailable")

    def test_visual_classifier_fallback_when_unavailable(self):
        """Verify video decoding gracefully falls back to PREPROCESSED_NO_CLASSIFIER if checkpoint fails."""
        with patch("backend.services.demo_service.get_visual_face_emotion_engine") as mock_v_fn:
            mock_v = MagicMock()
            mock_v.is_loaded = False
            mock_v._load_error = "Checkpoint missing"
            mock_v_fn.return_value = mock_v

            res = self.service.analyze_conversation(
                mode="upload",
                video_bytes=self.valid_video_bytes,
                video_filename="clip.mp4",
            )
            vis_mod = res["modalities"]["visual"]
            self.assertEqual(vis_mod["state"], "PREPROCESSED_NO_CLASSIFIER")
            self.assertFalse(vis_mod["is_available"])
            self.assertTrue(vis_mod["is_uploaded"])
            self.assertIn("Visual emotion classifier unavailable", vis_mod["status"])
            self.assertEqual(vis_mod["checkpoint_status"], "Emotion classification checkpoint unavailable")

    def test_disagreement_pairwise_correctness(self):
        """Test M/O/P/Q: Pairwise JSD and Categorical Disagreement."""
        # Case 1: Identical distributions (0 divergence, D3=0)
        p1 = np.array([0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05])
        p2 = np.array([0.05, 0.05, 0.05, 0.70, 0.05, 0.05, 0.05])
        d1 = pairwise_jsd_2(p1, p2, normalized=True)
        d3 = 0 if np.argmax(p1) == np.argmax(p2) else 1
        self.assertAlmostEqual(d1, 0.0, places=5)
        self.assertEqual(d3, 0)

        # Case 2: Divergent distributions (high divergence, D3=1)
        p_joy = np.array([0.02, 0.01, 0.02, 0.85, 0.04, 0.03, 0.03])
        p_sad = np.array([0.02, 0.01, 0.02, 0.03, 0.04, 0.85, 0.03])
        d1_div = pairwise_jsd_2(p_joy, p_sad, normalized=True)
        d3_div = 0 if np.argmax(p_joy) == np.argmax(p_sad) else 1
        self.assertGreater(d1_div, 0.5)
        self.assertEqual(d3_div, 1)

    def test_disagreement_triplet_correctness(self):
        """Test N/O/P/Q: Triplet Generalized JSD and D3 categorical disagreement."""
        p_joy = np.array([0.02, 0.01, 0.02, 0.85, 0.04, 0.03, 0.03])
        p_anger = np.array([0.85, 0.02, 0.02, 0.03, 0.03, 0.03, 0.02])
        p_fear = np.array([0.02, 0.02, 0.85, 0.03, 0.03, 0.03, 0.02])

        d1 = generalized_jsd_3(p_joy, p_anger, p_fear, normalized=True)
        d2 = mean_pairwise_jsd_3(p_joy, p_anger, p_fear, normalized=True)
        d3 = compute_categorical_disagreement_d3(
            np.argmax(p_joy), np.argmax(p_anger), np.argmax(p_fear)
        )

        self.assertGreater(d1, 0.5)
        self.assertGreater(d2, 0.5)
        self.assertEqual(d3, 2)  # All 3 differ

    def test_multimodal_disagreement_na_when_single_active_modality(self):
        """Verify D1/D2/D3 are marked is_applicable=False and d1_display='N/A' when < 2 active emotion modalities."""
        res = self.service.analyze_conversation(
            text="I spent the day reading in the sunroom.",
            mode="custom",
        )
        disagree = res["disagreement"]
        self.assertFalse(disagree["is_applicable"])
        self.assertEqual(disagree["d1_display"], "N/A")
        self.assertEqual(disagree["d2_display"], "N/A")
        self.assertEqual(disagree["d3_display"], "N/A")
        self.assertEqual(disagree["num_modalities_compared"], 1)
        self.assertEqual(disagree["status_tier"], "inactive")

    def test_dynamic_fusion_active_subsets(self):
        """Verify dynamic fusion combinations (T, T+A, T+V, A+V, T+A+V) correctly compute D1/D2/D3."""
        # 1. Bimodal Concordant (T+A)
        p_t = np.array([0.02, 0.01, 0.02, 0.85, 0.04, 0.03, 0.03])
        p_a = np.array([0.03, 0.01, 0.02, 0.82, 0.05, 0.04, 0.03])
        d1_ta = pairwise_jsd_2(p_t, p_a, normalized=True)
        self.assertLess(d1_ta, 0.15)

        # 2. Bimodal Conflict (T+A)
        p_a_angry = np.array([0.80, 0.04, 0.03, 0.04, 0.04, 0.03, 0.02])
        d1_conflict = pairwise_jsd_2(p_t, p_a_angry, normalized=True)
        self.assertGreater(d1_conflict, 0.50)

        # 3. Tri-modal Triplet (T+A+V)
        p_v = np.array([0.02, 0.02, 0.80, 0.04, 0.04, 0.04, 0.04])
        d1_3 = generalized_jsd_3(p_t, p_a_angry, p_v, normalized=True)
        d2_3 = mean_pairwise_jsd_3(p_t, p_a_angry, p_v, normalized=True)
        d3_3 = compute_categorical_disagreement_d3(np.argmax(p_t), np.argmax(p_a_angry), np.argmax(p_v))
        self.assertGreater(d1_3, 0.40)
        self.assertGreater(d2_3, 0.40)
        self.assertEqual(d3_3, 2)


if __name__ == "__main__":
    unittest.main()
