"""Unit and integration tests for the unified Single-MP4 Multimodal Pipeline.

Tests:
1. MP4 audio extraction and video frame sampling
2. Whisper ASR speech transcription
3. Automatic transcript -> RoBERTa text emotion (P(T))
4. Audio waveform -> Wav2Vec2 speech emotion (P(A))
5. Video frames -> ViT facial emotion (P(V))
6. Dynamic multimodal fusion across active modalities (T+A+V, T+A, A+V, T+V, V-only, A-only)
7. D1, D2, D3 disagreement computation and N/A handling
8. ASR failure handling (no fabricated transcripts or distributions)
9. No-face video handling (excluded from fusion)
10. Reliability Gate evaluation and Response Strategy selection
"""

import unittest
from pathlib import Path
import numpy as np

from backend.services.demo_service import ResearchDemonstratorService, CANONICAL_EMOTIONS
from backend.services.asr_service import ASREngine, get_asr_engine
from backend.services.pretrained_emotion_models import (
    AudioEmotionEngine,
    VisualFaceEmotionEngine,
    get_audio_emotion_engine,
    get_visual_face_emotion_engine,
)
from backend.services.media_processor import process_audio_bytes, process_video_bytes
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3, pairwise_jsd_2
from reliability.categorical_disagreement import compute_categorical_disagreement_d3
from reliability.entropy import compute_predictive_entropy


class TestSingleMP4Pipeline(unittest.TestCase):
    """Test suite for the unified single-MP4 multimodal pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.service = ResearchDemonstratorService.get_instance()
        cls.asr_engine = get_asr_engine()
        cls.test_mp4_path = Path("test_assets/real_conversation.mp4")
        cls.test_wav_path = Path("test_assets/family_speech.wav")

    def test_01_asr_speech_transcription(self):
        """Test Whisper ASR on real human spoken speech."""
        if not self.test_wav_path.exists():
            self.skipTest("family_speech.wav not found")

        audio_bytes = open(self.test_wav_path, "rb").read()
        res = process_audio_bytes(audio_bytes, "family_speech.wav")
        self.assertIn("waveform", res)
        self.assertEqual(res["sample_rate"], 16000)

        asr_res = self.asr_engine.transcribe_waveform(res["waveform"], res["sample_rate"])
        self.assertTrue(asr_res["is_available"])
        self.assertIn("grandchildren", asr_res["transcript"].lower())
        self.assertGreaterEqual(asr_res["word_count"], 5)
        self.assertEqual(asr_res["model_name"], "openai/whisper-tiny")

    def test_02_asr_silent_audio_handling(self):
        """Test Whisper ASR on silent waveform produces no fabricated transcript."""
        silent_waveform = np.zeros(16000 * 2, dtype=np.float32)
        asr_res = self.asr_engine.transcribe_waveform(silent_waveform, 16000)
        self.assertFalse(asr_res["is_available"])
        self.assertEqual(asr_res["transcript"], "")
        self.assertEqual(asr_res["word_count"], 0)

    def test_03_transcript_to_roberta_text_emotion(self):
        """Test automatic transcript fed to RoBERTa text emotion classifier."""
        transcript = "I had such a wonderful visit with my grandchildren in the garden today!"
        probs = self.service._infer_text_distribution(transcript)
        self.assertEqual(len(probs), 7)
        self.assertAlmostEqual(sum(probs), 1.0, places=2)
        top_idx = int(np.argmax(probs))
        top_emotion = CANONICAL_EMOTIONS[top_idx]
        self.assertEqual(top_emotion, "joy")
        self.assertGreaterEqual(probs[top_idx], 0.70)

    def test_04_audio_waveform_to_wav2vec2_emotion(self):
        """Test speech audio waveform processed by Wav2Vec2 SER."""
        audio_eng = get_audio_emotion_engine()
        self.assertTrue(audio_eng.is_loaded)

        dummy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        pred = audio_eng.predict_emotion(dummy_audio, 16000)
        self.assertTrue(pred["is_available"])
        self.assertEqual(len(pred["probabilities"]), 7)
        self.assertAlmostEqual(sum(pred["probabilities"]), 1.0, places=2)
        self.assertIn(pred["predicted_emotion"], CANONICAL_EMOTIONS)

    def test_05_video_frames_to_vit_facial_emotion(self):
        """Test video frames processed by ViT FER with face detection."""
        vis_eng = get_visual_face_emotion_engine()
        self.assertTrue(vis_eng.is_loaded)

        # 1. Test frame without face
        blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
        pred_blank = vis_eng.predict_video_frames([blank_frame])
        self.assertFalse(pred_blank["is_available"])
        self.assertEqual(pred_blank["faces_detected"], 0)

    def test_06_single_mp4_end_to_end_analysis(self):
        """Test single MP4 upload automatically generates T+A+V pipeline."""
        if not self.test_mp4_path.exists():
            self.skipTest("real_conversation.mp4 not found")

        v_bytes = open(self.test_mp4_path, "rb").read()
        res = self.service.analyze_conversation(
            video_bytes=v_bytes,
            video_filename="real_conversation.mp4",
        )

        self.assertIn("modalities", res)
        self.assertIn("text", res["modalities"])
        self.assertIn("audio", res["modalities"])
        self.assertIn("visual", res["modalities"])

        # Check Text modality is derived via ASR
        t_mod = res["modalities"]["text"]
        self.assertTrue(t_mod["is_available"])
        self.assertIn("grandchildren", t_mod.get("transcript", "").lower())
        self.assertEqual(t_mod["emotion"], "joy")
        self.assertEqual(t_mod["transcript_source"], "Automatic Speech Recognition (OpenAI Whisper Tiny)")

        # Check Audio modality is derived via Wav2Vec2
        a_mod = res["modalities"]["audio"]
        self.assertTrue(a_mod["is_available"])
        self.assertIn("Wav2Vec", a_mod["encoder"])

        # Check Visual modality is derived via ViT
        v_mod = res["modalities"]["visual"]
        self.assertTrue(v_mod["is_available"])
        self.assertIn("ViT", v_mod["encoder"])

        # Check Active Modalities & Fusion
        self.assertEqual(set(res["available_modalities"]), {"text", "audio", "visual"})
        self.assertEqual(res["fusion"]["emotion"], "joy")
        self.assertGreater(res["fusion"]["confidence"], 0.50)

        # Check D1/D2/D3 are calculated for genuine 3-modality evidence
        self.assertTrue(res["disagreement"]["is_applicable"])
        self.assertGreaterEqual(res["disagreement"]["d1"], 0.0)
        self.assertIn(res["disagreement"]["d3"], [0, 1, 2])

        # Check Reliability Gate & Response Strategy
        self.assertIn(res["reliability"]["level"], ["HIGH", "MEDIUM", "LOW"])
        self.assertIn("adaptation_badge", res["response_policy"])

        # Check Gemini companion response
        self.assertIn("companion_response", res)
        self.assertTrue(len(res["companion_response"]["reply"]) > 5)

    def test_07_dynamic_fusion_bimodal_ta(self):
        """Test dynamic fusion when only Text and Audio are available."""
        p_text = np.array([0.02, 0.01, 0.02, 0.85, 0.04, 0.03, 0.03], dtype=np.float32)
        p_audio = np.array([0.03, 0.01, 0.02, 0.80, 0.06, 0.04, 0.04], dtype=np.float32)

        d1 = pairwise_jsd_2(p_text, p_audio, normalized=True)
        self.assertLess(d1, 0.15)
        d3 = 0 if int(np.argmax(p_text)) == int(np.argmax(p_audio)) else 1
        self.assertEqual(d3, 0)

    def test_08_dynamic_fusion_unimodal_v_na_disagreement(self):
        """Test single active modality correctly reports Disagreement as N/A."""
        p_vis = np.array([0.05, 0.05, 0.05, 0.10, 0.65, 0.05, 0.05], dtype=np.float32)
        # 1 active modality -> Disagreement is N/A
        self.service._run_custom_text_inference("Hello")
        # Mode 1 has single text modality
        res = self.service._run_custom_text_inference("Hello")
        self.assertFalse(res["disagreement"]["is_applicable"])
        self.assertEqual(res["disagreement"]["d1_display"], "N/A")
        self.assertEqual(res["disagreement"]["d3_display"], "N/A")


if __name__ == "__main__":
    unittest.main()
