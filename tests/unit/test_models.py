"""Unit tests for Modality Models (Text, Audio, Visual) and Mock test harness."""

import unittest
import torch
import numpy as np

from models.text.roberta import RobertaEmotionClassifier
from models.audio.wav2vec2 import Wav2Vec2EmotionClassifier
from models.visual.vit import ViTEmotionClassifier


class TestModalityModels(unittest.TestCase):

    def setUp(self):
        self.text_model = RobertaEmotionClassifier(mock_mode=True)
        self.audio_model = Wav2Vec2EmotionClassifier(mock_mode=True)
        self.visual_model = ViTEmotionClassifier(mock_mode=True)

    def test_text_feature_extraction_and_forward(self):
        """Text model extracts 768-dim features and predicts 7-class logits."""
        text_input = "I am really glad we could talk today."
        features = self.text_model.extract_features(text_input)
        self.assertEqual(features.shape, torch.Size([1, 768]))

        logits = self.text_model(features)
        self.assertEqual(logits.shape, torch.Size([1, 7]))

        probs = self.text_model.predict_distribution(features)
        self.assertEqual(probs.shape, torch.Size([1, 7]))
        self.assertAlmostEqual(float(torch.sum(probs).item()), 1.0, places=5)

    def test_audio_feature_extraction_and_forward(self):
        """Audio model extracts 768-dim features and predicts 7-class logits."""
        dummy_audio = np.random.randn(16000).astype(np.float32) # 1 sec audio
        features = self.audio_model.extract_features(dummy_audio)
        self.assertEqual(features.shape, torch.Size([1, 768]))

        probs = self.audio_model.predict_distribution(features)
        self.assertEqual(probs.shape, torch.Size([1, 7]))
        self.assertAlmostEqual(float(torch.sum(probs).item()), 1.0, places=5)

    def test_visual_feature_extraction_and_forward(self):
        """Visual model extracts 768-dim features and predicts 7-class logits."""
        dummy_image = np.random.randn(3, 224, 224).astype(np.float32)
        features = self.visual_model.extract_features(dummy_image)
        self.assertEqual(features.shape, torch.Size([1, 768]))

        probs = self.visual_model.predict_distribution(features)
        self.assertEqual(probs.shape, torch.Size([1, 7]))
        self.assertAlmostEqual(float(torch.sum(probs).item()), 1.0, places=5)

    def test_predict_sample_dict(self):
        """predict_sample_dict returns properly formatted dictionary."""
        sample_dict = self.text_model.predict_sample_dict("Hello there")
        self.assertIn("probabilities", sample_dict)
        self.assertIn("predicted_emotion", sample_dict)
        self.assertIn("predicted_id", sample_dict)
        self.assertIn("confidence", sample_dict)
        self.assertEqual(len(sample_dict["probabilities"]), 7)
        self.assertAlmostEqual(sum(sample_dict["probabilities"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
