"""Unit tests for Calibration, Temperature Scaling, ECE, MCE, and Brier Score."""

import unittest
import numpy as np
import torch

from reliability.calibration import (
    TemperatureScaler,
    compute_ece_mce,
    compute_brier_score,
)
from reliability.confidence import compute_confidence, compute_high_confidence_error_rate


class TestCalibration(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

        # 100 synthetic predictions over 7 classes
        self.num_samples = 100
        self.num_classes = 7

        self.logits = np.random.randn(self.num_samples, self.num_classes).astype(np.float32)
        # Softmax probabilities
        exp_logits = np.exp(self.logits - np.max(self.logits, axis=-1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        self.targets = np.random.randint(0, self.num_classes, size=self.num_samples)

    def test_temperature_scaler_scaling(self):
        """Temperature scaling preserves argmax while smoothing probabilities when T > 1."""
        scaler = TemperatureScaler(initial_temperature=2.0)
        t_logits = torch.tensor(self.logits, dtype=torch.float32)

        scaled_probs = scaler.calibrate_probabilities(t_logits).detach().numpy()

        # Argmax should be identical
        orig_preds = np.argmax(self.probs, axis=-1)
        scaled_preds = np.argmax(scaled_probs, axis=-1)
        np.testing.assert_array_equal(orig_preds, scaled_preds)

        # Max confidence should decrease when T = 2.0 > 1.0
        orig_confs = np.max(self.probs, axis=-1)
        scaled_confs = np.max(scaled_probs, axis=-1)
        self.assertTrue(np.all(scaled_confs <= orig_confs + 1e-5))

    def test_temperature_scaler_fit(self):
        """Temperature optimizer learns a valid positive temperature."""
        scaler = TemperatureScaler()
        optimal_t = scaler.fit(self.logits, self.targets)
        self.assertGreater(optimal_t, 0.05)
        self.assertLess(optimal_t, 10.0)
        self.assertAlmostEqual(float(scaler.temperature.item()), optimal_t, places=4)

    def test_ece_mce_bounds(self):
        """ECE and MCE must lie in [0.0, 1.0]."""
        metrics = compute_ece_mce(self.probs, self.targets, num_bins=10)
        self.assertIn("ece", metrics)
        self.assertIn("mce", metrics)
        self.assertGreaterEqual(metrics["ece"], 0.0)
        self.assertLessEqual(metrics["ece"], 1.0)
        self.assertGreaterEqual(metrics["mce"], 0.0)
        self.assertLessEqual(metrics["mce"], 1.0)
        self.assertGreaterEqual(metrics["mce"], metrics["ece"] - 1e-6)

    def test_perfect_calibration(self):
        """A calibrated set with accuracy matching confidence has low ECE."""
        # Perfect 1-hot predictions that are 100% accurate
        perfect_probs = np.zeros((10, self.num_classes))
        perfect_targets = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])
        for i, t in enumerate(perfect_targets):
            perfect_probs[i, t] = 1.0

        metrics = compute_ece_mce(perfect_probs, perfect_targets, num_bins=10)
        self.assertAlmostEqual(metrics["ece"], 0.0, places=5)
        self.assertAlmostEqual(metrics["mce"], 0.0, places=5)

    def test_brier_score(self):
        """Brier score is 0.0 for perfect predictions and <= 2.0."""
        perfect_probs = np.zeros((7, 7))
        perfect_targets = np.arange(7)
        for i in range(7):
            perfect_probs[i, i] = 1.0

        brier_perfect = compute_brier_score(perfect_probs, perfect_targets, num_classes=7)
        self.assertAlmostEqual(brier_perfect, 0.0, places=5)

        brier_rand = compute_brier_score(self.probs, self.targets, num_classes=7)
        self.assertGreaterEqual(brier_rand, 0.0)
        self.assertLessEqual(brier_rand, 2.0)

    def test_high_confidence_error_rate(self):
        """High-confidence error rate calculation."""
        hc_stats = compute_high_confidence_error_rate(self.probs, self.targets, threshold=0.3)
        self.assertIn("high_conf_error_rate", hc_stats)
        self.assertIn("high_conf_coverage", hc_stats)
        self.assertGreaterEqual(hc_stats["high_conf_error_rate"], 0.0)
        self.assertLessEqual(hc_stats["high_conf_error_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
