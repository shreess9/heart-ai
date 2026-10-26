"""Unit tests for Predictive Entropy calculations."""

import unittest
import numpy as np
import torch

from reliability.entropy import compute_predictive_entropy


class TestPredictiveEntropy(unittest.TestCase):

    def setUp(self):
        self.k = 7
        self.p_onehot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.p_uniform = np.ones(self.k) / float(self.k)

    def test_onehot_zero_entropy(self):
        """Deterministic / one-hot distributions have 0.0 bits of entropy."""
        ent = compute_predictive_entropy(self.p_onehot)
        self.assertAlmostEqual(ent, 0.0, places=5)

    def test_uniform_max_entropy(self):
        """Uniform distribution has log2(K) bits of entropy, and 1.0 normalized entropy."""
        ent_raw = compute_predictive_entropy(self.p_uniform, normalized=False)
        self.assertAlmostEqual(ent_raw, np.log2(self.k), places=5)

        ent_norm = compute_predictive_entropy(self.p_uniform, normalized=True)
        self.assertAlmostEqual(ent_norm, 1.0, places=5)

    def test_torch_entropy(self):
        """PyTorch tensor batch matches NumPy values."""
        t_batch = torch.tensor(np.array([self.p_onehot, self.p_uniform]), dtype=torch.float32)
        ent_torch = compute_predictive_entropy(t_batch, normalized=True)

        self.assertEqual(ent_torch.shape, torch.Size([2]))
        self.assertAlmostEqual(float(ent_torch[0]), 0.0, places=5)
        self.assertAlmostEqual(float(ent_torch[1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
