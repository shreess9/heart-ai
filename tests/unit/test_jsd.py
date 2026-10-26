"""Unit tests for Jensen-Shannon Divergence calculations."""

import unittest
import numpy as np
import torch

from reliability.jsd import generalized_jsd_3, pairwise_jsd_2, mean_pairwise_jsd_3


class TestJSD(unittest.TestCase):

    def setUp(self):
        # 7-class discrete distributions
        self.p_uniform = np.ones(7) / 7.0
        self.p_onehot_0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.p_onehot_1 = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.p_onehot_2 = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        self.p_smooth_a = np.array([0.7, 0.1, 0.05, 0.05, 0.04, 0.03, 0.03])
        self.p_smooth_b = np.array([0.1, 0.6, 0.1, 0.05, 0.05, 0.05, 0.05])
        self.p_smooth_c = np.array([0.05, 0.05, 0.7, 0.05, 0.05, 0.05, 0.05])

    def test_identical_distributions_zero_jsd(self):
        """Identical distributions must have zero divergence."""
        jsd_val = generalized_jsd_3(self.p_smooth_a, self.p_smooth_a, self.p_smooth_a)
        self.assertAlmostEqual(jsd_val, 0.0, places=6)

        jsd_uniform = generalized_jsd_3(self.p_uniform, self.p_uniform, self.p_uniform)
        self.assertAlmostEqual(jsd_uniform, 0.0, places=6)

    def test_symmetry(self):
        """JSD must be symmetric under all permutations of input distributions."""
        jsd_abc = generalized_jsd_3(self.p_smooth_a, self.p_smooth_b, self.p_smooth_c)
        jsd_acb = generalized_jsd_3(self.p_smooth_a, self.p_smooth_c, self.p_smooth_b)
        jsd_bca = generalized_jsd_3(self.p_smooth_b, self.p_smooth_c, self.p_smooth_a)
        jsd_cab = generalized_jsd_3(self.p_smooth_c, self.p_smooth_a, self.p_smooth_b)

        self.assertAlmostEqual(jsd_abc, jsd_acb, places=6)
        self.assertAlmostEqual(jsd_abc, jsd_bca, places=6)
        self.assertAlmostEqual(jsd_abc, jsd_cab, places=6)

    def test_non_negativity_and_bounds(self):
        """Normalized JSD must lie in [0.0, 1.0]."""
        jsd_val = generalized_jsd_3(self.p_smooth_a, self.p_smooth_b, self.p_smooth_c, normalized=True)
        self.assertGreaterEqual(jsd_val, 0.0)
        self.assertLessEqual(jsd_val, 1.0)

        # Maximum divergence for 3 mutually orthogonal one-hot distributions
        jsd_max = generalized_jsd_3(self.p_onehot_0, self.p_onehot_1, self.p_onehot_2, normalized=True)
        self.assertAlmostEqual(jsd_max, 1.0, places=5)

    def test_pairwise_and_mean_jsd(self):
        """Pairwise JSD and mean pairwise JSD properties."""
        jsd_pair = pairwise_jsd_2(self.p_onehot_0, self.p_onehot_0)
        self.assertAlmostEqual(jsd_pair, 0.0, places=6)

        mean_jsd = mean_pairwise_jsd_3(self.p_smooth_a, self.p_smooth_b, self.p_smooth_c)
        self.assertGreater(mean_jsd, 0.0)
        self.assertLessEqual(mean_jsd, 1.0)

    def test_torch_tensor_batch_computation(self):
        """PyTorch tensor batch support matches NumPy."""
        t1 = torch.tensor(np.array([self.p_smooth_a, self.p_onehot_0]), dtype=torch.float32)
        t2 = torch.tensor(np.array([self.p_smooth_b, self.p_onehot_1]), dtype=torch.float32)
        t3 = torch.tensor(np.array([self.p_smooth_c, self.p_onehot_2]), dtype=torch.float32)

        res_torch = generalized_jsd_3(t1, t2, t3, normalized=True)
        self.assertEqual(res_torch.shape, torch.Size([2]))

        val_0 = float(generalized_jsd_3(self.p_smooth_a, self.p_smooth_b, self.p_smooth_c, normalized=True))
        val_1 = float(generalized_jsd_3(self.p_onehot_0, self.p_onehot_1, self.p_onehot_2, normalized=True))

        self.assertAlmostEqual(float(res_torch[0]), val_0, places=5)
        self.assertAlmostEqual(float(res_torch[1]), val_1, places=5)


if __name__ == "__main__":
    unittest.main()
