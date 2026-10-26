"""Unit tests for Bootstrap Resampling and Significance Testing."""

import unittest
import numpy as np

from reliability.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_delta


class TestBootstrap(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        self.n = 100
        self.targets = np.random.randint(0, 2, size=self.n)
        self.preds_a = self.targets.copy()
        # Introduce 10% errors in A
        corrupt_a = np.random.choice(self.n, size=10, replace=False)
        self.preds_a[corrupt_a] = 1 - self.preds_a[corrupt_a]

        # Model B has 30% errors
        self.preds_b = self.targets.copy()
        corrupt_b = np.random.choice(self.n, size=30, replace=False)
        self.preds_b[corrupt_b] = 1 - self.preds_b[corrupt_b]

        self.cluster_ids = np.repeat(np.arange(10), 10) # 10 dialogue clusters of size 10

    def test_sample_level_bootstrap_ci(self):
        """Sample-level bootstrap produces valid confidence interval around accuracy."""
        def acc_fn(preds, targets):
            return np.mean(preds == targets)

        ci_res = compute_bootstrap_ci(
            acc_fn, self.preds_a, self.targets,
            num_iterations=200, confidence_level=0.95, seed=42
        )

        self.assertAlmostEqual(ci_res["point_estimate"], 0.90, places=2)
        self.assertLessEqual(ci_res["ci_lower"], ci_res["point_estimate"])
        self.assertGreaterEqual(ci_res["ci_upper"], ci_res["point_estimate"])

    def test_dialogue_cluster_bootstrap_ci(self):
        """Cluster-level bootstrap resamples over dialogue clusters."""
        def acc_fn(preds, targets):
            return np.mean(preds == targets)

        ci_res = compute_bootstrap_ci(
            acc_fn, self.preds_a, self.targets,
            cluster_ids=self.cluster_ids,
            num_iterations=200, confidence_level=0.95, seed=42
        )

        self.assertAlmostEqual(ci_res["point_estimate"], 0.90, places=2)
        self.assertLessEqual(ci_res["ci_lower"], ci_res["point_estimate"])
        self.assertGreaterEqual(ci_res["ci_upper"], ci_res["point_estimate"])

    def test_paired_bootstrap_delta(self):
        """Paired bootstrap delta detects significant difference when Model A is better than Model B."""
        def acc_fn(preds, targets):
            return np.mean(preds == targets)

        # Delta: Metric(Model B) - Metric(Model A) = 0.70 - 0.90 = -0.20
        res = compute_paired_bootstrap_delta(
            acc_fn, self.preds_a, self.preds_b, self.targets,
            num_iterations=200, confidence_level=0.95, seed=42
        )

        self.assertAlmostEqual(res["point_delta"], -0.20, places=2)
        self.assertLess(res["ci_upper"], 0.0) # CI does not span zero
        self.assertLess(res["p_value"], 0.05)


if __name__ == "__main__":
    unittest.main()
