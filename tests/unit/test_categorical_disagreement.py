"""Unit tests for Categorical Disagreement (D3)."""

import unittest
import numpy as np
import torch

from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot


class TestCategoricalDisagreement(unittest.TestCase):

    def test_all_agree_level_zero(self):
        """When all three modalities predict the same class, D3 = 0."""
        # Single scalar index
        self.assertEqual(compute_categorical_disagreement_d3(4, 4, 4), 0)

        # NumPy batch
        yt = np.array([0, 1, 5])
        ya = np.array([0, 1, 5])
        yv = np.array([0, 1, 5])
        d3 = compute_categorical_disagreement_d3(yt, ya, yv)
        np.testing.assert_array_equal(d3, np.array([0, 0, 0]))

    def test_one_differs_level_one(self):
        """When exactly two agree and one differs, D3 = 1."""
        self.assertEqual(compute_categorical_disagreement_d3(0, 0, 1), 1)
        self.assertEqual(compute_categorical_disagreement_d3(0, 2, 0), 1)
        self.assertEqual(compute_categorical_disagreement_d3(3, 1, 1), 1)

    def test_all_differ_level_two(self):
        """When all three modalities predict distinct classes, D3 = 2."""
        self.assertEqual(compute_categorical_disagreement_d3(0, 1, 2), 2)
        self.assertEqual(compute_categorical_disagreement_d3(3, 4, 5), 2)

    def test_distribution_inputs(self):
        """Can compute D3 directly from class probability distributions."""
        p_text = np.array([0.8, 0.1, 0.1])    # class 0
        p_audio = np.array([0.2, 0.7, 0.1])   # class 1
        p_visual = np.array([0.1, 0.1, 0.8])  # class 2

        self.assertEqual(compute_categorical_disagreement_d3(p_text, p_audio, p_visual), 2)

        p_audio_agree = np.array([0.85, 0.1, 0.05]) # class 0
        self.assertEqual(compute_categorical_disagreement_d3(p_text, p_audio_agree, p_visual), 1)

    def test_one_hot_encoding(self):
        """Convert D3 values into 3-dimensional one-hot vectors."""
        one_hot_0 = d3_to_one_hot(0)
        np.testing.assert_array_equal(one_hot_0, np.array([1.0, 0.0, 0.0]))

        one_hot_1 = d3_to_one_hot(1)
        np.testing.assert_array_equal(one_hot_1, np.array([0.0, 1.0, 0.0]))

        one_hot_2 = d3_to_one_hot(2)
        np.testing.assert_array_equal(one_hot_2, np.array([0.0, 0.0, 1.0]))

        # Torch batch
        d3_batch = torch.tensor([0, 1, 2], dtype=torch.long)
        oh_torch = d3_to_one_hot(d3_batch)
        self.assertEqual(oh_torch.shape, torch.Size([3, 3]))
        self.assertEqual(oh_torch[0, 0].item(), 1.0)
        self.assertEqual(oh_torch[1, 1].item(), 1.0)
        self.assertEqual(oh_torch[2, 2].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
