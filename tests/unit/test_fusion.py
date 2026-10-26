"""Unit tests for Multimodal Fusion Architectures: Model A, Model B, and Model C."""

import unittest
import torch

from fusion.baseline import BaselineMultimodalMLP
from fusion.disagreement_aware import DisagreementAwareMLP
from fusion.reliability_aware import ReliabilityAwareFusion


class TestFusionModels(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.text_dim = 768
        self.audio_dim = 768
        self.visual_dim = 768
        self.num_classes = 7

        self.t_feat = torch.randn(self.batch_size, self.text_dim)
        self.a_feat = torch.randn(self.batch_size, self.audio_dim)
        self.v_feat = torch.randn(self.batch_size, self.visual_dim)

        self.t_probs = torch.softmax(torch.randn(self.batch_size, self.num_classes), dim=-1)
        self.a_probs = torch.softmax(torch.randn(self.batch_size, self.num_classes), dim=-1)
        self.v_probs = torch.softmax(torch.randn(self.batch_size, self.num_classes), dim=-1)

    def test_model_a_baseline_forward(self):
        """Model A forward pass produces [B, 7] logits."""
        model_a = BaselineMultimodalMLP(
            text_dim=self.text_dim,
            audio_dim=self.audio_dim,
            visual_dim=self.visual_dim,
            num_classes=self.num_classes,
        )
        logits = model_a(self.t_feat, self.a_feat, self.v_feat)
        self.assertEqual(logits.shape, torch.Size([self.batch_size, self.num_classes]))

        probs = model_a.predict_distribution(self.t_feat, self.a_feat, self.v_feat)
        self.assertEqual(probs.shape, torch.Size([self.batch_size, self.num_classes]))
        self.assertTrue(torch.allclose(torch.sum(probs, dim=-1), torch.ones(self.batch_size)))

    def test_model_b_disagreement_forward(self):
        """Model B forward pass with disagreement features produces [B, 7] logits."""
        model_b = DisagreementAwareMLP(
            text_dim=self.text_dim,
            audio_dim=self.audio_dim,
            visual_dim=self.visual_dim,
            num_classes=self.num_classes,
        )
        logits = model_b(
            self.t_feat, self.a_feat, self.v_feat,
            text_probs=self.t_probs,
            audio_probs=self.a_probs,
            visual_probs=self.v_probs,
        )
        self.assertEqual(logits.shape, torch.Size([self.batch_size, self.num_classes]))

    def test_model_c_reliability_forward(self):
        """Model C forward pass with attention and full reliability vector produces [B, 7] logits."""
        model_c = ReliabilityAwareFusion(
            text_dim=self.text_dim,
            audio_dim=self.audio_dim,
            visual_dim=self.visual_dim,
            num_classes=self.num_classes,
        )
        logits = model_c(
            self.t_feat, self.a_feat, self.v_feat,
            text_probs=self.t_probs,
            audio_probs=self.a_probs,
            visual_probs=self.v_probs,
        )
        self.assertEqual(logits.shape, torch.Size([self.batch_size, self.num_classes]))

        raw_logits, cal_probs = model_c.forward_calibrated(
            self.t_feat, self.a_feat, self.v_feat,
            text_probs=self.t_probs,
            audio_probs=self.a_probs,
            visual_probs=self.v_probs,
        )
        self.assertEqual(cal_probs.shape, torch.Size([self.batch_size, self.num_classes]))
        self.assertTrue(torch.allclose(torch.sum(cal_probs, dim=-1), torch.ones(self.batch_size)))

    def test_gradient_flow(self):
        """All three models support backpropagation without NaN gradients."""
        targets = torch.randint(0, self.num_classes, (self.batch_size,))
        criterion = torch.nn.CrossEntropyLoss()

        for model in [
            BaselineMultimodalMLP(),
            DisagreementAwareMLP(),
            ReliabilityAwareFusion(),
        ]:
            model.train()
            out = model(
                self.t_feat, self.a_feat, self.v_feat,
                text_probs=self.t_probs,
                audio_probs=self.a_probs,
                visual_probs=self.v_probs,
            )
            loss = criterion(out, targets)
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.assertFalse(torch.isnan(param.grad).any(), f"NaN grad in {name}")


if __name__ == "__main__":
    unittest.main()
