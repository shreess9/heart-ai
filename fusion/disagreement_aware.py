"""Model B: Disagreement-Aware Multimodal Fusion.

Combines modality features with:
1. Normalized generalized 3-distribution JSD (D1)
2. Mean pairwise JSD (D2)
3. One-hot categorical disagreement (D3: 0, 1, 2)
Total input dimension: 2304 + 5 = 2309
"""

from typing import Any, List, Optional
import torch
import torch.nn as nn
from fusion.base import BaseFusionModel
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot


class DisagreementAwareMLP(BaseFusionModel):
    """Model B: Disagreement-aware MLP fusion model."""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 768,
        disagreement_dim: int = 5, # D1 (1) + D2 (1) + D3 one-hot (3)
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 7,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__(
            name="Model_B_Disagreement_Aware_MLP",
            num_classes=num_classes,
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
        )
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.disagreement_dim = disagreement_dim
        self.input_dim = text_dim + audio_dim + visual_dim + disagreement_dim

        layers = []
        in_dim = self.input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation.lower() == "gelu":
                layers.append(nn.GELU())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def extract_disagreement_features(
        self,
        text_probs: torch.Tensor,
        audio_probs: torch.Tensor,
        visual_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Extract [D1, D2, D3_oh0, D3_oh1, D3_oh2] disagreement vector."""
        d1 = generalized_jsd_3(text_probs, audio_probs, visual_probs, normalized=True)
        d2 = mean_pairwise_jsd_3(text_probs, audio_probs, visual_probs, normalized=True)
        d3 = compute_categorical_disagreement_d3(text_probs, audio_probs, visual_probs)
        d3_oh = d3_to_one_hot(d3, num_levels=3)

        if d1.ndim == 0:
            d1 = d1.unsqueeze(0)
        if d2.ndim == 0:
            d2 = d2.unsqueeze(0)

        d1 = d1.unsqueeze(-1) if d1.ndim == 1 else d1
        d2 = d2.unsqueeze(-1) if d2.ndim == 1 else d2

        disagreement_vec = torch.cat([d1, d2, d3_oh], dim=-1)
        return disagreement_vec.to(text_probs.device)

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        text_probs: Optional[torch.Tensor] = None,
        audio_probs: Optional[torch.Tensor] = None,
        visual_probs: Optional[torch.Tensor] = None,
        disagreement_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass incorporating disagreement vector with feature embeddings."""
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        if audio_features.ndim == 1:
            audio_features = audio_features.unsqueeze(0)
        if visual_features.ndim == 1:
            visual_features = visual_features.unsqueeze(0)

        batch_size = text_features.shape[0]

        if disagreement_features is None:
            if text_probs is not None and audio_probs is not None and visual_probs is not None:
                disagreement_features = self.extract_disagreement_features(
                    text_probs, audio_probs, visual_probs
                )
            else:
                # Default zero disagreement vector if probabilities not supplied
                disagreement_features = torch.zeros(
                    batch_size, self.disagreement_dim, device=text_features.device
                )

        fused_input = torch.cat(
            [text_features, audio_features, visual_features, disagreement_features], dim=-1
        )
        logits = self.mlp(fused_input)
        return logits
