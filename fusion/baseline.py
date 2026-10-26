"""Model A: Controlled Baseline Multimodal Fusion (Feature Concatenation + MLP).

Concatenates Text (768), Audio (768), and Visual (768) embeddings (Total: 2304)
and feeds them through an MLP classifier.
Contains NO disagreement or uncertainty features.
"""

from typing import Any, List, Optional
import torch
import torch.nn as nn
from fusion.base import BaseFusionModel


class BaselineMultimodalMLP(BaseFusionModel):
    """Model A: Controlled baseline MLP fusion model."""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 768,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 7,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__(
            name="Model_A_Baseline_MLP",
            num_classes=num_classes,
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
        )
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = text_dim + audio_dim + visual_dim
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

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Concatenate features and forward through MLP."""
        # Ensure 2D shape [Batch, Dim]
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        if audio_features.ndim == 1:
            audio_features = audio_features.unsqueeze(0)
        if visual_features.ndim == 1:
            visual_features = visual_features.unsqueeze(0)

        fused_input = torch.cat([text_features, audio_features, visual_features], dim=-1)
        logits = self.mlp(fused_input)
        return logits
