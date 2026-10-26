"""Abstract Base Class for Multimodal Fusion Architectures.

Provides standard interface for Model A (Baseline), Model B (Disagreement-Aware),
and Model C (Reliability-Aware).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


class BaseFusionModel(nn.Module, ABC):
    """Abstract Base Class for Multimodal Emotion Fusion Models."""

    def __init__(
        self,
        name: str,
        num_classes: int = 7,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 768,
    ):
        super().__init__()
        self.name = name
        self.num_classes = num_classes
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.visual_dim = visual_dim

    @abstractmethod
    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass predicting emotion logits.

        Args:
            text_features: Tensor of shape [Batch, text_dim]
            audio_features: Tensor of shape [Batch, audio_dim]
            visual_features: Tensor of shape [Batch, visual_dim]

        Returns:
            torch.Tensor: Emotion logits of shape [Batch, num_classes]
        """
        pass

    def predict_distribution(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Predict softmax probability distribution."""
        logits = self.forward(text_features, audio_features, visual_features, **kwargs)
        return torch.softmax(logits, dim=-1)
