"""Abstract Base Class for Modality Encoders & Emotion Classifiers.

Defines the standard interface for Text, Audio, and Visual models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class BaseModalityModel(nn.Module, ABC):
    """Abstract Base Class for single-modality emotion recognition models."""

    def __init__(
        self,
        modality_name: str,
        feature_dim: int = 768,
        num_classes: int = 7,
        mock_mode: bool = False,
    ):
        super().__init__()
        self.modality_name = modality_name
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.mock_mode = mock_mode

    @abstractmethod
    def extract_features(self, raw_inputs: Any) -> torch.Tensor:
        """Extract continuous feature embeddings from raw inputs.

        Args:
            raw_inputs: Modality-specific raw input (text string/tokens, audio waveform, image tensor).

        Returns:
            torch.Tensor: Feature embeddings of shape [Batch, feature_dim].
        """
        pass

    @abstractmethod
    def forward(self, inputs: Any) -> torch.Tensor:
        """Forward pass computing raw class logits.

        Args:
            inputs: Either raw inputs or pre-extracted feature embeddings.

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes].
        """
        pass

    def predict_distribution(self, inputs: Any) -> torch.Tensor:
        """Compute softmax probability distribution over emotion classes.

        Returns:
            torch.Tensor: Softmax probabilities of shape [Batch, num_classes].
        """
        logits = self.forward(inputs)
        return torch.softmax(logits, dim=-1)

    def predict_sample_dict(
        self,
        inputs: Any,
        emotion_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Convenience method returning structured dictionary for a single sample.

        Returns:
            Dict containing 'probabilities', 'predicted_emotion', 'predicted_id', 'confidence'.
        """
        if emotion_labels is None:
            emotion_labels = [
                "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
            ]

        self.eval()
        with torch.no_grad():
            probs = self.predict_distribution(inputs)
            if probs.ndim > 1:
                probs_single = probs[0].detach().cpu().numpy()
            else:
                probs_single = probs.detach().cpu().numpy()

        pred_id = int(np.argmax(probs_single))
        conf = float(probs_single[pred_id])
        pred_label = emotion_labels[pred_id] if pred_id < len(emotion_labels) else f"class_{pred_id}"

        return {
            "probabilities": probs_single.tolist(),
            "predicted_emotion": pred_label,
            "predicted_id": pred_id,
            "confidence": conf,
            "is_mock": self.mock_mode,
        }
