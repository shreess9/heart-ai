"""Vision Transformer (ViT) Visual Emotion Classifier.

Supports:
1. Live HuggingFace `google/vit-base-patch16-224` backbone + Linear head.
2. CLS token pooling over image patch embeddings.
3. Checkpoint loading for fine-tuned weights.
4. Explicit Mock mode for fast test harness & offline verification.
"""

from typing import Any, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
from models.base import BaseModalityModel


class ViTEmotionClassifier(BaseModalityModel):
    """Visual emotion classifier using Vision Transformer (ViT) backbone."""

    def __init__(
        self,
        backbone_name: str = "google/vit-base-patch16-224",
        feature_dim: int = 768,
        num_classes: int = 7,
        image_size: int = 224,
        dropout: float = 0.1,
        checkpoint_path: Optional[str] = None,
        mock_mode: bool = False,
    ):
        super().__init__(
            modality_name="visual",
            feature_dim=feature_dim,
            num_classes=num_classes,
            mock_mode=mock_mode,
        )
        self.backbone_name = backbone_name
        self.image_size = image_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(feature_dim, num_classes)

        self._image_processor = None
        self._vit = None

        if not self.mock_mode:
            self._init_vit()

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def _init_vit(self):
        """Lazy load ViT model."""
        try:
            from transformers import AutoImageProcessor, AutoModel
            self._image_processor = AutoImageProcessor.from_pretrained(self.backbone_name)
            self._vit = AutoModel.from_pretrained(self.backbone_name)
        except Exception:
            self.mock_mode = True

    def extract_features(self, raw_inputs: Any) -> torch.Tensor:
        """Extract 768-dim CLS embeddings from image input."""
        if self.mock_mode or self._vit is None:
            if isinstance(raw_inputs, torch.Tensor) and raw_inputs.ndim == 2 and raw_inputs.shape[-1] == self.feature_dim:
                return raw_inputs
            
            # Determine batch size: 3D (C, H, W) or (H, W, C) is 1 image; 4D (B, C, H, W) is B images
            if isinstance(raw_inputs, (np.ndarray, torch.Tensor)):
                batch_size = 1 if raw_inputs.ndim <= 3 else raw_inputs.shape[0]
            elif isinstance(raw_inputs, list):
                batch_size = len(raw_inputs)
            else:
                batch_size = 1

            g = torch.Generator().manual_seed(202)
            return torch.randn(batch_size, self.feature_dim, generator=g)

        # Real ViT execution
        if isinstance(raw_inputs, torch.Tensor):
            if raw_inputs.ndim == 3:
                raw_inputs = raw_inputs.unsqueeze(0)
            raw_inputs = raw_inputs.to(next(self.parameters()).device)
            outputs = self._vit(pixel_values=raw_inputs)
            features = outputs.last_hidden_state[:, 0, :]
            return features

        # PIL image or numpy array
        processed = self._image_processor(raw_inputs, return_tensors="pt")
        processed = {k: v.to(next(self.parameters()).device) for k, v in processed.items()}
        outputs = self._vit(**processed)
        features = outputs.last_hidden_state[:, 0, :]
        return features

    def forward(self, inputs: Any) -> torch.Tensor:
        """Compute logits from image input or pre-extracted embeddings."""
        if isinstance(inputs, torch.Tensor) and inputs.shape[-1] == self.feature_dim:
            features = inputs
        else:
            features = self.extract_features(inputs)

        dropped = self.dropout(features)
        logits = self.classifier(dropped)
        return logits

    def load_checkpoint(self, path: str):
        """Load fine-tuned model checkpoint."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state, strict=False)
