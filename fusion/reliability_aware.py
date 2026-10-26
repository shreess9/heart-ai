"""Model C: Reliability-Aware Multimodal Fusion.

Combines:
1. Reliability-gated Cross-Modal Multi-Head Attention over feature embeddings.
2. Rich reliability feature vector:
   - D1: Generalized JSD (1)
   - D2: Pairwise Mean JSD (1)
   - D3: Categorical Disagreement one-hot (3)
   - Modality Confidences [c_t, c_a, c_v] (3)
   - Modality Predictive Entropies [H_t, H_a, H_v] (3)
   - Calibrated Modality Confidences [c_t_cal, c_a_cal, c_v_cal] (3)
   Total reliability dimension = 14
3. Post-hoc Temperature Scaling Calibration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from fusion.base import BaseFusionModel
from fusion.attention import ReliabilityGatedAttention
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot
from reliability.entropy import compute_predictive_entropy
from reliability.confidence import compute_confidence
from reliability.calibration import TemperatureScaler


class ReliabilityAwareFusion(BaseFusionModel):
    """Model C: Reliability-aware Attention + MLP fusion model."""

    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        visual_dim: int = 768,
        embed_dim: int = 768,
        reliability_feature_dim: int = 14,
        num_heads: int = 4,
        hidden_dims: Optional[List[int]] = None,
        num_classes: int = 7,
        dropout: float = 0.2,
        activation: str = "gelu",
        initial_temperature: float = 1.0,
    ):
        super().__init__(
            name="Model_C_Reliability_Aware_Attention",
            num_classes=num_classes,
            text_dim=text_dim,
            audio_dim=audio_dim,
            visual_dim=visual_dim,
        )
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.embed_dim = embed_dim
        self.reliability_feature_dim = reliability_feature_dim

        # Modality dimension projection if dims differ from embed_dim
        self.proj_text = nn.Identity() if text_dim == embed_dim else nn.Linear(text_dim, embed_dim)
        self.proj_audio = nn.Identity() if audio_dim == embed_dim else nn.Linear(audio_dim, embed_dim)
        self.proj_visual = nn.Identity() if visual_dim == embed_dim else nn.Linear(visual_dim, embed_dim)

        # Reliability Attention Module
        self.attention = ReliabilityGatedAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Classification MLP: [embed_dim + reliability_feature_dim] -> num_classes
        in_dim = embed_dim + reliability_feature_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if activation.lower() == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Built-in post-hoc temperature scaler
        self.temperature_scaler = TemperatureScaler(initial_temperature=initial_temperature)

    def extract_reliability_vector(
        self,
        text_probs: torch.Tensor,
        audio_probs: torch.Tensor,
        visual_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract full 14-dim reliability vector and per-modality reliability gates.

        Returns:
            rel_vector: [Batch, 14]
            rel_t: [Batch, 2] (confidence, 1 - norm_entropy)
            rel_a: [Batch, 2]
            rel_v: [Batch, 2]
        """
        # Disagreement features
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

        # Confidences
        conf_t = torch.max(text_probs, dim=-1, keepdim=True).values
        conf_a = torch.max(audio_probs, dim=-1, keepdim=True).values
        conf_v = torch.max(visual_probs, dim=-1, keepdim=True).values

        # Normalized Entropies
        ent_t = compute_predictive_entropy(text_probs, normalized=True)
        ent_a = compute_predictive_entropy(audio_probs, normalized=True)
        ent_v = compute_predictive_entropy(visual_probs, normalized=True)

        if ent_t.ndim == 0:
            ent_t = ent_t.unsqueeze(0)
        if ent_a.ndim == 0:
            ent_a = ent_a.unsqueeze(0)
        if ent_v.ndim == 0:
            ent_v = ent_v.unsqueeze(0)

        ent_t = ent_t.unsqueeze(-1) if ent_t.ndim == 1 else ent_t
        ent_a = ent_a.unsqueeze(-1) if ent_a.ndim == 1 else ent_a
        ent_v = ent_v.unsqueeze(-1) if ent_v.ndim == 1 else ent_v

        # Calibrated confidences (via modality temperature scaling or smoothed confidence)
        cal_conf_t = conf_t * (1.0 - 0.1 * ent_t)
        cal_conf_a = conf_a * (1.0 - 0.1 * ent_a)
        cal_conf_v = conf_v * (1.0 - 0.1 * ent_v)

        # Concatenate 14-dim reliability vector:
        # [d1 (1), d2 (1), d3_oh (3), conf_t, conf_a, conf_v (3), ent_t, ent_a, ent_v (3), cal_t, cal_a, cal_v (3)]
        rel_vector = torch.cat([
            d1, d2, d3_oh,
            conf_t, conf_a, conf_v,
            ent_t, ent_a, ent_v,
            cal_conf_t, cal_conf_a, cal_conf_v
        ], dim=-1).to(text_probs.device)

        # Per-modality 2D gate: (confidence, 1 - norm_entropy)
        rel_t = torch.cat([conf_t, 1.0 - ent_t], dim=-1).to(text_probs.device)
        rel_a = torch.cat([conf_a, 1.0 - ent_a], dim=-1).to(audio_probs.device)
        rel_v = torch.cat([conf_v, 1.0 - ent_v], dim=-1).to(visual_probs.device)

        return rel_vector, rel_t, rel_a, rel_v

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        text_probs: Optional[torch.Tensor] = None,
        audio_probs: Optional[torch.Tensor] = None,
        visual_probs: Optional[torch.Tensor] = None,
        reliability_features: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass with reliability attention and reliability feature conditioning."""
        if text_features.ndim == 1:
            text_features = text_features.unsqueeze(0)
        if audio_features.ndim == 1:
            audio_features = audio_features.unsqueeze(0)
        if visual_features.ndim == 1:
            visual_features = visual_features.unsqueeze(0)

        batch_size = text_features.shape[0]

        # Extract reliability gates
        rel_t, rel_a, rel_v = None, None, None
        if text_probs is not None and audio_probs is not None and visual_probs is not None:
            rel_vector, rel_t, rel_a, rel_v = self.extract_reliability_vector(
                text_probs, audio_probs, visual_probs
            )
        else:
            rel_vector = torch.zeros(batch_size, self.reliability_feature_dim, device=text_features.device)

        if reliability_features is not None:
            rel_vector = reliability_features

        # Project and attend
        t_proj = self.proj_text(text_features)
        a_proj = self.proj_audio(audio_features)
        v_proj = self.proj_visual(visual_features)

        attn_features, _ = self.attention(t_proj, a_proj, v_proj, rel_t, rel_a, rel_v)

        # Concatenate attended features with reliability features
        combined = torch.cat([attn_features, rel_vector], dim=-1)
        raw_logits = self.classifier(combined)
        return raw_logits

    def forward_calibrated(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        visual_features: torch.Tensor,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (raw_logits, calibrated_probabilities)."""
        raw_logits = self.forward(text_features, audio_features, visual_features, **kwargs)
        calibrated_probs = self.temperature_scaler.calibrate_probabilities(raw_logits)
        return raw_logits, calibrated_probs
