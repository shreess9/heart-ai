"""Reliability-Gated Cross-Modal Attention Mechanism.

Dynamically weights and mixes modality embeddings using modality confidence
and inverse predictive entropy.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class ReliabilityGatedAttention(nn.Module):
    """Multi-Head Cross-Modal Attention with Reliability Gating."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(2, embed_dim), # (confidence, 1 - norm_entropy) -> weight gate
            nn.Sigmoid(),
        )

    def forward(
        self,
        text_emb: torch.Tensor,
        audio_emb: torch.Tensor,
        visual_emb: torch.Tensor,
        text_rel: Optional[torch.Tensor] = None, # [B, 2] -> (conf, 1-entropy)
        audio_rel: Optional[torch.Tensor] = None,
        visual_rel: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply reliability gating followed by Multi-Head Self/Cross-Attention.

        Args:
            text_emb: [B, D]
            audio_emb: [B, D]
            visual_emb: [B, D]
            text_rel, audio_rel, visual_rel: Optional [B, 2] (confidence, 1-norm_entropy)

        Returns:
            fused_representation: [B, D]
            attention_weights: [B, 3, 3]
        """
        if text_emb.ndim == 1:
            text_emb = text_emb.unsqueeze(0)
        if audio_emb.ndim == 1:
            audio_emb = audio_emb.unsqueeze(0)
        if visual_emb.ndim == 1:
            visual_emb = visual_emb.unsqueeze(0)

        # Apply reliability gating if available
        if text_rel is not None:
            text_emb = text_emb * (1.0 + self.gate_fc(text_rel))
        if audio_rel is not None:
            audio_emb = audio_emb * (1.0 + self.gate_fc(audio_rel))
        if visual_rel is not None:
            visual_emb = visual_emb * (1.0 + self.gate_fc(visual_rel))

        # Stack modalities as sequence: [Batch, 3, embed_dim]
        modality_seq = torch.stack([text_emb, audio_emb, visual_emb], dim=1)

        # Multi-Head Attention
        attn_out, attn_weights = self.mha(modality_seq, modality_seq, modality_seq)
        out = self.layer_norm(modality_seq + attn_out)

        # Mean pooling across modalities
        pooled_out = torch.mean(out, dim=1) # [B, embed_dim]
        return pooled_out, attn_weights
