"""Predictive Uncertainty & Entropy Utilities.

Implements:
1. Shannon predictive entropy H(P) in bits.
2. Normalized predictive entropy H_norm(P) = H(P) / log2(K) bounded in [0, 1].
"""

from typing import Union
import numpy as np
import torch


def compute_predictive_entropy(
    probs: Union[np.ndarray, torch.Tensor],
    normalized: bool = False,
    eps: float = 1e-12,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute Shannon predictive entropy of a probability distribution.

    H(P) = - sum_{k=1}^K P(k) * log2(P(k))
    If normalized=True, divide by log2(K) so that H_norm in [0, 1].

    Args:
        probs: Probability distribution(s), shape [..., K].
        normalized: Whether to normalize by log2(num_classes).
        eps: Small constant to avoid log2(0).

    Returns:
        Entropy scalar or tensor/array matching batch shape.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.to(torch.float32)
        k = probs.shape[-1]
        probs_clamped = torch.clamp(probs, min=eps, max=1.0)
        probs_norm = probs_clamped / torch.sum(probs_clamped, dim=-1, keepdim=True)
        entropy = -torch.sum(probs_norm * torch.log2(probs_norm), dim=-1)
        entropy = torch.clamp(entropy, min=0.0)

        if normalized and k > 1:
            log2_k = float(np.log2(k))
            entropy = entropy / log2_k
            entropy = torch.clamp(entropy, min=0.0, max=1.0)

        if entropy.ndim == 0:
            return float(entropy.item())
        return entropy

    # NumPy path
    probs_arr = np.asarray(probs, dtype=np.float64)
    k = probs_arr.shape[-1]
    probs_clamped = np.clip(probs_arr, eps, 1.0)
    probs_norm = probs_clamped / np.sum(probs_clamped, axis=-1, keepdims=True)
    entropy = -np.sum(probs_norm * np.log2(probs_norm), axis=-1)
    entropy = np.clip(entropy, 0.0, None)

    if normalized and k > 1:
        entropy = entropy / np.log2(k)
        entropy = np.clip(entropy, 0.0, 1.0)

    if np.ndim(entropy) == 0:
        return float(entropy)
    return entropy
