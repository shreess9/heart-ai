"""Jensen-Shannon Divergence (JSD) Disagreement Metrics.

Implements:
1. Generalized 3-distribution JSD (D1) normalized by log2(3).
2. Mean pairwise JSD (D2) across text, audio, and visual distributions.
3. Batch-vectorized computation and PyTorch/NumPy interop.
"""

from typing import Tuple, Union
import numpy as np
import torch


def _shannon_entropy_np(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute Shannon entropy in bits (base 2) for probability vectors (NumPy)."""
    probs = np.clip(probs, eps, 1.0)
    # Normalize along last axis to ensure valid simplex
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return -np.sum(probs * np.log2(probs), axis=-1)


def _shannon_entropy_torch(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute Shannon entropy in bits (base 2) for probability tensors (PyTorch)."""
    probs = torch.clamp(probs, min=eps, max=1.0)
    probs = probs / torch.sum(probs, dim=-1, keepdim=True)
    return -torch.sum(probs * torch.log2(probs), dim=-1)


def generalized_jsd_3(
    p1: Union[np.ndarray, torch.Tensor],
    p2: Union[np.ndarray, torch.Tensor],
    p3: Union[np.ndarray, torch.Tensor],
    normalized: bool = True,
    eps: float = 1e-12,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute Generalized 3-distribution Jensen-Shannon Divergence (D1).

    JSD(P1, P2, P3) = H((P1 + P2 + P3) / 3) - 1/3 * (H(P1) + H(P2) + H(P3))
    Normalized by log2(3) so that D1 in [0.0, 1.0].

    Args:
        p1: Probability distribution for modality 1 (shape: [..., K])
        p2: Probability distribution for modality 2 (shape: [..., K])
        p3: Probability distribution for modality 3 (shape: [..., K])
        normalized: If True, divide by log2(3) to bound result in [0, 1].
        eps: Epsilon floor for numerical stability.

    Returns:
        Generalized JSD scalar or array/tensor matching input batch shape.
    """
    if isinstance(p1, torch.Tensor):
        p1 = p1.to(torch.float32)
        p2 = torch.as_tensor(p2, dtype=torch.float32, device=p1.device)
        p3 = torch.as_tensor(p3, dtype=torch.float32, device=p1.device)

        m = (p1 + p2 + p3) / 3.0
        h_m = _shannon_entropy_torch(m, eps=eps)
        h_avg = (_shannon_entropy_torch(p1, eps=eps) + 
                 _shannon_entropy_torch(p2, eps=eps) + 
                 _shannon_entropy_torch(p3, eps=eps)) / 3.0
        
        jsd = torch.clamp(h_m - h_avg, min=0.0)
        if normalized:
            log2_3 = float(np.log2(3.0))
            jsd = jsd / log2_3
        return jsd

    # NumPy path
    p1_arr = np.asarray(p1, dtype=np.float64)
    p2_arr = np.asarray(p2, dtype=np.float64)
    p3_arr = np.asarray(p3, dtype=np.float64)

    m = (p1_arr + p2_arr + p3_arr) / 3.0
    h_m = _shannon_entropy_np(m, eps=eps)
    h_avg = (_shannon_entropy_np(p1_arr, eps=eps) + 
             _shannon_entropy_np(p2_arr, eps=eps) + 
             _shannon_entropy_np(p3_arr, eps=eps)) / 3.0

    jsd = np.clip(h_m - h_avg, 0.0, None)
    if normalized:
        jsd = jsd / np.log2(3.0)
    
    if jsd.ndim == 0:
        return float(jsd)
    return jsd


def pairwise_jsd_2(
    p1: Union[np.ndarray, torch.Tensor],
    p2: Union[np.ndarray, torch.Tensor],
    normalized: bool = True,
    eps: float = 1e-12,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute pairwise Jensen-Shannon Divergence between two distributions.

    JSD(P1, P2) = H((P1 + P2) / 2) - 0.5 * (H(P1) + H(P2))
    Normalized by log2(2) = 1.0 bit.
    """
    if isinstance(p1, torch.Tensor):
        p1 = p1.to(torch.float32)
        p2 = torch.as_tensor(p2, dtype=torch.float32, device=p1.device)
        m = (p1 + p2) / 2.0
        h_m = _shannon_entropy_torch(m, eps=eps)
        h_avg = (_shannon_entropy_torch(p1, eps=eps) + _shannon_entropy_torch(p2, eps=eps)) / 2.0
        jsd = torch.clamp(h_m - h_avg, min=0.0)
        return jsd

    p1_arr = np.asarray(p1, dtype=np.float64)
    p2_arr = np.asarray(p2, dtype=np.float64)
    m = (p1_arr + p2_arr) / 2.0
    h_m = _shannon_entropy_np(m, eps=eps)
    h_avg = (_shannon_entropy_np(p1_arr, eps=eps) + _shannon_entropy_np(p2_arr, eps=eps)) / 2.0
    jsd = np.clip(h_m - h_avg, 0.0, None)
    if jsd.ndim == 0:
        return float(jsd)
    return jsd


def mean_pairwise_jsd_3(
    p_text: Union[np.ndarray, torch.Tensor],
    p_audio: Union[np.ndarray, torch.Tensor],
    p_visual: Union[np.ndarray, torch.Tensor],
    normalized: bool = True,
    eps: float = 1e-12,
) -> Union[float, np.ndarray, torch.Tensor]:
    """Compute Mean Pairwise JSD across all 3 pairs: (T,A), (T,V), (A,V) (D2)."""
    jsd_ta = pairwise_jsd_2(p_text, p_audio, normalized=normalized, eps=eps)
    jsd_tv = pairwise_jsd_2(p_text, p_visual, normalized=normalized, eps=eps)
    jsd_av = pairwise_jsd_2(p_audio, p_visual, normalized=normalized, eps=eps)
    mean_jsd = (jsd_ta + jsd_tv + jsd_av) / 3.0
    if isinstance(mean_jsd, (float, int)):
        return float(mean_jsd)
    return mean_jsd
