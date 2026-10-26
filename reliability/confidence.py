"""Confidence Analysis & Error Characterization.

Implements:
1. Modality confidence extraction: max_k P(k).
2. High-confidence error rate computation.
3. Confidence-accuracy binned evaluation.
"""

from typing import Dict, Tuple, Union
import numpy as np
import torch


def compute_confidence(
    probs: Union[np.ndarray, torch.Tensor],
) -> Union[float, np.ndarray, torch.Tensor]:
    """Extract prediction confidence (maximum class probability) from distribution."""
    if isinstance(probs, torch.Tensor):
        conf = torch.max(probs, dim=-1).values
        if conf.ndim == 0:
            return float(conf.item())
        return conf

    probs_arr = np.asarray(probs)
    conf = np.max(probs_arr, axis=-1)
    if np.ndim(conf) == 0:
        return float(conf)
    return conf


def compute_high_confidence_error_rate(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.80,
) -> Dict[str, float]:
    """Compute error rate specifically on high-confidence predictions (conf >= threshold).

    Args:
        probs: Predicted probability distributions, shape [N, K].
        targets: Ground truth class integer labels, shape [N].
        threshold: High confidence cutoff (e.g. 0.80).

    Returns:
        Dict with keys:
        - "high_conf_error_rate": error rate among high-confidence samples
        - "high_conf_coverage": proportion of total samples that are high-confidence
        - "high_conf_count": count of high-confidence samples
        - "high_conf_errors": count of errors in high-confidence subset
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    probs = np.asarray(probs)
    targets = np.asarray(targets)

    confidences = np.max(probs, axis=-1)
    preds = np.argmax(probs, axis=-1)
    is_correct = (preds == targets)

    high_conf_mask = (confidences >= threshold)
    high_conf_count = int(np.sum(high_conf_mask))
    total_count = len(targets)

    if high_conf_count == 0:
        return {
            "high_conf_error_rate": 0.0,
            "high_conf_coverage": 0.0,
            "high_conf_count": 0,
            "high_conf_errors": 0,
            "threshold": threshold,
        }

    high_conf_errors = int(np.sum(~is_correct[high_conf_mask]))
    high_conf_error_rate = float(high_conf_errors / high_conf_count)
    high_conf_coverage = float(high_conf_count / total_count)

    return {
        "high_conf_error_rate": high_conf_error_rate,
        "high_conf_coverage": high_conf_coverage,
        "high_conf_count": high_conf_count,
        "high_conf_errors": high_conf_errors,
        "threshold": threshold,
    }
