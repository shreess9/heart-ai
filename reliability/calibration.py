"""Confidence Calibration & Temperature Scaling.

Implements:
1. Temperature Scaling optimizer (fit T > 0 on validation logits via NLL).
2. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
3. Multi-class Brier Score.
4. Reliability binning analysis for calibration diagrams.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar


class TemperatureScaler(nn.Module):
    """Temperature scaling post-hoc calibration module for multi-class classification."""

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temperature)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by learned temperature parameter."""
        return self.scale_logits(logits)

    def scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Divide logits by temperature T."""
        t = torch.clamp(self.temperature, min=1e-3)
        return logits / t

    def calibrate_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled softmax probabilities."""
        scaled_logits = self.scale_logits(logits)
        return torch.softmax(scaled_logits, dim=-1)

    def fit(
        self,
        val_logits: Union[np.ndarray, torch.Tensor],
        val_targets: Union[np.ndarray, torch.Tensor],
        lr: float = 0.01,
        max_iter: int = 100,
    ) -> float:
        """Optimize temperature T on validation logits to minimize Negative Log-Likelihood (NLL)."""
        if isinstance(val_logits, np.ndarray):
            val_logits_t = torch.tensor(val_logits, dtype=torch.float32)
        else:
            val_logits_t = val_logits.detach().float()

        if isinstance(val_targets, np.ndarray):
            val_targets_t = torch.tensor(val_targets, dtype=torch.long)
        else:
            val_targets_t = val_targets.detach().long()

        nll_criterion = nn.CrossEntropyLoss()

        def _nll_loss_func(temp_val: float) -> float:
            if temp_val <= 0.001:
                return 1e9
            scaled = val_logits_t / temp_val
            loss = nll_criterion(scaled, val_targets_t).item()
            return loss

        # Scalar optimization over T in [0.05, 10.0]
        res = minimize_scalar(_nll_loss_func, bounds=(0.05, 10.0), method="bounded")
        optimal_t = float(res.x)
        self.temperature.data = torch.tensor([optimal_t], dtype=torch.float32)
        return optimal_t


def compute_ece_mce(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_bins: int = 15,
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        probs: Predicted probability distributions, shape [N, K].
        targets: Ground truth class integer labels, shape [N].
        num_bins: Number of confidence bins (standard is 10 or 15).

    Returns:
        Dict with keys "ece", "mce", and "bin_details".
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == targets).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    mce = 0.0
    n_samples = len(targets)
    bin_details = []

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # In interval [bin_lower, bin_upper) or [bin_lower, bin_upper] for the last bin
        if i == num_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = int(np.sum(in_bin))
        if bin_size > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)

            ece += (bin_size / n_samples) * bin_error
            mce = max(mce, bin_error)

            bin_details.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": bin_size,
                "bin_accuracy": bin_acc,
                "bin_confidence": bin_conf,
                "bin_error": bin_error,
            })
        else:
            bin_details.append({
                "bin_idx": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "bin_size": 0,
                "bin_accuracy": 0.0,
                "bin_confidence": 0.0,
                "bin_error": 0.0,
            })

    return {
        "ece": float(ece),
        "mce": float(mce),
        "bin_details": bin_details,
    }


def compute_brier_score(
    probs: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    num_classes: int = 7,
) -> float:
    """Compute multi-class Brier score: 1/N * sum_{i=1}^N sum_{k=1}^K (P_ik - Y_ik)^2."""
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    n_samples = len(targets)
    if n_samples == 0:
        return 0.0

    # One-hot encode targets
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float64)
    np.put_along_axis(one_hot, np.expand_dims(targets, axis=-1), 1.0, axis=-1)

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=-1))
    return float(brier)
