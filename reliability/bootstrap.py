"""Statistical Bootstrap Resampling Utilities.

Supports:
1. Standard sample-level non-parametric bootstrap.
2. Dialogue-cluster-level bootstrap resampling (clustered by Dialogue_ID).
3. 95% Confidence Interval calculation (percentile method).
4. Paired hypothesis testing / delta p-values.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    preds_or_probs: np.ndarray,
    targets: np.ndarray,
    cluster_ids: Optional[Union[np.ndarray, List[Union[str, int]]]] = None,
    num_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute non-parametric bootstrap confidence interval for a metric function.

    If cluster_ids is provided, performs cluster bootstrap resampling over unique dialogues.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(targets)
    alpha = (1.0 - confidence_level) / 2.0

    preds_or_probs = np.asarray(preds_or_probs)
    targets = np.asarray(targets)

    # Point estimate on entire dataset
    point_estimate = float(metric_fn(preds_or_probs, targets))

    boot_estimates = []

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)

        # Precompute mapping from cluster to sample indices
        cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

        for _ in range(num_iterations):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

            boot_preds = preds_or_probs[sampled_indices]
            boot_targets = targets[sampled_indices]
            boot_estimates.append(metric_fn(boot_preds, boot_targets))
    else:
        for _ in range(num_iterations):
            sampled_indices = rng.choice(n_samples, size=n_samples, replace=True)
            boot_preds = preds_or_probs[sampled_indices]
            boot_targets = targets[sampled_indices]
            boot_estimates.append(metric_fn(boot_preds, boot_targets))

    boot_estimates = np.array(boot_estimates, dtype=np.float64)
    ci_lower = float(np.percentile(boot_estimates, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_estimates, 100.0 * (1.0 - alpha)))
    std_err = float(np.std(boot_estimates))

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_err,
        "confidence_level": confidence_level,
        "num_iterations": num_iterations,
    }


def compute_paired_bootstrap_delta(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    model_a_preds: np.ndarray,
    model_b_preds: np.ndarray,
    targets: np.ndarray,
    cluster_ids: Optional[Union[np.ndarray, List[Union[str, int]]]] = None,
    num_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap distribution for the performance delta: Metric(Model B) - Metric(Model A)."""
    rng = np.random.RandomState(seed)
    n_samples = len(targets)
    alpha = (1.0 - confidence_level) / 2.0

    model_a_preds = np.asarray(model_a_preds)
    model_b_preds = np.asarray(model_b_preds)
    targets = np.asarray(targets)

    score_a = float(metric_fn(model_a_preds, targets))
    score_b = float(metric_fn(model_b_preds, targets))
    point_delta = score_b - score_a

    boot_deltas = []

    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}

        for _ in range(num_iterations):
            sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
            sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])

            s_a = metric_fn(model_a_preds[sampled_indices], targets[sampled_indices])
            s_b = metric_fn(model_b_preds[sampled_indices], targets[sampled_indices])
            boot_deltas.append(s_b - s_a)
    else:
        for _ in range(num_iterations):
            sampled_indices = rng.choice(n_samples, size=n_samples, replace=True)
            s_a = metric_fn(model_a_preds[sampled_indices], targets[sampled_indices])
            s_b = metric_fn(model_b_preds[sampled_indices], targets[sampled_indices])
            boot_deltas.append(s_b - s_a)

    boot_deltas = np.array(boot_deltas, dtype=np.float64)
    ci_lower = float(np.percentile(boot_deltas, 100.0 * alpha))
    ci_upper = float(np.percentile(boot_deltas, 100.0 * (1.0 - alpha)))

    # Empirical two-sided p-value against zero delta
    if point_delta >= 0:
        p_val = 2.0 * np.mean(boot_deltas <= 0)
    else:
        p_val = 2.0 * np.mean(boot_deltas >= 0)
    p_val = min(1.0, float(p_val))

    return {
        "model_a_score": score_a,
        "model_b_score": score_b,
        "point_delta": point_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "num_iterations": num_iterations,
    }
