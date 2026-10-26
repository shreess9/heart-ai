"""Reliability & Uncertainty Math Engine for H.E.A.R.T. AI."""

from reliability.jsd import generalized_jsd_3, pairwise_jsd_2, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot
from reliability.entropy import compute_predictive_entropy
from reliability.confidence import compute_confidence, compute_high_confidence_error_rate
from reliability.calibration import TemperatureScaler, compute_ece_mce, compute_brier_score
from reliability.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_delta

__all__ = [
    "generalized_jsd_3",
    "pairwise_jsd_2",
    "mean_pairwise_jsd_3",
    "compute_categorical_disagreement_d3",
    "d3_to_one_hot",
    "compute_predictive_entropy",
    "compute_confidence",
    "compute_high_confidence_error_rate",
    "TemperatureScaler",
    "compute_ece_mce",
    "compute_brier_score",
    "compute_bootstrap_ci",
    "compute_paired_bootstrap_delta",
]
