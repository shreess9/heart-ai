"""Unified Experiment Runner for Controlled Multimodal Emotion Benchmarking.

Evaluates Model A (Baseline), Model B (Disagreement-Aware), and Model C (Reliability-Aware)
under identical splits, seed policy, and evaluation metrics.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from fusion.baseline import BaselineMultimodalMLP
from fusion.disagreement_aware import DisagreementAwareMLP
from fusion.reliability_aware import ReliabilityAwareFusion
from reliability.calibration import compute_ece_mce, compute_brier_score, TemperatureScaler
from reliability.confidence import compute_high_confidence_error_rate
from reliability.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_delta
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3

EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class ExperimentBenchmarkRunner:
    """Standardized experiment runner comparing baseline and proposed fusion variants."""

    def __init__(
        self,
        num_classes: int = 7,
        feature_dim: int = 768,
        device: str = "cpu",
        random_seed: int = 42,
    ):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.random_seed = random_seed

    def generate_synthetic_benchmark_split(
        self,
        num_samples: int = 500,
        num_dialogues: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Generate a controlled test split harness for verification and pipeline testing."""
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        # Dialogue cluster IDs
        dialogue_ids = [f"dia_{i}" for i in rng.choice(num_dialogues, size=num_samples, replace=True)]
        targets = rng.choice(self.num_classes, size=num_samples)

        # Modality features with varying simulated noise
        t_features = torch.randn(num_samples, self.feature_dim)
        a_features = torch.randn(num_samples, self.feature_dim)
        v_features = torch.randn(num_samples, self.feature_dim)

        # Simulated modality probability distributions with realistic imbalance (Text > Audio/Visual)
        def _make_probs(signal_strength: float):
            logits = rng.randn(num_samples, self.num_classes)
            for i, target in enumerate(targets):
                logits[i, target] += signal_strength * rng.uniform(1.0, 3.0)
            exps = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            return torch.tensor(exps / np.sum(exps, axis=-1, keepdims=True), dtype=torch.float32)

        t_probs = _make_probs(signal_strength=2.2) # Text stronger
        a_probs = _make_probs(signal_strength=1.2) # Audio moderate
        v_probs = _make_probs(signal_strength=1.2) # Visual moderate

        return {
            "num_samples": num_samples,
            "dialogue_ids": np.array(dialogue_ids),
            "targets": np.array(targets),
            "t_features": t_features,
            "a_features": a_features,
            "v_features": v_features,
            "t_probs": t_probs,
            "a_probs": a_probs,
            "v_probs": v_probs,
        }

    def evaluate_model_predictions(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        dialogue_ids: Optional[np.ndarray] = None,
        hc_threshold: float = 0.80,
        bootstrap_iters: int = 500,
    ) -> Dict[str, Any]:
        """Compute full suite of recognition, reliability, calibration, and uncertainty metrics."""
        preds = np.argmax(probs, axis=-1)
        acc = float(accuracy_score(targets, preds))
        macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
        per_class_f1 = f1_score(targets, preds, average=None, zero_division=0).tolist()

        cal_metrics = compute_ece_mce(probs, targets, num_bins=15)
        brier = compute_brier_score(probs, targets, num_classes=self.num_classes)
        hc_stats = compute_high_confidence_error_rate(probs, targets, threshold=hc_threshold)

        # Bootstrap CI for Macro-F1
        def _f1_fn(p, y):
            pred_y = np.argmax(p, axis=-1) if p.ndim > 1 else p
            return f1_score(y, pred_y, average="macro", zero_division=0)

        boot_f1 = compute_bootstrap_ci(
            _f1_fn, probs, targets,
            cluster_ids=dialogue_ids,
            num_iterations=bootstrap_iters,
            seed=self.random_seed,
        )

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_f1": per_class_f1,
            "ece": cal_metrics["ece"],
            "mce": cal_metrics["mce"],
            "brier_score": brier,
            "high_conf_error_rate": hc_stats["high_conf_error_rate"],
            "high_conf_coverage": hc_stats["high_conf_coverage"],
            "macro_f1_ci": [boot_f1["ci_lower"], boot_f1["ci_upper"]],
        }

    def run_comparative_benchmark(
        self,
        dataset: Dict[str, Any],
        bootstrap_iters: int = 500,
    ) -> Dict[str, Any]:
        """Run controlled comparison across Model A, Model B, and Model C."""
        targets = dataset["targets"]
        dia_ids = dataset["dialogue_ids"]
        t_feat = dataset["t_features"].to(self.device)
        a_feat = dataset["a_features"].to(self.device)
        v_feat = dataset["v_features"].to(self.device)
        t_probs = dataset["t_probs"].to(self.device)
        a_probs = dataset["a_probs"].to(self.device)
        v_probs = dataset["v_probs"].to(self.device)

        # Instantiate models
        model_a = BaselineMultimodalMLP().to(self.device)
        model_b = DisagreementAwareMLP().to(self.device)
        model_c = ReliabilityAwareFusion().to(self.device)

        model_a.eval()
        model_b.eval()
        model_c.eval()

        with torch.no_grad():
            probs_a = model_a.predict_distribution(t_feat, a_feat, v_feat).cpu().numpy()
            probs_b = model_b.predict_distribution(
                t_feat, a_feat, v_feat,
                text_probs=t_probs, audio_probs=a_probs, visual_probs=v_probs
            ).cpu().numpy()
            _, cal_probs_c = model_c.forward_calibrated(
                t_feat, a_feat, v_feat,
                text_probs=t_probs, audio_probs=a_probs, visual_probs=v_probs
            )
            probs_c = cal_probs_c.cpu().numpy()

        results_a = self.evaluate_model_predictions(probs_a, targets, dia_ids, bootstrap_iters=bootstrap_iters)
        results_b = self.evaluate_model_predictions(probs_b, targets, dia_ids, bootstrap_iters=bootstrap_iters)
        results_c = self.evaluate_model_predictions(probs_c, targets, dia_ids, bootstrap_iters=bootstrap_iters)

        # Paired statistical comparisons against Model A baseline
        def _f1_fn(p, y):
            pred_y = np.argmax(p, axis=-1) if p.ndim > 1 else p
            return f1_score(y, pred_y, average="macro", zero_division=0)

        delta_b_vs_a = compute_paired_bootstrap_delta(
            _f1_fn, probs_a, probs_b, targets, cluster_ids=dia_ids, num_iterations=bootstrap_iters
        )
        delta_c_vs_a = compute_paired_bootstrap_delta(
            _f1_fn, probs_a, probs_c, targets, cluster_ids=dia_ids, num_iterations=bootstrap_iters
        )

        return {
            "model_a_baseline": results_a,
            "model_b_disagreement_aware": results_b,
            "model_c_reliability_aware": results_c,
            "paired_deltas": {
                "b_vs_a": delta_b_vs_a,
                "c_vs_a": delta_c_vs_a,
            },
            "num_samples": len(targets),
            "num_dialogue_clusters": len(np.unique(dia_ids)),
        }
