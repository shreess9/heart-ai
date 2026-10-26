"""Controlled Multimodal Fusion Training, Calibration, Evaluation, and Benchmark Suite.

Canonical Implementation adhering strictly to the reproducibility audit:
- Clean unimodal feature extraction and head training without label leakage.
- Seed-isolated initialization for Model A, Model B, and Model C.
- Validation-locked temperature scaling.
- Locked test evaluation (2,610 samples, 280 dialogue clusters).
- Paired cluster bootstrap hypothesis testing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_and_reproduce import (
    extract_clean_unimodal_data,
    train_and_eval_fusion_model,
    BaselineMultimodalMLP,
    DisagreementAwareMLP,
    ReliabilityAwareFusion,
    compute_paired_bootstrap_delta,
    set_seed,
    CANONICAL_EMOTIONS,
    NUM_CLASSES,
)


def run_full_comparative_multimodal_experiment(seed: int = 42) -> Dict[str, Any]:
    """Execute canonical controlled Model A vs Model B vs Model C training, calibration, and benchmarking."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print(f"CANONICAL MULTIMODAL FUSION BENCHMARK ON REAL MELD DATA (Seed: {seed})")
    print("=" * 70)

    splits, data_tensors, _ = extract_clean_unimodal_data(seed=seed, device=device)

    # 1. Model A
    print("\n[1/3] Training Model A: Controlled Baseline MLP...")
    set_seed(seed)
    model_a = BaselineMultimodalMLP()
    res_a = train_and_eval_fusion_model(model_a, "baseline", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
    print(f"Model A Test Acc: {res_a['accuracy']*100:.2f}% | Macro-F1: {res_a['macro_f1']:.4f} | Raw ECE: {res_a['raw_ece']*100:.2f}% -> Cal ECE: {res_a['calibrated_ece']*100:.2f}% (T={res_a['temperature']:.3f})")

    # 2. Model B
    print("\n[2/3] Training Model B: Disagreement-Aware MLP (+ D1, D2, D3)...")
    set_seed(seed)
    model_b = DisagreementAwareMLP()
    res_b = train_and_eval_fusion_model(model_b, "disagreement_aware", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
    print(f"Model B Test Acc: {res_b['accuracy']*100:.2f}% | Macro-F1: {res_b['macro_f1']:.4f} | Raw ECE: {res_b['raw_ece']*100:.2f}% -> Cal ECE: {res_b['calibrated_ece']*100:.2f}% (T={res_b['temperature']:.3f})")

    # 3. Model C
    print("\n[3/3] Training Model C: Reliability-Aware Attention Fusion...")
    set_seed(seed)
    model_c = ReliabilityAwareFusion()
    res_c = train_and_eval_fusion_model(model_c, "reliability_aware", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
    print(f"Model C Test Acc: {res_c['accuracy']*100:.2f}% | Macro-F1: {res_c['macro_f1']:.4f} | Raw ECE: {res_c['raw_ece']*100:.2f}% -> Cal ECE: {res_c['calibrated_ece']*100:.2f}% (T={res_c['temperature']:.3f})")

    # 4. Statistical Significance Tests
    print("\n[4/4] Computing Paired Dialogue-Cluster Bootstrap Deltas against Model A Baseline...")
    targets = res_a["targets"]
    dia_ids = res_a["dialogue_ids"]

    def _f1_fn(p, y):
        pred_y = np.argmax(p, axis=-1) if p.ndim > 1 else p
        return f1_score(y, pred_y, average="macro", zero_division=0)

    delta_ba = compute_paired_bootstrap_delta(_f1_fn, res_a["raw_probs"], res_b["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=seed)
    delta_ca = compute_paired_bootstrap_delta(_f1_fn, res_a["raw_probs"], res_c["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=seed)

    print(f"Model B vs Model A Delta Macro-F1: {delta_ba['point_delta']:+.4f} (p = {delta_ba['p_value']:.4f})")
    print(f"Model C vs Model A Delta Macro-F1: {delta_ca['point_delta']:+.4f} (p = {delta_ca['p_value']:.4f})")

    return {
        "model_a": res_a,
        "model_b": res_b,
        "model_c": res_c,
        "delta_ba": delta_ba,
        "delta_ca": delta_ca,
    }


if __name__ == "__main__":
    run_full_comparative_multimodal_experiment()
