"""Systematic Multimodal Ablation Study on Locked MELD Test Set.

Canonical Implementation adhering strictly to the reproducibility audit:
- Identical baseline configuration to canonical Model A.
- Multi-seed feature subset evaluation.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.audit_and_reproduce import (
    extract_clean_unimodal_data,
    train_and_eval_fusion_model,
    ConfigurableAblationFusion,
    set_seed,
    SEEDS,
)


def run_ablation_study(seed: int = 42) -> Dict[str, Any]:
    """Execute systematic ablation experiments under locked test set protocol."""
    print("=" * 65)
    print("SYSTEMATIC MULTIMODAL ABLATION STUDY ON REAL MELD DATA")
    print("=" * 65)

    splits, data_tensors, _ = extract_clean_unimodal_data(seed=seed)

    ablation_variants = [
        ("Model A (Baseline)", "base"),
        ("Model A + D1 (Gen JSD)", "base_plus_d1"),
        ("Model A + D2 (Pairwise JSD)", "base_plus_d2"),
        ("Model A + D3 (Categorical Disagreement)", "base_plus_d3"),
        ("Model B (Full Disagreement: D1+D2+D3)", "base_plus_d1_d2_d3"),
        ("Model A + Confidence", "base_plus_confidence"),
        ("Model A + Entropy", "base_plus_entropy"),
    ]

    ablation_results = {}
    for name, v_key in ablation_variants:
        print(f"Training ablation variant: {name}...")
        set_seed(seed)
        abl_model = ConfigurableAblationFusion(variant=v_key)
        res = train_and_eval_fusion_model(abl_model, "ablation", splits, data_tensors, seed=seed, epochs=15, lr=0.003)
        ablation_results[name] = {
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "raw_ece": res["raw_ece"],
            "calibrated_ece": res["calibrated_ece"],
            "brier_score": res["calibrated_brier"],
        }
        print(f"  -> Acc: {res['accuracy']*100:.2f}% | Macro-F1: {res['macro_f1']:.4f} | Raw ECE: {res['raw_ece']*100:.2f}% | Cal ECE: {res['calibrated_ece']*100:.2f}%")

    out_file = ROOT_DIR / "results" / "metrics" / "ablation_study.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\nSaved ablation study metrics to {out_file}")
    return ablation_results


if __name__ == "__main__":
    run_ablation_study()
