"""H.E.A.R.T. AI: Comprehensive Reproducibility and Experimental Audit Suite.

Executes:
1. Real Unimodal Baseline Training & Evaluation (Text, Audio, Visual) on canonical MELD splits.
2. Disagreement Analysis (D1, D2, D3, entropy correlation, Gate 2 correctness regression).
3. Multi-seed (42, 123, 456) Controlled Fusion Experiments:
   - Model A (Baseline Concatenation MLP)
   - Model B (Disagreement-Aware MLP: Base + D1 + D2 + D3)
   - Model C (Reliability-Aware Attention + Fusion)
   - Systematic Feature Ablations (Base, +D1, +D2, +D3, +D1+D2+D3, +Confidence, +Entropy)
4. Calibration Audit (Dev-locked Temperature Scaling, Raw vs Calibrated ECE/MCE/Brier/High-Conf Error).
5. Paired Dialogue-Cluster Bootstrap Statistical Significance Testing.
6. Generation of `results/metrics/canonical_ab_comparison.csv` and updated research reports.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.unimodal_engine import UnimodalEngine, UnimodalMLPHead
from fusion.baseline import BaselineMultimodalMLP
from fusion.disagreement_aware import DisagreementAwareMLP
from fusion.reliability_aware import ReliabilityAwareFusion
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot
from reliability.entropy import compute_predictive_entropy
from reliability.calibration import TemperatureScaler, compute_ece_mce, compute_brier_score
from reliability.confidence import compute_high_confidence_error_rate
from reliability.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_delta

CANONICAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_CLASSES = 7
SEEDS = [42, 123, 456]


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], t_feat, a_feat, v_feat, t_probs, a_probs, v_probs):
        self.records = records
        self.t_feat = t_feat
        self.a_feat = a_feat
        self.v_feat = v_feat
        self.t_probs = t_probs
        self.a_probs = a_probs
        self.v_probs = v_probs
        self.labels = torch.tensor([r["Emotion_ID"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return {
            "t_feat": self.t_feat[idx],
            "a_feat": self.a_feat[idx],
            "v_feat": self.v_feat[idx],
            "t_probs": self.t_probs[idx],
            "a_probs": self.a_probs[idx],
            "v_probs": self.v_probs[idx],
            "label": self.labels[idx],
            "record": self.records[idx],
        }


def extract_clean_unimodal_data(seed: int = 42, device: str = "cpu") -> Tuple[Dict[str, Any], Dict[str, Dict[str, torch.Tensor]], Dict[str, Any]]:
    """Train unimodal classifiers on train/dev and extract authentic feature embeddings and probabilities."""
    print(">>> [Phase 1] Extracting Clean Unimodal Representations...")
    u_engine = UnimodalEngine(device=device, seed=seed)
    splits = u_engine.splits

    # Extract continuous 768-dim embeddings
    embeddings = {}
    for s in ["train", "dev", "test"]:
        embeddings[s] = {
            "text": u_engine.extract_or_generate_modality_embeddings("text", s),
            "audio": u_engine.extract_or_generate_modality_embeddings("audio", s),
            "visual": u_engine.extract_or_generate_modality_embeddings("visual", s),
        }

    # Train unimodal heads on train split, best dev checkpoint
    unimodal_models = {}
    unimodal_test_metrics = {}
    for mod in ["text", "audio", "visual"]:
        set_seed(seed)
        model = u_engine.train_unimodal_classifier(
            mod,
            embeddings["train"][mod],
            embeddings["dev"][mod],
            epochs=15,
            lr=0.005,
            batch_size=64,
        )
        unimodal_models[mod] = model
        metrics, cache = u_engine.evaluate_and_cache_modality(mod, model, embeddings["test"][mod])
        unimodal_test_metrics[mod] = metrics
        print(f"    [{mod.upper()}] Acc: {metrics['accuracy']*100:.2f}% | Macro-F1: {metrics['macro_f1']:.4f} | ECE: {metrics['ece']*100:.2f}%")

    # Generate probabilities for all splits using the trained unimodal heads (NO label leakage)
    data_tensors = {}
    for s in ["train", "dev", "test"]:
        t_probs_list = []
        for mod in ["text", "audio", "visual"]:
            mod_model = unimodal_models[mod]
            mod_model.eval()
            with torch.no_grad():
                feat = embeddings[s][mod].to(device)
                logits = mod_model(feat)
                probs = torch.softmax(logits, dim=-1).cpu()
                t_probs_list.append(probs)

        data_tensors[s] = {
            "t_feat": embeddings[s]["text"],
            "a_feat": embeddings[s]["audio"],
            "v_feat": embeddings[s]["visual"],
            "t_probs": t_probs_list[0],
            "a_probs": t_probs_list[1],
            "v_probs": t_probs_list[2],
        }

    return splits, data_tensors, unimodal_test_metrics


def run_disagreement_analysis(splits: Dict[str, Any], data_tensors: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Perform rigorous cross-modal disagreement analysis on test split."""
    print(">>> [Phase 2] Running Disagreement & Uncertainty Mathematical Audit...")
    test_tensors = data_tensors["test"]
    t_probs = test_tensors["t_probs"].numpy()
    a_probs = test_tensors["a_probs"].numpy()
    v_probs = test_tensors["v_probs"].numpy()

    t_preds = np.argmax(t_probs, axis=-1)
    a_preds = np.argmax(a_probs, axis=-1)
    v_preds = np.argmax(v_probs, axis=-1)

    targets = np.array([r["Emotion_ID"] for r in splits["test"]])
    n_samples = len(targets)

    d1_list, d2_list, d3_list, ent_list = [], [], [], []
    for i in range(n_samples):
        d1 = float(generalized_jsd_3(t_probs[i], a_probs[i], v_probs[i], normalized=True))
        d2 = float(mean_pairwise_jsd_3(t_probs[i], a_probs[i], v_probs[i], normalized=True))
        d3 = int(compute_categorical_disagreement_d3(t_preds[i], a_preds[i], v_preds[i]))
        fused_p = (t_probs[i] + a_probs[i] + v_probs[i]) / 3.0
        ent = float(compute_predictive_entropy(fused_p, normalized=False))

        d1_list.append(d1)
        d2_list.append(d2)
        d3_list.append(d3)
        ent_list.append(ent)

    d1_arr = np.array(d1_list)
    d2_arr = np.array(d2_list)
    d3_arr = np.array(d3_list)
    ent_arr = np.array(ent_list)

    # 1. D3 distribution
    d3_counts = {int(k): int(np.sum(d3_arr == k)) for k in [0, 1, 2]}
    d3_pcts = {k: round(v / n_samples * 100.0, 2) for k, v in d3_counts.items()}
    mean_jsd_by_d3 = {k: float(np.mean(d1_arr[d3_arr == k])) for k in [0, 1, 2]}

    # 2. D1 vs D2 correlation
    d1_d2_rho, d1_d2_pval = stats.spearmanr(d1_arr, d2_arr)

    # 3. Disagreement vs Multimodal Predictive Entropy
    slope, intercept, r_value, p_value, std_err = stats.linregress(d1_arr, ent_arr)
    ent_rho, ent_rho_pval = stats.spearmanr(d1_arr, ent_arr)

    # 4. Gate 2: Correctness vs Text Confidence and Disagreement (Logistic Regression)
    text_conf = np.max(t_probs, axis=-1)
    correct = (t_preds == targets).astype(float)
    X = pd.DataFrame({"const": 1.0, "text_conf": text_conf, "d1": d1_arr})
    logit_mod = sm.Logit(correct, X)
    try:
        logit_res = logit_mod.fit(disp=False)
        t_conf_coef = float(logit_res.params["text_conf"])
        t_conf_pval = float(logit_res.pvalues["text_conf"])
        d1_coef = float(logit_res.params["d1"])
        d1_pval = float(logit_res.pvalues["d1"])
    except Exception as e:
        t_conf_coef, t_conf_pval, d1_coef, d1_pval = 0.0, 1.0, 0.0, 1.0

    disagreement_results = {
        "num_samples": n_samples,
        "d3_distribution": {
            "counts": d3_counts,
            "percentages": d3_pcts,
            "mean_jsd_by_d3": mean_jsd_by_d3,
        },
        "d1_d2_spearman_rho": float(d1_d2_rho),
        "d1_d2_pval": float(d1_d2_pval),
        "disagreement_vs_entropy": {
            "linear_slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "spearman_rho": float(ent_rho),
            "spearman_pval": float(ent_rho_pval),
        },
        "gate2_correctness_regression": {
            "text_conf_coef": t_conf_coef,
            "text_conf_pval": t_conf_pval,
            "d1_disagreement_coef": d1_coef,
            "d1_disagreement_pval": d1_pval,
        }
    }

    print(f"    D3 Distribution: Level 0 (All agree) = {d3_counts[0]} ({d3_pcts[0]}%, Mean JSD={mean_jsd_by_d3[0]:.4f}), Level 1 = {d3_counts[1]} ({d3_pcts[1]}%, Mean JSD={mean_jsd_by_d3[1]:.4f}), Level 2 = {d3_counts[2]} ({d3_pcts[2]}%, Mean JSD={mean_jsd_by_d3[2]:.4f})")
    print(f"    D1 vs D2 Spearman Rho: {d1_d2_rho:.5f} (p = {d1_d2_pval:.2e})")
    print(f"    D1 vs Entropy Spearman Rho: {ent_rho:.4f} (p = {ent_rho_pval:.2e}, Slope = {slope:.4f})")
    print(f"    Gate 2: Text Conf Coef = {t_conf_coef:.4f} (p={t_conf_pval:.2e}), D1 Coef = {d1_coef:.4f} (p={d1_pval:.2e})")

    with open(ROOT_DIR / "results" / "metrics" / "disagreement_analysis.json", "w", encoding="utf-8") as f:
        json.dump(disagreement_results, f, indent=2)

    return disagreement_results


def train_and_eval_fusion_model(
    model: nn.Module,
    model_type: str,
    splits: Dict[str, Any],
    data_tensors: Dict[str, Dict[str, torch.Tensor]],
    seed: int = 42,
    epochs: int = 15,
    lr: float = 0.003,
    batch_size: int = 64,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Train a fusion model on train split, select best dev checkpoint, calibrate on dev, evaluate on test."""
    set_seed(seed)
    train_data = MultimodalDataset(splits["train"], **data_tensors["train"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    dev_tensors = data_tensors["dev"]
    dev_y = torch.tensor([r["Emotion_ID"] for r in splits["dev"]], dtype=torch.long, device=device)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_dev_f1 = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for b in train_loader:
            t_f = b["t_feat"].to(device)
            a_f = b["a_feat"].to(device)
            v_f = b["v_feat"].to(device)
            t_p = b["t_probs"].to(device)
            a_p = b["a_probs"].to(device)
            v_p = b["v_probs"].to(device)
            by = b["label"].to(device)

            optimizer.zero_grad()
            if model_type == "baseline":
                logits = model(t_f, a_f, v_f)
            elif model_type in ["disagreement_aware", "reliability_aware"]:
                logits = model(t_f, a_f, v_f, text_probs=t_p, audio_probs=a_p, visual_probs=v_p)
            else: # custom ablation
                logits = model(t_f, a_f, v_f, text_probs=t_p, audio_probs=a_p, visual_probs=v_p)

            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Dev checkpoint selection
        model.eval()
        with torch.no_grad():
            dt_f = dev_tensors["t_feat"].to(device)
            da_f = dev_tensors["a_feat"].to(device)
            dv_f = dev_tensors["v_feat"].to(device)
            dt_p = dev_tensors["t_probs"].to(device)
            da_p = dev_tensors["a_probs"].to(device)
            dv_p = dev_tensors["v_probs"].to(device)

            if model_type == "baseline":
                dev_logits = model(dt_f, da_f, dv_f)
            else:
                dev_logits = model(dt_f, da_f, dv_f, text_probs=dt_p, audio_probs=da_p, visual_probs=dv_p)

            dev_preds = torch.argmax(dev_logits, dim=-1).cpu().numpy()
            dev_f1 = f1_score(dev_y.cpu().numpy(), dev_preds, average="macro", zero_division=0)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Validation Temperature Scaling
    model.eval()
    with torch.no_grad():
        dt_f = dev_tensors["t_feat"].to(device)
        da_f = dev_tensors["a_feat"].to(device)
        dv_f = dev_tensors["v_feat"].to(device)
        dt_p = dev_tensors["t_probs"].to(device)
        da_p = dev_tensors["a_probs"].to(device)
        dv_p = dev_tensors["v_probs"].to(device)
        if model_type == "baseline":
            dev_logits_np = model(dt_f, da_f, dv_f).cpu().numpy()
        else:
            dev_logits_np = model(dt_f, da_f, dv_f, text_probs=dt_p, audio_probs=da_p, visual_probs=dv_p).cpu().numpy()

    scaler = TemperatureScaler()
    optimal_t = float(scaler.fit(dev_logits_np, dev_y.cpu().numpy()))

    # Locked Test Evaluation
    test_tensors = data_tensors["test"]
    test_records = splits["test"]
    targets = np.array([r["Emotion_ID"] for r in test_records])
    dia_ids = np.array([r["Dialogue_ID"] for r in test_records])

    with torch.no_grad():
        tt_f = test_tensors["t_feat"].to(device)
        ta_f = test_tensors["a_feat"].to(device)
        tv_f = test_tensors["v_feat"].to(device)
        tt_p = test_tensors["t_probs"].to(device)
        ta_p = test_tensors["a_probs"].to(device)
        tv_p = test_tensors["v_probs"].to(device)

        if model_type == "baseline":
            raw_logits_t = model(tt_f, ta_f, tv_f).cpu()
        else:
            raw_logits_t = model(tt_f, ta_f, tv_f, text_probs=tt_p, audio_probs=ta_p, visual_probs=tv_p).cpu()

        raw_logits = raw_logits_t.numpy()
        raw_probs = torch.softmax(raw_logits_t, dim=-1).numpy()
        cal_probs = torch.softmax(raw_logits_t / optimal_t, dim=-1).numpy()

    preds = np.argmax(raw_probs, axis=-1)
    acc = float(accuracy_score(targets, preds))
    macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(targets, preds, average="weighted", zero_division=0))
    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(targets, preds, labels=list(range(NUM_CLASSES))).tolist()

    raw_cal = compute_ece_mce(raw_probs, targets, num_bins=15)
    cal_cal = compute_ece_mce(cal_probs, targets, num_bins=15)
    raw_brier = compute_brier_score(raw_probs, targets, num_classes=NUM_CLASSES)
    cal_brier = compute_brier_score(cal_probs, targets, num_classes=NUM_CLASSES)
    hc_raw = compute_high_confidence_error_rate(raw_probs, targets, threshold=0.80)
    hc_cal = compute_high_confidence_error_rate(cal_probs, targets, threshold=0.80)

    return {
        "model_type": model_type,
        "seed": seed,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {CANONICAL_EMOTIONS[i]: round(f1, 4) for i, f1 in enumerate(per_class_f1)},
        "confusion_matrix": cm,
        "temperature": round(optimal_t, 5),
        "raw_ece": raw_cal["ece"],
        "calibrated_ece": cal_cal["ece"],
        "raw_mce": raw_cal["mce"],
        "calibrated_mce": cal_cal["mce"],
        "raw_brier": raw_brier,
        "calibrated_brier": cal_brier,
        "high_conf_error_raw": hc_raw["high_conf_error_rate"],
        "high_conf_error_cal": hc_cal["high_conf_error_rate"],
        "high_conf_coverage_raw": hc_raw["high_conf_coverage"],
        "high_conf_coverage_cal": hc_cal["high_conf_coverage"],
        "raw_probs": raw_probs,
        "cal_probs": cal_probs,
        "targets": targets,
        "dialogue_ids": dia_ids,
    }


# Ablation wrapper model
class ConfigurableAblationFusion(nn.Module):
    def __init__(self, variant: str, hidden_dims: List[int] = [512, 256], num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        self.variant = variant
        base_dim = 2304
        if variant == "base":
            in_dim = base_dim
        elif variant in ["base_plus_d1", "base_plus_d2"]:
            in_dim = base_dim + 1
        elif variant == "base_plus_d3":
            in_dim = base_dim + 3
        elif variant == "base_plus_d1_d2_d3": # Model B
            in_dim = base_dim + 5
        elif variant == "base_plus_confidence":
            in_dim = base_dim + 3
        elif variant == "base_plus_entropy":
            in_dim = base_dim + 3
        else:
            in_dim = base_dim

        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, t_f, a_f, v_f, text_probs=None, audio_probs=None, visual_probs=None):
        base = torch.cat([t_f, a_f, v_f], dim=-1)
        if self.variant == "base":
            inp = base
        elif self.variant == "base_plus_d1":
            d1 = generalized_jsd_3(text_probs, audio_probs, visual_probs, normalized=True).unsqueeze(-1)
            inp = torch.cat([base, d1], dim=-1)
        elif self.variant == "base_plus_d2":
            d2 = mean_pairwise_jsd_3(text_probs, audio_probs, visual_probs, normalized=True).unsqueeze(-1)
            inp = torch.cat([base, d2], dim=-1)
        elif self.variant == "base_plus_d3":
            d3 = compute_categorical_disagreement_d3(text_probs, audio_probs, visual_probs)
            d3_oh = d3_to_one_hot(d3, num_levels=3)
            inp = torch.cat([base, d3_oh], dim=-1)
        elif self.variant == "base_plus_d1_d2_d3":
            d1 = generalized_jsd_3(text_probs, audio_probs, visual_probs, normalized=True).unsqueeze(-1)
            d2 = mean_pairwise_jsd_3(text_probs, audio_probs, visual_probs, normalized=True).unsqueeze(-1)
            d3 = compute_categorical_disagreement_d3(text_probs, audio_probs, visual_probs)
            d3_oh = d3_to_one_hot(d3, num_levels=3)
            inp = torch.cat([base, d1, d2, d3_oh], dim=-1)
        elif self.variant == "base_plus_confidence":
            c_t = torch.max(text_probs, dim=-1, keepdim=True).values
            c_a = torch.max(audio_probs, dim=-1, keepdim=True).values
            c_v = torch.max(visual_probs, dim=-1, keepdim=True).values
            inp = torch.cat([base, c_t, c_a, c_v], dim=-1)
        elif self.variant == "base_plus_entropy":
            e_t = compute_predictive_entropy(text_probs, normalized=True).unsqueeze(-1)
            e_a = compute_predictive_entropy(audio_probs, normalized=True).unsqueeze(-1)
            e_v = compute_predictive_entropy(visual_probs, normalized=True).unsqueeze(-1)
            inp = torch.cat([base, e_t, e_a, e_v], dim=-1)
        else:
            inp = base
        return self.net(inp)


def run_full_reproducibility_audit():
    print("=" * 75)
    print("H.E.A.R.T. AI — COMPREHENSIVE REPRODUCIBILITY AUDIT & BENCHMARK")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Hardware execution environment: {device.upper()} | PyTorch: {torch.__version__} | NumPy: {np.__version__}")

    # 1. Clean Unimodal Baselines & Feature Extraction
    splits, data_tensors, unimodal_test_metrics = extract_clean_unimodal_data(seed=42, device=device)

    # 2. Mathematical Disagreement Audit
    disagreement_results = run_disagreement_analysis(splits, data_tensors)

    # 3. Multi-Seed Controlled Benchmark (Model A, Model B, Model C)
    print("\n>>> [Phase 3] Running Controlled Multi-Seed Benchmark (Model A vs B vs C)...")
    model_records = []
    benchmark_runs = {s: {} for s in SEEDS}

    for seed in SEEDS:
        print(f"\n--- Running Seed: {seed} ---")
        # Model A
        set_seed(seed)
        model_a = BaselineMultimodalMLP()
        res_a = train_and_eval_fusion_model(model_a, "baseline", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
        benchmark_runs[seed]["Model_A"] = res_a
        print(f"  [Model A (Baseline)]       Acc: {res_a['accuracy']*100:.2f}% | Macro-F1: {res_a['macro_f1']:.4f} | Raw ECE: {res_a['raw_ece']*100:.2f}% -> Cal ECE: {res_a['calibrated_ece']*100:.2f}% (T={res_a['temperature']:.3f})")

        # Model B
        set_seed(seed)
        model_b = DisagreementAwareMLP()
        res_b = train_and_eval_fusion_model(model_b, "disagreement_aware", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
        benchmark_runs[seed]["Model_B"] = res_b
        print(f"  [Model B (Disagreement)]   Acc: {res_b['accuracy']*100:.2f}% | Macro-F1: {res_b['macro_f1']:.4f} | Raw ECE: {res_b['raw_ece']*100:.2f}% -> Cal ECE: {res_b['calibrated_ece']*100:.2f}% (T={res_b['temperature']:.3f})")

        # Model C
        set_seed(seed)
        model_c = ReliabilityAwareFusion()
        res_c = train_and_eval_fusion_model(model_c, "reliability_aware", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
        benchmark_runs[seed]["Model_C"] = res_c
        print(f"  [Model C (Reliability)]    Acc: {res_c['accuracy']*100:.2f}% | Macro-F1: {res_c['macro_f1']:.4f} | Raw ECE: {res_c['raw_ece']*100:.2f}% -> Cal ECE: {res_c['calibrated_ece']*100:.2f}% (T={res_c['temperature']:.3f})")

    # 4. Multi-Seed Systematic Feature Ablation Study
    print("\n>>> [Phase 4] Running Multi-Seed Feature Ablation Experiments...")
    ablation_variants = [
        ("Model A (Baseline)", "base"),
        ("Model A + D1 (Gen JSD)", "base_plus_d1"),
        ("Model A + D2 (Pairwise JSD)", "base_plus_d2"),
        ("Model A + D3 (Categorical Disagreement)", "base_plus_d3"),
        ("Model B (Full Disagreement: D1+D2+D3)", "base_plus_d1_d2_d3"),
        ("Model A + Confidence", "base_plus_confidence"),
        ("Model A + Entropy", "base_plus_entropy"),
    ]

    ablation_runs = {name: [] for name, _ in ablation_variants}
    for name, v_key in ablation_variants:
        for seed in SEEDS:
            set_seed(seed)
            abl_model = ConfigurableAblationFusion(variant=v_key)
            res = train_and_eval_fusion_model(abl_model, "ablation", splits, data_tensors, seed=seed, epochs=15, lr=0.003, device=device)
            ablation_runs[name].append(res)
        
        accs = [r["accuracy"] for r in ablation_runs[name]]
        f1s = [r["macro_f1"] for r in ablation_runs[name]]
        eces = [r["raw_ece"] for r in ablation_runs[name]]
        cal_eces = [r["calibrated_ece"] for r in ablation_runs[name]]
        briers = [r["raw_brier"] for r in ablation_runs[name]]
        print(f"  [{name:<40}] Acc: {np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}% | Macro-F1: {np.mean(f1s):.4f}±{np.std(f1s):.4f} | Raw ECE: {np.mean(eces)*100:.2f}% | Cal ECE: {np.mean(cal_eces)*100:.2f}% | Brier: {np.mean(briers):.4f}")

    # 5. Statistical Significance Testing on Canonical Seed (Seed 42)
    print("\n>>> [Phase 5] Computing Clustered Bootstrap Hypothesis Tests (Seed 42)...")
    res_a42 = benchmark_runs[42]["Model_A"]
    res_b42 = benchmark_runs[42]["Model_B"]
    res_c42 = benchmark_runs[42]["Model_C"]

    targets = res_a42["targets"]
    dia_ids = res_a42["dialogue_ids"]

    def _f1_fn(p, y):
        pred_y = np.argmax(p, axis=-1) if p.ndim > 1 else p
        return f1_score(y, pred_y, average="macro", zero_division=0)

    def _acc_fn(p, y):
        pred_y = np.argmax(p, axis=-1) if p.ndim > 1 else p
        return accuracy_score(y, pred_y)

    delta_ba_f1 = compute_paired_bootstrap_delta(_f1_fn, res_a42["raw_probs"], res_b42["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=42)
    delta_ba_acc = compute_paired_bootstrap_delta(_acc_fn, res_a42["raw_probs"], res_b42["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=42)

    delta_ca_f1 = compute_paired_bootstrap_delta(_f1_fn, res_a42["raw_probs"], res_c42["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=42)
    delta_ca_acc = compute_paired_bootstrap_delta(_acc_fn, res_a42["raw_probs"], res_c42["raw_probs"], targets, cluster_ids=dia_ids, num_iterations=1000, seed=42)

    print(f"  Model B vs Model A: Delta Macro-F1 = {delta_ba_f1['point_delta']:+.4f} (95% CI: [{delta_ba_f1['ci_lower']:+.4f}, {delta_ba_f1['ci_upper']:+.4f}], p = {delta_ba_f1['p_value']:.4f})")
    print(f"                      Delta Accuracy = {delta_ba_acc['point_delta']*100:+.2f}% (95% CI: [{delta_ba_acc['ci_lower']*100:+.2f}%, {delta_ba_acc['ci_upper']*100:+.2f}%], p = {delta_ba_acc['p_value']:.4f})")
    print(f"  Model C vs Model A: Delta Macro-F1 = {delta_ca_f1['point_delta']:+.4f} (95% CI: [{delta_ca_f1['ci_lower']:+.4f}, {delta_ca_f1['ci_upper']:+.4f}], p = {delta_ca_f1['p_value']:.4f})")
    print(f"                      Delta Accuracy = {delta_ca_acc['point_delta']*100:+.2f}% (95% CI: [{delta_ca_acc['ci_lower']*100:+.2f}%, {delta_ca_acc['ci_upper']*100:+.2f}%], p = {delta_ca_acc['p_value']:.4f})")

    # 6. Build Canonical Results CSV
    csv_rows = []
    exp_id = "EXP-MELD-AUDIT-2026-CANONICAL"
    git_commit = "meld-audit-canonical"
    ckpt_ver = "v1.0-canonical"

    for seed in SEEDS:
        for m_key, m_name in [("Model_A", "Model A (Baseline Concatenation)"), ("Model_B", "Model B (Disagreement-Aware)"), ("Model_C", "Model C (Reliability-Aware Attention)")]:
            r = benchmark_runs[seed][m_key]
            csv_rows.append({
                "experiment_id": exp_id,
                "model": m_name,
                "seed": seed,
                "accuracy": round(r["accuracy"], 6),
                "macro_f1": round(r["macro_f1"], 6),
                "weighted_f1": round(r["weighted_f1"], 6),
                "raw_ece": round(r["raw_ece"], 6),
                "calibrated_ece": round(r["calibrated_ece"], 6),
                "mce": round(r["calibrated_mce"], 6),
                "brier": round(r["calibrated_brier"], 6),
                "high_confidence_error": round(r["high_conf_error_cal"], 6),
                "temperature": round(r["temperature"], 5),
                "git_commit": git_commit,
                "checkpoint_version": ckpt_ver,
            })

    csv_df = pd.DataFrame(csv_rows)
    csv_path = ROOT_DIR / "results" / "metrics" / "canonical_ab_comparison.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"\n>>> Saved Canonical Results CSV to: {csv_path}")

    # 7. Update JSON Metrics Files
    # Save benchmark JSON
    benchmark_summary = {
        "canonical_seed_42": {
            "model_a": {k: v for k, v in res_a42.items() if not isinstance(v, (np.ndarray, list)) or k == "per_class_f1"},
            "model_b": {k: v for k, v in res_b42.items() if not isinstance(v, (np.ndarray, list)) or k == "per_class_f1"},
            "model_c": {k: v for k, v in res_c42.items() if not isinstance(v, (np.ndarray, list)) or k == "per_class_f1"},
            "paired_deltas": {
                "b_vs_a": {
                    "macro_f1": delta_ba_f1,
                    "accuracy": delta_ba_acc,
                },
                "c_vs_a": {
                    "macro_f1": delta_ca_f1,
                    "accuracy": delta_ca_acc,
                }
            }
        },
        "multi_seed_aggregate": {
            m_key: {
                "accuracy_mean": float(np.mean([benchmark_runs[s][m_key]["accuracy"] for s in SEEDS])),
                "accuracy_std": float(np.std([benchmark_runs[s][m_key]["accuracy"] for s in SEEDS])),
                "macro_f1_mean": float(np.mean([benchmark_runs[s][m_key]["macro_f1"] for s in SEEDS])),
                "macro_f1_std": float(np.std([benchmark_runs[s][m_key]["macro_f1"] for s in SEEDS])),
                "calibrated_ece_mean": float(np.mean([benchmark_runs[s][m_key]["calibrated_ece"] for s in SEEDS])),
                "calibrated_ece_std": float(np.std([benchmark_runs[s][m_key]["calibrated_ece"] for s in SEEDS])),
                "raw_ece_mean": float(np.mean([benchmark_runs[s][m_key]["raw_ece"] for s in SEEDS])),
                "calibrated_brier_mean": float(np.mean([benchmark_runs[s][m_key]["calibrated_brier"] for s in SEEDS])),
                "high_conf_error_cal_mean": float(np.mean([benchmark_runs[s][m_key]["high_conf_error_cal"] for s in SEEDS])),
            }
            for m_key in ["Model_A", "Model_B", "Model_C"]
        }
    }
    with open(ROOT_DIR / "results" / "metrics" / "model_abc_comparison.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Save ablation JSON
    ablation_summary = {
        name: {
            "accuracy_mean": float(np.mean([r["accuracy"] for r in ablation_runs[name]])),
            "accuracy_std": float(np.std([r["accuracy"] for r in ablation_runs[name]])),
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in ablation_runs[name]])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in ablation_runs[name]])),
            "raw_ece_mean": float(np.mean([r["raw_ece"] for r in ablation_runs[name]])),
            "calibrated_ece_mean": float(np.mean([r["calibrated_ece"] for r in ablation_runs[name]])),
            "brier_score_mean": float(np.mean([r["calibrated_brier"] for r in ablation_runs[name]])),
        }
        for name, _ in ablation_variants
    }
    with open(ROOT_DIR / "results" / "metrics" / "ablation_study.json", "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    # Cache prediction records for Seed 42
    pred_dir = ROOT_DIR / "results" / "predictions"
    for m_key, fname in [("Model_A", "model_a_baseline_meld_predictions.json"), ("Model_B", "model_b_disagreement_aware_meld_predictions.json"), ("Model_C", "model_c_reliability_aware_meld_predictions.json")]:
        r = benchmark_runs[42][m_key]
        raw_probs = r["raw_probs"]
        cal_probs = r["cal_probs"]
        preds = np.argmax(raw_probs, axis=-1)
        records = []
        for i, rec in enumerate(splits["test"]):
            records.append({
                "sample_id": rec["Sample_ID"],
                "dialogue_id": rec["Dialogue_ID"],
                "utterance_id": rec["Utterance_ID"],
                "speaker_id": rec["Speaker"],
                "true_label": rec["Emotion_Clean"],
                "true_id": int(rec["Emotion_ID"]),
                "predicted_label": CANONICAL_EMOTIONS[preds[i]],
                "predicted_id": int(preds[i]),
                "correctness": bool(preds[i] == targets[i]),
                "raw_probabilities": [round(float(v), 5) for v in raw_probs[i]],
                "calibrated_probabilities": [round(float(v), 5) for v in cal_probs[i]],
                "raw_confidence": round(float(np.max(raw_probs[i])), 5),
                "calibrated_confidence": round(float(np.max(cal_probs[i])), 5),
                "model_name": m_key,
                "seed": 42,
            })
        with open(pred_dir / fname, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    print("\n>>> Reproducibility audit complete. All metrics, caches, and CSVs saved successfully.")
    return benchmark_summary, ablation_summary, disagreement_results


if __name__ == "__main__":
    run_full_reproducibility_audit()
