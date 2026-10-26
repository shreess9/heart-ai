"""H.E.A.R.T. AI: Disagreement Feature Integration Sanity Experiments.

Evaluates whether the performance degradation in Model B is caused by:
A. The disagreement information itself, or
B. Poor numerical scaling / feature integration.

Experiments:
1. Model B-Norm: Standardized D1, D2, D3 using statistics computed ONLY from the training split.
2. Model B-Proj: Learned Disagreement Projection sub-network (Linear/MLP) before fusion.
3. Single Feature Controls: A + D1-Norm, A + D2-Norm, A + D3-Norm.
4. Random-Control Features: Model A + 3 random noise features matched to standardized scale.
5. Multi-Seed Fair Comparison: A, B-Raw, B-Norm, B-Proj, C across seeds 42, 123, 456.
6. Rigorous D1 Correctness Logistic Regression (coef, SE, 95% CI, p-value).
7. Modality Imbalance Analysis: D1 association with unimodal confidences and correctness.
8. Model C vs Model A In-Depth Reliability Trade-Off Evaluation.

Saves results to results/metrics/disagreement_integration_sanity.csv
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
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

from models.unimodal_engine import UnimodalEngine
from fusion.baseline import BaselineMultimodalMLP
from fusion.reliability_aware import ReliabilityAwareFusion
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3, d3_to_one_hot
from reliability.entropy import compute_predictive_entropy
from reliability.calibration import TemperatureScaler, compute_ece_mce, compute_brier_score
from reliability.confidence import compute_high_confidence_error_rate
from reliability.bootstrap import compute_bootstrap_ci, compute_paired_bootstrap_delta

SEEDS = [42, 123, 456]
CANONICAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_CLASSES = 7


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


# Model Architectures for Sanity Experiments
class ModelBNorm(nn.Module):
    """Model B-Norm: Concatenates 2304 features + 3 standardized disagreement features [D1', D2', D3']."""

    def __init__(self, in_dim: int = 2304 + 3, hidden_dims: List[int] = [512, 256], num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, t_f, a_f, v_f, norm_disagreement):
        fused = torch.cat([t_f, a_f, v_f, norm_disagreement], dim=-1)
        return self.net(fused)


class ModelBProj(nn.Module):
    """Model B-Proj: Projects 3 disagreement features into a 64-dim embedding before concatenating with multimodal features."""

    def __init__(
        self,
        feat_dim: int = 2304,
        disagree_dim: int = 3,
        proj_dim: int = 64,
        hidden_dims: List[int] = [512, 256],
        num_classes: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Projection sub-network
        self.disagree_proj = nn.Sequential(
            nn.Linear(disagree_dim, 32),
            nn.ReLU(),
            nn.Linear(32, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )
        # Main classification MLP
        total_dim = feat_dim + proj_dim
        layers = []
        cur = total_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, t_f, a_f, v_f, norm_disagreement):
        d_embed = self.disagree_proj(norm_disagreement)
        fused = torch.cat([t_f, a_f, v_f, d_embed], dim=-1)
        return self.classifier(fused)


class ModelSingleFeature(nn.Module):
    """Model A + single standardized feature (2304 + 1)."""

    def __init__(self, in_dim: int = 2305, hidden_dims: List[int] = [512, 256], num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, t_f, a_f, v_f, single_feature):
        fused = torch.cat([t_f, a_f, v_f, single_feature], dim=-1)
        return self.net(fused)


class ModelRandomControl(nn.Module):
    """Model A + 3 random noise features (2304 + 3) to test if adding any 3 features causes instability."""

    def __init__(self, in_dim: int = 2307, hidden_dims: List[int] = [512, 256], num_classes: int = 7, dropout: float = 0.2):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, t_f, a_f, v_f, random_features):
        fused = torch.cat([t_f, a_f, v_f, random_features], dim=-1)
        return self.net(fused)


def compute_split_disagreement_features(t_probs, a_probs, v_probs) -> torch.Tensor:
    """Compute [D1, D2, D3] raw scalar features for a tensor set."""
    n = len(t_probs)
    t_p_np = t_probs.numpy()
    a_p_np = a_probs.numpy()
    v_p_np = v_probs.numpy()

    t_preds = np.argmax(t_p_np, axis=-1)
    a_preds = np.argmax(a_p_np, axis=-1)
    v_preds = np.argmax(v_p_np, axis=-1)

    d_mat = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        d1 = float(generalized_jsd_3(t_p_np[i], a_p_np[i], v_p_np[i], normalized=True))
        d2 = float(mean_pairwise_jsd_3(t_p_np[i], a_p_np[i], v_p_np[i], normalized=True))
        d3 = float(compute_categorical_disagreement_d3(t_preds[i], a_preds[i], v_preds[i]))
        d_mat[i] = [d1, d2, d3]
    return torch.tensor(d_mat, dtype=torch.float32)


def run_sanity_experiments():
    print("=" * 80)
    print("H.E.A.R.T. AI — DISAGREEMENT FEATURE INTEGRATION SANITY STUDY")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()} | PyTorch: {torch.__version__}")

    # 1. Load Clean Unimodal Embeddings and Probability Tensors
    u_engine = UnimodalEngine(device=device, seed=42)
    splits = u_engine.splits

    embeddings = {}
    for s in ["train", "dev", "test"]:
        embeddings[s] = {
            "text": u_engine.extract_or_generate_modality_embeddings("text", s),
            "audio": u_engine.extract_or_generate_modality_embeddings("audio", s),
            "visual": u_engine.extract_or_generate_modality_embeddings("visual", s),
        }

    unimodal_models = {}
    for mod in ["text", "audio", "visual"]:
        set_seed(42)
        model = u_engine.train_unimodal_classifier(
            mod,
            embeddings["train"][mod],
            embeddings["dev"][mod],
            epochs=15,
            lr=0.005,
            batch_size=64,
        )
        unimodal_models[mod] = model

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

    # 2. Compute Raw Disagreement and TRAIN-ONLY Standardization Statistics
    print("\n>>> Computing Disagreement Statistics Strictly on Train Split...")
    raw_d_train = compute_split_disagreement_features(data_tensors["train"]["t_probs"], data_tensors["train"]["a_probs"], data_tensors["train"]["v_probs"])
    raw_d_dev = compute_split_disagreement_features(data_tensors["dev"]["t_probs"], data_tensors["dev"]["a_probs"], data_tensors["dev"]["v_probs"])
    raw_d_test = compute_split_disagreement_features(data_tensors["test"]["t_probs"], data_tensors["test"]["a_probs"], data_tensors["test"]["v_probs"])

    # Standardization parameters from TRAIN ONLY
    d_mean_train = torch.mean(raw_d_train, dim=0, keepdim=True)
    d_std_train = torch.std(raw_d_train, dim=0, keepdim=True)
    d_std_train = torch.clamp(d_std_train, min=1e-5)

    print(f"    Train D1: mean = {d_mean_train[0,0]:.4f}, std = {d_std_train[0,0]:.4f}")
    print(f"    Train D2: mean = {d_mean_train[0,1]:.4f}, std = {d_std_train[0,1]:.4f}")
    print(f"    Train D3: mean = {d_mean_train[0,2]:.4f}, std = {d_std_train[0,2]:.4f}")

    norm_d_train = (raw_d_train - d_mean_train) / d_std_train
    norm_d_dev = (raw_d_dev - d_mean_train) / d_std_train
    norm_d_test = (raw_d_test - d_mean_train) / d_std_train

    # Fixed random noise features for Experiment 4 (mean=0, std=1)
    g_rnd = torch.Generator().manual_seed(42)
    rnd_feat_train = torch.randn(len(raw_d_train), 3, generator=g_rnd)
    rnd_feat_dev = torch.randn(len(raw_d_dev), 3, generator=g_rnd)
    rnd_feat_test = torch.randn(len(raw_d_test), 3, generator=g_rnd)

    # 3. Model Training & Evaluation Harness
    train_y = torch.tensor([r["Emotion_ID"] for r in splits["train"]], dtype=torch.long, device=device)
    dev_y = torch.tensor([r["Emotion_ID"] for r in splits["dev"]], dtype=torch.long, device=device)
    test_y = np.array([r["Emotion_ID"] for r in splits["test"]])
    dia_ids = np.array([r["Dialogue_ID"] for r in splits["test"]])

    def train_eval_variant(model_builder, extra_train, extra_dev, extra_test, seed, epochs=15, lr=0.003, batch_size=64):
        set_seed(seed)
        model = model_builder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        tr_tf = data_tensors["train"]["t_feat"].to(device)
        tr_af = data_tensors["train"]["a_feat"].to(device)
        tr_vf = data_tensors["train"]["v_feat"].to(device)
        tr_ex = extra_train.to(device) if extra_train is not None else None

        dev_tf = data_tensors["dev"]["t_feat"].to(device)
        dev_af = data_tensors["dev"]["a_feat"].to(device)
        dev_vf = data_tensors["dev"]["v_feat"].to(device)
        dev_ex = extra_dev.to(device) if extra_dev is not None else None

        best_dev_f1 = -1.0
        best_state = None

        n_train = len(train_y)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(n_train)
            for b_idx in range(0, n_train, batch_size):
                idxs = perm[b_idx:b_idx+batch_size]
                bx_tf, bx_af, bx_vf, by = tr_tf[idxs], tr_af[idxs], tr_vf[idxs], train_y[idxs]
                bx_ex = tr_ex[idxs] if tr_ex is not None else None

                optimizer.zero_grad()
                if bx_ex is not None:
                    out = model(bx_tf, bx_af, bx_vf, bx_ex)
                else:
                    out = model(bx_tf, bx_af, bx_vf)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

            # Dev evaluation
            model.eval()
            with torch.no_grad():
                if dev_ex is not None:
                    dev_out = model(dev_tf, dev_af, dev_vf, dev_ex)
                else:
                    dev_out = model(dev_tf, dev_af, dev_vf)
                dev_preds = torch.argmax(dev_out, dim=-1).cpu().numpy()
                dev_f1 = f1_score(dev_y.cpu().numpy(), dev_preds, average="macro", zero_division=0)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Fit validation temperature
        model.eval()
        with torch.no_grad():
            if dev_ex is not None:
                dev_logits = model(dev_tf, dev_af, dev_vf, dev_ex).cpu().numpy()
            else:
                dev_logits = model(dev_tf, dev_af, dev_vf).cpu().numpy()
        scaler = TemperatureScaler()
        opt_t = float(scaler.fit(dev_logits, dev_y.cpu().numpy()))

        # Evaluate on Test
        test_tf = data_tensors["test"]["t_feat"].to(device)
        test_af = data_tensors["test"]["a_feat"].to(device)
        test_vf = data_tensors["test"]["v_feat"].to(device)
        test_ex = extra_test.to(device) if extra_test is not None else None

        with torch.no_grad():
            if test_ex is not None:
                raw_logits_t = model(test_tf, test_af, test_vf, test_ex).cpu()
            else:
                raw_logits_t = model(test_tf, test_af, test_vf).cpu()
            raw_probs = torch.softmax(raw_logits_t, dim=-1).numpy()
            cal_probs = torch.softmax(raw_logits_t / opt_t, dim=-1).numpy()

        preds = np.argmax(raw_probs, axis=-1)
        acc = float(accuracy_score(test_y, preds))
        macro_f1 = float(f1_score(test_y, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(test_y, preds, average="weighted", zero_division=0))

        raw_cal = compute_ece_mce(raw_probs, test_y, num_bins=15)
        cal_cal = compute_ece_mce(cal_probs, test_y, num_bins=15)
        raw_brier = compute_brier_score(raw_probs, test_y, num_classes=7)
        cal_brier = compute_brier_score(cal_probs, test_y, num_classes=7)
        hc_raw = compute_high_confidence_error_rate(raw_probs, test_y, threshold=0.80)
        hc_cal = compute_high_confidence_error_rate(cal_probs, test_y, threshold=0.80)

        return {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "raw_ece": raw_cal["ece"],
            "calibrated_ece": cal_cal["ece"],
            "raw_brier": raw_brier,
            "calibrated_brier": cal_brier,
            "high_conf_error_cal": hc_cal["high_conf_error_rate"],
            "temperature": opt_t,
            "raw_probs": raw_probs,
            "cal_probs": cal_probs,
        }

    # 4. Execute Multi-Seed Comparison across Variants
    print("\n>>> Running Controlled Variant Benchmark (Seeds: 42, 123, 456)...")
    variants_to_test = [
        ("Model A (Baseline Concatenation)", lambda: BaselineMultimodalMLP(), None, None, None),
        ("Model B-Raw (Direct Scalar Concatenation)", lambda: ModelBNorm(in_dim=2307), raw_d_train, raw_d_dev, raw_d_test),
        ("Model B-Norm (Standardized D1, D2, D3)", lambda: ModelBNorm(in_dim=2307), norm_d_train, norm_d_dev, norm_d_test),
        ("Model B-Proj (Learned Disagreement Projection)", lambda: ModelBProj(disagree_dim=3, proj_dim=64), norm_d_train, norm_d_dev, norm_d_test),
        ("Single Feature: A + D1-Norm", lambda: ModelSingleFeature(in_dim=2305), norm_d_train[:, 0:1], norm_d_dev[:, 0:1], norm_d_test[:, 0:1]),
        ("Single Feature: A + D2-Norm", lambda: ModelSingleFeature(in_dim=2305), norm_d_train[:, 1:2], norm_d_dev[:, 1:2], norm_d_test[:, 1:2]),
        ("Single Feature: A + D3-Norm", lambda: ModelSingleFeature(in_dim=2305), norm_d_train[:, 2:3], norm_d_dev[:, 2:3], norm_d_test[:, 2:3]),
        ("Random-Control: A + 3 Random Features", lambda: ModelRandomControl(in_dim=2307), rnd_feat_train, rnd_feat_dev, rnd_feat_test),
    ]

    results_table = {}
    csv_rows = []

    for name, builder, ex_tr, ex_dv, ex_ts in variants_to_test:
        print(f"\nEvaluating Variant: {name}")
        seed_runs = []
        for s in SEEDS:
            res = train_eval_variant(builder, ex_tr, ex_dv, ex_ts, seed=s)
            seed_runs.append(res)
            csv_rows.append({
                "experiment": "disagreement_integration_sanity",
                "variant": name,
                "seed": s,
                "accuracy": round(res["accuracy"], 6),
                "macro_f1": round(res["macro_f1"], 6),
                "weighted_f1": round(res["weighted_f1"], 6),
                "raw_ece": round(res["raw_ece"], 6),
                "calibrated_ece": round(res["calibrated_ece"], 6),
                "raw_brier": round(res["raw_brier"], 6),
                "calibrated_brier": round(res["calibrated_brier"], 6),
                "high_conf_error_cal": round(res["high_conf_error_cal"], 6),
                "temperature": round(res["temperature"], 5),
            })

        accs = [r["accuracy"] for r in seed_runs]
        f1s = [r["macro_f1"] for r in seed_runs]
        raw_eces = [r["raw_ece"] for r in seed_runs]
        cal_eces = [r["calibrated_ece"] for r in seed_runs]
        briers = [r["calibrated_brier"] for r in seed_runs]
        hc_errs = [r["high_conf_error_cal"] for r in seed_runs]

        results_table[name] = {
            "acc_mean": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
            "raw_ece_mean": float(np.mean(raw_eces)),
            "cal_ece_mean": float(np.mean(cal_eces)),
            "brier_mean": float(np.mean(briers)),
            "hc_err_mean": float(np.mean(hc_errs)),
            "seed_42_res": seed_runs[0],
        }

        print(f"  -> Acc: {np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}% | Macro-F1: {np.mean(f1s):.4f}±{np.std(f1s):.4f} | Raw ECE: {np.mean(raw_eces)*100:.2f}% | Cal ECE: {np.mean(cal_eces)*100:.2f}% | Cal Brier: {np.mean(briers):.4f}")

    # Also record Model C from canonical benchmark
    with open(ROOT_DIR / "results" / "metrics" / "model_abc_comparison.json", "r", encoding="utf-8") as f:
        m_abc = json.load(f)
    c_multi = m_abc["multi_seed_aggregate"]["Model_C"]
    results_table["Model C (Reliability-Aware Attention)"] = {
        "acc_mean": c_multi["accuracy_mean"], "acc_std": c_multi["accuracy_std"],
        "f1_mean": c_multi["macro_f1_mean"], "f1_std": c_multi["macro_f1_std"],
        "raw_ece_mean": c_multi["raw_ece_mean"],
        "cal_ece_mean": c_multi["calibrated_ece_mean"],
        "brier_mean": c_multi["calibrated_brier_mean"],
        "hc_err_mean": c_multi["high_conf_error_cal_mean"],
    }

    # Save to CSV
    csv_df = pd.DataFrame(csv_rows)
    sanity_csv_path = ROOT_DIR / "results" / "metrics" / "disagreement_integration_sanity.csv"
    csv_df.to_csv(sanity_csv_path, index=False)
    print(f"\nSaved sanity experiment CSV to {sanity_csv_path}")

    # 5. Experiment 6: Detailed D1 Correctness Logistic Regression
    print("\n>>> [Experiment 6] Detailed Logistic Regression: Correctness ~ Text Confidence + D1...")
    t_probs_test = data_tensors["test"]["t_probs"].numpy()
    t_preds_test = np.argmax(t_probs_test, axis=-1)
    text_conf = np.max(t_probs_test, axis=-1)
    correct_test = (t_preds_test == test_y).astype(float)
    d1_test = raw_d_test[:, 0].numpy()

    df_logit = pd.DataFrame({
        "const": 1.0,
        "text_confidence": text_conf,
        "d1_disagreement": d1_test,
    })
    logit_model = sm.Logit(correct_test, df_logit)
    logit_res = logit_model.fit(disp=False)

    coef_conf = float(logit_res.params["text_confidence"])
    se_conf = float(logit_res.bse["text_confidence"])
    ci_conf = [float(logit_res.conf_int().loc["text_confidence", 0]), float(logit_res.conf_int().loc["text_confidence", 1])]
    pval_conf = float(logit_res.pvalues["text_confidence"])

    coef_d1 = float(logit_res.params["d1_disagreement"])
    se_d1 = float(logit_res.bse["d1_disagreement"])
    ci_d1 = [float(logit_res.conf_int().loc["d1_disagreement", 0]), float(logit_res.conf_int().loc["d1_disagreement", 1])]
    pval_d1 = float(logit_res.pvalues["d1_disagreement"])

    print(f"  Text Confidence: Coef = {coef_conf:+.4f} (SE = {se_conf:.4f}, 95% CI: [{ci_conf[0]:+.4f}, {ci_conf[1]:+.4f}], z = {coef_conf/se_conf:.2f}, p = {pval_conf:.2e})")
    print(f"  D1 Disagreement: Coef = {coef_d1:+.4f} (SE = {se_d1:.4f}, 95% CI: [{ci_d1[0]:+.4f}, {ci_d1[1]:+.4f}], z = {coef_d1/se_d1:.2f}, p = {pval_d1:.2e})")

    # 6. Experiment 7: Modality Imbalance & What D1 Represents
    print("\n>>> [Experiment 7] Modality Confidence & Correctness Association Analysis...")
    a_probs_test = data_tensors["test"]["a_probs"].numpy()
    v_probs_test = data_tensors["test"]["v_probs"].numpy()

    a_preds_test = np.argmax(a_probs_test, axis=-1)
    v_preds_test = np.argmax(v_probs_test, axis=-1)

    audio_conf = np.max(a_probs_test, axis=-1)
    visual_conf = np.max(v_probs_test, axis=-1)

    t_correct = (t_preds_test == test_y)
    a_correct = (a_preds_test == test_y)
    v_correct = (v_preds_test == test_y)

    # Correlation of D1 with confidences
    r_d1_tc, p_d1_tc = stats.pearsonr(d1_test, text_conf)
    r_d1_ac, p_d1_ac = stats.pearsonr(d1_test, audio_conf)
    r_d1_vc, p_d1_vc = stats.pearsonr(d1_test, visual_conf)

    # Mean D1 by modality correctness states
    # State 1: All 3 correct
    all_3_corr = t_correct & a_correct & v_correct
    # State 2: Text correct, but Audio & Visual incorrect (Noise in secondary modalities)
    t_corr_sec_inc = t_correct & (~a_correct) & (~v_correct)
    # State 3: Text incorrect, secondary correct
    t_inc_sec_corr = (~t_correct) & (a_correct | v_correct)
    # State 4: All 3 incorrect
    all_3_inc = (~t_correct) & (~a_correct) & (~v_correct)

    print(f"  D1 vs Text Confidence Correlation:   r = {r_d1_tc:+.4f} (p = {p_d1_tc:.2e})")
    print(f"  D1 vs Audio Confidence Correlation:  r = {r_d1_ac:+.4f} (p = {p_d1_ac:.2e})")
    print(f"  D1 vs Visual Confidence Correlation: r = {r_d1_vc:+.4f} (p = {p_d1_vc:.2e})")
    print(f"\n  Mean D1 across Correctness States:")
    print(f"    - All 3 Modalities Correct (N={np.sum(all_3_corr)}):          Mean D1 = {np.mean(d1_test[all_3_corr]):.4f}")
    print(f"    - Text Correct, Audio/Visual Wrong (N={np.sum(t_corr_sec_inc)}):  Mean D1 = {np.mean(d1_test[t_corr_sec_inc]):.4f}")
    print(f"    - Text Wrong, Secondary Correct (N={np.sum(t_inc_sec_corr)}):     Mean D1 = {np.mean(d1_test[t_inc_sec_corr]):.4f}")
    print(f"    - All 3 Modalities Wrong (N={np.sum(all_3_inc)}):           Mean D1 = {np.mean(d1_test[all_3_inc]):.4f}")

    # Summary JSON (clean primitive types)
    clean_experiments = {}
    for k, v in results_table.items():
        clean_experiments[k] = {
            "acc_mean": v["acc_mean"], "acc_std": v["acc_std"],
            "f1_mean": v["f1_mean"], "f1_std": v["f1_std"],
            "raw_ece_mean": v["raw_ece_mean"],
            "cal_ece_mean": v["cal_ece_mean"],
            "brier_mean": v["brier_mean"],
            "hc_err_mean": v["hc_err_mean"],
        }

    sanity_summary = {
        "experiments": clean_experiments,
        "regression_d1_correctness": {
            "text_conf": {"coef": coef_conf, "se": se_conf, "ci": ci_conf, "pval": pval_conf},
            "d1": {"coef": coef_d1, "se": se_d1, "ci": ci_d1, "pval": pval_d1},
        },
        "modality_analysis": {
            "corr_d1_text_conf": float(r_d1_tc),
            "corr_d1_audio_conf": float(r_d1_ac),
            "corr_d1_visual_conf": float(r_d1_vc),
            "mean_d1_all_3_correct": float(np.mean(d1_test[all_3_corr])),
            "mean_d1_text_corr_sec_inc": float(np.mean(d1_test[t_corr_sec_inc])),
            "mean_d1_text_inc_sec_corr": float(np.mean(d1_test[t_inc_sec_corr])),
            "mean_d1_all_3_incorrect": float(np.mean(d1_test[all_3_inc])),
        }
    }

    with open(ROOT_DIR / "results" / "metrics" / "disagreement_integration_summary.json", "w", encoding="utf-8") as f:
        json.dump(sanity_summary, f, indent=2)

    return sanity_summary


if __name__ == "__main__":
    run_sanity_experiments()

