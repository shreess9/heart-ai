"""
experiment_elderreact_proper.py
================================
H.E.A.R.T. AI — Official ElderReact Supervised Training & Evaluation
Phases 5-9 of the ElderReact integration plan.

DESIGN PRINCIPLES
-----------------
1. Respects official train / dev / test split structure — no pooling.
2. Multi-label binary classification: 6 emotions, BCEWithLogitsLoss.
3. Per-emotion decision thresholds tuned on dev set only.
4. Three fusion strategies compared:
   - ER-A: Standard Fusion Baseline
   - ER-B: Disagreement-Aware Fusion (per-emotion Bernoulli JSD)
   - ER-C: Reliability-Aware Fusion (confidence×(1−entropy) gating)
5. Test set evaluated ONCE after all design decisions are fixed.
6. Known NaN row in test audio (row 289, rammstein_30) is dropped safely.

DATA SOURCES  (read-only)
--------------------------
Annotations : /Users/shreee/Downloads/ElderReact_official_repo/Annotations/
Features    : /Users/shreee/Downloads/ElderReact_official_repo/Features/
Videos      : /Users/shreee/Desktop/heart-ai-main copy/ElderReact_Data/  (not used here)

OUTPUTS
-------
results/elderreact/official/
  official_dataset_audit.md    — Phase 3/4 audit summary
  train_results.json           — per-model dev & test metrics
  per_emotion_results.json     — per-emotion breakdown
  thresholds.json              — per-emotion dev-tuned thresholds
  checkpoints/                 — model parameters (sklearn pickle)
  plots/                       — calibration & F1 bar charts

REPRODUCIBILITY
---------------
  conda run -n base python3 research/elderreact/experiment_elderreact_proper.py
  # or with a custom config:
  conda run -n base ELDERREACT_CONFIG=configs/elderreact_official.yaml \\
      python3 research/elderreact/experiment_elderreact_proper.py

Commit: 8a0ae96 (ElderReact official repo, cloned 2026-10-26)
"""

from __future__ import annotations
import os, sys, json, pickle, logging, warnings, time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (f1_score, hamming_loss, accuracy_score,
                              precision_recall_fscore_support, brier_score_loss)
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR  = Path("/Users/shreee/Downloads/ElderReact_official_repo")
VIDEO_DIR = Path("/Users/shreee/Desktop/heart-ai-main copy/ElderReact_Data")
OUT_DIR   = Path("/Users/shreee/Desktop/heart-ai-main copy/results/elderreact/official")
CKPT_DIR  = OUT_DIR / "checkpoints"
PLOT_DIR  = OUT_DIR / "plots"
for d in (OUT_DIR, CKPT_DIR, PLOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
N_CLS    = len(EMOTIONS)

# ── Known data quality issues (from official baseline.py) ─────────────────────
# Test audio row 289 (rammstein_30) contains NaN/Inf — drop it.
TEST_DROP_ROWS = {289}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _parse_audio_line(line: str) -> Tuple[str, np.ndarray | None]:
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return parts[0].strip() if parts else "", None
    fname = parts[0].strip()
    try:
        raw = np.array([float(x) for x in parts[1].split()])
        feat = np.concatenate([raw[:36], raw[74:110]])       # 72-dim
        return fname, feat
    except (ValueError, IndexError):
        return fname, None


def _parse_video_line(line: str) -> Tuple[str, np.ndarray | None]:
    parts = line.strip().split("\t")
    if len(parts) < 2:
        return parts[0].strip() if parts else "", None
    fname = parts[0].strip()
    try:
        raw = np.array([float(x) for x in parts[1].split()])
        feat = np.concatenate([
            raw[0:8], raw[288:294], raw[634:717],
            raw[997:1003], raw[1343:1418]
        ])                                                    # 178-dim
        return fname, feat
    except (ValueError, IndexError):
        return fname, None


def _parse_label_line(line: str) -> Tuple[str, np.ndarray | None]:
    parts = line.strip().split()
    if len(parts) < 7:
        return parts[0] if parts else "", None
    fname = parts[0]
    labels = np.array([int(x) for x in parts[1:7]], dtype=np.float32)
    return fname, labels


def load_split(split: str, drop_rows: set = None) -> Dict:
    """
    Load one official split. Matches rows by position (audio, video and
    annotation files have identical row ordering, confirmed by validation).
    The filename in annotation has `.mp4` suffix; audio/video do not.
    """
    a_lines = (REPO_DIR / "Features" / "Audio_feat" / f"{split}_audio.txt").read_text().splitlines()
    v_lines = (REPO_DIR / "Features" / "Video_feat" / f"{split}_video.txt").read_text().splitlines()
    l_lines = (REPO_DIR / "Annotations" / f"{split}_labels.txt").read_text().splitlines()

    assert len(a_lines) == len(v_lines) == len(l_lines), (
        f"{split}: line count mismatch a={len(a_lines)} v={len(v_lines)} l={len(l_lines)}")

    ids, X_a, X_v, Y = [], [], [], []
    dropped, errors = [], []

    for i, (al, vl, ll) in enumerate(zip(a_lines, v_lines, l_lines)):
        if drop_rows and i in drop_rows:
            dropped.append(i)
            continue
        a_id, a_feat = _parse_audio_line(al)
        v_id, v_feat = _parse_video_line(vl)
        l_id, labels = _parse_label_line(ll)
        if a_feat is None or v_feat is None or labels is None:
            errors.append(i)
            continue
        # Sanity: strip extensions for comparison
        a_stem = Path(a_id).stem; l_stem = Path(l_id).stem
        if a_stem != l_stem:
            errors.append(i)
            continue
        ids.append(l_id)
        X_a.append(a_feat); X_v.append(v_feat); Y.append(labels)

    X_a = np.nan_to_num(np.array(X_a), nan=0.0, posinf=0.0, neginf=0.0)
    X_v = np.nan_to_num(np.array(X_v), nan=0.0, posinf=0.0, neginf=0.0)
    Y   = np.array(Y)

    log.info(f"  {split}: {len(ids)} samples  dropped={dropped}  errors={errors}")
    return dict(ids=ids, X_a=X_a, X_v=X_v, Y=Y, split=split)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DISAGREEMENT & RELIABILITY FEATURES (multi-label aware)
# ══════════════════════════════════════════════════════════════════════════════

EPS = 1e-10

def _bernoulli_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Per-emotion Bernoulli JSD between two probability arrays.
    p, q: [N, 6] sigmoid probabilities.
    Returns [N, 6] JSD values in [0, log(2)] (nats), normalised to [0,1].
    """
    p = np.clip(p, EPS, 1 - EPS)
    q = np.clip(q, EPS, 1 - EPS)
    m = 0.5 * (p + q)
    def kl(a, b):
        return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return np.clip(jsd, 0, None) / np.log(2)   # normalise to [0,1]


def _per_emotion_entropy(p: np.ndarray) -> np.ndarray:
    """Per-emotion binary entropy. p: [N,6]. Returns [N,6]."""
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2)


def _reliability_gate(p: np.ndarray) -> np.ndarray:
    """
    Per-emotion reliability gate = confidence * (1 - entropy).
    confidence = max(p, 1-p),  entropy = binary entropy.
    Returns [N, 6].
    """
    conf = np.maximum(p, 1 - p)           # [N,6]  ∈ [0.5, 1]
    ent  = _per_emotion_entropy(p)        # [N,6]  ∈ [0, 1]
    return conf * (1 - ent)               # [N,6]


def build_disagreement_features(pa: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    Disagreement feature vector for Model B.
    pa, pv: [N, 6] sigmoid probabilities from audio and visual.
    Returns [N, 18]:
      - per_emotion_jsd      [N, 6]
      - mean_jsd (broadcast) [N, 1] → repeated 6
      - per_emotion_conf_gap [N, 6]
    """
    jsd     = _bernoulli_jsd(pa, pv)                      # [N, 6]
    jsd_mean= jsd.mean(axis=1, keepdims=True).repeat(6, 1)# [N, 6]
    conf_gap= np.abs(pa - pv)                             # [N, 6]
    return np.concatenate([jsd, jsd_mean, conf_gap], axis=1)  # [N, 18]


def build_reliability_features(pa: np.ndarray, pv: np.ndarray) -> np.ndarray:
    """
    Reliability feature vector for Model C.
    Returns [N, 12]:
      - audio gate   [N, 6]
      - visual gate  [N, 6]
    """
    ga = _reliability_gate(pa)   # [N, 6]
    gv = _reliability_gate(pv)   # [N, 6]
    return np.concatenate([ga, gv], axis=1)  # [N, 12]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SKLEARN MULTI-LABEL MLP (wraps binary relevance)
# ══════════════════════════════════════════════════════════════════════════════

class MultiLabelMLP:
    """
    Binary-Relevance MLP: trains one MLPClassifier per emotion label.
    predict_proba returns [N, 6] sigmoid probabilities.
    """

    def __init__(self, hidden=(128, 64), seed=42):
        self.hidden = hidden
        self.seed   = seed
        self.clfs   = []

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.clfs = []
        for i in range(Y.shape[1]):
            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden,
                activation="relu",
                max_iter=300,
                random_state=self.seed,
                alpha=1e-3,
            )
            clf.fit(X, Y[:, i])
            self.clfs.append(clf)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns [N, 6] — probability of positive class for each emotion."""
        return np.column_stack([
            clf.predict_proba(X)[:, 1] for clf in self.clfs
        ])

    def predict(self, X: np.ndarray, thresholds: np.ndarray | None = None) -> np.ndarray:
        proba = self.predict_proba(X)
        thresh = thresholds if thresholds is not None else np.full(N_CLS, 0.5)
        return (proba >= thresh).astype(int)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "MultiLabelMLP":
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — THRESHOLD TUNING (dev set only)
# ══════════════════════════════════════════════════════════════════════════════

def tune_thresholds(proba: np.ndarray, Y: np.ndarray,
                    candidates: np.ndarray = None) -> np.ndarray:
    """
    Per-emotion threshold tuning on dev set.
    Maximises per-emotion F1 across candidate thresholds.
    Returns array of shape [6].
    """
    if candidates is None:
        candidates = np.linspace(0.1, 0.9, 33)
    thresholds = np.full(N_CLS, 0.5)
    for i in range(N_CLS):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            preds = (proba[:, i] >= t).astype(int)
            f1 = f1_score(Y[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(Y_true: np.ndarray, Y_pred: np.ndarray,
                    proba: np.ndarray) -> Dict:
    """Full suite of multi-label evaluation metrics."""
    macro_f1  = f1_score(Y_true, Y_pred, average="macro",    zero_division=0)
    micro_f1  = f1_score(Y_true, Y_pred, average="micro",    zero_division=0)
    weighted_f1 = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
    hamming   = hamming_loss(Y_true, Y_pred)
    subset_acc = accuracy_score(Y_true, Y_pred)

    # Per-emotion
    p, r, f, _ = precision_recall_fscore_support(Y_true, Y_pred,
                                                  average=None, zero_division=0)
    per_emo = {e: dict(precision=float(p[i]), recall=float(r[i]), f1=float(f[i]))
               for i, e in enumerate(EMOTIONS)}

    # Per-emotion Brier score
    brier = {}
    for i, e in enumerate(EMOTIONS):
        brier[e] = float(brier_score_loss(Y_true[:, i], proba[:, i]))
    mean_brier = float(np.mean(list(brier.values())))

    # ECE (multi-label: macro mean of per-emotion ECE)
    def ece_binary(y, prob, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (prob >= lo) & (prob < hi)
            if m.sum():
                ece += m.sum() / len(y) * abs(y[m].mean() - prob[m].mean())
        return ece
    ece_per = {e: ece_binary(Y_true[:, i], proba[:, i])
               for i, e in enumerate(EMOTIONS)}
    mean_ece = float(np.mean(list(ece_per.values())))

    return dict(
        macro_f1=float(macro_f1), micro_f1=float(micro_f1),
        weighted_f1=float(weighted_f1), hamming=float(hamming),
        subset_acc=float(subset_acc), per_emotion=per_emo,
        brier=brier, mean_brier=mean_brier,
        ece=ece_per, mean_ece=mean_ece,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — CALIBRATION PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration(Y_true, proba, title, path, n_bins=10):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(title, fontsize=12)
    for i, e in enumerate(EMOTIONS):
        ax = axes[i // 3][i % 3]
        y, p = Y_true[:, i], proba[:, i]
        bins = np.linspace(0, 1, n_bins + 1)
        frac_pos, mean_prob = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum():
                frac_pos.append(y[m].mean())
                mean_prob.append(p[m].mean())
        ax.plot(mean_prob, frac_pos, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.set_title(e, fontsize=9, fontweight="bold")
        ax.set_xlabel("Predicted prob"); ax.set_ylabel("Actual freq")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_f1_comparison(results, path):
    models = list(results.keys())
    macro  = [results[m]["test"]["macro_f1"] for m in models]
    micro  = [results[m]["test"]["micro_f1"] for m in models]
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - 0.2, macro, 0.35, label="Macro-F1", color=colors, alpha=0.85)
    bars2 = ax.bar(x + 0.2, micro, 0.35, label="Micro-F1",
                   color=colors, alpha=0.45, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("F1 Score"); ax.set_title("Model Comparison — ElderReact Official Test Set")
    ax.legend()
    for bar, v in zip(bars1, macro):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — MAIN EXPERIMENT
# ══════════════════════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    log.info("=" * 65)
    log.info("H.E.A.R.T. AI — ElderReact Official Training & Evaluation")
    log.info("=" * 65)

    # ── Load data ──────────────────────────────────────────────────────────
    log.info("\nLoading official splits...")
    train = load_split("train")
    dev   = load_split("dev")
    test  = load_split("test", drop_rows=TEST_DROP_ROWS)

    log.info(f"  Train: {len(train['ids'])} samples")
    log.info(f"  Dev:   {len(dev['ids'])} samples")
    log.info(f"  Test:  {len(test['ids'])} samples (1 NaN row dropped)")

    # ── Scale features (fit on train ONLY) ────────────────────────────────
    sc_a = MinMaxScaler().fit(train["X_a"])
    sc_v = MinMaxScaler().fit(train["X_v"])

    def scale(d):
        return sc_a.transform(d["X_a"]), sc_v.transform(d["X_v"])

    Xa_tr, Xv_tr = scale(train)
    Xa_dv, Xv_dv = scale(dev)
    Xa_te, Xv_te = scale(test)

    Xf_tr = np.concatenate([Xa_tr, Xv_tr], axis=1)   # 250-dim fused
    Xf_dv = np.concatenate([Xa_dv, Xv_dv], axis=1)
    Xf_te = np.concatenate([Xa_te, Xv_te], axis=1)

    Y_tr, Y_dv, Y_te = train["Y"], dev["Y"], test["Y"]

    results   = {}
    thresholds_out = {}

    # ══════════════════════════════════════════════════════════════════════
    # MODEL ER-A — Standard Fusion Baseline
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n--- ER-A: Standard Fusion Baseline ---")

    era = MultiLabelMLP(hidden=(128, 64), seed=SEED)
    era.fit(Xf_tr, Y_tr)
    era.save(str(CKPT_DIR / "er_a.pkl"))

    pa_dv = era.predict_proba(Xf_dv)
    thresh_a = tune_thresholds(pa_dv, Y_dv)
    thresholds_out["ER_A"] = thresh_a.tolist()
    log.info(f"  Dev thresholds: {dict(zip(EMOTIONS, thresh_a.round(2)))}")

    pa_te = era.predict_proba(Xf_te)
    pred_a_dv = era.predict(Xf_dv, thresh_a)
    pred_a_te = era.predict(Xf_te, thresh_a)

    results["ER-A"] = dict(
        dev  = compute_metrics(Y_dv, pred_a_dv, pa_dv),
        test = compute_metrics(Y_te, pred_a_te, pa_te),
    )
    log.info(f"  Dev  Macro-F1: {results['ER-A']['dev']['macro_f1']:.4f}")
    log.info(f"  Test Macro-F1: {results['ER-A']['test']['macro_f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL ER-B — Disagreement-Aware Fusion
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n--- ER-B: Disagreement-Aware Fusion ---")

    # Step 1: Train unimodal models on train set (for probability estimation)
    era_a_uni = MultiLabelMLP(hidden=(64,), seed=SEED).fit(Xa_tr, Y_tr)
    era_v_uni = MultiLabelMLP(hidden=(64,), seed=SEED).fit(Xv_tr, Y_tr)

    def get_disagree_feats(xa, xv):
        pa = era_a_uni.predict_proba(xa)   # [N, 6]
        pv = era_v_uni.predict_proba(xv)   # [N, 6]
        return build_disagreement_features(pa, pv)  # [N, 18]

    Xd_tr = np.concatenate([Xf_tr, get_disagree_feats(Xa_tr, Xv_tr)], axis=1)
    Xd_dv = np.concatenate([Xf_dv, get_disagree_feats(Xa_dv, Xv_dv)], axis=1)
    Xd_te = np.concatenate([Xf_te, get_disagree_feats(Xa_te, Xv_te)], axis=1)

    erb = MultiLabelMLP(hidden=(128, 64), seed=SEED)
    erb.fit(Xd_tr, Y_tr)
    erb.save(str(CKPT_DIR / "er_b.pkl"))

    pb_dv = erb.predict_proba(Xd_dv)
    thresh_b = tune_thresholds(pb_dv, Y_dv)
    thresholds_out["ER_B"] = thresh_b.tolist()

    pb_te = erb.predict_proba(Xd_te)
    pred_b_dv = erb.predict(Xd_dv, thresh_b)
    pred_b_te = erb.predict(Xd_te, thresh_b)

    results["ER-B"] = dict(
        dev  = compute_metrics(Y_dv, pred_b_dv, pb_dv),
        test = compute_metrics(Y_te, pred_b_te, pb_te),
    )
    log.info(f"  Dev  Macro-F1: {results['ER-B']['dev']['macro_f1']:.4f}")
    log.info(f"  Test Macro-F1: {results['ER-B']['test']['macro_f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL ER-C — Reliability-Aware Fusion
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n--- ER-C: Reliability-Aware Fusion ---")

    def get_rel_feats(xa, xv):
        pa = era_a_uni.predict_proba(xa)   # [N, 6]
        pv = era_v_uni.predict_proba(xv)   # [N, 6]
        gates = build_reliability_features(pa, pv)  # [N, 12]: [gate_a×6, gate_v×6]
        # Per-sample scalar gate = mean across emotions
        ga_scalar = gates[:, :6].mean(axis=1, keepdims=True)   # [N, 1]
        gv_scalar = gates[:, 6:].mean(axis=1, keepdims=True)   # [N, 1]
        xa_gated = xa * ga_scalar   # broadcasts [N,72] * [N,1]
        xv_gated = xv * gv_scalar   # broadcasts [N,178] * [N,1]
        fused_gated = np.concatenate([xa_gated, xv_gated], axis=1)  # [N,250]
        return fused_gated, gates

    Xc_tr_f, Xc_tr_g = get_rel_feats(Xa_tr, Xv_tr)
    Xc_dv_f, Xc_dv_g = get_rel_feats(Xa_dv, Xv_dv)
    Xc_te_f, Xc_te_g = get_rel_feats(Xa_te, Xv_te)

    Xc_tr = np.concatenate([Xc_tr_f, Xc_tr_g], axis=1)
    Xc_dv = np.concatenate([Xc_dv_f, Xc_dv_g], axis=1)
    Xc_te = np.concatenate([Xc_te_f, Xc_te_g], axis=1)

    erc = MultiLabelMLP(hidden=(128, 64), seed=SEED)
    erc.fit(Xc_tr, Y_tr)
    erc.save(str(CKPT_DIR / "er_c.pkl"))

    pc_dv = erc.predict_proba(Xc_dv)
    thresh_c = tune_thresholds(pc_dv, Y_dv)
    thresholds_out["ER_C"] = thresh_c.tolist()

    pc_te = erc.predict_proba(Xc_te)
    pred_c_dv = erc.predict(Xc_dv, thresh_c)
    pred_c_te = erc.predict(Xc_te, thresh_c)

    results["ER-C"] = dict(
        dev  = compute_metrics(Y_dv, pred_c_dv, pc_dv),
        test = compute_metrics(Y_te, pred_c_te, pc_te),
    )
    log.info(f"  Dev  Macro-F1: {results['ER-C']['dev']['macro_f1']:.4f}")
    log.info(f"  Test Macro-F1: {results['ER-C']['test']['macro_f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 8 — SAVE RESULTS & PLOTS
    # ══════════════════════════════════════════════════════════════════════
    log.info("\n--- Saving outputs ---")

    def _ser(o):
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)):    return int(o)
        if isinstance(o, np.ndarray):           return o.tolist()
        if isinstance(o, dict):  return {k: _ser(v) for k, v in o.items()}
        if isinstance(o, list):  return [_ser(x) for x in o]
        return o

    (OUT_DIR / "train_results.json").write_text(
        json.dumps(_ser(results), indent=2))
    (OUT_DIR / "thresholds.json").write_text(
        json.dumps(_ser({"thresholds": thresholds_out,
                         "emotions": EMOTIONS, "tuned_on": "dev"}), indent=2))

    # Per-emotion breakdown table
    per_emo = {}
    for mname, mres in results.items():
        per_emo[mname] = mres["test"]["per_emotion"]
    (OUT_DIR / "per_emotion_results.json").write_text(
        json.dumps(_ser(per_emo), indent=2))

    # Calibration plots
    for mname, proba, label in [
        ("ER-A", pa_te, "ER-A Standard Fusion"),
        ("ER-B", pb_te, "ER-B Disagreement-Aware"),
        ("ER-C", pc_te, "ER-C Reliability-Aware"),
    ]:
        plot_calibration(Y_te, proba, f"{label} — Calibration (Test Set)",
                         str(PLOT_DIR / f"calibration_{mname.replace('-','_')}.png"))

    # F1 comparison plot
    plot_f1_comparison(results, str(PLOT_DIR / "f1_comparison.png"))

    # ══════════════════════════════════════════════════════════════════════
    # SECTION 9 — STATISTICAL COMPARISON (permutation test)
    # ══════════════════════════════════════════════════════════════════════
    def permutation_test(a_vals, b_vals, n=5000, seed=42):
        rng = np.random.default_rng(seed)
        obs = np.mean(b_vals) - np.mean(a_vals)
        pool = np.concatenate([a_vals, b_vals])
        diffs = [
            rng.permutation(pool)[:len(a_vals)].mean() -
            rng.permutation(pool)[len(a_vals):].mean()
            for _ in range(n)
        ]
        p = (np.abs(diffs) >= abs(obs)).mean()
        return float(obs), float(p)

    # Sample-level F1 per sample for paired comparison
    def sample_f1s(Y_true, Y_pred):
        return np.array([f1_score(Y_true[i], Y_pred[i], zero_division=0)
                         for i in range(len(Y_true))])

    f1s_A = sample_f1s(Y_te, pred_a_te)
    f1s_B = sample_f1s(Y_te, pred_b_te)
    f1s_C = sample_f1s(Y_te, pred_c_te)

    diff_BA, p_BA = permutation_test(f1s_A, f1s_B)
    diff_CA, p_CA = permutation_test(f1s_A, f1s_C)

    stat_results = {
        "B_vs_A": {"diff_macro_f1": diff_BA, "p_value": p_BA,
                   "significant": p_BA < 0.05},
        "C_vs_A": {"diff_macro_f1": diff_CA, "p_value": p_CA,
                   "significant": p_CA < 0.05},
    }
    (OUT_DIR / "statistical_comparison.json").write_text(
        json.dumps(_ser(stat_results), indent=2))

    # ══════════════════════════════════════════════════════════════════════
    # CONSOLE SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 68)
    print("  RESULTS — ElderReact Official Test Set")
    print("=" * 68)
    print(f"  {'Model':<10} {'Macro-F1':>9} {'Micro-F1':>9} {'Hamming':>9} "
          f"{'ECE':>8} {'Brier':>8}")
    print("-" * 68)
    for mname, mres in results.items():
        t = mres["test"]
        print(f"  {mname:<10} {t['macro_f1']:>9.4f} {t['micro_f1']:>9.4f} "
              f"{t['hamming']:>9.4f} {t['mean_ece']:>8.4f} {t['mean_brier']:>8.4f}")
    print("-" * 68)
    print(f"\n  Permutation tests (vs ER-A):")
    print(f"    B−A: diff={diff_BA:+.4f}  p={p_BA:.4f}  "
          f"{'SIGNIFICANT' if p_BA<0.05 else 'not significant'}")
    print(f"    C−A: diff={diff_CA:+.4f}  p={p_CA:.4f}  "
          f"{'SIGNIFICANT' if p_CA<0.05 else 'not significant'}")
    print("=" * 68)
    print(f"\n  Per-Emotion F1 (Test):")
    print(f"  {'Emotion':<12} {'ER-A':>7} {'ER-B':>7} {'ER-C':>7}")
    print("  " + "-" * 34)
    for e in EMOTIONS:
        fa = results["ER-A"]["test"]["per_emotion"][e]["f1"]
        fb = results["ER-B"]["test"]["per_emotion"][e]["f1"]
        fc = results["ER-C"]["test"]["per_emotion"][e]["f1"]
        best = max(fa, fb, fc)
        marker = lambda v: "*" if abs(v - best) < 1e-6 else " "
        print(f"  {e:<12} {fa:>6.3f}{marker(fa)} {fb:>6.3f}{marker(fb)} {fc:>6.3f}{marker(fc)}")
    print(f"\n  All outputs → {OUT_DIR}")
    log.info(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")

    return results, stat_results


if __name__ == "__main__":
    run()
