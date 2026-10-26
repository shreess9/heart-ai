"""
experiment_elderreact_rich_features.py
========================================
H.E.A.R.T. AI — ElderReact: Richer Pretrained Feature Experiment (Phase 8)

PURPOSE
-------
Compare richer pretrained audio/visual embeddings against ER-A (COVAREP+OpenFace)
on the same official ElderReact train/dev/test split.

This is an exploratory experiment, NOT a claim of improvement.
All comparisons use the same official test split as ER-A.
Calibration and thresholds fitted on train/dev only.

STATUS: SCAFFOLD — NOT YET RUN
--------------------------------
The following sections are ready:
  [x] Data split loading (reuses experiment_elderreact_proper.py helpers)
  [x] Wav2Vec2 audio feature extraction (CPU-only, may be slow)
  [x] OpenFace raw features as visual fallback (immediate, no GPU)
  [ ] ViT visual extraction (requires download, GPU recommended)
  [ ] Full training run (depends on feature extraction completing)
  [ ] Statistical comparison vs ER-A

REQUIREMENTS
------------
  pip install transformers soundfile librosa

USAGE
-----
  conda run -n base python3 research/elderreact/experiment_elderreact_rich_features.py \
      --audio_model wav2vec2 --visual_model openface_raw --max_iter 500

IMPORTANT CONSTRAINTS
----------------------
- Fit scaler, classifier, and thresholds on TRAIN only.
- Tune thresholds on DEV only.
- Evaluate on TEST once, AFTER all design decisions are fixed.
- Do not use test labels for any selection decision.
- Report differences vs ER-A with permutation test (n=5000).
- Do not claim improvement unless p < 0.05 on the permutation test.
"""

from __future__ import annotations
import argparse, logging, json, os, time, warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, hamming_loss
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_DIR  = Path("/Users/shreee/Downloads/ElderReact_official_repo")
VIDEO_DIR = Path("/Users/shreee/Desktop/heart-ai-main copy/ElderReact_Data")
OUT_DIR   = Path("/Users/shreee/Desktop/heart-ai-main copy/results/elderreact/rich_features")
BASELINE_JSON = Path("/Users/shreee/Desktop/heart-ai-main copy/results/elderreact/official/train_results.json")

EMOTIONS = ["Anger","Disgust","Fear","Happiness","Sadness","Surprise"]
SEED     = 42
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Audio Feature Extraction (Wav2Vec2)
# ══════════════════════════════════════════════════════════════════════════════

def extract_wav2vec2_features(mp4_paths: List[Path],
                               model_name: str = "facebook/wav2vec2-base",
                               layer: int = -1) -> np.ndarray:
    """
    Extract clip-level Wav2Vec2 embeddings from MP4 audio tracks.

    Process:
      1. Extract audio waveform from MP4 (ffmpeg via soundfile/librosa).
      2. Resample to 16 kHz (Wav2Vec2 requirement).
      3. Pass through Wav2Vec2 feature extractor.
      4. Pool hidden states across time (mean pooling).
      5. Return [N, 768] embeddings.

    NOTES
    -----
    - This requires: pip install transformers soundfile librosa
    - CPU inference on 1,323 clips ≈ 20–40 minutes depending on hardware.
    - For GPU: pass device="cuda" when creating the model.
    - Embeddings are NOT emotion classifiers. We train a classification head on top.
    """
    try:
        import torch
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        import librosa
    except ImportError as e:
        raise ImportError(
            f"Required package missing: {e}. "
            "Install with: pip install transformers librosa soundfile"
        ) from e

    log.info(f"Loading Wav2Vec2 model: {model_name}")
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name)
    model.eval()

    feats = []
    for i, mp4_path in enumerate(mp4_paths):
        if i % 100 == 0:
            log.info(f"  Wav2Vec2 progress: {i}/{len(mp4_paths)}")
        try:
            # Extract audio as 16kHz mono waveform
            wav, sr = librosa.load(str(mp4_path), sr=16000, mono=True)
            inputs = extractor(wav, sampling_rate=16000, return_tensors="pt",
                               padding=True)
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            # Mean-pool the selected hidden layer
            h = out.hidden_states[layer]  # [1, T, 768]
            feat = h[0].mean(0).numpy()   # [768]
        except Exception as e:
            log.warning(f"  Failed {mp4_path.name}: {e} — using zeros")
            feat = np.zeros(768)
        feats.append(feat)

    return np.array(feats)   # [N, 768]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Visual Feature Extraction (OpenFace raw or ViT)
# ══════════════════════════════════════════════════════════════════════════════

def load_openface_raw(split: str, drop_rows=None) -> np.ndarray:
    """
    Load full 1418-dim OpenFace features (no subset selection).
    This is a richer visual representation than the 178-dim subset used in ER-A.
    """
    lines = (REPO_DIR / "Features" / "Video_feat" / f"{split}_video.txt").read_text().splitlines()
    feats = []
    for i, line in enumerate(lines):
        if drop_rows and i in drop_rows:
            continue
        parts = line.strip().split("\t")
        if len(parts) < 2:
            feats.append(np.zeros(1418))
            continue
        feat = np.array([float(x) for x in parts[1].split()])
        feats.append(feat)
    return np.nan_to_num(np.array(feats))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Multi-label training (reused from ER-A architecture)
# ══════════════════════════════════════════════════════════════════════════════

class MultiLabelMLP:
    """Binary-relevance MLP: one classifier per emotion."""
    def __init__(self, hidden=(256, 128), seed=42):
        self.hidden = hidden; self.seed = seed; self.clfs = []

    def fit(self, X, Y):
        self.clfs = []
        for i in range(Y.shape[1]):
            clf = MLPClassifier(hidden_layer_sizes=self.hidden, activation="relu",
                                max_iter=500, random_state=self.seed, alpha=1e-3,
                                early_stopping=True, validation_fraction=0.1)
            clf.fit(X, Y[:, i]); self.clfs.append(clf)
        return self

    def predict_proba(self, X):
        return np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])

    def predict(self, X, thresholds=None):
        p = self.predict_proba(X)
        t = thresholds if thresholds is not None else np.full(len(self.clfs), 0.5)
        return (p >= t).astype(int)


def tune_thresholds(proba, Y, candidates=None):
    if candidates is None:
        candidates = np.linspace(0.1, 0.9, 33)
    thresholds = np.full(len(EMOTIONS), 0.5)
    for i in range(len(EMOTIONS)):
        best_f1, best_t = -1, 0.5
        for t in candidates:
            f1 = f1_score(Y[:, i], (proba[:, i] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[i] = best_t
    return thresholds


def permutation_test(a_vals, b_vals, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    obs = np.mean(b_vals) - np.mean(a_vals)
    pool = np.concatenate([a_vals, b_vals])
    diffs = [rng.permutation(pool)[:len(a_vals)].mean() -
             rng.permutation(pool)[len(a_vals):].mean() for _ in range(n)]
    return float(obs), float((np.abs(diffs) >= abs(obs)).mean())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Annotation loading (identical to official split)
# ══════════════════════════════════════════════════════════════════════════════

def load_labels(split, drop_rows=None):
    lines = (REPO_DIR / "Annotations" / f"{split}_labels.txt").read_text().splitlines()
    ids, Y = [], []
    for i, line in enumerate(lines):
        if drop_rows and i in drop_rows: continue
        parts = line.strip().split()
        if len(parts) < 7: continue
        ids.append(parts[0])
        Y.append([int(x) for x in parts[1:7]])
    return ids, np.array(Y, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Main experiment runner
# ══════════════════════════════════════════════════════════════════════════════

def run(audio_model: str = "openface_subset",
        visual_model: str = "openface_full",
        max_iter: int = 500):
    """
    Parameters
    ----------
    audio_model : "wav2vec2" | "covarep_subset" (72-dim, same as ER-A)
    visual_model : "openface_full" (1418-dim) | "openface_subset" (178-dim, same as ER-A)
    max_iter : MLP max iterations
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Rich features experiment: audio={audio_model}, visual={visual_model}")
    log.info("=" * 65)

    # ── Load labels ──────────────────────────────────────────────────────
    _, Y_tr = load_labels("train")
    _, Y_dv = load_labels("dev")
    _, Y_te = load_labels("test", drop_rows={289})
    log.info(f"Labels: train={len(Y_tr)} dev={len(Y_dv)} test={len(Y_te)}")

    # ── Load visual features ─────────────────────────────────────────────
    if visual_model == "openface_full":
        log.info("Loading full OpenFace features (1418-dim)...")
        Xv_tr = load_openface_raw("train")
        Xv_dv = load_openface_raw("dev")
        Xv_te = load_openface_raw("test", drop_rows={289})
        v_dim = 1418
    else:
        # Fallback: same 178-dim subset as ER-A (for ablation)
        def _load_sub(split, drop=None):
            lines = (REPO_DIR/"Features"/"Video_feat"/f"{split}_video.txt").read_text().splitlines()
            feats = []
            for i, l in enumerate(lines):
                if drop and i in drop: continue
                p = l.strip().split("\t")
                r = np.array([float(x) for x in p[1].split()]) if len(p)>1 else np.zeros(1418)
                feats.append(np.concatenate([r[0:8],r[288:294],r[634:717],r[997:1003],r[1343:1418]]))
            return np.nan_to_num(np.array(feats))
        Xv_tr = _load_sub("train"); Xv_dv = _load_sub("dev")
        Xv_te = _load_sub("test", {289}); v_dim = 178

    # ── Load audio features ──────────────────────────────────────────────
    if audio_model == "wav2vec2":
        log.info("Extracting Wav2Vec2 audio features — this may take 20-40 min on CPU...")
        def _mp4s(split, drop=None):
            ann_ids = [(l.strip().split()[0], i) for i, l in enumerate(
                (REPO_DIR/"Annotations"/f"{split}_labels.txt").read_text().splitlines()) if l.strip()]
            return [VIDEO_DIR/f"ElderReact_{split}"/fname
                    for i, (fname, idx) in enumerate(ann_ids) if not (drop and idx in drop)]
        Xa_tr = extract_wav2vec2_features(_mp4s("train"))
        Xa_dv = extract_wav2vec2_features(_mp4s("dev"))
        Xa_te = extract_wav2vec2_features(_mp4s("test", {289}))
        a_dim = 768
    else:
        # COVAREP 72-dim subset (same as ER-A — for ablation of only visual)
        def _load_covarep(split, drop=None):
            lines = (REPO_DIR/"Features"/"Audio_feat"/f"{split}_audio.txt").read_text().splitlines()
            feats = []
            for i, l in enumerate(lines):
                if drop and i in drop: continue
                p = l.strip().split("\t")
                r = np.array([float(x) for x in p[1].split()]) if len(p)>1 else np.zeros(148)
                feats.append(np.concatenate([r[:36], r[74:110]]))
            return np.nan_to_num(np.array(feats))
        Xa_tr = _load_covarep("train"); Xa_dv = _load_covarep("dev")
        Xa_te = _load_covarep("test", {289}); a_dim = 72

    log.info(f"Features: audio={a_dim}-dim, visual={v_dim}-dim, fused={a_dim+v_dim}-dim")

    # ── Scale (fit on train ONLY) ─────────────────────────────────────────
    sc_a = StandardScaler().fit(Xa_tr)
    sc_v = StandardScaler().fit(Xv_tr)
    Xf_tr = np.concatenate([sc_a.transform(Xa_tr), sc_v.transform(Xv_tr)], axis=1)
    Xf_dv = np.concatenate([sc_a.transform(Xa_dv), sc_v.transform(Xv_dv)], axis=1)
    Xf_te = np.concatenate([sc_a.transform(Xa_te), sc_v.transform(Xv_te)], axis=1)

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("Training rich-features model (ER-Rich)...")
    model = MultiLabelMLP(hidden=(256, 128), seed=SEED)
    model.fit(Xf_tr, Y_tr)

    # ── Tune thresholds on DEV ONLY ───────────────────────────────────────
    proba_dv = model.predict_proba(Xf_dv)
    thresholds = tune_thresholds(proba_dv, Y_dv)
    log.info(f"Dev thresholds: {dict(zip(EMOTIONS, thresholds.round(3)))}")

    # ── Evaluate on TEST (ONCE) ───────────────────────────────────────────
    proba_te = model.predict_proba(Xf_te)
    pred_te  = (proba_te >= thresholds).astype(int)

    macro_f1 = f1_score(Y_te, pred_te, average="macro", zero_division=0)
    micro_f1 = f1_score(Y_te, pred_te, average="micro", zero_division=0)
    hamming  = hamming_loss(Y_te, pred_te)
    per_emo  = f1_score(Y_te, pred_te, average=None, zero_division=0)

    # ── Compare vs ER-A (permutation test) ───────────────────────────────
    er_a_macro = json.loads(BASELINE_JSON.read_text())["ER-A"]["test"]["macro_f1"]
    sample_f1s_rich = np.array([f1_score(Y_te[i], pred_te[i], zero_division=0)
                                  for i in range(len(Y_te))])

    # Load ER-A sample-level F1s by reproducing (requires ER-A checkpoint)
    # For now, report overall comparison only
    diff = macro_f1 - er_a_macro
    log.info(f"ER-A Macro-F1 (baseline):  {er_a_macro:.4f}")
    log.info(f"ER-Rich Macro-F1:          {macro_f1:.4f}  (Δ = {diff:+.4f})")

    results = dict(
        model="ER-Rich",
        audio_model=audio_model,
        visual_model=visual_model,
        test=dict(
            macro_f1=float(macro_f1),
            micro_f1=float(micro_f1),
            hamming=float(hamming),
            per_emotion={e: float(f) for e, f in zip(EMOTIONS, per_emo)},
        ),
        comparison_vs_ERA=dict(
            era_macro_f1=float(er_a_macro),
            diff=float(diff),
            note="Permutation test requires ER-A sample-level F1s — run full comparison script"
        ),
        thresholds=thresholds.tolist(),
        features=dict(audio_dim=a_dim, visual_dim=v_dim),
    )

    out_path = OUT_DIR / f"results_{audio_model}_{visual_model}.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info(f"\nResults saved: {out_path}")

    print("\n" + "=" * 60)
    print(f"  ER-Rich ({audio_model} + {visual_model})")
    print("=" * 60)
    print(f"  Macro-F1: {macro_f1:.4f}  (ER-A baseline: {er_a_macro:.4f}, Δ={diff:+.4f})")
    print(f"  Micro-F1: {micro_f1:.4f}  Hamming: {hamming:.4f}")
    print(f"\n  Per-Emotion F1 (Test):")
    for e, f in zip(EMOTIONS, per_emo):
        print(f"    {e:<12}: {f:.3f}")
    if diff > 0:
        print(f"\n  NOTE: +{diff:.4f} over baseline — run full statistical comparison to confirm.")
    else:
        print(f"\n  NOTE: No improvement over ER-A with this feature configuration.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ElderReact rich features experiment")
    parser.add_argument("--audio_model",   default="covarep_subset",
                        choices=["covarep_subset", "wav2vec2"],
                        help="Audio feature extractor")
    parser.add_argument("--visual_model",  default="openface_full",
                        choices=["openface_subset", "openface_full"],
                        help="Visual feature extractor")
    parser.add_argument("--max_iter",      type=int, default=500)
    args = parser.parse_args()

    # Quick sanity run first: ablation with full OpenFace (covarep unchanged)
    run(audio_model=args.audio_model,
        visual_model=args.visual_model,
        max_iter=args.max_iter)
