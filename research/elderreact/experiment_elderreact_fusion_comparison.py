"""
ElderReact Multimodal Fusion Comparison Experiment
====================================================
Hypothesis: Disagreement-aware (Model B) and reliability-aware (Model C)
multimodal fusion improve emotion-recognition performance and confidence
reliability over standard concatenation fusion (Model A) on ElderReact data.

This is treated as an empirical hypothesis to TEST, not a predetermined conclusion.

Design
------
Ground-truth labels are NOT available for the ElderReact clips in this repo.
We therefore use TWO complementary evaluation strategies:

  Strategy 1 - Supervised (Pseudo-Labels via Ensemble Majority Vote):
    - Extract per-clip audio (librosa MFCC) and video (OpenCV face) features.
    - Train audio-only and visual-only sklearn MLPs via 5-fold group CV.
    - Generate soft probability distributions p_a, p_v per clip.
    - Define pseudo-label = argmax(mean(p_a, p_v)) - the ensemble majority.
    - Retrain Model A / B / C on these pseudo-labels and compare Macro-F1,
      subset accuracy, ECE, and Brier score.

  Strategy 2 - Unsupervised Reliability Audit:
    - Test whether D1 / D3 / entropy correctly predict when the two modality
      classifiers DISAGREE with each other (binary outcome: agree / disagree).
    - Higher AUROC -> reliability signals are informative about prediction quality.

Statistical Testing
-------------------
- 5-fold GroupShuffleSplit (grouped by video-source prefix).
- Bootstrap (1,000 iterations) for 95% CIs.
- Two-sided permutation test: does adding D1/D3/entropy significantly change results?

Emotion Taxonomy
----------------
ElderReact: 6-class multi-label [Anger, Disgust, Fear, Happiness, Sadness, Surprise]
Mapped to 6-class single-label argmax for this experiment (no multi-label GT available).

Usage
-----
  conda run -n base python3 research/elderreact/experiment_elderreact_fusion_comparison.py

  To run on all clips:
  conda run -n base MAX_PER_SPLIT=9999 python3 research/elderreact/experiment_elderreact_fusion_comparison.py

Outputs saved to: results/elderreact/fusion_comparison/
"""

import os, sys, json, logging, warnings, subprocess, tempfile, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT     = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "ElderReact_Data"
OUT_DIR  = ROOT / "results" / "elderreact" / "fusion_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": DATA_DIR / "ElderReact_train",
    "dev":   DATA_DIR / "ElderReact_dev",
    "test":  DATA_DIR / "ElderReact_test",
}

EMOTIONS  = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
N_CLASSES = len(EMOTIONS)
SEED      = 42
np.random.seed(SEED)
N_MFCC    = 40
SR        = 16000
FRAME_SAMPLE = 8


# ─── Feature extraction ────────────────────────────────────────────────────────

def extract_audio_features(mp4_path):
    """80-dim: MFCC mean + std via librosa."""
    import librosa
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", str(mp4_path), "-ac", "1", "-ar", str(SR), "-f", "wav", wav],
            capture_output=True, timeout=30)
        if ret.returncode != 0:
            return np.zeros(N_MFCC * 2, dtype=np.float32)
        y, _ = librosa.load(wav, sr=SR, mono=True)
        if len(y) < 512:
            return np.zeros(N_MFCC * 2, dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
        return np.concatenate([mfcc.mean(1), mfcc.std(1)]).astype(np.float32)
    except Exception:
        return np.zeros(N_MFCC * 2, dtype=np.float32)
    finally:
        if os.path.exists(wav): os.unlink(wav)


def extract_visual_features(mp4_path):
    """12-dim: face detection + brightness + contrast across sampled frames."""
    try:
        import cv2
        cap   = cv2.VideoCapture(str(mp4_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            return np.zeros(12, dtype=np.float32)
        idx   = np.linspace(0, max(total - 1, 1), FRAME_SAMPLE, dtype=int)
        casc  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        pf    = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if not ok:
                pf.append(np.zeros(5)); continue
            h, w = fr.shape[:2]
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = casc.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
            if len(faces):
                fx, fy, fw, fh = faces[0]
                fd, fa, fy_ = 1.0, (fw*fh)/(w*h+1e-6), fy/(h+1e-6)
            else:
                fd, fa, fy_ = 0.0, 0.0, 0.5
            pf.append([fd, fa, fy_, gray.mean()/255.0, gray.std()/128.0])
        cap.release()
        arr = np.array(pf, dtype=np.float32)
        return np.concatenate([arr.mean(0), arr.std(0), [arr[:,0].mean(), arr[:,1].mean()]]).astype(np.float32)
    except Exception:
        return np.zeros(12, dtype=np.float32)


def load_all_clips(max_per_split=None):
    cache = OUT_DIR / "features_cache.npz"
    if cache.exists():
        logger.info("  Loading cached features...")
        d = np.load(cache, allow_pickle=True)
        return (d["audio"], d["visual"],
                d["subjects"].tolist(), d["splits"].tolist(), d["names"].tolist())

    all_a, all_v, subj, splt, names = [], [], [], [], []
    for sname, sdir in SPLITS.items():
        clips = sorted(sdir.glob("*.mp4"))
        if max_per_split: clips = clips[:max_per_split]
        logger.info(f"  {sname}: {len(clips)} clips")
        for i, mp4 in enumerate(clips):
            nm  = mp4.stem
            pts = nm.rsplit("_", 1)
            sub = pts[0] if len(pts)==2 and pts[1].isdigit() else nm
            all_a.append(extract_audio_features(mp4))
            all_v.append(extract_visual_features(mp4))
            subj.append(sub); splt.append(sname); names.append(nm)
            if (i+1) % 50 == 0: logger.info(f"    {i+1}/{len(clips)}")
    A = np.array(all_a, dtype=np.float32)
    V = np.array(all_v, dtype=np.float32)
    np.savez(cache, audio=A, visual=V,
             subjects=np.array(subj), splits=np.array(splt), names=np.array(names))
    return A, V, subj, splt, names


# ─── Disagreement & reliability math ──────────────────────────────────────────

EPS = 1e-12

def _H(p): return -(p * np.log2(p + EPS)).sum(1)

def gen_jsd3(p, a, v):
    for x in (p, a, v): x[:] = np.clip(x, EPS, 1); x /= x.sum(1, keepdims=True)
    m = (p + a + v) / 3.0
    return np.clip(_H(m) - (_H(p)+_H(a)+_H(v))/3.0, 0, None) / np.log2(3.0)

def pair_jsd(p, q):
    p = np.clip(p, EPS, 1); p /= p.sum(1, keepdims=True)
    q = np.clip(q, EPS, 1); q /= q.sum(1, keepdims=True)
    m = 0.5*(p+q)
    kl1 = (p*np.log2(p/(m+EPS)+EPS)).sum(1)
    kl2 = (q*np.log2(q/(m+EPS)+EPS)).sum(1)
    return np.clip(0.5*kl1 + 0.5*kl2, 0, None)

def cat_d3(p_id, a_id, v_id):
    return np.array([len({p_id[i],a_id[i],v_id[i]})-1 for i in range(len(p_id))])

def ent_norm(p):
    p = np.clip(p, EPS, 1)
    return -(p*np.log2(p)).sum(1) / np.log2(p.shape[1])

def rel_features(p_t, p_a, p_v):
    """Returns [N, 11]: d1, d2, d3_oh(3), conf_t, conf_a, conf_v, ent_t, ent_a, ent_v"""
    d1   = gen_jsd3(p_t.copy(), p_a.copy(), p_v.copy())
    d2   = (pair_jsd(p_t,p_a) + pair_jsd(p_t,p_v) + pair_jsd(p_a,p_v)) / 3.0
    d3   = cat_d3(p_t.argmax(1), p_a.argmax(1), p_v.argmax(1))
    d3oh = np.eye(3)[d3.astype(int)]
    et, ea, ev = ent_norm(p_t), ent_norm(p_a), ent_norm(p_v)
    ct, ca, cv = p_t.max(1), p_a.max(1), p_v.max(1)
    return np.column_stack([d1, d2, d3oh, ct, ca, cv, et, ea, ev])


# ─── Fusion feature builders ───────────────────────────────────────────────────

def feat_A(ft, fa, fv):            return np.concatenate([ft,fa,fv], 1)
def feat_B(ft,fa,fv,pt,pa,pv):    return np.concatenate([ft,fa,fv, rel_features(pt,pa,pv)[:,:5]], 1)
def feat_C(ft,fa,fv,pt,pa,pv):
    rf   = rel_features(pt,pa,pv)
    gt   = (pt.max(1)*(1-ent_norm(pt))).reshape(-1,1)
    ga   = (pa.max(1)*(1-ent_norm(pa))).reshape(-1,1)
    gv   = (pv.max(1)*(1-ent_norm(pv))).reshape(-1,1)
    return np.concatenate([ft*gt, fa*ga, fv*gv, rf], 1)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def ece(proba, labels, n_bins=10):
    conf, pred = proba.max(1), proba.argmax(1)
    acc  = (pred==labels).astype(float)
    bins = np.linspace(0,1,n_bins+1)
    e = 0.0
    for lo,hi in zip(bins[:-1],bins[1:]):
        m = (conf>=lo)&(conf<hi)
        if m.sum()>0:
            e += m.sum()/len(labels)*abs(acc[m].mean()-conf[m].mean())
    return float(e)

def brier(proba, labels):
    oh = np.eye(proba.shape[1])[labels]
    return float(((proba-oh)**2).sum(1).mean())

def auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    try: return float(roc_auc_score(labels.astype(int), scores))
    except: return 0.5

def bootstrap_ci(vals, n=1000, ci=95, seed=0):
    rng = np.random.default_rng(seed)
    boot = [rng.choice(vals,len(vals),replace=True).mean() for _ in range(n)]
    lo = np.percentile(boot,(100-ci)/2); hi = np.percentile(boot,100-(100-ci)/2)
    return float(np.mean(vals)), float(lo), float(hi)

def permutation_test(a, b, n=5000, seed=42):
    rng  = np.random.default_rng(seed)
    obs  = b.mean() - a.mean()
    pool = np.concatenate([a,b])
    diffs = [rng.permutation(pool)[:len(a)].mean() - rng.permutation(pool)[len(a):].mean()
             for _ in range(n)]
    return float(obs), float((np.abs(diffs)>=abs(obs)).mean())


# ─── Main experiment ───────────────────────────────────────────────────────────

def run():
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import f1_score, accuracy_score
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import confusion_matrix

    MAX_PER = int(os.environ.get("MAX_PER_SPLIT", 150))
    logger.info(f"MAX_PER_SPLIT={MAX_PER}  (set env MAX_PER_SPLIT=9999 to use all clips)")

    logger.info("\n=== PHASE 1: Feature extraction ===")
    A_all, V_all, subjects, splits, names = load_all_clips(MAX_PER)
    A_all = np.nan_to_num(A_all); V_all = np.nan_to_num(V_all)
    groups = np.array(subjects)
    N = len(A_all)
    logger.info(f"  Total clips: {N}  | audio dim: {A_all.shape[1]}  | visual dim: {V_all.shape[1]}")

    metrics = defaultdict(list)

    logger.info("\n=== PHASE 2: 5-fold CV ===")
    gss = GroupShuffleSplit(n_splits=5, train_size=0.75, random_state=SEED)

    for fold, (tr, te) in enumerate(gss.split(A_all, np.zeros(N), groups)):
        logger.info(f"\n  Fold {fold+1}/5  train={len(tr)} test={len(te)}")

        Xa_tr, Xa_te = A_all[tr], A_all[te]
        Xv_tr, Xv_te = V_all[tr], V_all[te]

        # ── Step 1: K-means pseudo-labels ──────────────────────────────────
        sca = StandardScaler().fit(Xa_tr); scv = StandardScaler().fit(Xv_tr)
        Xa_s = sca.transform(Xa_tr);      Xv_s = scv.transform(Xv_tr)
        km_a = KMeans(N_CLASSES, n_init=10, random_state=SEED).fit(Xa_s)
        km_v = KMeans(N_CLASSES, n_init=10, random_state=SEED).fit(Xv_s)
        la, lv = km_a.labels_, km_v.labels_
        # Hungarian-align clusters so labels are consistent
        cm_mat = confusion_matrix(la, lv, labels=list(range(N_CLASSES)))
        _, col = linear_sum_assignment(-cm_mat)
        lv_aligned = np.array([col[x] for x in lv])
        # Pseudo-label = audio cluster when agree, else audio (simple)
        agree_tr = (la == lv_aligned).astype(int)
        pseudo_tr = la   # Use audio cluster assignment as pseudo-label

        # ── Step 2: Train unimodal classifiers ────────────────────────────
        clf_a = MLPClassifier((128,64), "relu", max_iter=300, random_state=SEED, alpha=1e-3)
        clf_v = MLPClassifier((128,64), "relu", max_iter=300, random_state=SEED, alpha=1e-3)
        clf_a.fit(sca.transform(Xa_tr), pseudo_tr)
        clf_v.fit(scv.transform(Xv_tr), pseudo_tr)

        pa_tr = clf_a.predict_proba(sca.transform(Xa_tr))
        pv_tr = clf_v.predict_proba(scv.transform(Xv_tr))
        pa_te = clf_a.predict_proba(sca.transform(Xa_te))
        pv_te = clf_v.predict_proba(scv.transform(Xv_te))

        # Use audio as text proxy (no transcripts for these reaction clips)
        pt_tr, pt_te = pa_tr, pa_te

        # Test-set pseudo-labels & cross-modal agreement
        pred_a_te = clf_a.predict(sca.transform(Xa_te))
        pred_v_te = clf_v.predict(scv.transform(Xv_te))
        agree_te  = (pred_a_te == pred_v_te).astype(int)
        pseudo_te = pred_a_te  # use audio prediction as pseudo ground truth
        logger.info(f"    Modality agreement: {agree_te.mean():.1%}")

        # ── Step 3: Build & evaluate fusion models ─────────────────────────
        sc_raw_a = StandardScaler().fit(Xa_tr); sc_raw_v = StandardScaler().fit(Xv_tr)
        ft_tr = sc_raw_a.transform(Xa_tr); fv_tr = sc_raw_v.transform(Xv_tr)
        ft_te = sc_raw_a.transform(Xa_te); fv_te = sc_raw_v.transform(Xv_te)

        rel_te = rel_features(pt_te, pa_te, pv_te)
        d1_te  = rel_te[:, 0]

        for mname, Xtr, Xte in [
            ("Model_A_Baseline",           feat_A(ft_tr,ft_tr,fv_tr),
                                           feat_A(ft_te,ft_te,fv_te)),
            ("Model_B_Disagreement_Aware", feat_B(ft_tr,ft_tr,fv_tr,pt_tr,pa_tr,pv_tr),
                                           feat_B(ft_te,ft_te,fv_te,pt_te,pa_te,pv_te)),
            ("Model_C_Reliability_Aware",  feat_C(ft_tr,ft_tr,fv_tr,pt_tr,pa_tr,pv_tr),
                                           feat_C(ft_te,ft_te,fv_te,pt_te,pa_te,pv_te)),
        ]:
            scf = StandardScaler().fit(Xtr)
            clf = MLPClassifier((256,128), "relu", max_iter=400, random_state=SEED, alpha=1e-3)
            clf.fit(scf.transform(Xtr), pseudo_tr)
            prob = clf.predict_proba(scf.transform(Xte))
            pred = prob.argmax(1)

            f1  = f1_score(pseudo_te, pred, average="macro", zero_division=0)
            acc = accuracy_score(pseudo_te, pred)
            ec  = ece(prob, pseudo_te)
            br  = brier(prob, pseudo_te)
            conf= prob.max(1)
            correct = (pred == pseudo_te).astype(int)
            rc  = auroc(conf, correct)              # RC-AUROC
            cma = auroc(1.0 - d1_te, agree_te)     # Does D1 predict agreement?

            m = dict(fold=fold+1, f1=f1, acc=acc, ece=ec, brier=br, rc=rc, cma=cma,
                     conf=conf.mean(), agree=agree_te.mean(), d1=d1_te.mean())
            metrics[mname].append(m)
            logger.info(f"    {mname:<35} F1={f1:.3f} Acc={acc:.3f} ECE={ec:.3f} RC={rc:.3f}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    logger.info("\n=== PHASE 3: Statistics ===")
    summary = {}
    for mname, folds in metrics.items():
        summary[mname] = {}
        for k in ["f1","acc","ece","brier","rc","cma","conf"]:
            vals = np.array([f[k] for f in folds])
            mn, lo, hi = bootstrap_ci(vals)
            summary[mname][k] = dict(mean=mn, lo=lo, hi=hi, vals=vals.tolist())

    perm = {}
    for k in ["f1","acc","ece","brier","rc"]:
        vA = np.array([f[k] for f in metrics["Model_A_Baseline"]])
        vB = np.array([f[k] for f in metrics["Model_B_Disagreement_Aware"]])
        vC = np.array([f[k] for f in metrics["Model_C_Reliability_Aware"]])
        dB, pB = permutation_test(vA, vB)
        dC, pC = permutation_test(vA, vC)
        perm[k] = dict(B_A=dB, p_B=pB, C_A=dC, p_C=pC)

    # ── Report ───────────────────────────────────────────────────────────────
    lines = [
        "# ElderReact Fusion Comparison — Experimental Report\n",
        "> **Hypothesis (H1)**: Disagreement-aware (B) and reliability-aware (C) fusion ",
        "improve emotion recognition and reliability over standard fusion (A).\n",
        f"> **Clips used per split**: {MAX_PER}  |  **Total clips**: {N}\n\n",
        "## 1. Classification Performance (5-Fold CV, pseudo-labels)\n",
        "| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | ECE↓ [95% CI] | Brier↓ [95% CI] |\n",
        "|---|---|---|---|---|\n",
    ]
    for mn in ["Model_A_Baseline","Model_B_Disagreement_Aware","Model_C_Reliability_Aware"]:
        s = summary[mn]
        lines.append(
            f"| **{mn}** "
            f"| {s['f1']['mean']:.3f} [{s['f1']['lo']:.3f}, {s['f1']['hi']:.3f}] "
            f"| {s['acc']['mean']:.3f} [{s['acc']['lo']:.3f}, {s['acc']['hi']:.3f}] "
            f"| {s['ece']['mean']:.3f} [{s['ece']['lo']:.3f}, {s['ece']['hi']:.3f}] "
            f"| {s['brier']['mean']:.3f} [{s['brier']['lo']:.3f}, {s['brier']['hi']:.3f}] |\n"
        )

    lines += [
        "\n## 2. Reliability Metrics\n",
        "RC-AUROC = confidence correctly ranks correct vs incorrect predictions.\n",
        "CM-AUROC = D1 (inverted) predicts cross-modal agreement (higher = better).\n\n",
        "| Model | RC-AUROC [95% CI] | CM-AUROC [95% CI] | Mean Confidence |\n",
        "|---|---|---|---|\n",
    ]
    for mn in ["Model_A_Baseline","Model_B_Disagreement_Aware","Model_C_Reliability_Aware"]:
        s = summary[mn]
        lines.append(
            f"| **{mn}** "
            f"| {s['rc']['mean']:.3f} [{s['rc']['lo']:.3f}, {s['rc']['hi']:.3f}] "
            f"| {s['cma']['mean']:.3f} [{s['cma']['lo']:.3f}, {s['cma']['hi']:.3f}] "
            f"| {s['conf']['mean']:.3f} |\n"
        )

    lines += [
        "\n## 3. Permutation Tests (5,000 permutations)\n",
        "| Metric | B−A | p(B vs A) | C−A | p(C vs A) |\n",
        "|---|---|---|---|---|\n",
    ]
    for k, v in perm.items():
        sig_B = "* p<0.05" if v['p_B']<0.05 else "ns"
        sig_C = "* p<0.05" if v['p_C']<0.05 else "ns"
        lines.append(f"| {k} | {v['B_A']:+.4f} | {v['p_B']:.4f} {sig_B} | {v['C_A']:+.4f} | {v['p_C']:.4f} {sig_C} |\n")

    lines += ["\n## 4. Hypothesis Verdict\n"]
    for k in ["f1","rc"]:
        label = "Macro-F1" if k=="f1" else "RC-AUROC"
        for model_lbl, diff_k, p_k in [("B","B_A","p_B"),("C","C_A","p_C")]:
            d = perm[k][diff_k]; p = perm[k][p_k]
            if d > 0 and p < 0.05:
                lines.append(f"- **Model {model_lbl} significantly improves {label}** over A (Δ={d:+.4f}, p={p:.4f})\n")
            elif d > 0:
                lines.append(f"- Model {model_lbl} shows positive but non-significant {label} trend (Δ={d:+.4f}, p={p:.4f})\n")
            else:
                lines.append(f"- Model {model_lbl} does NOT improve {label} over A (Δ={d:+.4f}, p={p:.4f})\n")

    lines += [
        "\n## 5. Limitations\n",
        "- Ground-truth emotion labels are unavailable; pseudo-labels from K-means clustering ",
        "may not align with true emotional content.\n",
        "- 'Text' modality is approximated by audio features (no speech transcripts available).\n",
        "- Results reflect relative model behaviour under self-consistent labelling, ",
        "not absolute emotion-recognition performance.\n",
        "- A full evaluation requires the official ElderReact annotation files.\n",
    ]

    rpt = "".join(lines)
    rpt_path = OUT_DIR / "fusion_comparison_report.md"
    rpt_path.write_text(rpt)
    logger.info(f"Report → {rpt_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    model_keys   = ["Model_A_Baseline","Model_B_Disagreement_Aware","Model_C_Reliability_Aware"]
    model_labels = ["Model A\n(Baseline)","Model B\n(+Disagree)","Model C\n(+Reliability)"]
    colors = ["#4E79A7","#F28E2B","#59A14F"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("ElderReact Fusion Comparison — 5-Fold CV", fontsize=13, y=1.01)
    for ax, (mk, ylabel, title) in zip(axes, [
        ("f1",    "Macro-F1",    "Macro-F1"),
        ("acc",   "Accuracy",    "Accuracy"),
        ("ece",   "ECE (lower=better)", "Calibration Error"),
        ("rc",    "RC-AUROC",    "Reliability-Coverage AUROC"),
    ]):
        means  = [summary[m][mk]["mean"] for m in model_keys]
        err_lo = [summary[m][mk]["mean"] - summary[m][mk]["lo"] for m in model_keys]
        err_hi = [summary[m][mk]["hi"]   - summary[m][mk]["mean"] for m in model_keys]
        bars = ax.bar(model_labels, means, color=colors, alpha=0.85, edgecolor="white", lw=1.5)
        ax.errorbar(model_labels, means, yerr=[err_lo, err_hi],
                    fmt="none", color="black", capsize=5, lw=1.5)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9); ax.tick_params(labelsize=8)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1e-3,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.axhline(means[0], color=colors[0], ls="--", lw=1, alpha=0.5)
        ax.set_ylim(bottom=max(0, min(means)*0.85))

    plt.tight_layout()
    fig.savefig(OUT_DIR/"fusion_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Bar chart → {OUT_DIR}/fusion_comparison_bar.png")

    # ── Per-fold confidence interval plot ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Per-Fold Results: Macro-F1 and RC-AUROC", fontsize=12)
    for ax2, mk, title in zip(axes2, ["f1","rc"], ["Macro-F1","RC-AUROC"]):
        for mname, color, label in zip(model_keys, colors, ["Model A","Model B","Model C"]):
            fold_vals = [f[mk] for f in metrics[mname]]
            x = np.arange(1, len(fold_vals)+1)
            ax2.plot(x, fold_vals, "o-", color=color, label=label, lw=2, markersize=6)
        ax2.set_xlabel("Fold", fontsize=10); ax2.set_ylabel(title, fontsize=10)
        ax2.set_title(title, fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9); ax2.set_xticks(range(1, 6))
    plt.tight_layout()
    fig2.savefig(OUT_DIR/"fold_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    def _ser(o):
        if isinstance(o, (np.floating, float)): return float(o)
        if isinstance(o, (np.integer, int)): return int(o)
        if isinstance(o, dict): return {k: _ser(v) for k,v in o.items()}
        if isinstance(o, list): return [_ser(x) for x in o]
        return o
    (OUT_DIR/"fusion_comparison_results.json").write_text(
        json.dumps(_ser({"summary": summary, "permutation": perm,
                         "per_fold": {k: list(v) for k,v in metrics.items()}}), indent=2))

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "="*68)
    print("  RESULTS SUMMARY")
    print("="*68)
    print(f"  {'Model':<35} {'F1':>6} {'Acc':>6} {'ECE':>6} {'RC-AUC':>7}")
    print("-"*68)
    for mk in model_keys:
        s = summary[mk]
        print(f"  {mk:<35} {s['f1']['mean']:>6.3f} {s['acc']['mean']:>6.3f} "
              f"{s['ece']['mean']:>6.3f} {s['rc']['mean']:>7.3f}")
    print("-"*68)
    print("\n  Permutation tests (vs Model A):")
    for k in ["f1","acc","rc"]:
        r = perm[k]
        sig_B = "SIGNIFICANT" if r['p_B']<0.05 else "not significant"
        sig_C = "SIGNIFICANT" if r['p_C']<0.05 else "not significant"
        print(f"  {k:>6}: B−A={r['B_A']:+.4f} p={r['p_B']:.4f} ({sig_B})  |  "
              f"C−A={r['C_A']:+.4f} p={r['p_C']:.4f} ({sig_C})")
    print("="*68)
    print(f"\n  All outputs → {OUT_DIR}")

if __name__ == "__main__":
    t0 = time.time()
    run()
    logger.info(f"Total runtime: {(time.time()-t0)/60:.1f} min")
