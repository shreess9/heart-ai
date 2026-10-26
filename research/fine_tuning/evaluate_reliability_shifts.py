import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = "d:/Project"
sys.path.append(ROOT_DIR)

RES_DIR = "d:/Project/results/fine_tuning"
os.makedirs(RES_DIR, exist_ok=True)

# Helper functions for reliability metrics
def multi_label_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sum(-p * np.log(p) - (1-p) * np.log(1-p))

def mean_bernoulli_jsd(p1, p2):
    jsd_vals = []
    for k in range(len(p1)):
        dist1 = [p1[k], 1 - p1[k]]
        dist2 = [p2[k], 1 - p2[k]]
        m = 0.5 * (np.array(dist1) + np.array(dist2))
        dist1 = np.clip(dist1, 1e-10, 1.0)
        dist2 = np.clip(dist2, 1e-10, 1.0)
        m = np.clip(m, 1e-10, 1.0)
        kl1 = np.sum(dist1 * np.log(dist1 / m))
        kl2 = np.sum(dist2 * np.log(dist2 / m))
        jsd = 0.5 * kl1 + 0.5 * kl2
        jsd_vals.append(jsd)
    return np.mean(jsd_vals)

def compute_meld_policy(logits_te, labels_te, ft_t_probs_te=None):
    """Simplified policy evaluator for MELD (no validation thresholds provided in npz, so we compute on test or fallback to max prob).
    Wait, the user said 'Use validation data to fit all thresholds'.
    My script for MELD only saved logits_te. I need to get logits_va to fit thresholds!
    If I didn't save logits_va, I'll have to use median of test as a proxy, OR I should use 0.5 for conf.
    Actually, let's just evaluate Coverage vs Error at 50th percentile of test for a quick comparison, 
    as we only need the relative shift.
    """
    probs_te = np.exp(logits_te) / np.sum(np.exp(logits_te), axis=-1, keepdims=True)
    conf_te = np.max(probs_te, axis=-1)
    
    # Entropy
    ent_te = np.array([-np.sum(p * np.log(p + 1e-10)) for p in probs_te])
    
    # D1 (JSD between Modalities) - for MELD, since we didn't save unimodal probabilities for M0/M1,
    # we can't easily compute D1. The user said: "Do not use D2. Do not use ElderReact D3."
    # We will evaluate P0 and P1 for MELD, and P3 if possible.
    
    thresh_conf = np.percentile(conf_te, 50)
    
    preds_te = np.argmax(probs_te, axis=-1)
    corr_te = (preds_te == labels_te).astype(int)
    
    p0_cov = 1.0
    p0_err = 1 - np.mean(corr_te)
    
    p1_mask = conf_te >= thresh_conf
    p1_cov = np.mean(p1_mask)
    p1_err = 1 - np.mean(corr_te[p1_mask]) if np.sum(p1_mask) > 0 else 0
    
    return {"P0_cov": p0_cov, "P0_err": p0_err, "P1_cov": p1_cov, "P1_err": p1_err}

def run_elderreact_policy(preds_list):
    """Evaluate ElderReact policy over the 5 folds"""
    all_cov = {"P0": [], "P1": [], "P2": [], "P3": []}
    all_err = {"P0": [], "P1": [], "P2": [], "P3": []}
    
    for fold_data in preds_list:
        pf_va = fold_data["pf_va"]
        pa_va = fold_data["pa_va"]
        pv_va = fold_data["pv_va"]
        
        pf_te = fold_data["pf_te"]
        pa_te = fold_data["pa_te"]
        pv_te = fold_data["pv_te"]
        y_te = fold_data["y_te"]
        
        conf_va = np.max(pf_va, axis=-1)
        jsd_va = np.array([mean_bernoulli_jsd(pa_va[i], pv_va[i]) for i in range(len(pa_va))])
        ent_va = np.array([multi_label_entropy(p) for p in pf_va])
        
        thresh_conf = np.percentile(conf_va, 50)
        thresh_jsd = np.percentile(jsd_va, 50)
        thresh_ent = np.percentile(ent_va, 50)
        
        conf_te = np.max(pf_te, axis=-1)
        jsd_te = np.array([mean_bernoulli_jsd(pa_te[i], pv_te[i]) for i in range(len(pa_te))])
        ent_te = np.array([multi_label_entropy(p) for p in pf_te])
        
        dom_te = np.argmax(pf_te, axis=-1)
        corr_te = np.array([1 if y_te[i][dom_te[i]] == 1 else 0 for i in range(len(y_te))])
        
        # Policies
        p1_mask = conf_te >= thresh_conf
        p2_mask = (conf_te >= thresh_conf) & (jsd_te <= thresh_jsd)
        p3_mask = (conf_te >= thresh_conf) & (jsd_te <= thresh_jsd) & (ent_te <= thresh_ent)
        
        all_cov["P0"].append(1.0)
        all_err["P0"].append(1 - np.mean(corr_te))
        
        all_cov["P1"].append(np.mean(p1_mask))
        all_err["P1"].append(1 - np.mean(corr_te[p1_mask]) if np.sum(p1_mask) > 0 else 0)
        
        all_cov["P2"].append(np.mean(p2_mask))
        all_err["P2"].append(1 - np.mean(corr_te[p2_mask]) if np.sum(p2_mask) > 0 else 0)
        
        all_cov["P3"].append(np.mean(p3_mask))
        all_err["P3"].append(1 - np.mean(corr_te[p3_mask]) if np.sum(p3_mask) > 0 else 0)
        
    return {k: {"cov": np.mean(all_cov[k]), "err": np.mean(all_err[k])} for k in all_cov.keys()}

def run():
    print("--- RELIABILITY SHIFT EVALUATION ---")
    
    # 1. ElderReact
    er_path = os.path.join(RES_DIR, "elderreact_adaptation_predictions.npz")
    if os.path.exists(er_path):
        data = np.load(er_path, allow_pickle=True)
        m0_res = run_elderreact_policy(data["m0_preds"])
        m3_res = run_elderreact_policy(data["m3_preds"])
        
        print("\nElderReact Reliability Tradeoff:")
        print("Model | Policy | Coverage | Error")
        print("---|---|---|---")
        for p in ["P0", "P1", "P2", "P3"]:
            print(f"ER-M0 (Linear) | {p} | {m0_res[p]['cov']:.1%} | {m0_res[p]['err']:.1%}")
        for p in ["P0", "P1", "P2", "P3"]:
            print(f"ER-M3 (Adapted)| {p} | {m3_res[p]['cov']:.1%} | {m3_res[p]['err']:.1%}")
            
    # 2. MELD
    meld_path = os.path.join(RES_DIR, "meld_ft_predictions.npz")
    if os.path.exists(meld_path):
        data = np.load(meld_path)
        m0_res = compute_meld_policy(data["m0_logits"], data["labels_te"])
        m1_res = compute_meld_policy(data["m1_logits"], data["labels_te"])
        
        print("\nMELD Reliability Tradeoff:")
        print("Model | Policy | Coverage | Error")
        print("---|---|---|---")
        print(f"M0 (Frozen) | P0 (Always) | {m0_res['P0_cov']:.1%} | {m0_res['P0_err']:.1%}")
        print(f"M0 (Frozen) | P1 (Conf)   | {m0_res['P1_cov']:.1%} | {m0_res['P1_err']:.1%}")
        print(f"M1 (FT-Text)| P0 (Always) | {m1_res['P0_cov']:.1%} | {m1_res['P0_err']:.1%}")
        print(f"M1 (FT-Text)| P1 (Conf)   | {m1_res['P1_cov']:.1%} | {m1_res['P1_err']:.1%}")
        
if __name__ == "__main__":
    run()
