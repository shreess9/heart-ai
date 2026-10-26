import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import jensenshannon
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)

DATA_DIR = r"d:\Project\data\raw\ElderReact"
RES_DIR = r"d:\Project\results\elderreact"
os.makedirs(RES_DIR, exist_ok=True)

emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

def load_all_data():
    X_a, X_v, y_all, subjects = [], [], [], []
    
    splits = ["train", "dev", "test"]
    for split in splits:
        with open(os.path.join(DATA_DIR, "Annotations", f"{split}_labels.txt")) as f:
            labels = f.readlines()
        with open(os.path.join(DATA_DIR, "Features", "Audio_feat", f"{split}_audio.txt")) as f:
            audio_feats = f.readlines()
        with open(os.path.join(DATA_DIR, "Features", "Video_feat", f"{split}_video.txt")) as f:
            video_feats = f.readlines()
            
        for i in range(len(labels)):
            parts_l = labels[i].strip().split()
            if not parts_l: continue
            
            # audio
            a_parts = audio_feats[i].strip().split("\t")
            if len(a_parts) < 2: continue
            a_v = [float(x) for x in a_parts[1].split()]
            a_feat = a_v[:36] + a_v[74:110]
            
            # video
            v_parts = video_feats[i].strip().split("\t")
            if len(v_parts) < 2: continue
            v_v = [float(x) for x in v_parts[1].split()]
            v_feat = v_v[0:8] + v_v[288:294] + v_v[634:717] + v_v[997:1003] + v_v[1343:]
            
            # labels
            y = [int(x) for x in parts_l[1:7]]
            
            filename = parts_l[0]
            subject_id = filename.split('_')[0] if '_' in filename else filename
            
            X_a.append(a_feat)
            X_v.append(v_feat)
            y_all.append(y)
            subjects.append(subject_id)
    X_a_arr = np.nan_to_num(np.array(X_a, dtype=np.float64))
    X_v_arr = np.nan_to_num(np.array(X_v, dtype=np.float64))
    X_a_arr = np.clip(X_a_arr, -1e6, 1e6)
    X_v_arr = np.clip(X_v_arr, -1e6, 1e6)
    return X_a_arr, X_v_arr, np.array(y_all), np.array(subjects)

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

def multi_label_entropy(p):
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return np.sum(-p * np.log(p) - (1-p) * np.log(1-p))

def run_experiment():
    print("Loading data...")
    X_a, X_v, y, subjects = load_all_data()
    print(f"Total samples: {len(y)}")
    
    metrics = {
        "audio_macro_f1": [], "visual_macro_f1": [], "fusion_macro_f1": [],
        "audio_subset_acc": [], "visual_subset_acc": [], "fusion_subset_acc": []
    }
    
    policy_results = {
        "P0_cov": [], "P0_err": [],
        "P1_cov": [], "P1_err": [],
        "P2_cov": [], "P2_err": [],
        "P3_cov": [], "P3_err": []
    }
    
    all_test_conf = []
    all_test_corr = []
    all_test_jsd = []
    
    gss1 = GroupShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
    
    for fold, (train_val_idx, test_idx) in enumerate(gss1.split(X_a, y, subjects)):
        print(f"Fold {fold+1}...")
        
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
        train_idx_sub, val_idx_sub = next(gss2.split(X_a[train_val_idx], y[train_val_idx], subjects[train_val_idx]))
        
        train_idx = train_val_idx[train_idx_sub]
        val_idx = train_val_idx[val_idx_sub]
        
        scaler_a = MinMaxScaler().fit(X_a[train_idx])
        scaler_v = MinMaxScaler().fit(X_v[train_idx])
        
        Xa_tr, Xa_va, Xa_te = scaler_a.transform(X_a[train_idx]), scaler_a.transform(X_a[val_idx]), scaler_a.transform(X_a[test_idx])
        Xv_tr, Xv_va, Xv_te = scaler_v.transform(X_v[train_idx]), scaler_v.transform(X_v[val_idx]), scaler_v.transform(X_v[test_idx])
        
        y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
        
        Xf_tr = np.concatenate([Xa_tr, Xv_tr], axis=1)
        Xf_va = np.concatenate([Xa_va, Xv_va], axis=1)
        Xf_te = np.concatenate([Xa_te, Xv_te], axis=1)
        
        model_a = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=200, random_state=42)
        model_v = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=200, random_state=42)
        model_f = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42)
        
        model_a.fit(Xa_tr, y_tr)
        model_v.fit(Xv_tr, y_tr)
        model_f.fit(Xf_tr, y_tr)
        
        pa_va = model_a.predict_proba(Xa_va)
        pv_va = model_v.predict_proba(Xv_va)
        pf_va = model_f.predict_proba(Xf_va)
        
        conf_va = np.max(pf_va, axis=1)
        ent_va = np.array([multi_label_entropy(p) for p in pf_va])
        jsd_va = np.array([mean_bernoulli_jsd(pa_va[i], pv_va[i]) for i in range(len(pa_va))])
        
        thresh_conf = np.percentile(conf_va, 50)
        thresh_jsd = np.percentile(jsd_va, 50)
        thresh_ent = np.percentile(ent_va, 50)
        
        pa_te = model_a.predict_proba(Xa_te)
        pv_te = model_v.predict_proba(Xv_te)
        pf_te = model_f.predict_proba(Xf_te)
        
        metrics["audio_macro_f1"].append(f1_score(y_te, pa_te > 0.5, average='macro'))
        metrics["visual_macro_f1"].append(f1_score(y_te, pv_te > 0.5, average='macro'))
        metrics["fusion_macro_f1"].append(f1_score(y_te, pf_te > 0.5, average='macro'))
        
        metrics["audio_subset_acc"].append(accuracy_score(y_te, pa_te > 0.5))
        metrics["visual_subset_acc"].append(accuracy_score(y_te, pv_te > 0.5))
        metrics["fusion_subset_acc"].append(accuracy_score(y_te, pf_te > 0.5))
        
        conf_te = np.max(pf_te, axis=1)
        ent_te = np.array([multi_label_entropy(p) for p in pf_te])
        jsd_te = np.array([mean_bernoulli_jsd(pa_te[i], pv_te[i]) for i in range(len(pa_te))])
        
        dom_te = np.argmax(pf_te, axis=1)
        corr_te = np.array([1 if y_te[i][dom_te[i]] == 1 else 0 for i in range(len(y_te))])
        
        all_test_conf.extend(conf_te)
        all_test_corr.extend(corr_te)
        all_test_jsd.extend(jsd_te)
        
        for i in range(len(y_te)):
            policy0 = True
            policy1 = conf_te[i] >= thresh_conf
            policy2 = (conf_te[i] >= thresh_conf) and (jsd_te[i] <= thresh_jsd)
            policy3 = (conf_te[i] >= thresh_conf) and (jsd_te[i] <= thresh_jsd) and (ent_te[i] <= thresh_ent)
            
            if policy0:
                policy_results["P0_cov"].append(1)
                policy_results["P0_err"].append(1 - corr_te[i])
            else:
                policy_results["P0_cov"].append(0)
                
            if policy1:
                policy_results["P1_cov"].append(1)
                policy_results["P1_err"].append(1 - corr_te[i])
            else:
                policy_results["P1_cov"].append(0)
                
            if policy2:
                policy_results["P2_cov"].append(1)
                policy_results["P2_err"].append(1 - corr_te[i])
            else:
                policy_results["P2_cov"].append(0)
                
            if policy3:
                policy_results["P3_cov"].append(1)
                policy_results["P3_err"].append(1 - corr_te[i])
            else:
                policy_results["P3_cov"].append(0)
                
    p0_cov = np.mean(policy_results["P0_cov"])
    p0_err = np.mean([e for i, e in enumerate(policy_results["P0_err"]) if policy_results["P0_cov"][i] == 1]) if sum(policy_results["P0_cov"]) > 0 else 0
    p1_cov = np.mean(policy_results["P1_cov"])
    p1_err = np.mean([e for i, e in enumerate(policy_results["P1_err"]) if policy_results["P1_cov"][i] == 1]) if sum(policy_results["P1_cov"]) > 0 else 0
    p2_cov = np.mean(policy_results["P2_cov"])
    p2_err = np.mean([e for i, e in enumerate(policy_results["P2_err"]) if policy_results["P2_cov"][i] == 1]) if sum(policy_results["P2_cov"]) > 0 else 0
    p3_cov = np.mean(policy_results["P3_cov"])
    p3_err = np.mean([e for i, e in enumerate(policy_results["P3_err"]) if policy_results["P3_cov"][i] == 1]) if sum(policy_results["P3_cov"]) > 0 else 0

    report = "# ElderReact A+V Validation Results\n\n"
    report += "## 1. Classification Performance (5-Fold CV)\n"
    report += "| Model | Macro-F1 | Subset Acc |\n|---|---|---|\n"
    report += f"| Audio Only | {np.mean(metrics['audio_macro_f1']):.3f} | {np.mean(metrics['audio_subset_acc']):.3f} |\n"
    report += f"| Visual Only | {np.mean(metrics['visual_macro_f1']):.3f} | {np.mean(metrics['visual_subset_acc']):.3f} |\n"
    report += f"| A+V Fusion | {np.mean(metrics['fusion_macro_f1']):.3f} | {np.mean(metrics['fusion_subset_acc']):.3f} |\n\n"
    
    report += "## 2. Response Policy Evaluation\n"
    report += "| Policy | Strong-Adaptation Coverage | Dominant-Emotion Error Rate |\n|---|---|---|\n"
    report += f"| P0: Always Adapt | {p0_cov:.1%} | {p0_err:.1%} |\n"
    report += f"| P1: Confidence Only | {p1_cov:.1%} | {p1_err:.1%} |\n"
    report += f"| P2: Conf + Disagreement | {p2_cov:.1%} | {p2_err:.1%} |\n"
    report += f"| P3: Full Reliability | {p3_cov:.1%} | {p3_err:.1%} |\n\n"
    
    with open(os.path.join(RES_DIR, "elderreact_results.md"), "w") as f:
        f.write(report)
        
    plt.figure(figsize=(8,6))
    sns.boxplot(x=all_test_corr, y=all_test_conf)
    plt.title("Confidence vs Dominant-Emotion Correctness")
    plt.xlabel("Correctness (1=Correct, 0=Incorrect)")
    plt.ylabel("Confidence (Max Probability)")
    plt.savefig(os.path.join(RES_DIR, "confidence_vs_correctness.png"))
    plt.close()
    
    plt.figure(figsize=(8,6))
    sns.boxplot(x=all_test_corr, y=all_test_jsd)
    plt.title("Mean Bernoulli JSD vs Dominant-Emotion Correctness")
    plt.xlabel("Correctness (1=Correct, 0=Incorrect)")
    plt.ylabel("Mean Bernoulli JSD")
    plt.savefig(os.path.join(RES_DIR, "jsd_vs_correctness.png"))
    plt.close()
    
    plt.figure(figsize=(8,6))
    plt.hist(all_test_jsd, bins=30, alpha=0.7)
    plt.title("Distribution of Mean Bernoulli JSD")
    plt.xlabel("JSD")
    plt.ylabel("Count")
    plt.savefig(os.path.join(RES_DIR, "jsd_distribution.png"))
    plt.close()
    
    print("Experiment completed. Results saved to", RES_DIR)

if __name__ == "__main__":
    run_experiment()
