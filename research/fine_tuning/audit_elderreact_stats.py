import os
import sys
import numpy as np
import scipy.stats as stats
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

DATA_DIR = r"d:\Project\data\raw\ElderReact"
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
            
            a_parts = audio_feats[i].strip().split("\t")
            if len(a_parts) < 2: continue
            a_v = [float(x) for x in a_parts[1].split()]
            a_feat = a_v[:36] + a_v[74:110]
            
            v_parts = video_feats[i].strip().split("\t")
            if len(v_parts) < 2: continue
            v_v = [float(x) for x in v_parts[1].split()]
            v_feat = v_v[0:8] + v_v[288:294] + v_v[634:717] + v_v[997:1003] + v_v[1343:]
            
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

def run_audit():
    X_a, X_v, y, subjects = load_all_data()
    
    gss1 = GroupShuffleSplit(n_splits=5, train_size=0.8, random_state=42)
    
    m0_macro, m3_macro = [], []
    m0_micro, m3_micro = [], []
    
    m0_pclass = {e: [] for e in emotions}
    m3_pclass = {e: [] for e in emotions}
    
    p0_m0_err, p1_m0_err, p2_m0_err, p3_m0_err = [], [], [], []
    p0_m3_err, p1_m3_err, p2_m3_err, p3_m3_err = [], [], [], []
    
    p0_m0_cov, p1_m0_cov, p2_m0_cov, p3_m0_cov = [], [], [], []
    p0_m3_cov, p1_m3_cov, p2_m3_cov, p3_m3_cov = [], [], [], []
    
    # Store absolute counts
    m0_counts = {0: {"total": 0, "adapt": 0, "correct": 0}, 1: {"total": 0, "adapt": 0, "correct": 0}, 2: {"total": 0, "adapt": 0, "correct": 0}, 3: {"total": 0, "adapt": 0, "correct": 0}}
    m3_counts = {0: {"total": 0, "adapt": 0, "correct": 0}, 1: {"total": 0, "adapt": 0, "correct": 0}, 2: {"total": 0, "adapt": 0, "correct": 0}, 3: {"total": 0, "adapt": 0, "correct": 0}}
    
    print("--- SPLIT AUDIT ---")
    
    for fold, (train_val_idx, test_idx) in enumerate(gss1.split(X_a, y, subjects)):
        gss2 = GroupShuffleSplit(n_splits=1, train_size=0.75, random_state=42)
        train_idx_sub, val_idx_sub = next(gss2.split(X_a[train_val_idx], y[train_val_idx], subjects[train_val_idx]))
        
        train_idx = train_val_idx[train_idx_sub]
        val_idx = train_val_idx[val_idx_sub]
        
        tr_sub = set(subjects[train_idx])
        va_sub = set(subjects[val_idx])
        te_sub = set(subjects[test_idx])
        
        overlap = tr_sub.intersection(va_sub) | tr_sub.intersection(te_sub) | va_sub.intersection(te_sub)
        print(f"Fold {fold+1} Overlap: {len(overlap)} participants")
        
        scaler_a = MinMaxScaler().fit(X_a[train_idx])
        scaler_v = MinMaxScaler().fit(X_v[train_idx])
        
        Xa_tr, Xa_va, Xa_te = scaler_a.transform(X_a[train_idx]), scaler_a.transform(X_a[val_idx]), scaler_a.transform(X_a[test_idx])
        Xv_tr, Xv_va, Xv_te = scaler_v.transform(X_v[train_idx]), scaler_v.transform(X_v[val_idx]), scaler_v.transform(X_v[test_idx])
        y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]
        
        Xf_tr = np.concatenate([Xa_tr, Xv_tr], axis=1)
        Xf_va = np.concatenate([Xa_va, Xv_va], axis=1)
        Xf_te = np.concatenate([Xa_te, Xv_te], axis=1)
        
        # Train M0 (Linear)
        m0_a = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=200, random_state=42).fit(Xa_tr, y_tr)
        m0_v = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=200, random_state=42).fit(Xv_tr, y_tr)
        m0_f = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=200, random_state=42).fit(Xf_tr, y_tr)
        
        # Train M3 (MLP)
        m3_a = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=200, random_state=42).fit(Xa_tr, y_tr)
        m3_v = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=200, random_state=42).fit(Xv_tr, y_tr)
        m3_f = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42).fit(Xf_tr, y_tr)
        
        pf_te_m0 = m0_f.predict_proba(Xf_te)
        preds_te_m0 = (pf_te_m0 > 0.5).astype(int)
        
        pf_te_m3 = m3_f.predict_proba(Xf_te)
        preds_te_m3 = (pf_te_m3 > 0.5).astype(int)
        
        m0_macro.append(f1_score(y_te, preds_te_m0, average='macro'))
        m3_macro.append(f1_score(y_te, preds_te_m3, average='macro'))
        
        m0_micro.append(f1_score(y_te, preds_te_m0, average='micro'))
        m3_micro.append(f1_score(y_te, preds_te_m3, average='micro'))
        
        pc_m0 = f1_score(y_te, preds_te_m0, average=None)
        pc_m3 = f1_score(y_te, preds_te_m3, average=None)
        
        for i, e in enumerate(emotions):
            m0_pclass[e].append(pc_m0[i])
            m3_pclass[e].append(pc_m3[i])
            
        # Policies
        for m_name, ma, mv, mf, perr, pcov, m_counts in [
            ("M0", m0_a, m0_v, m0_f, 
             [p0_m0_err, p1_m0_err, p2_m0_err, p3_m0_err], 
             [p0_m0_cov, p1_m0_cov, p2_m0_cov, p3_m0_cov], m0_counts),
            ("M3", m3_a, m3_v, m3_f,
             [p0_m3_err, p1_m3_err, p2_m3_err, p3_m3_err],
             [p0_m3_cov, p1_m3_cov, p2_m3_cov, p3_m3_cov], m3_counts)
        ]:
            pa_va = ma.predict_proba(Xa_va)
            pv_va = mv.predict_proba(Xv_va)
            pf_va = mf.predict_proba(Xf_va)
            
            pa_te = ma.predict_proba(Xa_te)
            pv_te = mv.predict_proba(Xv_te)
            pf_te = mf.predict_proba(Xf_te)
            
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
            
            p1_mask = conf_te >= thresh_conf
            p2_mask = (conf_te >= thresh_conf) & (jsd_te <= thresh_jsd)
            p3_mask = (conf_te >= thresh_conf) & (jsd_te <= thresh_jsd) & (ent_te <= thresh_ent)
            
            perr[0].append(1 - np.mean(corr_te))
            pcov[0].append(1.0)
            m_counts[0]["total"] += len(corr_te)
            m_counts[0]["adapt"] += len(corr_te)
            m_counts[0]["correct"] += np.sum(corr_te)
            
            perr[1].append(1 - np.mean(corr_te[p1_mask]) if np.sum(p1_mask) > 0 else 0)
            pcov[1].append(np.mean(p1_mask))
            m_counts[1]["total"] += len(corr_te)
            m_counts[1]["adapt"] += np.sum(p1_mask)
            m_counts[1]["correct"] += np.sum(corr_te[p1_mask])
            
            perr[2].append(1 - np.mean(corr_te[p2_mask]) if np.sum(p2_mask) > 0 else 0)
            pcov[2].append(np.mean(p2_mask))
            m_counts[2]["total"] += len(corr_te)
            m_counts[2]["adapt"] += np.sum(p2_mask)
            m_counts[2]["correct"] += np.sum(corr_te[p2_mask])
            
            perr[3].append(1 - np.mean(corr_te[p3_mask]) if np.sum(p3_mask) > 0 else 0)
            pcov[3].append(np.mean(p3_mask))
            m_counts[3]["total"] += len(corr_te)
            m_counts[3]["adapt"] += np.sum(p3_mask)
            m_counts[3]["correct"] += np.sum(corr_te[p3_mask])
            
    print("\n--- STATISTICAL RESULTS ---")
    
    m0_m_mac, m0_s_mac = np.mean(m0_macro), np.std(m0_macro)
    m3_m_mac, m3_s_mac = np.mean(m3_macro), np.std(m3_macro)
    
    mac_diff = np.array(m3_macro) - np.array(m0_macro)
    t_mac, p_mac = stats.ttest_rel(m3_macro, m0_macro)
    ci_mac = stats.t.interval(0.95, len(mac_diff)-1, loc=np.mean(mac_diff), scale=stats.sem(mac_diff))
    
    print(f"Macro-F1: M0={m0_m_mac:.4f}±{m0_s_mac:.4f} | M3={m3_m_mac:.4f}±{m3_s_mac:.4f}")
    print(f"Diff: {np.mean(mac_diff):.4f} (95% CI: [{ci_mac[0]:.4f}, {ci_mac[1]:.4f}])")
    print(f"Paired t-test: t={t_mac:.4f}, p={p_mac:.4f}")
    
    # Same for P1 Error
    p1_diff = np.array(p1_m3_err) - np.array(p1_m0_err)
    t_p1, p_p1 = stats.ttest_rel(p1_m3_err, p1_m0_err)
    ci_p1 = stats.t.interval(0.95, len(p1_diff)-1, loc=np.mean(p1_diff), scale=stats.sem(p1_diff))
    
    print("\nPolicy 1 Error Comparison:")
    print(f"M0 P1 Error: {np.mean(p1_m0_err):.4f}±{np.std(p1_m0_err):.4f}")
    print(f"M3 P1 Error: {np.mean(p1_m3_err):.4f}±{np.std(p1_m3_err):.4f}")
    print(f"Diff: {np.mean(p1_diff):.4f} (95% CI: [{ci_p1[0]:.4f}, {ci_p1[1]:.4f}])")
    print(f"Paired t-test: t={t_p1:.4f}, p={p_p1:.4f}")
    
    print("\nPer-Emotion F1 Comparison (M0 vs M3):")
    for e in emotions:
        m_m0 = np.mean(m0_pclass[e])
        m_m3 = np.mean(m3_pclass[e])
        t_e, p_e = stats.ttest_rel(m3_pclass[e], m0_pclass[e])
        print(f"{e:10s} | M0: {m_m0:.4f} | M3: {m_m3:.4f} | p={p_e:.4f}")
        
    print("\n--- DETAILED POLICY TABLES ---")
    
    def print_absolute_policy_table(model_name, m_counts):
        print(f"\nTABLE {model_name} Reliability Policy:")
        for i, pname in enumerate(["P0 (Always Adapt)", "P1 (Confidence)", "P2 (Conf + JSD)", "P3 (Conf + JSD + Ent)"]):
            total = m_counts[i]["total"]
            adapt = m_counts[i]["adapt"]
            correct = m_counts[i]["correct"]
            incorrect = adapt - correct
            
            cov = (adapt / total) * 100 if total > 0 else 0
            acc = (correct / adapt) * 100 if adapt > 0 else 0
            err = 100 - acc if adapt > 0 else 0
            
            print(f"{pname} | Cov: {cov:.2f}% | Adapt: {adapt}/{total} | Corr: {correct} | Incorr: {incorrect} | Acc: {acc:.2f}% | Err: {err:.2f}%")
            
    print_absolute_policy_table("ER-M0", m0_counts)
    print_absolute_policy_table("ER-M3", m3_counts)
    
if __name__ == "__main__":
    run_audit()
