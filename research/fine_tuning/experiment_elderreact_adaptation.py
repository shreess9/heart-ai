import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

np.random.seed(42)

ROOT_DIR = "d:/Project"
sys.path.append(ROOT_DIR)

DATA_DIR = r"d:\Project\data\raw\ElderReact"
RES_DIR = r"d:\Project\results\fine_tuning"
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

def get_model(type_, name):
    if type_ == "baseline":
        return MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=200, random_state=42)
    elif type_ == "adapted":
        if name in ["audio", "visual"]:
            return MLPClassifier(hidden_layer_sizes=(64,), activation='relu', max_iter=200, random_state=42)
        else: # fusion
            return MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=200, random_state=42)
    raise ValueError()

def run_experiment():
    print("Loading ElderReact data...")
    X_a, X_v, y, subjects = load_all_data()
    print(f"Total samples: {len(y)}")
    
    # metrics store
    configs = ["ER-M0", "ER-M1", "ER-M2", "ER-M3"]
    metrics = {c: {"macro_f1": [], "micro_f1": [], "hamming": []} for c in configs}
    
    # Reliability metrics to dump for comparison
    all_preds_m0 = []
    all_preds_m3 = []
    
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
        
        for config in configs:
            if config == "ER-M0":
                m_a = get_model("baseline", "audio")
                m_v = get_model("baseline", "visual")
                m_f = get_model("baseline", "fusion")
            elif config == "ER-M1":
                m_a = get_model("adapted", "audio")
                m_v = get_model("baseline", "visual")
                m_f = get_model("baseline", "fusion")
            elif config == "ER-M2":
                m_a = get_model("baseline", "audio")
                m_v = get_model("adapted", "visual")
                m_f = get_model("baseline", "fusion")
            elif config == "ER-M3":
                m_a = get_model("adapted", "audio")
                m_v = get_model("adapted", "visual")
                m_f = get_model("adapted", "fusion")
                
            m_a.fit(Xa_tr, y_tr)
            m_v.fit(Xv_tr, y_tr)
            m_f.fit(Xf_tr, y_tr)
            
            pa_te = m_a.predict_proba(Xa_te)
            pv_te = m_v.predict_proba(Xv_te)
            pf_te = m_f.predict_proba(Xf_te)
            
            preds_te = (pf_te > 0.5).astype(int)
            
            metrics[config]["macro_f1"].append(f1_score(y_te, preds_te, average='macro'))
            metrics[config]["micro_f1"].append(f1_score(y_te, preds_te, average='micro'))
            metrics[config]["hamming"].append(hamming_loss(y_te, preds_te))
            
            if config == "ER-M0":
                pa_va = m_a.predict_proba(Xa_va)
                pv_va = m_v.predict_proba(Xv_va)
                pf_va = m_f.predict_proba(Xf_va)
                all_preds_m0.append({
                    "pf_te": pf_te, "pa_te": pa_te, "pv_te": pv_te, "y_te": y_te,
                    "pf_va": pf_va, "pa_va": pa_va, "pv_va": pv_va
                })
            elif config == "ER-M3":
                pa_va = m_a.predict_proba(Xa_va)
                pv_va = m_v.predict_proba(Xv_va)
                pf_va = m_f.predict_proba(Xf_va)
                all_preds_m3.append({
                    "pf_te": pf_te, "pa_te": pa_te, "pv_te": pv_te, "y_te": y_te,
                    "pf_va": pf_va, "pa_va": pa_va, "pv_va": pv_va
                })

    print(f"\n--- ElderReact FINAL RESULTS ---")
    for config in configs:
        mac = np.mean(metrics[config]["macro_f1"])
        mic = np.mean(metrics[config]["micro_f1"])
        ham = np.mean(metrics[config]["hamming"])
        print(f"{config} | Macro-F1: {mac:.4f} | Micro-F1: {mic:.4f} | Hamming: {ham:.4f}")
        
    np.savez(os.path.join(RES_DIR, "elderreact_adaptation_predictions.npz"), 
             m0_preds=all_preds_m0, m3_preds=all_preds_m3)
    
if __name__ == "__main__":
    run_experiment()
