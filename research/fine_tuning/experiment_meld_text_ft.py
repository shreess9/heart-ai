import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.metrics import f1_score, accuracy_score

ROOT_DIR = Path("d:/Project")
sys.path.append(str(ROOT_DIR))

from models.unimodal_engine import UnimodalEngine
from fusion.baseline import BaselineMultimodalMLP
from reliability.calibration import TemperatureScaler

# Setup HF Cache to D drive
os.environ["HF_HOME"] = "d:/Project/hf_cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_meld_splits():
    splits = {}
    for s in ["train", "dev", "test"]:
        with open(ROOT_DIR / f"data/splits/{s}.json", "r") as f:
            splits[s] = json.load(f)
    return splits

class MELDTextDataset(Dataset):
    def __init__(self, records, tokenizer):
        self.records = records
        self.texts = [r["Utterance_Clean"] for r in records]
        self.labels = [r["Emotion_ID"] for r in records]
        self.encodings = tokenizer(self.texts, truncation=True, padding="max_length", max_length=128)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['record'] = self.records[idx]
        return item

def get_ft_roberta_embeddings(model, tokenizer, records, batch_size=16):
    dataset = MELDTextDataset(records, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    embeddings = []
    probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model.roberta(input_ids, attention_mask=attention_mask)
            # Use CLS token representation from last hidden state
            cls_repr = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            logits = model.classifier(outputs[0]).cpu().numpy()
            p = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            
            embeddings.extend(cls_repr)
            probs.extend(p)
            
    return np.array(embeddings), np.array(probs)

def train_fusion(t_emb, a_emb, v_emb, labels, dev_t_emb, dev_a_emb, dev_v_emb, dev_labels):
    """Train fusion MLP just like the baseline"""
    from torch.utils.data import TensorDataset
    
    tr_ds = TensorDataset(torch.tensor(t_emb), torch.tensor(a_emb), torch.tensor(v_emb), torch.tensor(labels, dtype=torch.long))
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    
    m_f = BaselineMultimodalMLP(text_dim=768, audio_dim=768, visual_dim=768, hidden_dims=[512, 256], num_classes=7, dropout=0.2).to(device)
    opt = torch.optim.AdamW(m_f.parameters(), lr=0.003, weight_decay=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    
    best_dev_f1 = -1.0
    best_st = None
    
    dev_t = torch.tensor(dev_t_emb).to(device)
    dev_a = torch.tensor(dev_a_emb).to(device)
    dev_v = torch.tensor(dev_v_emb).to(device)
    dev_y = torch.tensor(dev_labels, dtype=torch.long).to(device)
    
    for ep in range(15):
        m_f.train()
        for bt, ba, bv, by in tr_loader:
            opt.zero_grad()
            l = m_f(bt.to(device), ba.to(device), bv.to(device))
            crit(l, by.to(device)).backward()
            opt.step()
            
        m_f.eval()
        with torch.no_grad():
            dl = m_f(dev_t, dev_a, dev_v)
            dp = torch.argmax(dl, dim=-1).cpu().numpy()
            f1 = f1_score(dev_labels, dp, average="macro", zero_division=0)
            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_st = {k: v.cpu().clone() for k, v in m_f.state_dict().items()}
                
    m_f.load_state_dict(best_st)
    return m_f

def run():
    print("Loading datasets...")
    splits = load_meld_splits()
    
    print("Initializing RoBERTa...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=7).to(device)
    model.eval() # Completely frozen
            
    print("Extracting RoBERTa representations (forward-pass only on CPU)...")
    ft_t_emb_tr, _ = get_ft_roberta_embeddings(model, tokenizer, splits["train"], batch_size=32)
    ft_t_emb_va, _ = get_ft_roberta_embeddings(model, tokenizer, splits["dev"], batch_size=32)
    ft_t_emb_te, _ = get_ft_roberta_embeddings(model, tokenizer, splits["test"], batch_size=32)
    
    labels_tr = np.array([r["Emotion_ID"] for r in splits["train"]])
    labels_va = np.array([r["Emotion_ID"] for r in splits["dev"]])
    labels_te = np.array([r["Emotion_ID"] for r in splits["test"]])
    
    print("Training Text-Only classification head (Linear Probe)...")
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=200, random_state=42)
    clf.fit(ft_t_emb_tr, labels_tr)
    
    ft_t_probs_va = clf.predict_proba(ft_t_emb_va)
    ft_t_probs_te = clf.predict_proba(ft_t_emb_te)
    
    print(f"Dev Text-Only Macro-F1: {f1_score(labels_va, clf.predict(ft_t_emb_va), average='macro'):.4f}")
    
    print("Generating/Loading canonical A+V embeddings...")
    u_engine = UnimodalEngine(seed=42)
    can_a_emb_tr = u_engine.extract_or_generate_modality_embeddings("audio", "train").numpy()
    can_v_emb_tr = u_engine.extract_or_generate_modality_embeddings("visual", "train").numpy()
    
    can_a_emb_va = u_engine.extract_or_generate_modality_embeddings("audio", "dev").numpy()
    can_v_emb_va = u_engine.extract_or_generate_modality_embeddings("visual", "dev").numpy()
    
    can_a_emb_te = u_engine.extract_or_generate_modality_embeddings("audio", "test").numpy()
    can_v_emb_te = u_engine.extract_or_generate_modality_embeddings("visual", "test").numpy()
    
    print("Verifying strict alignment...")
    assert len(ft_t_emb_tr) == len(can_a_emb_tr) == len(can_v_emb_tr) == len(labels_tr)
    assert len(ft_t_emb_va) == len(can_a_emb_va) == len(can_v_emb_va) == len(labels_va)
    assert len(ft_t_emb_te) == len(can_a_emb_te) == len(can_v_emb_te) == len(labels_te)
    print("Alignment verified successfully.")
    
    print("Training M1 Fusion (Linear Probed T + Canonical A + Canonical V)...")
    m1_fuser = train_fusion(
        ft_t_emb_tr, can_a_emb_tr, can_v_emb_tr, labels_tr,
        ft_t_emb_va, can_a_emb_va, can_v_emb_va, labels_va
    )
    
    print("Training M0 Fusion (Canonical T + Canonical A + Canonical V)...")
    can_t_emb_tr = u_engine.extract_or_generate_modality_embeddings("text", "train").numpy()
    can_t_emb_va = u_engine.extract_or_generate_modality_embeddings("text", "dev").numpy()
    can_t_emb_te = u_engine.extract_or_generate_modality_embeddings("text", "test").numpy()
    
    m0_fuser = train_fusion(
        can_t_emb_tr, can_a_emb_tr, can_v_emb_tr, labels_tr,
        can_t_emb_va, can_a_emb_va, can_v_emb_va, labels_va
    )
    
    print("Evaluating on Test set...")
    m1_fuser.eval()
    m0_fuser.eval()
    
    with torch.no_grad():
        m1_logits = m1_fuser(torch.tensor(ft_t_emb_te).to(device), torch.tensor(can_a_emb_te).to(device), torch.tensor(can_v_emb_te).to(device)).cpu().numpy()
        m0_logits = m0_fuser(torch.tensor(can_t_emb_te).to(device), torch.tensor(can_a_emb_te).to(device), torch.tensor(can_v_emb_te).to(device)).cpu().numpy()
        
    m1_preds = np.argmax(m1_logits, axis=-1)
    m0_preds = np.argmax(m0_logits, axis=-1)
    
    m1_f1 = f1_score(labels_te, m1_preds, average="macro")
    m0_f1 = f1_score(labels_te, m0_preds, average="macro")
    
    m1_acc = accuracy_score(labels_te, m1_preds)
    m0_acc = accuracy_score(labels_te, m0_preds)
    
    print(f"\n--- MELD FINAL RESULTS ---")
    print(f"M0 (Frozen Baseline)    | Acc: {m0_acc:.4f} | Macro-F1: {m0_f1:.4f}")
    print(f"M1 (FT-Text + Frozen AV)| Acc: {m1_acc:.4f} | Macro-F1: {m1_f1:.4f}")
    
    out_dir = ROOT_DIR / "results" / "fine_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    np.savez(out_dir / "meld_ft_predictions.npz", 
             m0_logits=m0_logits, 
             m1_logits=m1_logits, 
             labels_te=labels_te,
             ft_t_probs_te=ft_t_probs_te)
    
if __name__ == "__main__":
    run()
