#!/usr/bin/env python3
"""H.E.A.R.T. AI: Deterministic Canonical Prediction Artifact Generator.

Generates:
1. unimodal_text.json
2. unimodal_audio.json
3. unimodal_visual.json
4. model_a_baseline_meld_predictions.json

Using:
- Fixed seed = 42
- MD5-based deterministic modality seeds
- Exact MELD canonical train (9,989), dev (1,109), test (2,610) splits
- Exact model architectures & training protocol
- Locked test evaluation
- Zero test-set leakage
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fusion.baseline import BaselineMultimodalMLP
from models.unimodal_engine import UnimodalEngine
from reliability.calibration import TemperatureScaler, compute_ece_mce
from reliability.entropy import compute_predictive_entropy

CANONICAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
NUM_CLASSES = 7


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultimodalDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], t_feat: torch.Tensor, a_feat: torch.Tensor, v_feat: torch.Tensor):
        self.records = records
        self.t_feat = t_feat
        self.a_feat = a_feat
        self.v_feat = v_feat
        self.labels = torch.tensor([r["Emotion_ID"] for r in records], dtype=torch.long)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        return {
            "t_feat": self.t_feat[idx],
            "a_feat": self.a_feat[idx],
            "v_feat": self.v_feat[idx],
            "label": self.labels[idx],
            "record": self.records[idx],
        }


def generate_deterministic_artifacts(output_dir: Path, seed: int = 42, device: str = "cpu") -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    print(f"=== Generating Deterministic Canonical Artifacts (Seed={seed}) ===")
    print(f"Output Directory: {output_dir}")

    # 1. Unimodal Engine & Embeddings
    u_engine = UnimodalEngine(device=device, seed=seed)
    splits = u_engine.splits

    embeddings = {}
    for s in ["train", "dev", "test"]:
        embeddings[s] = {
            "text": u_engine.extract_or_generate_modality_embeddings("text", s),
            "audio": u_engine.extract_or_generate_modality_embeddings("audio", s),
            "visual": u_engine.extract_or_generate_modality_embeddings("visual", s),
        }

    # 2. Train Unimodal Heads and Save JSONs
    unimodal_models = {}
    unimodal_caches = {}
    unimodal_metrics = {}

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

        # Evaluate on test
        model.eval()
        test_feat = embeddings["test"][mod].to(device)
        test_records = splits["test"]
        targets = np.array([r["Emotion_ID"] for r in test_records])

        with torch.no_grad():
            raw_logits = model(test_feat).cpu()
            probs = torch.softmax(raw_logits, dim=-1).numpy()
            logits_np = raw_logits.numpy()

        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)
        entropies = compute_predictive_entropy(probs, normalized=False)
        norm_entropies = compute_predictive_entropy(probs, normalized=True)

        acc = float(accuracy_score(targets, preds))
        macro_f1 = float(f1_score(targets, preds, average="macro", zero_division=0))

        cache_records = []
        for i, r in enumerate(test_records):
            cache_records.append({
                "sample_id": r["Sample_ID"],
                "dialogue_id": r["Dialogue_ID"],
                "utterance_id": r["Utterance_ID"],
                "speaker_id": r["Speaker"],
                "true_label": r["Emotion_Clean"],
                "true_id": int(r["Emotion_ID"]),
                "predicted_label": CANONICAL_EMOTIONS[preds[i]],
                "predicted_id": int(preds[i]),
                "correct": bool(preds[i] == targets[i]),
                "logits": [round(float(v), 5) for v in logits_np[i]],
                "probabilities": [round(float(v), 5) for v in probs[i]],
                "confidence": round(float(confs[i]), 5),
                "entropy": round(float(entropies[i]), 5),
                "normalized_entropy": round(float(norm_entropies[i]), 5),
                "modality": mod,
                "model_checkpoint": f"{mod}_roberta_w2v_vit_meld_head",
                "model_version": "0.1.0-deterministic",
                "seed": seed,
            })

        out_file = output_dir / f"unimodal_{mod}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cache_records, f, indent=2)

        print(f"[{mod.upper()}] Acc: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f} -> Saved {len(cache_records)} records to {out_file.name}")
        unimodal_caches[mod] = cache_records
        unimodal_metrics[mod] = {"accuracy": acc, "macro_f1": macro_f1}

    # 3. Train Model A Baseline
    set_seed(seed)
    train_ds = MultimodalDataset(splits["train"], embeddings["train"]["text"], embeddings["train"]["audio"], embeddings["train"]["visual"])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    dev_x_t = embeddings["dev"]["text"].to(device)
    dev_x_a = embeddings["dev"]["audio"].to(device)
    dev_x_v = embeddings["dev"]["visual"].to(device)
    dev_y = torch.tensor([r["Emotion_ID"] for r in splits["dev"]], dtype=torch.long, device=device)

    model_a = BaselineMultimodalMLP(text_dim=768, audio_dim=768, visual_dim=768, hidden_dims=[512, 256], num_classes=NUM_CLASSES, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_dev_f1 = -1.0
    best_state = None

    for epoch in range(15):
        model_a.train()
        for batch in train_loader:
            bt = batch["t_feat"].to(device)
            ba = batch["a_feat"].to(device)
            bv = batch["v_feat"].to(device)
            by = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model_a(bt, ba, bv)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

        # Dev checkpoint selection
        model_a.eval()
        with torch.no_grad():
            dev_logits = model_a(dev_x_t, dev_x_a, dev_x_v)
            dev_preds = torch.argmax(dev_logits, dim=-1).cpu().numpy()
            dev_f1 = f1_score(dev_y.cpu().numpy(), dev_preds, average="macro", zero_division=0)
            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_state = {k: v.cpu().clone() for k, v in model_a.state_dict().items()}

    if best_state is not None:
        model_a.load_state_dict(best_state)

    # Dev Temperature Calibration
    model_a.eval()
    with torch.no_grad():
        dev_logits_np = model_a(dev_x_t, dev_x_a, dev_x_v).cpu().numpy()

    scaler = TemperatureScaler()
    optimal_t = float(scaler.fit(dev_logits_np, dev_y.cpu().numpy()))

    # Test Evaluation
    test_records = splits["test"]
    test_x_t = embeddings["test"]["text"].to(device)
    test_x_a = embeddings["test"]["audio"].to(device)
    test_x_v = embeddings["test"]["visual"].to(device)
    test_targets = np.array([r["Emotion_ID"] for r in test_records])

    with torch.no_grad():
        test_logits_t = model_a(test_x_t, test_x_a, test_x_v).cpu()
        test_logits_np = test_logits_t.numpy()
        test_raw_probs = torch.softmax(test_logits_t, dim=-1).numpy()
        test_cal_probs = torch.softmax(test_logits_t / optimal_t, dim=-1).numpy()

    test_preds = np.argmax(test_raw_probs, axis=-1)
    test_confs = np.max(test_cal_probs, axis=-1)
    test_raw_confs = np.max(test_raw_probs, axis=-1)
    test_entropies = compute_predictive_entropy(test_cal_probs, normalized=False)
    test_norm_entropies = compute_predictive_entropy(test_cal_probs, normalized=True)

    m_a_acc = float(accuracy_score(test_targets, test_preds))
    m_a_f1 = float(f1_score(test_targets, test_preds, average="macro", zero_division=0))

    model_a_records = []
    for i, r in enumerate(test_records):
        model_a_records.append({
            "sample_id": r["Sample_ID"],
            "dialogue_id": r["Dialogue_ID"],
            "utterance_id": r["Utterance_ID"],
            "speaker_id": r["Speaker"],
            "true_label": r["Emotion_Clean"],
            "true_id": int(r["Emotion_ID"]),
            "predicted_label": CANONICAL_EMOTIONS[test_preds[i]],
            "predicted_id": int(test_preds[i]),
            "correct": bool(test_preds[i] == test_targets[i]),
            "logits": [round(float(v), 5) for v in test_logits_np[i]],
            "raw_probabilities": [round(float(v), 5) for v in test_raw_probs[i]],
            "calibrated_probabilities": [round(float(v), 5) for v in test_cal_probs[i]],
            "raw_confidence": round(float(test_raw_confs[i]), 5),
            "calibrated_confidence": round(float(test_confs[i]), 5),
            "entropy": round(float(test_entropies[i]), 5),
            "normalized_entropy": round(float(test_norm_entropies[i]), 5),
            "model_name": "Model_A_Baseline_MLP",
            "checkpoint_version": "v1.0-deterministic",
            "temperature": round(optimal_t, 5),
            "seed": seed,
        })

    model_a_file = output_dir / "model_a_baseline_meld_predictions.json"
    with open(model_a_file, "w", encoding="utf-8") as f:
        json.dump(model_a_records, f, indent=2)

    print(f"[MODEL A] Acc: {m_a_acc*100:.2f}% | Macro-F1: {m_a_f1:.4f} | Optimal T: {optimal_t:.4f} -> Saved {len(model_a_records)} records to {model_a_file.name}")

    return {
        "unimodal_metrics": unimodal_metrics,
        "model_a_metrics": {"accuracy": m_a_acc, "macro_f1": m_a_f1, "temperature": optimal_t},
        "num_records": len(test_records),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Deterministic Canonical Prediction Artifacts")
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "results" / "predictions" / "canonical_deterministic"), help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device")
    args = parser.parse_args()

    generate_deterministic_artifacts(output_dir=Path(args.output_dir), seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
