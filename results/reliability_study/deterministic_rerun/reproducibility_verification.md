# Reproducibility Verification Report

## 1. Experimental Setup & Protocol Summary
- **Random Seed**: `42` (global PyTorch & NumPy seed)
- **Deterministic Modality Seed Derivation**: `seed_int = (seed + int(hashlib.md5(f"{modality}_{split_name}".encode("utf-8")).hexdigest(), 16)) % 1000000`
- **MELD Split Sizes**:
  - `train`: 9,989 utterances (1,038 dialogues)
  - `dev`: 1,109 utterances (114 dialogues)
  - `test`: 2,610 utterances (280 dialogues)
- **Evaluation Subset**: Locked MELD test split (2,610 utterances across 280 dialogue clusters; zero exclusions).
- **Leakage Prevention**: No ground-truth labels used as input features; temperature calibration and thresholds fitted strictly on `dev`.

## 2. Model Architecture & Checkpoint Specifications
- **Unimodal Classifiers**:
  - Architecture: `UnimodalMLPHead` (Linear 768 $\to$ 256, ReLU, Dropout 0.2, Linear 256 $\to$ 7)
  - Optimizer: AdamW (lr = 0.005, weight decay = $1\times 10^{-4}$), batch size = 64, epochs = 15
  - Selection Metric: Best Macro-F1 on `dev` split
- **Multimodal Baseline (Model A)**:
  - Architecture: `BaselineMultimodalMLP` (Linear 2304 $\to$ 512, ReLU, Dropout 0.2, Linear 512 $\to$ 256, ReLU, Dropout 0.2, Linear 256 $\to$ 7)
  - Optimizer: AdamW (lr = 0.003, weight decay = $1\times 10^{-4}$), batch size = 64, epochs = 15
  - Calibration: Validation Temperature Scaling fitted via NLL minimization on `dev` logits

## 3. Independent Process Determinism Verification (Two Process Runs)
Two independent Python processes were executed with distinct process environments (`PYTHONHASHSEED=12345` vs `PYTHONHASHSEED=24690`).

| Artifact File | Sample Count | Sample ID Match | True Label Match | Prediction Match | Prob Vector Max $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `unimodal_text.json` | 2,610 | 2,610 / 2,610 (100%) | 2,610 / 2,610 (100%) | 2,610 / 2,610 (**100.0%**) | **0.00000000** |
| `unimodal_audio.json` | 2,610 | 2,610 / 2,610 (100%) | 2,610 / 2,610 (100%) | 2,610 / 2,610 (**100.0%**) | **0.00000000** |
| `unimodal_visual.json` | 2,610 | 2,610 / 2,610 (100%) | 2,610 / 2,610 (100%) | 2,610 / 2,610 (**100.0%**) | **0.00000000** |
| `model_a_baseline_meld_predictions.json` | 2,610 | 2,610 / 2,610 (100%) | 2,610 / 2,610 (100%) | 2,610 / 2,610 (**100.0%**) | **0.00000000** |

### D1 Cross-Modal Disagreement Verification:
- **Mean $D_1$ Across All 2,610 Test Samples**: `0.39454`
- **Max Absolute Difference in $D_1$ Between Process 1 and Process 2**: **`0.00000000e+00`** ($0.0$)

## 4. Conclusion
The canonical prediction generation pipeline is **100% deterministic and bit-level reproducible** across distinct Python processes and environments.
