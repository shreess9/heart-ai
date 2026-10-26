# H.E.A.R.T. AI
**Human-centered Emotion-Aware Reliability-aware Technology for AI Companionship**

---

## 1. Project Overview

H.E.A.R.T. AI is a research-driven, reliability-aware multimodal conversational companion system designed primarily for older adults. The system integrates:
- **Multimodal Perception**: Text (RoBERTa), Speech/Audio (Wav2Vec 2.0), and Facial/Visual (Vision Transformer - ViT) emotion estimation over the 7-class MELD taxonomy (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
- **Reliability & Uncertainty Engine**: Generalized Jensen-Shannon Divergence ($D_1$), Pairwise JSD ($D_2$), Categorical Disagreement ($D_3$), Predictive Shannon Entropy, Temperature-Scaled Probability Calibration, and Confidence-Correctness Analysis.
- **Controlled Fusion Architectures**:
  - **Model A (Baseline)**: Feature Concatenation + MLP
  - **Model B (Disagreement-Aware)**: Features + JSD ($D_1, D_2$) + Categorical Disagreement ($D_3$) + MLP
  - **Model C (Reliability-Aware)**: Features + Disagreement + Predictive Entropy + Calibrated Confidence + Attention/MLP
- **Affective Dialogue Response Policy**: Calibrated, safety-first conversational adaptation ensuring the assistant acts with epistemic humility under conflicting or uncertain evidence.
- **Elder & Caregiver Applications**: Large-format, voice/text conversational interface and privacy-conscious aggregate caregiver dashboard.

---

## 2. Core Research Hypothesis

> **Central Research Question**: *Does explicitly incorporating cross-modal disagreement, predictive uncertainty, and calibrated reliability into multimodal fusion improve emotion recognition quality, calibration (ECE/MCE), high-confidence error reduction, and safety of conversational adaptation?*

We do **not** assume proposed fusion methods outperform controlled baselines. All variants are empirically compared under identical evaluation splits with cluster-level bootstrap statistical testing.

---

## 3. Repository Architecture

```
HEART-AI/
├── configs/          # YAML configurations (emotions, models, fusion, experiment, dialogue)
├── data/             # Benchmark data management & splits
├── models/           # Modality encoders (RoBERTa, Wav2Vec 2.0, ViT)
├── fusion/           # Fusion models (Baseline, Disagreement-aware, Reliability-aware, Attention)
├── reliability/      # Mathematical reliability core (JSD, D3, Entropy, Calibration, Bootstrap)
├── dialogue/         # Structured affective evidence schemas & response policy engine
├── backend/          # FastAPI REST API services
├── experiments/      # Evaluation suites & benchmark comparisons
├── scripts/          # Runnable scripts (run_pipeline.py, run_benchmark.py)
├── tests/            # Comprehensive unit and integration test suites
├── docs/             # Research findings, methodology, and Responsible AI principles
└── results/          # Machine-readable evaluation metrics, figures, and tables
```

---

## 4. Quickstart Guide

### Installation
```bash
# Clone and enter directory
cd HEART-AI

# Install dependencies
pip install -r requirements.txt
```

### Running Unit & Integration Tests
```bash
# Run all unit tests
python -m unittest discover -s tests/unit -p "test_*.py"

# Run integration tests
python -m unittest discover -s tests/integration -p "test_*.py"
```

### Executing the Vertical Proof-of-Concept Pipeline
```bash
# Run multimodal inference & reliability analysis on a sample utterance
python scripts/run_pipeline.py --mock
```

### Running Benchmark Comparison (Model A vs Model B vs Model C)
```bash
python scripts/run_benchmark.py --mock
```

---

## 5. Development & Research Status Taxonomy
All components in this repository are explicitly tagged with their lifecycle status:
- `[PROPOSED]`: Conceptual or under review.
- `[IMPLEMENTED]`: Code complete in repository.
- `[TESTED]`: Passes unit and/or integration test suite.
- `[EXPERIMENTAL]`: Under active empirical evaluation.
- `[VALIDATED]`: Verified by statistical experimental evidence.
- `[BLOCKED]`: Waiting on dependency, dataset access, or upstream prerequisite.
