# ElderReact Fusion Comparison — Experimental Report
> **Hypothesis (H1)**: Disagreement-aware (B) and reliability-aware (C) fusion improve emotion recognition and reliability over standard fusion (A).
> **Clips used per split**: 150  |  **Total clips**: 450

## 1. Classification Performance (5-Fold CV, pseudo-labels)
| Model | Macro-F1 [95% CI] | Accuracy [95% CI] | ECE↓ [95% CI] | Brier↓ [95% CI] |
|---|---|---|---|---|
| **Model_A_Baseline** | 0.870 [0.847, 0.904] | 0.895 [0.866, 0.923] | 0.052 [0.048, 0.054] | 0.146 [0.108, 0.184] |
| **Model_B_Disagreement_Aware** | 0.823 [0.761, 0.871] | 0.877 [0.858, 0.895] | 0.072 [0.055, 0.085] | 0.182 [0.156, 0.202] |
| **Model_C_Reliability_Aware** | 0.677 [0.628, 0.727] | 0.721 [0.700, 0.744] | 0.219 [0.195, 0.242] | 0.511 [0.451, 0.564] |

## 2. Reliability Metrics
RC-AUROC = confidence correctly ranks correct vs incorrect predictions.
CM-AUROC = D1 (inverted) predicts cross-modal agreement (higher = better).

| Model | RC-AUROC [95% CI] | CM-AUROC [95% CI] | Mean Confidence |
|---|---|---|---|
| **Model_A_Baseline** | 0.892 [0.866, 0.928] | 0.742 [0.668, 0.805] | 0.896 |
| **Model_B_Disagreement_Aware** | 0.887 [0.859, 0.915] | 0.742 [0.668, 0.805] | 0.941 |
| **Model_C_Reliability_Aware** | 0.567 [0.344, 0.790] | 0.742 [0.668, 0.805] | 0.966 |

## 3. Permutation Tests (5,000 permutations)
| Metric | B−A | p(B vs A) | C−A | p(C vs A) |
|---|---|---|---|---|
| f1 | -0.0472 | 0.0948 ns | -0.1925 | 0.0000 * p<0.05 |
| acc | -0.0180 | 0.1988 ns | -0.1738 | 0.0000 * p<0.05 |
| ece | +0.0202 | 0.0084 * p<0.05 | +0.1670 | 0.0000 * p<0.05 |
| brier | +0.0363 | 0.0600 ns | +0.3656 | 0.0000 * p<0.05 |
| rc | -0.0056 | 0.7452 ns | -0.3258 | 0.0014 * p<0.05 |

## 4. Hypothesis Verdict
- Model B does NOT improve Macro-F1 over A (Δ=-0.0472, p=0.0948)
- Model C does NOT improve Macro-F1 over A (Δ=-0.1925, p=0.0000)
- Model B does NOT improve RC-AUROC over A (Δ=-0.0056, p=0.7452)
- Model C does NOT improve RC-AUROC over A (Δ=-0.3258, p=0.0014)

## 5. Limitations
- Ground-truth emotion labels are unavailable; pseudo-labels from K-means clustering may not align with true emotional content.
- 'Text' modality is approximated by audio features (no speech transcripts available).
- Results reflect relative model behaviour under self-consistent labelling, not absolute emotion-recognition performance.
- A full evaluation requires the official ElderReact annotation files.
