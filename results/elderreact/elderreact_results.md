# ElderReact A+V Validation Results

## 1. Classification Performance (5-Fold CV)
| Model | Macro-F1 | Subset Acc |
|---|---|---|
| Audio Only | 0.237 | 0.212 |
| Visual Only | 0.309 | 0.201 |
| A+V Fusion | 0.365 | 0.201 |

## 2. Response Policy Evaluation
| Policy | Strong-Adaptation Coverage | Dominant-Emotion Error Rate |
|---|---|---|
| P0: Always Adapt | 100.0% | 43.6% |
| P1: Confidence Only | 34.8% | 16.8% |
| P2: Conf + Disagreement | 16.5% | 23.1% |
| P3: Full Reliability | 12.7% | 12.8% |

