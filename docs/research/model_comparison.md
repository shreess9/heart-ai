# Controlled Multimodal Fusion Model Benchmark (Model A vs B vs C)

This report records the canonical, multi-seed controlled benchmark on the locked MELD test split (2,610 samples, 280 dialogue clusters), comparing Model A (Baseline Concatenation MLP), Model B (Disagreement-Aware MLP), and Model C (Reliability-Aware Attention + Fusion).

---

## 1. Multi-Seed Benchmark Summary Table (Seeds: 42, 123, 456)

| Architecture | Accuracy (Mean $\pm$ Std) | Macro-F1 (Mean $\pm$ Std) | Raw ECE (Mean) | Calibrated ECE (Mean $\pm$ Std) | Calibrated Brier | High-Conf Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline Concatenation)** | **68.01% $\pm$ 0.90%** | 0.4808 $\pm$ 0.0077 | 20.54% | **3.86% $\pm$ 0.86%** | **0.4497** | 10.23% |
| **Model B (Disagreement-Aware MLP)** | 53.86% $\pm$ 1.79% | 0.4186 $\pm$ 0.0119 | 25.25% | 4.00% $\pm$ 0.58% | 0.5866 | 9.80% |
| **Model C (Reliability-Aware Attention)** | 64.80% $\pm$ 1.53% | **0.4937 $\pm$ 0.0097** | **14.22%** | 6.96% $\pm$ 1.48% | 0.4830 | **9.41%** |

---

## 2. Canonical Seed 42 Breakdown

| Architecture | Accuracy | Macro-F1 | Weighted-F1 | Raw ECE | Calibrated ECE | Dev $T$ | High-Conf Error | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline MLP)** | 66.82% | 0.4787 | 0.6633 | 25.36% | **2.98%** | 3.720 | 11.16% | 0.4588 |
| **Model B (Disagreement-Aware)** | 54.06% | 0.4255 | 0.5735 | 26.12% | 4.66% | 3.173 | 9.72% | 0.5878 |
| **Model C (Reliability-Aware)** | 66.67% | **0.4849** | **0.6774** | **10.56%** | 6.83% | 1.954 | **9.26%** | 0.4628 |

---

## 3. Paired Clustered Statistical Significance (vs Baseline Model A on Seed 42)

- **Model B vs Model A**:
  - $\Delta \text{Macro-F1} = -0.0531$ (95% CI: $[-0.0825, -0.0248]$, $p < 0.0001$, statistically significant drop).
  - $\Delta \text{Accuracy} = -12.76\%$ (95% CI: $[-14.87\%, -10.67\%]$, $p < 0.0001$).
- **Model C vs Model A**:
  - $\Delta \text{Macro-F1} = +0.0063$ (95% CI: $[-0.0247, +0.0386]$, $p = 0.6960$, directional positive trend, non-significant).
  - $\Delta \text{Accuracy} = -0.15\%$ (95% CI: $[-1.96\%, +1.71\%]$, $p = 0.8780$).

---

## 4. Key Takeaways
1. Naively appending disagreement features to feature embeddings without attention gates (**Model B**) catastrophically degrades accuracy because MELD is heavily imbalanced toward Text. Audio and visual channels are noisy and cause the network to over-penalize valid text signals.
2. **Model C (Reliability-Aware Attention)** uses attention and reliability gating to prevent this collapse, yielding slightly higher Macro-F1 across seeds ($0.4937 \pm 0.0097$ vs $0.4808 \pm 0.0077$) and significantly reducing raw (pre-scaled) calibration error from 20.54% down to 14.22%.
3. In post-hoc calibration, temperature scaling effectively calibrates all models to ~3%–7% ECE.
