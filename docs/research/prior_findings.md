# Historical Research Context & Prior Findings

This document formally records prior research investigations and established baseline numbers. These historical results serve as research context and must **not** be treated as results of new code until formally reproduced under the current configuration.

---

## 1. MELD Benchmark Configuration
- **Test Set**: 2,610 test utterances across 280 dialogue clusters.
- **Canonical Taxonomy**: 7 classes (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).

### Historical Modality Performance
- **Text (RoBERTa)**: $\approx 65.98\%$ Accuracy
- **Audio (Wav2Vec 2.0)**: $\approx 47.89\%$ Accuracy
- **Visual (ViT)**: $\approx 48.12\%$ Accuracy
- **Existing Multimodal MLP**: $\approx 66.21\%$ Accuracy

> [!NOTE]
> **Modality Imbalance Finding**: Text is substantially stronger than audio and visual modalities. Therefore, cross-modal disagreement cannot be assumed to represent pure emotional ambiguity; it may reflect modality capability differences.

---

## 2. Phase 3: Disagreement vs Predictive Uncertainty
- **Initial Hypothesis**: Higher cross-modal disagreement ($JSD$) directly implies higher predictive uncertainty ($H(P)$).
- **Observed Result**: The linear relationship was in the opposite direction ($\beta \approx -1.141, p \approx 6.79 \times 10^{-95}$).
- **Conclusion**: The simple hypothesis that disagreement is a monotonic linear proxy for predictive uncertainty was **rejected**.

---

## 3. Calibration & Temperature Scaling
- **Historical Optimal Temperature**: $T \approx 1.249188$
- **Historical Test Calibration Metrics**:
  - Raw ECE: $\approx 8.21\%$ $\rightarrow$ Calibrated ECE: $\approx 1.93\%$
  - Raw MCE: $\approx 28.95\%$ $\rightarrow$ Calibrated MCE: $\approx 6.11\%$

---

## 4. Gate 2: Disagreement & Correctness Relationship
- **Experiment**: Tested whether, at comparable confidence levels, higher disagreement predicts lower correctness.
- **Reported Coefficient**: $+3.871$ ($p \approx 0.3909$, not statistically significant).
- **Conclusion**: There is no statistical basis to claim that disagreement alone predicts lower correctness at fixed confidence.

---

## 5. DE-1: Disagreement Representations
Three representations of cross-modal conflict were analyzed:
1. **$D_1$**: Normalized generalized 3-distribution Jensen-Shannon Divergence.
2. **$D_2$**: Mean pairwise JSD across $(T, A)$, $(T, V)$, and $(A, V)$.
3. **$D_3$**: Categorical decision-level disagreement from $\text{argmax}$ predictions:
   - $D_3 = 0$ (All 3 agree): 1,188 samples ($\approx 45.52\%$, mean JSD $\approx 0.0964$)
   - $D_3 = 1$ (One differs): 1,175 samples ($\approx 45.02\%$, mean JSD $\approx 0.1872$)
   - $D_3 = 2$ (All 3 differ): 247 samples ($\approx 9.46\%$, mean JSD $\approx 0.2164$)

- **Correlation**: $D_1$ and $D_2$ Spearman rank correlation $\rho \approx 0.99827$, indicating near-equivalent distribution-level information.
- **Conclusion**: Disagreement provides a descriptive characterization of cross-modal consistency, but its utility in fusion remains an open experimental question to be validated against controlled baselines.
