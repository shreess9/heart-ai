# H.E.A.R.T. AI: Comprehensive MELD Multimodal Reproducibility Audit & Research Findings

---

## 1. Executive Research Summary & Scientific Audit

This report documents the canonical, audited empirical evaluation of the **H.E.A.R.T. AI** framework on the canonical **Multimodal EmotionLines Dataset (MELD)** benchmark across 2,610 locked test utterances from 280 multi-speaker dialogue clusters, evaluated under multi-seed validation (Seeds: 42, 123, 456).

### Core Research Question Addressed
> *"Can cross-modal disagreement, together with calibrated predictive reliability, provide a useful signal for improving the reliability of multimodal emotion-aware conversational systems?"*

### Primary Scientific Conclusions
1. **Modality Capability Imbalance**:
   - **Text (RoBERTa)** achieves **69.00% accuracy** and **0.4808 Macro-F1** (ECE: 5.14%), decisively outperforming **Speech/Audio** (42.68% Acc, 0.2188 Macro-F1, ECE: 35.16%) and **Facial/Visual** (43.49% Acc, 0.2305 Macro-F1, ECE: 35.78%).
   - Audio and visual classifiers struggle on minority classes (e.g., disgust, fear F1 $\approx 0.0$).
   - **Critical Takeaway**: Cross-modal disagreement ($D_1, D_2, D_3$) cannot be assumed to represent pure conversational ambiguity; it predominantly reflects acoustic/visual noise and unimodal capacity disparity.

2. **The Hazard of Naive Disagreement Appending (Model B)**:
   - When disagreement features ($D_1, D_2, D_3$) are naively concatenated into an MLP (**Model B**), performance severely degrades: accuracy drops from **68.01% (Model A)** to **53.86% (Model B)** (Macro-F1 drops from 0.4808 to 0.4186, $\Delta = -0.0531, p < 0.0001$).
   - **Mechanistic Cause**: Because 77.13% of utterances exhibit cross-modal disagreement ($D_3 \ge 1$) due to weak audio/visual signals, the un-gated MLP interprets widespread non-text disagreement as overall uncertainty, suppressing the highly reliable text predictions.

3. **Reliability-Aware Attention & Gating (Model C)**:
   - **Model C (Reliability-Aware Attention + Fusion)** uses reliability gates and multi-head attention over confidence and entropy to protect the dominant modality while incorporating affective nuance.
   - **Performance**: Achieves **0.4937 $\pm$ 0.0097 Macro-F1** (Peak: 0.5073) and **64.80% $\pm$ 1.53% Accuracy**.
   - **Raw Calibration**: Model C dramatically improves **raw (pre-scaling) calibration**, dropping Raw ECE from **20.54% (Model A)** down to **14.22% $\pm$ 5.10% (as low as 10.56% on Seed 42)**.
   - **Statistical Test (Seed 42)**: $\Delta \text{Macro-F1} = +0.0063$ [95% CI: $-0.0247, +0.0386$, $p = 0.6960$]. While positive on minority emotional expressions, the gain over baseline is directional and not statistically significant for global classification accuracy.

4. **Probability Calibration Behavior**:
   - Post-hoc temperature scaling fitted strictly on validation (dev) logits effectively reduces ECE across all architectures:
     - Model A: Calibrated ECE = **3.86% $\pm$ 0.86%** ($T \approx 2.08 - 3.72$)
     - Model B: Calibrated ECE = **4.00% $\pm$ 0.58%** ($T \approx 2.59 - 3.30$)
     - Model C: Calibrated ECE = **6.96% $\pm$ 1.48%** ($T \approx 1.75 - 3.91$)
   - Temperature scaling works effectively across all architectures; Model C improves uncalibrated probabilistic alignment, but does not outperform post-hoc calibrated baseline ECE.

5. **Application & Policy Value**:
   - Cross-modal disagreement and reliability estimates ($D_1, D_3, H(P), c_{\text{cal}}$) are **not** magic classification boosters.
   - Their true scientific value lies in **epistemic safety gating**: enabling the companion system to detect when sensor signals are conflicted, gracefully reducing conversational assumptions, avoiding false confidence, and prioritizing user clarification.

---

## 2. Canonical Unimodal Baselines Audit

Evaluated on 2,610 locked MELD test utterances from 280 dialogue clusters:

| Modality | Backbone / Architecture | Accuracy | Macro-F1 | Weighted-F1 | Raw ECE | Raw Brier | High-Conf Error ($>0.80$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Text** | RoBERTa-Base + MLP Head | **69.00%** | **0.4808** | **0.6787** | **5.14%** | **0.4439** | **9.82%** |
| **Audio** | Wav2Vec 2.0 Head | 42.68% | 0.2188 | 0.4042 | 35.16% | 0.8248 | 38.46% |
| **Visual** | ViT-Base Head | 43.49% | 0.2305 | 0.4121 | 35.78% | 0.8122 | 37.14% |

---

## 3. Disagreement & Uncertainty Mathematical Audit ($D_1, D_2, D_3$)

### Categorical Disagreement Distribution ($D_3$)
- **Level 0 (All 3 modalities agree)**: 597 samples (**22.87%**), Mean Generalized JSD ($D_1$) = **0.1129**
- **Level 1 (Exactly one modality differs)**: 1,406 samples (**53.87%**), Mean Generalized JSD ($D_1$) = **0.4007**
- **Level 2 (All 3 modalities differ)**: 607 samples (**23.26%**), Mean Generalized JSD ($D_1$) = **0.5858**

### Mathematical Properties
1. **$D_1$ vs $D_2$ Equivalence**: Spearman rank correlation $\rho = 0.99575$ ($p < 10^{-100}$), confirming Generalized JSD and Mean Pairwise JSD provide identical distributional divergence rankings.
2. **Disagreement vs Mixture Entropy ($H(P)$)**:
   - Spearman $\rho = +0.6174$ ($p = 3.62 \times 10^{-274}$)
   - Linear regression: $\text{Entropy} = 1.8430 \times D_1 + 0.7727$ ($R^2 = 0.538$)
   - Confirms that higher cross-modal divergence correlates strongly with mixture entropy.
3. **Gate 2 Correctness Regression**:
   - Logistic regression: $\text{Correctness} \sim \text{Text Confidence} + D_1$
   - Text Confidence coefficient: $\beta = +4.6021$ ($p = 1.61 \times 10^{-91}$)
   - $D_1$ Disagreement coefficient: $\beta = -1.6580$ ($p = 8.42 \times 10^{-11}$)
   - Controlling for text confidence, higher cross-modal disagreement exerts a significant negative penalty on prediction correctness.

---

## 4. Controlled Multimodal Fusion Model Benchmark (Model A vs B vs C)

Aggregated across 3 canonical random seeds (42, 123, 456) on the locked test split:

| Model Architecture | Accuracy (Mean $\pm$ Std) | Macro-F1 (Mean $\pm$ Std) | Raw ECE (Mean) | Calibrated ECE (Mean $\pm$ Std) | Calibrated Brier | High-Conf Error |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline Concatenation)** | **68.01% $\pm$ 0.90%** | 0.4808 $\pm$ 0.0077 | 20.54% | **3.86% $\pm$ 0.86%** | **0.4497** | 10.23% |
| **Model B (Disagreement-Aware MLP)** | 53.86% $\pm$ 1.79% | 0.4186 $\pm$ 0.0119 | 25.25% | 4.00% $\pm$ 0.58% | 0.5866 | 9.80% |
| **Model C (Reliability-Aware Attention)** | 64.80% $\pm$ 1.53% | **0.4937 $\pm$ 0.0097** | **14.22%** | 6.96% $\pm$ 1.48% | 0.4830 | **9.41%** |

### Paired Clustered Bootstrap Hypothesis Testing (Seed 42, 1,000 resamples)
- **Model B vs Model A**:
  - $\Delta \text{Macro-F1} = -0.0531$ [95% CI: $-0.0825, -0.0248$, $p < 0.0001$] (Statistically significant drop)
  - $\Delta \text{Accuracy} = -12.76\%$ [95% CI: $-14.87\%, -10.67\%$, $p < 0.0001$]
- **Model C vs Model A**:
  - $\Delta \text{Macro-F1} = +0.0063$ [95% CI: $-0.0247, +0.0386$, $p = 0.6960$] (Directional positive trend, non-significant)
  - $\Delta \text{Accuracy} = -0.15\%$ [95% CI: $-1.96\%, +1.71\%$, $p = 0.8780$]

---

## 5. Systematic Feature Ablation Study (3 Seeds Aggregated)

| Feature Configuration | Input Dim | Accuracy | Macro-F1 | Raw ECE | Calibrated ECE | Brier Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (Baseline Concatenation)** | 2304 | 68.01% $\pm$ 0.90% | 0.4808 $\pm$ 0.0077 | 20.54% | 3.86% | 0.5212 |
| **Model A + $D_1$ (Generalized JSD)** | 2305 | 57.68% $\pm$ 1.25% | 0.4462 $\pm$ 0.0125 | 20.40% | 4.11% | 0.6128 |
| **Model A + $D_2$ (Pairwise JSD)** | 2305 | 56.08% $\pm$ 1.70% | 0.4321 $\pm$ 0.0127 | 26.04% | 3.35% | 0.6655 |
| **Model A + $D_3$ (Categorical Conflict)** | 2307 | 58.95% $\pm$ 1.92% | 0.4460 $\pm$ 0.0102 | 26.40% | 5.40% | 0.6522 |
| **Model B (Full Disagreement $D_1+D_2+D_3$)**| 2309 | 53.86% $\pm$ 1.79% | 0.4186 $\pm$ 0.0119 | 25.25% | 4.00% | 0.6877 |
| **Model A + Unimodal Confidence** | 2307 | 67.89% $\pm$ 0.33% | 0.4732 $\pm$ 0.0045 | 21.71% | 3.44% | 0.5207 |
| **Model A + Unimodal Entropy** | 2307 | 45.47% $\pm$ 0.28% | 0.3649 $\pm$ 0.0033 | 39.56% | 3.22% | 0.8979 |

---

## 6. Audit Explanations for Previous Discrepancies

The audit revealed the exact causes for the previously reported numerical discrepancies:
1. **66.02% vs 67.24% (Model A Baseline Discrepancy)**:
   - `train_evaluate_multimodal.py` trained Model A for 15 epochs with PyTorch `DataLoader(shuffle=True)`.
   - `ablation_study.py` trained its baseline for 12 epochs with manual `torch.randperm` indexing.
   - Neither script re-seeded RNG per-model, causing RNG state drift between sequential training routines.
2. **Artificial Label Leakage in `_est_probs`**:
   - An intermediate version of `_prepare_multimodal_tensors()` contained a synthetic boost term `logits[i, y] += boost` indexed on the ground truth `y`, which caused artificial probability overconfidence.
   - The canonical audit completely removed this leak, training authentic unimodal classifiers strictly on `train` and checkpointing on `dev`.

---

## 7. Review II Guidance: What to Present vs What NOT to Claim

### What to Present in Review II:
1. **Defensible Empirical Metrics**: Present the multi-seed canonical results (Model A: 68.01% Acc / 0.4808 F1, Model C: 64.80% Acc / 0.4937 F1).
2. **Modality Capability Imbalance**: Highlight that Text dominates MELD, and un-gated fusion of noisy modalities causes catastrophic failure (Model B drops to 53.86%).
3. **Reliability Gating Value**: Model C successfully uses cross-modal attention and reliability gating to prevent this collapse and reduce raw calibration error (Raw ECE 14.22% vs 20.54%).
4. **Affective Dialogue Policy**: The true strength of $D_1, D_3, H(P)$ is safety gating in conversation (epistemic humility, asking for clarification under conflict).

### What MUST NOT Be Claimed:
1. DO NOT claim that Model C achieves a statistically significant accuracy or Macro-F1 improvement over Model A ($p = 0.6960$).
2. DO NOT claim that adding raw disagreement features automatically boosts classification accuracy (it severely harms naive MLPs).
3. DO NOT claim that reliability-aware fusion outperforms post-hoc temperature scaling in calibrated ECE.
4. DO NOT claim emotional inferences represent diagnostic mental health ground truth.
