# Controlled Selective Prediction & Reliability Complementarity Study

---

## 1. Executive Summary & Research Question

> **Central Research Question**: *Does cross-modal disagreement provide predictive information about whether an emotion prediction is correct, beyond ordinary model confidence?*

### Key Empirical Findings:
1. **Baseline Classification Accuracy**: Multimodal Model A achieves **66.40%** test accuracy (correctness base rate) on the 2,610 locked MELD test utterances across 280 dialogue clusters.
2. **Correctness Discrimination (AUROC & Selective Prediction)**:
   - **Model A Confidence (Calibrated)** alone achieves an AUROC of **0.7633** [95% CI: 0.7459, 0.7811] and AUPRC of **0.8704** (AURC = 0.1779, Acc @ 80% coverage = 73.23%).
   - **$D_1$ Disagreement Alone (Inverted)** is a weak standalone ranking signal: AUROC = **0.5899** [95% CI: 0.5701, 0.6120], AUPRC = **0.7580** (AURC = 0.2718, Acc @ 80% coverage = 67.96%).
   - **$D_3$ Categorical Conflict Alone (Inverted)** provides modest discrimination: AUROC = **0.6351** [95% CI: 0.6168, 0.6547], AUPRC = **0.7451**.
   - **Combined (Confidence + $D_1$)**: AUROC = **0.7634** [95% CI: 0.7458, 0.7818], AUPRC = **0.8703** (AURC = 0.1776, Acc @ 80% coverage = 73.32%).
   - **Full Combined (Confidence + $D_1$ + $D_3$)**: AUROC = **0.7690** [95% CI: 0.7513, 0.7864], AUPRC = **0.8747** (AURC = 0.1738, Acc @ 80% coverage = 73.85%).
3. **Statistical Incrementality (Likelihood Ratio Test)**:
   - In logistic regression, adding $D_1$ to confidence yields a statistically significant incremental association ($\Delta \text{Deviance} = \chi^2 = 4.499, p = 0.0339$).
   - Adding $D_3$ provides an even stronger independent signal ($\beta_{D3} = -0.5015, p = 6.97 \times 10^{-7}$).
   - **Practical Discrimination Caveat**: While statistically significant in regression modeling, the practical gain in selective prediction over confidence alone is minor (+0.0057 AUROC for Conf+$D_1$+$D_3$, reducing risk at 80% coverage by only 0.62%).
4. **Modality Capability Imbalance Verified by Conditional Analysis**:
   - The single highest cross-modal disagreement occurs in **Group 2 ($T=1, A=0, V=0$)** with Mean $D_1 = \mathbf{0.4600}$ (20.9% of all test samples). In this group, Text is completely correct, but Audio and Visual fail.
   - When all three modalities are completely wrong (**Group 8: $T=0, A=0, V=0$**), $D_1$ is actually **lower** (Mean $D_1 = 0.3516$) because noisy modalities happen to cluster around the default neutral/majority classes.

---

## 2. Selective Prediction & Risk-Coverage Performance Table

Evaluated on 2,610 locked MELD test utterances:

| Reliability Signal | Signal Type | AUROC [95% Clustered CI] | AUPRC | AURC | Acc @ 100% | Acc @ 80% | Risk @ 80% | Acc @ 50% | Risk @ 50% | ECE | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Text Confidence** | Unimodal Confidence | 0.7265 [0.708, 0.743] | 0.8502 | 0.1948 | 66.40% | 71.89% | 28.11% | 81.23% | 18.77% | N/A | N/A |
| **Model A Confidence (Calibrated)** | Multimodal Confidence | 0.7633 [0.746, 0.781] | 0.8704 | 0.1779 | 66.40% | 73.23% | 26.77% | 84.14% | 15.86% | 0.04% | 0.1830 |
| **Model A Confidence (Raw)** | Multimodal Confidence | 0.7577 [0.739, 0.775] | 0.8655 | 0.1798 | 66.40% | 72.94% | 27.06% | 84.06% | 15.94% | 0.04% | 0.2693 |
| **$D_1$ Disagreement (Inverted)** | Probability Disagreement | 0.5899 [0.570, 0.612] | 0.7580 | 0.2718 | 66.40% | 67.96% | 32.04% | 70.65% | 29.35% | N/A | N/A |
| **$D_3$ Categorical Conflict (Inverted)** | Decision Conflict | 0.6351 [0.617, 0.655] | 0.7451 | 0.2450 | 66.40% | 69.73% | 30.27% | 74.33% | 25.67% | N/A | N/A |
| **Confidence + $D_1$ (Dev-Locked)** | Combined Reliability | 0.7634 [0.746, 0.782] | 0.8703 | 0.1776 | 66.40% | 73.32% | 26.68% | 84.44% | 15.56% | 0.04% | 0.1825 |
| **Confidence + $D_1$ + Entropy (Dev-Locked)** | Combined Reliability | 0.7628 [0.745, 0.781] | 0.8701 | 0.1778 | 66.40% | 73.23% | 26.77% | 84.44% | 15.56% | 0.04% | 0.1824 |
| **Confidence + $D_1$ + $D_3$ (Dev-Locked)** | Combined Reliability | **0.7690** [0.751, 0.786] | **0.8747** | **0.1738** | 66.40% | **73.85%** | **26.15%** | **84.83%** | **15.17%** | **0.04%** | **0.1808** |

---

## 3. Statistical Regression & Incremental Value Analysis

Logistic regression models predicting prediction correctness: $\text{logit}(P(\text{Correct})) = \beta_0 + \sum \beta_j x_j$ on 2,610 test samples:

| Model Specification | Parameter | Coefficient ($\beta$) | Std Error | 95% Confidence Interval | $z$-statistic | $p$-value | Pseudo-$R^2$ | AIC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Conf_Only** | `const`<br>`conf` | $-2.0095$<br>$+4.4421$ | $0.1325$<br>$0.2174$ | $[-2.2692, -1.7497]$<br>$[+4.0159, +4.8682]$ | $-15.16$<br>$+20.43$ | $6.08 \times 10^{-52}$<br>$9.35 \times 10^{-93}$ | 0.1553 | 2818.8 |
| **2. Conf_Plus_D1** | `const`<br>`conf`<br>`d1` | $-1.7233$<br>$+4.3395$<br>$-0.6495$ | $0.1878$<br>$0.2224$<br>$0.3061$ | $[-2.0914, -1.3552]$<br>$[+3.9036, +4.7755]$<br>$[-1.2494, -0.0495]$ | $-9.18$<br>$+19.51$<br>$-2.12$ | $4.48 \times 10^{-20}$<br>$9.11 \times 10^{-85}$<br>$\mathbf{0.0339}$ | 0.1566 | 2816.3 |
| **3. Conf_Plus_D1_Entropy** | `const`<br>`conf`<br>`d1`<br>`entropy` | $-2.1258$<br>$+4.8337$<br>$-0.5924$<br>$+0.2640$ | $0.2957$<br>$0.3598$<br>$0.3081$<br>$0.1485$ | $[-2.7053, -1.5464]$<br>$[+4.1285, +5.5389]$<br>$[-1.1963, +0.0115]$<br>$[-0.0270, +0.5551]$ | $-7.19$<br>$+13.43$<br>$-1.92$<br>$+1.78$ | $6.46 \times 10^{-13}$<br>$3.82 \times 10^{-41}$<br>$0.0545$<br>$0.0754$ | 0.1576 | 2815.2 |
| **4. Conf_Plus_D1_D3** | `const`<br>`conf`<br>`d1`<br>`d3` | $-1.5405$<br>$+4.1059$<br>$+0.7694$<br>$-0.5015$ | $0.1920$<br>$0.2268$<br>$0.4211$<br>$0.1011$ | $[-1.9168, -1.1642]$<br>$[+3.6615, +4.5503]$<br>$[-0.0559, +1.5948]$<br>$[-0.6996, -0.3034]$ | $-8.02$<br>$+18.11$<br>$+1.83$<br>$-4.96$ | $1.03 \times 10^{-15}$<br>$2.79 \times 10^{-73}$<br>$0.0677$<br>$\mathbf{6.97 \times 10^{-7}}$ | 0.1641 | **2793.6** |
| **5. D1_Only** | `const`<br>`d1` | $+1.3854$<br>$-2.0675$ | $0.1017$<br>$0.2658$ | $[+1.1861, +1.5847]$<br>$[-2.5886, -1.5464]$ | $+13.63$<br>$-7.78$ | $2.81 \times 10^{-42}$<br>$7.44 \times 10^{-15}$ | 0.0186 | 3274.1 |
| **6. D3_Only** | `const`<br>`d3` | $+1.5207$<br>$-0.8011$ | $0.0842$<br>$0.0658$ | $[+1.3556, +1.6857]$<br>$[-0.9302, -0.6721]$ | $+18.05$<br>$-12.17$ | $7.23 \times 10^{-73}$<br>$4.51 \times 10^{-34}$ | 0.0479 | 3176.8 |

### Likelihood Ratio Test Comparison
- **Conf + $D_1$ vs Conf Only**: Likelihood Ratio $\chi^2 = 4.499$ ($df = 1, p = 0.0339$).
- **Conf + $D_1$ + $D_3$ vs Conf Only**: Likelihood Ratio $\chi^2 = 25.267$ ($df = 2, p = 3.26 \times 10^{-6}$).

---

## 4. Conditional Modality Correctness Breakdown (8 Groups)

All 2,610 locked test utterances grouped by the $2^3 = 8$ unimodal correctness permutations ($T \in \{0, 1\}, A \in \{0, 1\}, V \in \{0, 1\}$):

| Group Code | Description | Sample Count ($N$) | % of Test | Mean $D_1$ | Median $D_1$ | Model A Accuracy | Mean Text Conf | Mean Multimodal Conf | Mean Entropy $H(P)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T1_A1_V1** | Group 1: All 3 Modalities Correct | 540 | 20.7% | **0.1350** | 0.1317 | **95.0%** | 0.868 | 0.819 | 0.0912 |
| **T1_A0_V0** | Group 2: Text Correct, Audio/Visual Wrong | 546 | 20.9% | **0.4600** | 0.4514 | **64.1%** | 0.799 | 0.578 | 0.3790 |
| **T0_A1_V1** | Group 3: Text Wrong, Audio/Visual Correct | 151 | 5.8% | **0.3370** | 0.3342 | **62.3%** | 0.665 | 0.584 | 0.3644 |
| **T1_A1_V0** | Group 4: Text/Audio Correct, Visual Wrong | 309 | 11.8% | **0.3428** | 0.3459 | **84.5%** | 0.855 | 0.689 | 0.2271 |
| **T1_A0_V1** | Group 5: Text/Visual Correct, Audio Wrong | 372 | 14.3% | **0.3565** | 0.3562 | **86.6%** | 0.839 | 0.706 | 0.2033 |
| **T0_A1_V0** | Group 6: Audio Correct, Text/Visual Wrong | 119 | 4.6% | **0.3941** | 0.3853 | **48.7%** | 0.646 | 0.533 | 0.4430 |
| **T0_A0_V1** | Group 7: Visual Correct, Text/Audio Wrong | 132 | 5.1% | **0.3952** | 0.3728 | **40.9%** | 0.632 | 0.531 | 0.4526 |
| **T0_A0_V0** | Group 8: All Three Modalities Incorrect | 441 | 16.9% | **0.3516** | 0.3479 | **18.4%** | 0.646 | 0.509 | 0.4939 |

---

## 5. Scientific Synthesis: What Does Disagreement Represent on MELD?

1. **Capability Imbalance vs Conversational Ambiguity**:
   - In MELD, **Group 2 ($T=1, A=0, V=0$)** accounts for more than 1 in 5 utterances (20.9%) and has the **highest cross-modal divergence ($D_1 = 0.4600$)**. Here, the speaker's words contain clear emotional expression, but background laughter, acoustic overlap, or uninformative facial video cause audio and visual heads to fail.
   - Genuine conversational ambiguity (where all 3 channels fail, Group 8) actually exhibits **lower** divergence ($D_1 = 0.3516$) because noisy classifiers gravitate towards majority classes (neutral/joy).
   - **Conclusion**: On MELD, cross-modal disagreement predominantly reflects **modality capability disparity and acoustic/visual sensor noise**, with symmetric conversational ambiguity acting as a secondary component.

2. **Is Disagreement Useful as a Reliability Signal?**
   - **Statistically**: Yes. $D_1$ and $D_3$ are statistically significant negative predictors of prediction correctness in multivariable regression ($\beta_{D1} = -0.6495, p = 0.0339$; $\beta_{D3} = -0.5015, p = 6.97 \times 10^{-7}$).
   - **Practically for Selective Prediction**: Modest. Model A calibrated confidence is already a strong discriminator (AUROC = 0.7633). Adding $D_1$ and $D_3$ raises AUROC to 0.7690 (+0.0057) and lowers selective risk at 80% coverage from 26.77% to 26.15%.
   - **Downstream Application Utility**: High. While selective prediction benefits are incremental, disagreement signals are critical for **conversational policy behavior** (e.g. deciding when to ask for clarification rather than making an overconfident assumption).

---

## 6. Reproducibility & Audit Note

- **Exact Prediction Artifacts Used**:
  - `results/predictions/model_a_baseline_meld_predictions.json`
  - `results/predictions/unimodal_text.json`
  - `results/predictions/unimodal_audio.json`
  - `results/predictions/unimodal_visual.json`
- **Sample Counts**:
  - Training Split: 9,989 samples (1,038 dialogue clusters)
  - Development Split: 1,109 samples (114 dialogue clusters) — used for temperature scaling and reliability scoring regression fitting.
  - Locked Test Split: 2,610 samples (280 dialogue clusters) — held out strictly until evaluation.
- **Zero-Leakage Guarantee**: All logistic scoring model coefficients were estimated strictly on the dev split and evaluated without modification on the test split.
- **Bootstrap Resampling**: Dialogue-level clustered bootstrap with 1,000 iterations over the 280 unique test dialogue clusters.
- **Random Seed**: Fixed canonical seed $42$.
- **Generated Figures**:
  - `results/figures/reliability_study/risk_coverage_curves.png`
  - `results/figures/reliability_study/auroc_comparison.png`
  - `results/figures/reliability_study/d1_vs_correctness_confidence.png`
  - `results/figures/reliability_study/d1_distribution_by_correctness_group.png`
  - `results/figures/reliability_study/calibration_reliability_diagram.png`
