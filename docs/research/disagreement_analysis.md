# Cross-Modal Disagreement & Uncertainty Mathematical Audit

This document records the mathematical audit of cross-modal disagreement ($D_1, D_2, D_3$) and predictive uncertainty on 2,610 locked MELD test samples.

---

## 1. Disagreement Metrics Definitions
1. **$D_1$ (Generalized JSD)**: Normalized 3-distribution Jensen-Shannon Divergence across text ($P_T$), audio ($P_A$), and visual ($P_V$) probability distributions:
   $$D_1 = \frac{JSD(P_T, P_A, P_V)}{\ln(3)}$$
2. **$D_2$ (Mean Pairwise JSD)**: Mean pairwise normalized JSD across $(T, A)$, $(T, V)$, and $(A, V)$:
   $$D_2 = \frac{1}{3}\left(\frac{JSD(P_T, P_A)}{\ln 2} + \frac{JSD(P_T, P_V)}{\ln 2} + \frac{JSD(P_A, P_V)}{\ln 2}\right)$$
3. **$D_3$ (Categorical Conflict Level)**: Decision-level agreement among $\text{argmax}$ unimodal predictions:
   - $D_3 = 0$: All 3 modalities predict the identical emotion class.
   - $D_3 = 1$: Exactly one modality differs from the other two.
   - $D_3 = 2$: All 3 modalities predict different emotion classes.

---

## 2. Empirical Distribution on MELD Test Set (2,610 Samples)

| Disagreement Level ($D_3$) | Count | Percentage | Mean Generalized JSD ($D_1$) | Mean Pairwise JSD ($D_2$) |
| :--- | :--- | :--- | :--- | :--- |
| **Level 0 (All 3 Agree)** | 597 | **22.87%** | **0.1129** | 0.1085 |
| **Level 1 (One Differs)** | 1,406 | **53.87%** | **0.4007** | 0.3892 |
| **Level 2 (All 3 Differ)** | 607 | **23.26%** | **0.5858** | 0.5734 |

---

## 3. Mathematical Consistency & Correlation Findings

1. **$D_1$ vs $D_2$ Rank Equivalence**:
   - Spearman rank correlation: $\rho = 0.99575$ ($p < 10^{-100}$)
   - **Conclusion**: Generalized 3-distribution JSD and Mean Pairwise JSD provide mathematically equivalent ranking of cross-modal divergence.

2. **$D_1$ vs Multimodal Mixture Entropy ($H(P)$)**:
   - Spearman rank correlation: $\rho = +0.6174$ ($p = 3.62 \times 10^{-274}$)
   - Linear regression: $\text{Entropy} = 1.8430 \times D_1 + 0.7727$ ($R^2 = 0.538$)
   - **Conclusion**: Cross-modal divergence strongly correlates with predictive uncertainty in the multimodal mixture distribution.

3. **Gate 2: Correctness vs Text Confidence and Disagreement ($D_1$)**:
   - Logistic regression: $\text{logit}(P(\text{Correct})) = \beta_0 + \beta_{\text{conf}} \cdot c_T + \beta_{\text{disagree}} \cdot D_1$
   - Text Confidence: $\beta_{\text{conf}} = +4.6021$ ($p = 1.61 \times 10^{-91}$)
   - $D_1$ Disagreement: $\beta_{\text{disagree}} = -1.6580$ ($p = 8.42 \times 10^{-11}$)
   - **Conclusion**: Text confidence is the primary positive determinant of correctness; cross-modal disagreement exerts a significant negative penalty, confirming that disagreement acts as a valid risk/uncertainty indicator.
