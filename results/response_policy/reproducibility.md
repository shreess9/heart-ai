# Response Policy Simulation: Reproducibility Protocol

## 1. Canonical Artifact Provenance
All inputs are strictly loaded from verified deterministic prediction files:
- `results/predictions/canonical_deterministic/unimodal_text.json`
- `results/predictions/canonical_deterministic/unimodal_audio.json`
- `results/predictions/canonical_deterministic/unimodal_visual.json`
- `results/predictions/canonical_deterministic/model_a_baseline_meld_predictions.json`

## 2. Dataset & Split Specifications
- **MELD Train**: 9,989 utterances (1,038 dialogues)
- **MELD Dev**: 1,109 utterances (114 dialogues) — **Thresholds & weights fitted strictly here**
- **MELD Test**: 2,610 utterances (280 dialogues) — **Locked evaluation; zero threshold tuning**

## 3. Mathematical Policy Formulations & DEV Thresholds
- **Policy 0 (Always Adapt)**: Action = `STRONG_ADAPTATION` $\forall x$.
- **Policy 1 (Confidence-Only)**:
  - $\tau_{\text{high}} = 0.66591$
  - $\tau_{\text{low}} = 0.43124$
- **Policy 2 (Confidence + Disagreement)**:
  - Clarification: $D_3 = 2 \land D_1 \ge 0.53085$
  - Trust Score: $\text{logit}(P) = -1.5949 + (4.0346\cdot c) + (0.5757\cdot D_1) + (-0.3404\cdot D_3)$
  - Cutoffs: $\theta_{\text{high}} = 0.72412, \theta_{\text{low}} = 0.49161$
- **Policy 3 (Full Reliability + Entropy)**:
  - Reliability Score: $\text{logit}(P) = -0.3298 + (2.8905\cdot c) + (0.5985\cdot D_1) + (-0.3310\cdot D_3) + (-0.3669\cdot H)$
  - Cutoffs: $\theta_{\text{high}} = 0.71844, \theta_{\text{low}} = 0.48835$

## 4. Bootstrap Resampling Configuration
- Resampling Unit: Dialogue Cluster (`Dialogue_ID`)
- Iterations: 1,000
- Random Seed: `42`
- Confidence Interval: Two-sided empirical 95% percentiles [2.5%, 97.5%]
