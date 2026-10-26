# Response Policy Simulation: Limitations & Scientific Scope

## 1. Simulation Nature & Lack of Clinical / Safety Labels
> [!IMPORTANT]
> **This experiment is a computational policy simulation. It does not establish human-perceived response appropriateness, clinical efficacy, or real-world conversational safety.**

- MELD contains multi-party television dialogue annotations for 7 categorical emotions; it does **NOT** contain ground-truth labels for "safe response", "unsafe response", "caregiver appropriateness", or "emotional harm".
- In this simulation, "incorrect strong adaptation" is defined strictly as an utterance where Model A predicted the wrong emotion label ($P(\text{pred} \neq \text{true})$) and the policy nevertheless authorized full emotional response adaptation.
- While emotion classification errors correlate with conversational incongruity, human emotional interactions are nuanced: a misclassified emotion (e.g. confusing surprise for joy) may be benign in some contexts and disruptive in others.

## 2. Explicit Assumptions in the Policy Simulation
1. **Tiered Response Model Assumption**: We assume a conversational architecture where:
   - `STRONG_ADAPTATION` generates emotion-conditioned empathetic responses.
   - `CAUTIOUS_ADAPTATION` softens emotion assertions (e.g., "It sounds like you might be feeling...").
   - `NEUTRAL_RESPONSE` uses generic supportive conversational grounding.
   - `CLARIFICATION` asks targeted disambiguation questions.
2. **Fixed Threshold Optimization**: Thresholds were fitted on MELD DEV to satisfy an empirical 50% coverage constraint. In clinical or assistive deployments, threshold selection must be driven by explicit risk tolerances determined by domain experts.
3. **Synthetic Secondary Modality Representation**: While Text representations are based on RoBERTa contextual embeddings, Audio and Visual continuous representations are generated under controlled signal-to-noise dynamics matching literature benchmarks.

## 3. Requirement for Human-in-the-Loop Evaluation
To validate whether reliability gating translates to perceived conversational safety:
- **A controlled human rater study is required.**
- Future work should evaluate generated dialogue responses paired with MELD multi-modal clips across human evaluators assessing:
  1. Perceived Empathy & Appropriateness
  2. Frequency of Jarring / Harmful Emotional Misattributions
  3. Conversational Fluency and Naturalness.
