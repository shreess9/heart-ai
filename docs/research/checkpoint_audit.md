# Modality Checkpoint Audit & Specification Report

This document formally records the exact model backbones, feature extractors, classification heads, training/fine-tuning protocols, and checkpoint provenance for the H.E.A.R.T. AI research pipeline.

---

## 1. Modality Checkpoint Taxonomy

| Modality | Model Backbone | Provenance / Checkpoint | Feature Dim | Pooling Method | Classification Head | Fine-Tuning Status | Output Type |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Text** | RoBERTa-Base | `roberta-base` (HuggingFace Hub) | 768 | CLS token | Linear(768, 7) | Fine-tuned on MELD train (9,989) / dev (1,109) | 7-class emotion logits & 768-dim embeddings |
| **Audio** | Wav2Vec 2.0 | `facebook/wav2vec2-base-960h` / SER Head | 768 | Temporal Mean Pooling | Linear(768, 7) + Dropout(0.1) | Fine-tuned on MELD speech features | 7-class emotion logits & 768-dim embeddings |
| **Visual** | Vision Transformer | `google/vit-base-patch16-224` / FER Head | 768 | CLS token | Linear(768, 7) + Dropout(0.1) | Fine-tuned on MELD facial expressions | 7-class emotion logits & 768-dim embeddings |

---

## 2. Distinction Between Backbones and Emotion Classifiers

> [!IMPORTANT]
> **Research Integrity Principle**: Pretrained foundation backbones (e.g. `roberta-base`, `wav2vec2-base-960h`, `vit-base-patch16-224`) are generic feature extractors, **not** out-of-the-box emotion classifiers. 
> To obtain valid emotion probability distributions over the 7 MELD classes (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`), task-specific classification heads are explicitly trained on the canonical MELD training split (9,989 samples) with checkpoint selection governed strictly by validation performance on the development split (1,109 samples). The test set (2,610 samples) remains held-out.

---

## 3. Modality Capability Imbalance Note

In accordance with literature findings (Geng Tu et al., 2026; Cai et al., 2026):
- Text features exhibit substantially higher classification capacity on MELD than Audio or Visual cues.
- Audio signals in MELD suffer from varied acoustic environments, laughter tracks, and background noise.
- Visual signals are constrained by face detection availability, occlusion, and head pose.
- Consequently, cross-modal disagreement ($D_1, D_2, D_3$) must **not** be naively interpreted as pure emotional ambiguity; it may partly reflect modality capability imbalance or acoustic/visual noise.
