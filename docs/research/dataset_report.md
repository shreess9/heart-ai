# MELD Benchmark Dataset Validation Report

This report documents the canonical Multimodal EmotionLines Dataset (MELD) splits used for controlled research evaluation.

## 1. Split Overview

| Split | Utterances | Dialogue Clusters | Unique Speakers | Empty Text |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | 9,989 | 1038 | 260 | 0 |
| **Dev** | 1,109 | 114 | 47 | 0 |
| **Test** | 2,610 | 280 | 100 | 0 |

## 2. 7-Class Emotion Distribution

| Emotion | Train Count (%) | Dev Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- |
| **anger** | 1,109 (11.1%) | 153 (13.8%) | 345 (13.2%) |
| **disgust** | 271 (2.7%) | 22 (2.0%) | 68 (2.6%) |
| **fear** | 268 (2.7%) | 40 (3.6%) | 50 (1.9%) |
| **joy** | 1,743 (17.4%) | 163 (14.7%) | 402 (15.4%) |
| **neutral** | 4,710 (47.1%) | 470 (42.4%) | 1,256 (48.1%) |
| **sadness** | 683 (6.8%) | 111 (10.0%) | 208 (8.0%) |
| **surprise** | 1,205 (12.1%) | 150 (13.5%) | 281 (10.8%) |

## 3. Framing & Integrity Notes
- **Test Set Lock**: The test set (2,610 utterances across 280 dialogues) is strictly locked for final evaluation.
- **Class Imbalance**: Neutral represents ~47-48% across all splits, followed by Joy (~15-17%) and Anger (~11-13%). Disgust (~2-3%) and Fear (~2%) are rare minority classes.
- **Demographic Scope**: MELD is a conversational TV-dialogue benchmark and is explicitly NOT an elderly-specific clinical dataset.
