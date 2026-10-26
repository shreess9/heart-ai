# H.E.A.R.T. AI — Reliability-Aware Multimodal Emotion Demonstrator
## Multimodal Research Demonstrator & Architecture Guide

---

### 1. Project Identity & Research Core

- **Project Title**: **H.E.A.R.T. AI — Emotion-Aware Companion for Older Adults**
- **Technical Research Focus**: Reliability-Aware Multimodal Emotion Understanding and Response Adaptation.
- **Core Scientific Thesis**:
  > *"The companion does not just recognise an emotion; it considers how trustworthy that inference is before deciding how strongly to adapt its response."*

---

### 2. Demonstrator Purpose & Objectives

This research demonstrator provides an interactive, full-stack implementation of the H.E.A.R.T. AI pipeline. Its primary objective is to make the internal multimodal perception, decomposition, cross-modal disagreement calculation, uncertainty calibration, reliability gating, and conversational response adaptation visible and interactive.

```
Older Adult Input (Text + Audio + Visual)
      │
      ▼
Modality Encoders & Media Processors (RoBERTa, Wav2Vec 2.0, ViT)
      │
      ▼
Modality Emotion Evidence (Probability Distributions P_text, P_audio, P_visual)
      ├──► Dynamic Multimodal Fusion ──► Emotion Prediction & Calibrated Confidence
      │
      └──► Cross-Modal Disagreement Calculation (D1, D2, D3)
                  │
                  ▼
         Reliability Assessment (Confidence + Disagreement + Entropy)
                  │
                  ▼
         Reliability Gate (HIGH / MEDIUM / LOW)
                  │
                  ▼
         Conversational Manager Policy (Strong / Cautious / Clarification)
                  │
                  ▼
         H.E.A.R.T. AI Companion Attuned Response + "Why this response?" Explainability
```

---

### 3. How to Launch & Run the Demonstrator

#### Prerequisites
- Python 3.10+ with repository virtual environment active.
- Dependencies installed (`fastapi`, `uvicorn`, `torch`, `transformers`, `numpy`, `scipy`, `scikit-learn`, `soundfile`, `opencv-python`, `moviepy`).

#### Start the Unified Server (Backend & Frontend)
From the repository root (`d:/Project`):

```bash
# Start FastAPI backend with static UI mounting
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Open the Demonstrator UI
Open your web browser and navigate to:
```
http://127.0.0.1:8000/
```
(or `http://127.0.0.1:8000/demo`)

---

### 4. Demonstrator Modes: Four Distinct Analysis Pipelines

The demonstrator provides **four clearly separated modes**:

| Mode | **Mode 1: Manual Text** | **Mode 2: Upload Multimodal Input** | **Mode 3: Select MELD Utterance** | **Mode 4: Controlled Demonstration** |
|---|---|---|---|---|
| **Input Source** | User types an arbitrary text utterance. | Real user-uploaded Audio (.wav, .mp3) and/or Video (.mp4, .mov) with optional transcript. | User selects an authentic sample from the locked MELD test split (2,610 samples). | User selects one of 4 predefined benchmark scenarios. |
| **Modality Handling** | Text is analyzed via RoBERTa; absent audio/visual sensor channels are explicitly marked as **"Not Provided"** without fake sensor data. | Real audio waveform extraction (16kHz mono) and video frame sampling ($224 \times 224$ patches) + audio track separation. Dynamic fusion across active modalities. | Real multimodal predictions ($P_{\text{text}}, P_{\text{audio}}, P_{\text{visual}}$) from canonical deterministic MELD run. | Pre-calibrated distributions illustrating specific theoretical contrasts. |
| **Disagreement** | Single modality ($D_1=0, D_2=0, D_3=0$, marked N/A). | $D_1, D_2, D_3$ computed dynamically across active modalities (1, 2, or 3 channels). | $D_1, D_2, D_3$ computed authentically from canonical modality evidence. | $D_1, D_2, D_3$ computed from scenario distributions. |
| **Notice / Badge** | `Mode 1: Text-Only Inference` | `Mode 2: Upload Multimodal Input` | `Mode 3: MELD Test Sample` | `Mode 4: Controlled Demonstration Scenario` |

---

### 5. Mode 2: Real Multimodal Upload Processing Pipeline

#### Pretrained Emotion Classifier Checkpoints Integrated:
1. **Audio Emotion Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
   - **Architecture**: `Wav2Vec2Model` + fine-tuned `AudioEmotionHead` (1024-dim dense tanh projection + 8-class linear output layer).
   - **Provenance**: RAVDESS-trained (Ryerson Audio-Visual Database of Emotional Speech and Song).
   - **Class Mapping (8 $\to$ 7 Canonical)**:
     - `angry` (0) $\to$ `anger` (0)
     - `disgust` (2) $\to$ `disgust` (1)
     - `fearful` (3) $\to$ `fear` (2)
     - `happy` (4) $\to$ `joy` (3)
     - `calm` (1) + `neutral` (5) $\to$ `neutral` (4)
     - `sad` (6) $\to$ `sadness` (5)
     - `surprised` (7) $\to$ `surprise` (6)
     - Normalized to sum to $1.0$.

2. **Visual Face Emotion Model**: `jihedjabnoun/vit-face-emotion-recognition`
   - **Architecture**: Vision Transformer (`ViTForImageClassification`, Google ViT-base patch16 224).
   - **Provenance**: FER2013 / EmoSet Facial Emotion Recognition.
   - **Face Detection & Cropping**: OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) crops detected frontal faces with a 15% bounding-box margin. Frames with no detected human face are strictly excluded from emotion inference (returns `NO_FACES_DETECTED` when 0 faces exist).
   - **Temporal Aggregation**: Arithmetic mean probability averaging across all frames with detected faces:
     $$P_{\text{visual}} = \frac{1}{N_{\text{faces}}} \sum_{i=1}^{N_{\text{faces}}} P_{\text{frame}_i}, \quad \sum P_{\text{visual}} = 1.0$$
   - **Class Mapping (7 $\to$ 7 Canonical)**:
     `Anger(0) -> anger, Disgust(1) -> disgust, Fear(2) -> fear, Happiness(3) -> joy, Neutral(4) -> neutral, Sadness(5) -> sadness, Surprise(6) -> surprise`.

#### Supported File Formats & Limits
- **Audio Formats**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`, `.wma`
- **Video Formats**: `.mp4`, `.mov`, `.avi`, `.webm`, `.mkv`, `.m4v`, `.flv`
- **Maximum File Size**: 50 MB per file.
- **Dynamic Multimodal Combinations**:
  - Audio only ($A$)
  - Video only ($V$ / $A+V$ if audio track present)
  - Text + Audio ($T+A$)
  - Text + Video ($T+V$)
  - Audio + Video ($A+V$)
  - Text + Audio + Video ($T+A+V$)
  - Text only ($T$)
  - Missing modalities are marked `NOT_PROVIDED` or `NO_FACES_DETECTED` and never inserted as fake neutral vectors.

---

### 6. Mathematical Definitions of Disagreement Metrics

Cross-modal disagreement metrics are computed directly across unimodal probability distributions ($P_{\text{text}}, P_{\text{audio}}, P_{\text{visual}}$):

1. **$D_1$ — Distribution-Level Disagreement (Generalized JSD)**:
   - For 3 modalities:
     $$D_1 = \text{JSD}(P_T, P_A, P_V) = H\left(\frac{P_T + P_A + P_V}{3}\right) - \frac{H(P_T) + H(P_A) + H(P_V)}{3}$$
     Normalized by $\log_2(3)$ into $[0, 1]$.
   - For 2 modalities:
     $$D_1 = \text{JSD}(P_1, P_2) = H\left(\frac{P_1 + P_2}{2}\right) - \frac{H(P_1) + H(P_2)}{2}$$
     Normalized by $\log_2(2) = 1.0$.
   - For 1 modality: $D_1 = 0.0$ (Not Applicable).

2. **$D_2$ — Pairwise Probability Disagreement**:
   - For 3 modalities: $D_2 = \frac{1}{3}\big(\text{JSD}(P_T, P_A) + \text{JSD}(P_T, P_V) + \text{JSD}(P_A, P_V)\big)$.
   - For 2 modalities: $D_2 = \text{JSD}(P_1, P_2) = D_1$.
   - For 1 modality: $D_2 = 0.0$ (Not Applicable).

3. **$D_3$ — Categorical Conflict Level**:
   - For 3 modalities: $D_3 \in \{0, 1, 2\}$:
     - $0$ = All 3 modalities agree ($\hat{y}_T = \hat{y}_A = \hat{y}_V$).
     - $1$ = Exactly two agree and one differs.
     - $2$ = All 3 modalities predict different emotions.
   - For 2 modalities: $D_3 = 0 \text{ if } \hat{y}_1 = \hat{y}_2 \text{ else } 1$.
   - For 1 modality: $D_3 = 0$ (Not Applicable).

---

### 7. Reliability Gating Logic & Response Policies

The Reliability Gate synthesizes **Calibrated Confidence ($\hat{P}_{\text{max}}$)**, **Distribution Disagreement ($D_1$)**, **Categorical Conflict ($D_3$)**, and **Predictive Entropy ($H$)**:

$$\text{Reliability Level} = \begin{cases} \text{HIGH} & \text{if } \hat{P}_{\text{max}} \ge 0.70 \land D_1 \le 0.15 \land D_3 = 0 \\ \text{LOW} & \text{if } (D_3 \ge 1 \land D_1 \ge 0.35) \lor \hat{P}_{\text{max}} < 0.40 \\ \text{MEDIUM} & \text{otherwise} \end{cases}$$

#### Conversational Response Tiers:
- **HIGH Reliability $\to$ Strong Emotion Adaptation**: Expressive empathetic companionship when evidence is concordant and trustworthy.
- **MEDIUM Reliability $\to$ Cautious Adaptation**: Gentle, tentative phrasing to avoid emotional misattribution under mild divergence.
- **LOW Reliability (Conflict) $\to$ Clarification Query**: Explicitly asks the older adult how they are feeling rather than guessing an emotional state.
- **LOW Reliability (Ambiguity / High Entropy) $\to$ Neutral Supportive Presence**: Warm conversational presence without asserting emotion labels.

---

### 8. Controlled Demonstration Scenarios

Four benchmark demonstration scenarios illustrate key gate transitions:

1. **Strong Agreement**: Joy across Text, Audio, Visual ($D_1 \approx 0.04, D_3 = 0$) $\to$ **HIGH Reliability $\to$ Strong Adaptation**.
2. **Moderate Disagreement**: Sadness in Text, Neutral in Audio, Sadness in Visual ($D_1 \approx 0.28, D_3 = 1$) $\to$ **MEDIUM Reliability $\to$ Cautious Adaptation**.
3. **Strong Cross-Modal Conflict**: Joy in Text, Anger in Audio, Fear in Visual ($D_1 \approx 0.58, D_3 = 2$) $\to$ **LOW Reliability $\to$ Clarification**.
4. **Low Confidence / Ambiguity**: Diffuse distributions with elevated entropy ($H \approx 2.7$) $\to$ **LOW Reliability $\to$ Neutral Supportive**.

---

### 8. Dynamic Gemini / OpenAI LLM Companion & Strategy Control

A central architectural requirement of H.E.A.R.T. AI is decoupling **affective perception & reliability evaluation** from **conversational language generation**:

> **Core Architectural Principle**:
> *"Gemini is used only for dynamic conversational response generation. Emotion inference and reliability assessment remain separate components of the H.E.A.R.T. AI architecture."*

```
                    OLDER ADULT
                         │
                 CONVERSATION INPUT
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
            TEXT       AUDIO      VIDEO
              │          │          │
           Emotion     Emotion    Emotion
           Model       Model      Model
              └──────────┼──────────┘
                         │
                 MULTIMODAL FUSION
                         │
               EMOTION PREDICTION
                         │
             CONFIDENCE + D1/D2/D3
                         │
                RELIABILITY GATE
                         │
                 RESPONSE STRATEGY
                         │
                      GEMINI
                         │
              DYNAMIC COMPANION
                   RESPONSE
```

#### Provider Priority Hierarchy:
1. **Priority 1: `GEMINI_API_KEY`** $\to$ `GeminiLLMProvider`
   - Uses Google's official OpenAI-compatible endpoint: `https://generativelanguage.googleapis.com/v1beta/openai/`
   - Supported models: `gemini-flash-lite-latest` (default), `gemini-flash-latest`, `gemini-3.8-flash`, etc. configured via `GEMINI_MODEL`.
   - UI Status Badge: `Gemini LLM — Connected`.
2. **Priority 2: `OPENAI_API_KEY` or `LLM_API_KEY`** $\to$ `OpenAILLMProvider`
   - Configurable for OpenAI (`gpt-4o-mini`), Groq, Ollama, OpenRouter, or LocalAI.
   - UI Status Badge: `OpenAI LLM — Connected`.
3. **Priority 3: No Key Configured** $\to$ `RuleBasedDynamicFallbackProvider`
   - Deterministic context-aware offline generator.
   - UI Status Badge: `Demo Fallback — LLM Unconfigured`.

#### Error Handling & Fail-Safe Transparency:
If Gemini encounters network failure, rate limits (HTTP 429), or invalid models, the system catches the exception cleanly and routes to the offline fallback engine. The UI and API explicitly label the response as `"Demo Fallback Response (LLM Error: ...)"` and mark `is_llm_generated: false`. The system **never** silently claims a fallback response was generated by Gemini.

#### Response Strategy Directives Injected into Gemini:
The Reliability Gate computes the Allowed Response Strategy **before** calling Gemini:
- **`STRONG_ADAPTATION` (HIGH Reliability)**:
  > *"High reliability concordant evidence detected '{emotion}'. You ARE authorized to warmly acknowledge the emotional tone, offer attuned comfort or share joy, and ask a supportive, empathetic follow-up question."*
- **`CAUTIOUS_ADAPTATION` (MEDIUM Reliability)**:
  > *"Moderate reliability with mild divergence detected '{emotion}'. Use gentle, tentative phrasing. Do NOT assert the inferred emotion as certainty or fact. Keep the topic open-ended and let the user guide the emotional depth."*
- **`CLARIFICATION` (LOW Reliability / Conflict)**:
  > *"High incongruity between verbal and non-verbal cues. Do NOT assume or assert the user's emotion. Respond naturally to their spoken words and gently ask an open clarification question to understand how they are feeling."*
- **`NEUTRAL_RESPONSE` (LOW Reliability / Diffuse Uncertainty)**:
  > *"Emotion evidence is diffuse or uncertain. Provide a warm, respectful, neutral conversational presence without making assumptions about their emotional state."*

#### Bounded Conversation History:
The `DialogueManager` maintains the recent conversation context bounded to the last **5–10 turns**, appending prior turns into the Gemini payload while preventing runaway context growth.

---

### 9. Audio & Visual Emotion Checkpoint Audit & Transparency

The repository was comprehensively audited for unimodal emotion classification weights across all checkpoint extensions (`.pt`, `.pth`, `.bin`, `.safetensors`, `.ckpt`, `.onnx`, `.h5`):

1. **Audio Pipeline (`wav2vec2-base`)**:
   - Audio decoding (16,000 Hz mono float32), duration, sample count, and RMS energy calculation function as intended.
   - However, a fine-tuned 7-class emotion classification head/weights file on top of Wav2Vec2 is currently absent on disk.
   - **Honest Status**: Reported as `"Audio emotion checkpoint unavailable"`, and audio evidence is excluded from multimodal fusion for uploaded media.
2. **Visual Pipeline (`vit-base-patch16-224`)**:
   - Video container decoding, audio track separation, and OpenCV keyframe sampling ($224 \times 224$ patches) function as intended.
   - However, a fine-tuned 7-class facial emotion classification checkpoint on top of ViT is absent on disk.
   - **Honest Status**: Reported as `"Visual emotion classification checkpoint unavailable"`, and visual evidence is excluded from fusion for uploaded media.
3. **MELD Research Mode**:
   - Uses genuine, locked canonical deterministic prediction distributions ($P_T, P_A, P_V$) generated from the canonical MELD test evaluation split (`results/predictions/canonical_deterministic/`).
4. **Label Space Integrity**:
   - All models adhere strictly to the canonical 7-class MELD label ordering: `["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]`.
   - Modality probability distributions strictly satisfy $\sum p_i \approx 1$, $p_i \ge 0$, with no NaNs.

| Modality State | Meaning in Demonstrator | UI Indicator & Handling |
|---|---|---|
| **AVAILABLE** | Sensor input provided AND fine-tuned 7-class emotion classifier executed. | `✓ Emotion Inferred` &bull; Distribution used in multimodal fusion and disagreement. |
| **PREPROCESSED_NO_CLASSIFIER** | Media file decoded / frames extracted, but fine-tuned emotion classification checkpoint is absent. | `⚠ Processed (Classifier Unavailable)` &bull; Modality telemetry displayed; **channel excluded from fusion**. |
| **NOT_PROVIDED** | User did not supply this sensor channel for the current turn. | `✗ Not Provided` &bull; Modality inactive; no data fabricated. |

---

### 10. Responsible AI & Older Adult Care Guardrails

The companion system is governed by strict non-clinical safety boundaries:
1. **Zero Medical/Psychiatric Diagnosis**: NEVER diagnoses depression, cognitive decline, anxiety, dementia, or any clinical condition.
2. **Zero Impersonation**: NEVER claims to be human, a doctor, or a therapist.
3. **No Emotional Dependency**: Promotes grounded support and family/community connection without fostering dependency.
4. **Uncertainty-First Grounding**: Defaults to neutral supportive dialogue whenever confidence is low or sensor channels conflict.
5. **No Metric Exposure**: Mathematical formulas ($D_1, D_2, D_3$, entropy, logits) are translated into high-level behavioral directives and never dumped into conversational text.

---

### 11. Future Scope & Older-Adult Datasets

- **ElderReact Evaluation**: Extension to dedicated older-adult conversational affective datasets such as **ElderReact** is planned strictly as **Future Scope**, subject to dataset availability, licensing, and multimodal compatibility.
- **Noisy Sensor Robustness**: Handling real-world microphone noise, acoustic reverberation, and partial visual occlusion.
- **Asynchronous Modality Arrival**: Processing conversational turns when audio, video, or speech arrive at different latencies.
- **Longitudinal Personalization**: Attuning reliability thresholds to individual baseline acoustic and facial patterns over time.

---

### 12. Automated Test Suite Verification

The complete test suite contains **98 automated unit and integration tests**:

```bash
python -m unittest discover -s tests -p "test_*.py"
# Ran 98 tests in 5.946s, OK (100% Passing)
```

Tests verify:
- Gemini LLM Provider configuration, OpenAI-compatible endpoint payloads, and error handling.
- Provider selection priority (`GEMINI_API_KEY` $\to$ `OPENAI_API_KEY` $\to$ Fallback).
- Strategy directive enforcement (High, Medium, Conflict, Low Confidence).
- Dialogue history bounding (up to 10 turns) and session resets.
- Audio/video format and size validation with HTTP 400 error reporting.
- Video frame extraction (OpenCV) and audio track separation.
- Dynamic missing-modality combinations ($T, A, V, T+A, T+V, A+V, T+A+V$).
- Mathematical correctness of $D_1, D_2, D_3$ across 1, 2, and 3 active channels.
- Offline fallback provider integrity and telemetry reporting.
- Non-clinical older adult safety framing throughout.

