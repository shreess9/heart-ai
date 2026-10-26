"""H.E.A.R.T. AI: Pretrained Audio & Visual Emotion Inference Engines.

Integrates real pretrained models for live multimodal media uploads:
1. Audio Model: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
   - Architecture: Wav2Vec2ForSequenceClassification
   - Training Dataset: RAVDESS (Speech Emotion Recognition)
   - Classes (8 classes): [angry(0), calm(1), disgust(2), fearful(3), happy(4), neutral(5), sad(6), surprised(7)]
   - Mapped to Canonical 7 Classes: [anger, disgust, fear, joy, neutral (calm+neutral), sadness, surprise]
   - Expected Input: 16,000 Hz mono float32 waveform

2. Visual Model: `jihedjabnoun/vit-face-emotion-recognition`
   - Architecture: Vision Transformer (ViTForImageClassification)
   - Training Dataset: FER2013 / EmoSet (Facial Emotion Recognition)
   - Classes (7 classes): [Anger(0), Disgust(1), Fear(2), Happiness(3)->Joy, Neutral(4), Sadness(5), Surprise(6)]
   - Preprocessing: Face detection & bounding-box crop (OpenCV Haar Cascade) -> (224, 224, 3) -> ViT
   - Aggregation: Temporal probability averaging across detected face frames

Canonical Project Emotion Ordering:
["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Force cache directories onto D: drive before importing torch/transformers
DEFAULT_HF_CACHE = Path("d:/Project/hf_cache")
DEFAULT_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(DEFAULT_HF_CACHE)
os.environ["HF_HUB_CACHE"] = str(DEFAULT_HF_CACHE / "hub")
os.environ["TORCH_HOME"] = str(DEFAULT_HF_CACHE / "torch")
TMP_DIR = Path("d:/Project/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TMP_DIR)
os.environ["TMP"] = str(TMP_DIR)

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

CANONICAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


class AudioEmotionEngine:
    """Speech Emotion Recognition model using pretrained Wav2Vec2 audio classification."""

    def __init__(self, model_id: str = "superb/wav2vec2-base-superb-er"):
        self.model_id = model_id
        self.model = None
        self.feature_extractor = None
        self.is_loaded = False
        self._load_error = None
        self._try_load()

    def _try_load(self):
        try:
            import os
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            # Try loading from local cache first, then download
            try:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id, local_files_only=True)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id, local_files_only=True)
            except Exception:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
                self.model = AutoModelForAudioClassification.from_pretrained(self.model_id)

            self.model.eval()
            self.is_loaded = True
            logger.info(f"AudioEmotionEngine successfully loaded [{self.model_id}].")
        except Exception as e:
            self._load_error = str(e)
            self.is_loaded = False
            logger.warning(f"AudioEmotionEngine failed to load [{self.model_id}]: {e}")

    def predict_emotion(self, waveform: np.ndarray, sample_rate: int = 16000) -> Dict[str, Any]:
        """Run genuine Wav2Vec2 emotion inference on 16kHz mono audio waveform."""
        if not self.is_loaded:
            self._try_load()

        if not self.is_loaded or self.model is None or self.feature_extractor is None:
            raise RuntimeError(f"Audio emotion classifier unavailable: {self._load_error}")

        import torch

        # Ensure 1D float32 numpy array
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)
        # Resample to 16000 Hz if necessary
        if sample_rate != 16000:
            try:
                from scipy.signal import resample
                target_len = int(len(waveform) * 16000 / sample_rate)
                waveform = resample(waveform, target_len).astype(np.float32)
            except Exception:
                old_idx = np.linspace(0, len(waveform) - 1, len(waveform))
                new_idx = np.linspace(0, len(waveform) - 1, int(len(waveform) * 16000 / sample_rate))
                waveform = np.interp(new_idx, old_idx, waveform).astype(np.float32)
            sample_rate = 16000

        # Truncate / pad if needed (up to 10 seconds)
        max_samples = sample_rate * 10
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        # Feature extraction
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

        with torch.no_grad():
            outputs = self.model(inputs.input_values)
            logits = outputs.logits.squeeze(0).cpu().numpy()

        # Apply softmax to get raw probabilities
        exp_logits = np.exp(logits - np.max(logits))
        model_probs = exp_logits / np.sum(exp_logits)

        # Canonical project emotion ordering:
        # ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
        canon_probs = np.zeros(7, dtype=np.float32)

        # Check model labels
        id2label = getattr(self.model.config, "id2label", {})
        if id2label and any("neu" in str(v).lower() for v in id2label.values()):
            # superb/wav2vec2-base-superb-er: {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}
            for idx, prob in enumerate(model_probs):
                lbl = str(id2label.get(idx, "")).lower()
                if "ang" in lbl:
                    canon_probs[0] += float(prob)
                elif "dis" in lbl:
                    canon_probs[1] += float(prob)
                elif "fea" in lbl:
                    canon_probs[2] += float(prob)
                elif "hap" in lbl or "joy" in lbl:
                    canon_probs[3] += float(prob)
                elif "neu" in lbl or "calm" in lbl:
                    canon_probs[4] += float(prob)
                elif "sad" in lbl:
                    canon_probs[5] += float(prob)
                elif "sur" in lbl:
                    canon_probs[6] += float(prob)
        elif len(model_probs) == 8:
            # ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition: [angry(0), calm(1), disgust(2), fearful(3), happy(4), neutral(5), sad(6), surprised(7)]
            canon_probs[0] = float(model_probs[0])  # angry -> anger
            canon_probs[1] = float(model_probs[2])  # disgust -> disgust
            canon_probs[2] = float(model_probs[3])  # fearful -> fear
            canon_probs[3] = float(model_probs[4])  # happy -> joy
            canon_probs[4] = float(model_probs[1]) + float(model_probs[5])  # calm + neutral -> neutral
            canon_probs[5] = float(model_probs[6])  # sad -> sadness
            canon_probs[6] = float(model_probs[7])  # surprised -> surprise
        else:
            for idx, prob in enumerate(model_probs[:7]):
                canon_probs[idx] = float(prob)

        # Normalize so sum == 1.0 (with slight smoothing if zero)
        if np.sum(canon_probs) < 1e-6:
            canon_probs[4] = 1.0
        else:
            canon_probs = canon_probs / np.sum(canon_probs)

        top_id = int(np.argmax(canon_probs))
        top_emotion = CANONICAL_EMOTIONS[top_id]
        confidence = float(canon_probs[top_id])

        # Predictive entropy
        probs_safe = np.clip(canon_probs, 1e-12, 1.0)
        entropy = float(-np.sum(probs_safe * np.log2(probs_safe)))

        return {
            "predicted_emotion": top_emotion,
            "predicted_id": top_id,
            "confidence": round(confidence, 4),
            "confidence_raw": round(confidence, 4),
            "probabilities": [round(float(v), 4) for v in canon_probs],
            "distribution": {CANONICAL_EMOTIONS[i]: round(float(v), 4) for i, v in enumerate(canon_probs)},
            "predictive_entropy": round(entropy, 4),
            "model_name": self.model_id,
            "provenance": "IEMOCAP/RAVDESS-trained (Wav2Vec2 Speech Emotion Recognition)",
            "checkpoint_status": "AVAILABLE",
            "is_available": True,
        }


class VisualFaceEmotionEngine:
    """ViT 7-class facial emotion recognition model with OpenCV face cropping."""

    def __init__(self, model_id: str = "jihedjabnoun/vit-face-emotion-recognition"):
        self.model_id = model_id
        self.model = None
        self.image_processor = None
        self.face_cascade = None
        self.is_loaded = False
        self._load_error = None
        self._try_load()

    def _try_load(self):
        try:
            import cv2
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_id, local_files_only=True)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id, local_files_only=True)
            except Exception:
                self.image_processor = AutoImageProcessor.from_pretrained(self.model_id)
                self.model = AutoModelForImageClassification.from_pretrained(self.model_id)

            self.model.eval()

            # Load OpenCV Haar cascade face detector
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            self.is_loaded = True
            logger.info(f"VisualFaceEmotionEngine successfully loaded [{self.model_id}].")
        except Exception as e:
            self._load_error = str(e)
            self.is_loaded = False
            logger.warning(f"VisualFaceEmotionEngine failed to load [{self.model_id}]: {e}")

    def _detect_and_crop_face(self, frame_rgb: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """Detect frontal face in frame and crop with margin; returns (crop, True) or (None, False)."""
        import cv2
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )

        h, w, _ = frame_rgb.shape
        if len(faces) > 0:
            # Pick largest face by bounding area
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, fw, fh = faces[0]
            # Add 15% margin
            margin_x = int(fw * 0.15)
            margin_y = int(fh * 0.15)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(w, x + fw + margin_x)
            y2 = min(h, y + fh + margin_y)
            face_crop = frame_rgb[y1:y2, x1:x2]
            return cv2.resize(face_crop, (224, 224)), True
        else:
            return None, False

    def predict_video_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Run ViT facial emotion inference on sampled video frames with face detection and temporal aggregation."""
        if not self.is_loaded:
            self._try_load()

        if not self.is_loaded or self.model is None or self.image_processor is None:
            raise RuntimeError(f"Visual emotion classifier unavailable: {self._load_error}")

        import torch
        from PIL import Image

        if not frames:
            raise ValueError("No video frames provided for visual emotion inference.")

        frame_prob_list = []
        faces_detected_count = 0

        for frame in frames:
            # If frame is float in [0, 1], convert to uint8 [0, 255]
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame_uint8 = frame.astype(np.uint8)

            face_img, face_detected = self._detect_and_crop_face(frame_uint8)
            if not face_detected or face_img is None:
                continue

            faces_detected_count += 1
            pil_img = Image.fromarray(face_img)
            inputs = self.image_processor(images=pil_img, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits.squeeze(0).cpu().numpy()

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)

            # Map jihedjabnoun/vit-face-emotion-recognition order:
            # 0: Anger, 1: Disgust, 2: Fear, 3: Happiness, 4: Neutral, 5: Sadness, 6: Surprise
            # Matches canonical: [anger, disgust, fear, joy (from Happiness), neutral, sadness, surprise]
            canon_frame_probs = np.zeros(7, dtype=np.float32)
            canon_frame_probs[0] = float(probs[0])  # Anger
            canon_frame_probs[1] = float(probs[1])  # Disgust
            canon_frame_probs[2] = float(probs[2])  # Fear
            canon_frame_probs[3] = float(probs[3])  # Happiness -> Joy
            canon_frame_probs[4] = float(probs[4])  # Neutral
            canon_frame_probs[5] = float(probs[5])  # Sadness
            canon_frame_probs[6] = float(probs[6])  # Surprise
            canon_frame_probs /= np.sum(canon_frame_probs)

            frame_prob_list.append(canon_frame_probs)

        # Handle case where no faces were detected in any sampled frame
        if faces_detected_count == 0 or len(frame_prob_list) == 0:
            return {
                "predicted_emotion": "No Face Detected",
                "predicted_id": -1,
                "confidence": 0.0,
                "confidence_raw": 0.0,
                "probabilities": [0.0] * 7,
                "distribution": {CANONICAL_EMOTIONS[i]: 0.0 for i in range(7)},
                "predictive_entropy": 0.0,
                "frames_sampled": len(frames),
                "faces_detected": 0,
                "frames_used": 0,
                "model_name": self.model_id,
                "provenance": "FER2013 / EmoSet-trained (Facial Emotion Recognition)",
                "checkpoint_status": "NO_FACES_DETECTED",
                "is_available": False,
                "warning": "No human faces detected in sampled video frames. Visual channel excluded from emotion fusion.",
            }

        # Temporal Probability Averaging (arithmetic mean across detected face frames)
        aggregated_probs = np.mean(frame_prob_list, axis=0)
        aggregated_probs /= np.sum(aggregated_probs)

        top_id = int(np.argmax(aggregated_probs))
        top_emotion = CANONICAL_EMOTIONS[top_id]
        confidence = float(aggregated_probs[top_id])

        # Predictive entropy
        probs_safe = np.clip(aggregated_probs, 1e-12, 1.0)
        entropy = float(-np.sum(probs_safe * np.log2(probs_safe)))

        return {
            "predicted_emotion": top_emotion,
            "predicted_id": top_id,
            "confidence": round(confidence, 4),
            "confidence_raw": round(confidence, 4),
            "probabilities": [round(float(v), 4) for v in aggregated_probs],
            "distribution": {CANONICAL_EMOTIONS[i]: round(float(v), 4) for i, v in enumerate(aggregated_probs)},
            "predictive_entropy": round(entropy, 4),
            "frames_sampled": len(frames),
            "faces_detected": faces_detected_count,
            "frames_used": len(frame_prob_list),
            "model_name": self.model_id,
            "provenance": "FER2013 / EmoSet-trained (Facial Emotion Recognition)",
            "checkpoint_status": "AVAILABLE",
            "is_available": True,
        }


# Singleton instances
_audio_engine: Optional[AudioEmotionEngine] = None
_visual_engine: Optional[VisualFaceEmotionEngine] = None


def get_audio_emotion_engine() -> AudioEmotionEngine:
    global _audio_engine
    if _audio_engine is None:
        _audio_engine = AudioEmotionEngine()
    return _audio_engine


def get_visual_face_emotion_engine() -> VisualFaceEmotionEngine:
    global _visual_engine
    if _visual_engine is None:
        _visual_engine = VisualFaceEmotionEngine()
    return _visual_engine
