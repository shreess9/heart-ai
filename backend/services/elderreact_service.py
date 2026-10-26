"""
backend/services/elderreact_service.py
========================================
ElderReact multi-label emotion inference service.

Lifecycle
---------
- Loaded ONCE at application startup via `get_elderreact_service()`.
- Does NOT retrain or refit any parameters.
- Does NOT touch the MELD pipeline_service or existing routes.

Inference chain for raw media uploads
---------------------------------------
  MP4 / WAV upload
      │
      ├── Audio track → librosa resample → COVAREP-compatible proxy features (72-dim)
      │   NOTE: We do NOT have the COVAREP MATLAB toolbox.
      │   We produce a 72-dim Mel-spectrogram proxy (mean+std of 36 Mel bands),
      │   which matches the expected input dimension but is NOT identical to COVAREP.
      │   This is documented explicitly in every response as `audio_feature_source`.
      │
      └── Video frames → OpenFace-compatible proxy features (178-dim)
          NOTE: We do NOT have the OpenFace binary.
          We produce a 178-dim visual proxy from cv2 face detection + Haar features,
          matching the expected dimension but NOT identical to OpenFace.
          This is documented explicitly in every response as `video_feature_source`.

Both proxies produce valid-shaped input to the trained model, but the model was
trained on COVAREP + OpenFace features. Distribution mismatch means predictions
on proxy features should be treated as exploratory, not production-quality.

If pre-extracted 148-dim audio and/or 1418-dim video features are uploaded as
.npy files, the exact trained feature pipeline is used instead.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Paths (resolved relative to project root) ─────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_CKPT_DIR   = _ROOT / "results" / "elderreact" / "official" / "checkpoints"
_THRESH_PATH = _ROOT / "results" / "elderreact" / "official" / "thresholds.json"

ELDERREACT_EMOTIONS: List[str] = [
    "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
]
N_EMOTIONS = 6

# Maximum upload sizes
MAX_AUDIO_BYTES = 30 * 1024 * 1024   # 30 MB
MAX_VIDEO_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_NPY_BYTES   =  2 * 1024 * 1024   #  2 MB

SUPPORTED_AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}
SUPPORTED_NPY_EXT   = {".npy"}

# ── Minimal MultiLabelMLP re-definition for unpickling ────────────────────────
class _MultiLabelMLP:
    """Pickle-compatible re-definition of the trained model wrapper."""
    def __init__(self, hidden=(128, 64), seed=42):
        self.hidden = hidden
        self.seed   = seed
        self.clfs   = []

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])

    def predict(self, X: np.ndarray, thresholds: Optional[np.ndarray] = None) -> np.ndarray:
        p = self.predict_proba(X)
        t = thresholds if thresholds is not None else np.full(len(self.clfs), 0.5)
        return (p >= t).astype(int)


def _inject_for_unpickling() -> None:
    """Inject _MultiLabelMLP into __main__ so pickle.load resolves it."""
    import sys
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "MultiLabelMLP"):
        setattr(main_mod, "MultiLabelMLP", _MultiLabelMLP)


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_audio_subset(raw_148: np.ndarray) -> np.ndarray:
    """Select the 72-dim COVAREP subset: [:36] + [74:110]."""
    return np.concatenate([raw_148[..., :36], raw_148[..., 74:110]], axis=-1)


def _extract_video_subset(raw_1418: np.ndarray) -> np.ndarray:
    """Select the 178-dim OpenFace subset from the full 1418-dim vector."""
    return np.concatenate([
        raw_1418[..., 0:8],
        raw_1418[..., 288:294],
        raw_1418[..., 634:717],
        raw_1418[..., 997:1003],
        raw_1418[..., 1343:1418],
    ], axis=-1)


def _audio_bytes_to_proxy_72(audio_bytes: bytes, suffix: str) -> Tuple[np.ndarray, str]:
    """
    Produce a 72-dim proxy from raw audio bytes using librosa Mel features.

    Proxy design: 36 Mel bands, compute mean and std across time.
    Concatenating [mean(36), std(36)] gives the 72-dim vector matching
    the COVAREP-trained input dimension.

    Returns (features_72, warning_message).
    WARNING: These are NOT COVAREP features. Distribution mismatch exists.
    """
    try:
        import librosa
    except ImportError:
        raise RuntimeError("librosa is required for audio processing. pip install librosa")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        y, sr = librosa.load(tmp, sr=16000, mono=True, duration=30.0)
        if len(y) < 256:
            raise ValueError("Audio clip too short (< 256 samples at 16kHz).")
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=36, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        feat = np.concatenate([mel_db.mean(axis=1), mel_db.std(axis=1)])  # [72]
        feat = np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        warning = (
            "Audio features are a Mel-spectrogram proxy (72-dim). "
            "The model was trained on COVAREP features. "
            "Predictions from proxy features are exploratory only."
        )
        return feat, warning
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def _video_bytes_to_proxy_178(video_bytes: bytes, suffix: str) -> Tuple[np.ndarray, str]:
    """
    Produce a 178-dim proxy from raw video bytes using cv2 frame statistics.

    Proxy design: sample up to 16 evenly-spaced frames, extract mean and std
    of RGB channels (3), LBP-like texture (1), edge density (1), and Haar face
    score (1) per frame, pooled across frames to fill 178 dims.

    Returns (features_178, warning_message).
    WARNING: These are NOT OpenFace features. Distribution mismatch exists.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python is required. pip install opencv-python-headless")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(video_bytes)
        tmp = f.name
    try:
        cap = cv2.VideoCapture(tmp)
        if not cap.isOpened():
            raise ValueError("Could not open video file.")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_sample = min(16, max(1, total))
        indices  = np.linspace(0, max(0, total - 1), n_sample, dtype=int)

        # Load Haar face cascade (bundled with OpenCV)
        face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

        frame_features = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                frame_features.append(np.zeros(11, dtype=np.float32))
                continue
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (64, 64))
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Per-channel mean + std (6), edge density (1), brightness (1),
            # face detection score (1), aspect ratio invariant stats (2)
            r_m, r_s = rgb[:, :, 0].mean() / 255., rgb[:, :, 0].std() / 255.
            g_m, g_s = rgb[:, :, 1].mean() / 255., rgb[:, :, 1].std() / 255.
            b_m, b_s = rgb[:, :, 2].mean() / 255., rgb[:, :, 2].std() / 255.
            edges     = cv2.Canny(small, 50, 150).mean() / 255.
            brightness = small.mean() / 255.
            faces     = face_cascade.detectMultiScale(small, 1.1, 3, minSize=(10, 10))
            face_score = min(1.0, len(faces) * 0.5)
            laplacian  = cv2.Laplacian(small, cv2.CV_64F).var() / 10000.0
            hist_std   = np.std(np.histogram(small, 16)[0]) / small.size

            frame_features.append(np.array([
                r_m, r_s, g_m, g_s, b_m, b_s,
                edges, brightness, face_score,
                min(laplacian, 1.0), min(hist_std, 1.0),
            ], dtype=np.float32))
        cap.release()

        # Pool: mean + std across frames → [22] → tile to [178]
        stacked = np.array(frame_features)  # [n_sample, 11]
        pooled  = np.concatenate([stacked.mean(0), stacked.std(0)])  # [22]
        reps    = 178 // len(pooled)
        rem     = 178 % len(pooled)
        feat    = np.concatenate([np.tile(pooled, reps), pooled[:rem]])  # [178]
        feat    = np.nan_to_num(feat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        warning = (
            "Video features are a cv2 frame-statistics proxy (178-dim). "
            "The model was trained on OpenFace AU/landmark features. "
            "Predictions from proxy features are exploratory only."
        )
        return feat, warning
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Service class
# ══════════════════════════════════════════════════════════════════════════════

class ElderReactService:
    """
    Singleton service wrapping the ElderReact inference pipeline.

    Model loaded once at startup — never retrained during inference.
    Exposes `predict()` accepting raw media bytes or pre-extracted .npy features.
    """

    _instance: Optional["ElderReactService"] = None

    def __init__(self, model_variant: str = "ER-A"):
        import json

        self.model_variant = model_variant
        self._model        = None
        self._scaler_a     = None
        self._scaler_v     = None
        self._thresholds   = np.full(N_EMOTIONS, 0.5)
        self._load_error: Optional[str] = None
        self.is_ready      = False

        # ── Load thresholds ──────────────────────────────────────────────
        try:
            data = json.loads(_THRESH_PATH.read_text())
            key  = model_variant.replace("-", "_")
            self._thresholds = np.array(data["thresholds"][key])
            log.info(f"Thresholds loaded (tuned on dev): "
                     f"{dict(zip(ELDERREACT_EMOTIONS, self._thresholds.round(3)))}")
        except Exception as e:
            self._load_error = f"Threshold load failed: {e}"
            log.error(self._load_error)
            return

        # ── Load scalers ─────────────────────────────────────────────────
        scaler_path = _CKPT_DIR / "scalers.pkl"
        try:
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            self._scaler_a = scalers["audio"]
            self._scaler_v = scalers["video"]
            log.info("Scalers loaded (fitted on train split).")
        except Exception as e:
            self._load_error = f"Scaler load failed: {e}"
            log.error(self._load_error)
            return

        # ── Load checkpoint ──────────────────────────────────────────────
        variant_map = {"ER-A": "er_a.pkl", "ER-B": "er_b.pkl", "ER-C": "er_c.pkl"}
        pkl_name    = variant_map.get(model_variant)
        if not pkl_name:
            self._load_error = f"Unknown model_variant: {model_variant}"
            log.error(self._load_error)
            return

        ckpt_path = _CKPT_DIR / pkl_name
        if not ckpt_path.exists():
            self._load_error = f"Checkpoint not found: {ckpt_path}"
            log.error(self._load_error)
            return

        try:
            _inject_for_unpickling()
            with open(ckpt_path, "rb") as f:
                self._model = pickle.load(f)
            log.info(f"Checkpoint loaded: {ckpt_path} "
                     f"({len(self._model.clfs)} classifiers)")
            self.is_ready = True
        except Exception as e:
            self._load_error = f"Checkpoint load failed: {e}"
            log.error(self._load_error)

    # ── Singleton factory ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, model_variant: str = "ER-A") -> "ElderReactService":
        if cls._instance is None:
            cls._instance = ElderReactService(model_variant=model_variant)
        return cls._instance

    # ── Public prediction API ────────────────────────────────────────────────

    def predict(
        self,
        audio_bytes:       Optional[bytes] = None,
        audio_filename:    Optional[str]   = None,
        video_bytes:       Optional[bytes] = None,
        video_filename:    Optional[str]   = None,
        audio_npy_bytes:   Optional[bytes] = None,
        video_npy_bytes:   Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Run inference. Exactly one of the following input combinations is supported:

        A. audio_npy_bytes + video_npy_bytes  → exact COVAREP+OpenFace features
        B. audio_bytes + video_bytes           → proxy features (documented mismatch)
        C. audio_bytes only                    → UNSUPPORTED (model requires both)
        D. video_bytes only                    → UNSUPPORTED (model requires both)

        Returns a dict with keys defined in EmotionPredictionResponse schema.
        """
        if not self.is_ready:
            return self._error_response(
                f"Model not ready: {self._load_error}",
                status_code=503,
            )

        warnings: List[str] = []
        audio_feature_source = "unknown"
        video_feature_source = "unknown"

        # ── Path A: pre-extracted .npy features ─────────────────────────
        if audio_npy_bytes is not None and video_npy_bytes is not None:
            try:
                audio_feat, video_feat, src_a, src_v = self._load_npy_features(
                    audio_npy_bytes, video_npy_bytes
                )
                audio_feature_source = src_a
                video_feature_source = src_v
            except (ValueError, RuntimeError) as e:
                return self._error_response(str(e), status_code=422)

        # ── Path B: raw media bytes ──────────────────────────────────────
        elif audio_bytes is not None and video_bytes is not None:
            audio_filename = audio_filename or "upload.wav"
            video_filename = video_filename or "upload.mp4"

            # Validate
            err = self._validate_file(audio_bytes, audio_filename,
                                      SUPPORTED_AUDIO_EXT, MAX_AUDIO_BYTES, "audio")
            if err:
                return self._error_response(err, status_code=422)
            err = self._validate_file(video_bytes, video_filename,
                                      SUPPORTED_VIDEO_EXT, MAX_VIDEO_BYTES, "video")
            if err:
                return self._error_response(err, status_code=422)

            # Extract proxy features
            try:
                suffix_a = Path(audio_filename).suffix or ".wav"
                audio_feat, w_a = _audio_bytes_to_proxy_72(audio_bytes, suffix_a)
                warnings.append(w_a)
                audio_feature_source = "mel-proxy-72dim"
            except Exception as e:
                return self._error_response(f"Audio processing failed: {e}", status_code=422)

            try:
                suffix_v = Path(video_filename).suffix or ".mp4"
                video_feat, w_v = _video_bytes_to_proxy_178(video_bytes, suffix_v)
                warnings.append(w_v)
                video_feature_source = "cv2-proxy-178dim"
            except Exception as e:
                return self._error_response(f"Video processing failed: {e}", status_code=422)

        # ── Path C/D: unsupported single-modality ───────────────────────
        elif audio_bytes is not None or video_bytes is not None:
            return self._error_response(
                "The ElderReact model requires BOTH audio and video inputs. "
                "Single-modality inference is not supported. "
                "Provide both audio_file and video_file, "
                "or both audio_features.npy and video_features.npy.",
                status_code=422,
            )
        else:
            return self._error_response(
                "No input provided. Supply either: "
                "(audio_file + video_file) or (audio_features.npy + video_features.npy).",
                status_code=422,
            )

        # ── Scale ────────────────────────────────────────────────────────
        audio_feat = np.nan_to_num(audio_feat.flatten()[:72], nan=0.0, posinf=0.0, neginf=0.0)
        video_feat = np.nan_to_num(video_feat.flatten()[:178], nan=0.0, posinf=0.0, neginf=0.0)

        audio_scaled = self._scaler_a.transform(audio_feat.reshape(1, -1))[0]
        video_scaled = self._scaler_v.transform(video_feat.reshape(1, -1))[0]
        X = np.concatenate([audio_scaled, video_scaled]).reshape(1, -1)

        # ── Predict ──────────────────────────────────────────────────────
        proba = self._model.predict_proba(X)[0]   # [6] independent probs
        active = [
            e for e, p, t in zip(ELDERREACT_EMOTIONS, proba, self._thresholds)
            if p >= t
        ]

        # ── Reliability assessment ────────────────────────────────────────
        level, note = self._assess_reliability(proba)

        return {
            "is_mock":               False,
            "model_variant":         self.model_variant,
            "emotion_labels":        ELDERREACT_EMOTIONS,
            "probabilities":         dict(zip(ELDERREACT_EMOTIONS, proba.tolist())),
            "active_emotions":       active,
            "thresholds":            dict(zip(ELDERREACT_EMOTIONS, self._thresholds.tolist())),
            "reliability_level":     level,
            "reliability_note":      note,
            "audio_feature_source":  audio_feature_source,
            "video_feature_source":  video_feature_source,
            "warnings":              warnings,
            "error":                 None,
        }

    def predict_mock(self) -> Dict[str, Any]:
        """Return clearly-labelled mock output for testing. Probabilities are random."""
        rng   = np.random.default_rng()
        proba = rng.uniform(0.05, 0.95, N_EMOTIONS).astype(float)
        return {
            "is_mock":               True,
            "model_variant":         self.model_variant,
            "emotion_labels":        ELDERREACT_EMOTIONS,
            "probabilities":         dict(zip(ELDERREACT_EMOTIONS, proba.tolist())),
            "active_emotions":       [],
            "thresholds":            dict(zip(ELDERREACT_EMOTIONS, self._thresholds.tolist())),
            "reliability_level":     "low",
            "reliability_note":      (
                "MOCK MODE — probabilities are random and have no diagnostic value. "
                "Do not use for emotional adaptation."
            ),
            "audio_feature_source":  "mock",
            "video_feature_source":  "mock",
            "warnings":              ["MOCK MODE active — not a real prediction."],
            "error":                 None,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_npy_features(
        self, audio_npy: bytes, video_npy: bytes
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Load pre-extracted .npy feature arrays and select correct subsets."""
        if len(audio_npy) > MAX_NPY_BYTES:
            raise ValueError(f"audio_features.npy exceeds {MAX_NPY_BYTES//1024}KB limit.")
        if len(video_npy) > MAX_NPY_BYTES:
            raise ValueError(f"video_features.npy exceeds {MAX_NPY_BYTES//1024}KB limit.")

        a_arr = np.load(__import__("io").BytesIO(audio_npy)).flatten().astype(np.float32)
        v_arr = np.load(__import__("io").BytesIO(video_npy)).flatten().astype(np.float32)

        if a_arr.shape[0] == 148:
            a_feat = _extract_audio_subset(a_arr)
            src_a  = "covarep-148dim-subset-72dim"
        elif a_arr.shape[0] == 72:
            a_feat = a_arr
            src_a  = "covarep-72dim-direct"
        else:
            raise ValueError(f"audio .npy must be 148-dim or 72-dim, got {a_arr.shape[0]}.")

        if v_arr.shape[0] == 1418:
            v_feat = _extract_video_subset(v_arr)
            src_v  = "openface-1418dim-subset-178dim"
        elif v_arr.shape[0] == 178:
            v_feat = v_arr
            src_v  = "openface-178dim-direct"
        else:
            raise ValueError(f"video .npy must be 1418-dim or 178-dim, got {v_arr.shape[0]}.")

        return a_feat, v_feat, src_a, src_v

    def _assess_reliability(self, proba: np.ndarray) -> Tuple[str, str]:
        """
        Conservative reliability assessment.

        ECE for ER-A on ElderReact test set = 0.212 (moderate miscalibration).
        Do NOT describe raw probabilities as calibrated confidence scores.
        """
        ECE_ER_A = 0.212
        margins  = proba - self._thresholds
        active   = margins > 0
        n_active = int(active.sum())

        if n_active == 0:
            return "low", (
                "No emotion reached its dev-tuned threshold. "
                "Prediction is inconclusive. Do not infer emotional state."
            )

        max_margin = float(margins[active].max()) if n_active else 0.0
        max_prob   = float(proba[active].max()) if n_active else 0.0
        active_str = ", ".join(e for e, m in zip(ELDERREACT_EMOTIONS, active) if m)

        # ECE 0.212 exceeds the HIGH threshold (0.20) — cap at moderate
        if max_margin >= 0.25 and max_prob >= 0.65 and ECE_ER_A <= 0.20:
            return "high", (
                f"Active emotions: {active_str} with well-separated confidence. "
                "Acknowledge emotional tone empathetically, without clinical claims."
            )
        if max_margin >= 0.10 and max_prob >= 0.45:
            return "moderate", (
                f"Possible emotions: {active_str} (moderate confidence). "
                f"Model ECE={ECE_ER_A:.3f} (moderate miscalibration). "
                "Use open-ended supportive language. Avoid strong emotional assertions."
            )
        return "low", (
            f"Active emotion probabilities ({active_str}) are close to threshold. "
            f"Model ECE={ECE_ER_A:.3f}. Low reliability — maintain neutral supportive tone."
        )

    @staticmethod
    def _validate_file(
        data: bytes,
        filename: str,
        allowed_ext: set,
        max_bytes: int,
        label: str,
    ) -> Optional[str]:
        if not data:
            return f"{label} file is empty."
        if len(data) > max_bytes:
            return (f"{label} file ({len(data)//1024}KB) exceeds "
                    f"{max_bytes//1024//1024}MB limit.")
        ext = Path(filename).suffix.lower()
        if ext not in allowed_ext:
            return (f"Unsupported {label} format '{ext}'. "
                    f"Supported: {sorted(allowed_ext)}")
        return None

    @staticmethod
    def _error_response(msg: str, status_code: int = 500) -> Dict[str, Any]:
        return {
            "is_mock":               False,
            "model_variant":         "N/A",
            "emotion_labels":        ELDERREACT_EMOTIONS,
            "probabilities":         {},
            "active_emotions":       [],
            "thresholds":            {},
            "reliability_level":     "low",
            "reliability_note":      msg,
            "audio_feature_source":  "none",
            "video_feature_source":  "none",
            "warnings":              [],
            "error":                 msg,
            "_status_code":          status_code,
        }


# ── Module-level accessor ────────────────────────────────────────────────────

_service_instance: Optional[ElderReactService] = None


def get_elderreact_service() -> ElderReactService:
    """Get or create the singleton ElderReact service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ElderReactService(model_variant="ER-A")
    return _service_instance
