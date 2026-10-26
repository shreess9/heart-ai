"""H.E.A.R.T. AI: Automatic Speech Recognition (ASR) Service.

Transcribes audio tracks extracted from uploaded conversation videos (MP4)
using OpenAI Whisper (via Hugging Face Transformers pipeline on local CPU).

The resulting transcript feeds directly into the RoBERTa text emotion classifier,
achieving the primary multimodal architecture:
One MP4 -> Audio Track -> ASR -> Transcript -> RoBERTa -> P(Text Emotion).

IMPORTANT: This service operates in STRICT LOCAL-ONLY mode.
- The Whisper model MUST be pre-downloaded into D:/Project/hf_cache.
- No network requests are made during model loading or inference.
- If the local cache is missing, the service returns ASR_MODEL_UNAVAILABLE
  rather than retrying Hugging Face Hub repeatedly.
"""

import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configure Hugging Face cache to use D:\Project\hf_cache exclusively
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
HF_HUB_DIR = HF_CACHE_DIR / "hub"

# Force HF to use our local cache and never reach out to the network
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_DIR)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Suppress the TensorFlow oneDNN informational message
# (this is NOT a Hugging Face error — just TF logging about CPU optimizations)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

DEFAULT_MODEL_ID = "openai/whisper-tiny"


def _verify_local_cache(model_id: str) -> bool:
    """Check whether the model snapshot directory exists and contains
    the minimum required files for offline loading.

    Returns True if the model appears to be fully cached locally.
    """
    safe_name = model_id.replace("/", "--")
    model_dir = HF_HUB_DIR / f"models--{safe_name}"

    if not model_dir.exists():
        logger.error(
            f"ASR model cache directory not found: {model_dir}. "
            f"Run `python -c \"from transformers import pipeline; "
            f"pipeline('automatic-speech-recognition', model='{model_id}')\"` "
            f"once with internet access to download it."
        )
        return False

    # Check for snapshot directory with actual model files
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        logger.error(f"ASR model snapshots directory missing: {snapshots_dir}")
        return False

    # Find the latest snapshot (there should be exactly one commit hash dir)
    snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
    if not snapshot_dirs:
        logger.error(f"ASR model has no snapshot subdirectories in: {snapshots_dir}")
        return False

    snapshot = snapshot_dirs[0]
    required_files = ["config.json", "model.safetensors", "tokenizer.json"]
    missing = [f for f in required_files if not (snapshot / f).exists()]

    if missing:
        logger.error(
            f"ASR model snapshot at {snapshot} is incomplete. "
            f"Missing files: {missing}. Re-download the model."
        )
        return False

    model_size = (snapshot / "model.safetensors").stat().st_size
    logger.info(
        f"ASR model cache verified: {snapshot.name[:12]}... "
        f"({model_size / 1e6:.1f} MB, {len(list(snapshot.iterdir()))} files)"
    )
    return True


class ASREngine:
    """Whisper Automatic Speech Recognition Engine.

    Operates in strict local-only mode. If the model files are not
    present in D:\\Project\\hf_cache, the engine reports
    ASR_MODEL_UNAVAILABLE without making any network requests.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.pipeline = None
        self.is_loaded = False
        self._load_error: Optional[str] = None
        self._load_attempted = False
        self._try_load()

    def _try_load(self):
        """Attempt to load Whisper from local cache. Never retries on failure."""
        if self._load_attempted:
            return  # Don't retry — avoids repeated HF Hub timeouts
        self._load_attempted = True

        # Step 1: Verify local cache integrity BEFORE importing transformers
        if not _verify_local_cache(self.model_id):
            self._load_error = (
                f"ASR_MODEL_UNAVAILABLE: Local cache for [{self.model_id}] "
                f"not found or incomplete under {HF_CACHE_DIR}. "
                f"Download the model once with internet access first."
            )
            logger.warning(self._load_error)
            return

        # Step 2: Load from local cache only — no network fallback
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(
                f"Loading ASR model [{self.model_id}] from local cache "
                f"(local_files_only=True, TRANSFORMERS_OFFLINE=1)..."
            )
            t0 = time.time()

            self.pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                device=-1,  # CPU
                model_kwargs={"local_files_only": True},
            )

            self.is_loaded = True
            elapsed = time.time() - t0
            logger.info(
                f"ASREngine loaded [{self.model_id}] successfully in {elapsed:.2f}s "
                f"(strict local-only, no network requests)."
            )

        except OSError as e:
            # OSError is what HF raises when local_files_only=True fails
            self._load_error = (
                f"ASR_MODEL_UNAVAILABLE: Could not load [{self.model_id}] "
                f"from local cache: {e}"
            )
            self.is_loaded = False
            logger.error(self._load_error)

        except Exception as e:
            self._load_error = f"ASR_LOAD_FAILED: Unexpected error loading [{self.model_id}]: {e}"
            self.is_loaded = False
            logger.error(self._load_error)

    def transcribe_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe a 16kHz mono audio waveform into text.

        Args:
            waveform: 1D numpy float32 array normalized to [-1.0, 1.0].
            sample_rate: Audio sampling rate (default: 16000 Hz).

        Returns:
            Dict containing transcript, confidence/status, and metadata.
        """
        # Do NOT retry loading — if it failed once, it stays failed
        if not self.is_loaded or self.pipeline is None:
            return {
                "transcript": "",
                "is_available": False,
                "state": "ASR_MODEL_UNAVAILABLE",
                "status": self._load_error or "ASR model not loaded (local cache missing).",
                "model_name": self.model_id,
                "duration_seconds": float(len(waveform) / max(sample_rate, 1)),
                "word_count": 0,
            }

        # Ensure 1D float32
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=0)
        waveform = waveform.astype(np.float32)

        duration = float(len(waveform) / max(sample_rate, 1))
        rms = float(np.sqrt(np.mean(waveform ** 2)))

        # If audio is virtually silent, don't run heavy decoding
        if rms < 0.005 or duration < 0.3:
            return {
                "transcript": "",
                "is_available": False,
                "status": "Audio is silent or too short for speech transcription.",
                "model_name": self.model_id,
                "provenance": "OpenAI Whisper Tiny (local cache, no network)",
                "duration_seconds": round(duration, 3),
                "word_count": 0,
            }

        try:
            t0 = time.time()
            audio_input = {"sampling_rate": sample_rate, "raw": waveform}
            result = self.pipeline(
                audio_input,
                generate_kwargs={"task": "transcribe", "language": "english"},
            )
            raw_text = (
                result.get("text", "").strip()
                if isinstance(result, dict)
                else str(result).strip()
            )
            elapsed = time.time() - t0

            # Filter trivial hallucinations or empty output
            cleaned_text = raw_text.strip()
            is_valid = bool(cleaned_text and len(cleaned_text) >= 2)

            return {
                "transcript": cleaned_text if is_valid else "",
                "is_available": is_valid,
                "status": (
                    "Speech Transcribed Successfully"
                    if is_valid
                    else "No intelligible speech detected in audio track."
                ),
                "model_name": self.model_id,
                "provenance": "OpenAI Whisper Tiny (local cache, no network)",
                "duration_seconds": round(duration, 3),
                "inference_time_seconds": round(elapsed, 3),
                "word_count": len(cleaned_text.split()) if is_valid else 0,
            }
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
            return {
                "transcript": "",
                "is_available": False,
                "status": f"Transcription error: {e}",
                "model_name": self.model_id,
                "duration_seconds": round(duration, 3),
                "word_count": 0,
            }


# ---------------------------------------------------------------------------
# Singleton instance
# ---------------------------------------------------------------------------
_asr_engine: Optional[ASREngine] = None


def get_asr_engine() -> ASREngine:
    """Retrieve singleton ASREngine instance."""
    global _asr_engine
    if _asr_engine is None:
        _asr_engine = ASREngine()
    return _asr_engine
