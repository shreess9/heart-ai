"""H.E.A.R.T. AI: Safe Multimodal Media Processor.

Handles:
1. Validation of uploaded audio and video formats & sizes.
2. Safe temporary storage with deterministic UUID filenames and auto-cleanup.
3. Audio decoding, mono conversion, 16kHz resampling, RMS energy & duration extraction.
4. Video decoding, controlled multi-frame sampling, resizing to 224x224, and audio-track extraction.
"""

import logging
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np
import soundfile as sf
import cv2

logger = logging.getLogger(__name__)

# Supported MIME extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".flv"}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit

# Ensure media temp files are written to spacious workspace disk
MEDIA_TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp_media"
MEDIA_TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_file_extension(filename: str) -> str:
    """Extract lowercase file extension including dot."""
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def validate_audio_file(filename: str, file_bytes: bytes, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """Validate audio file extension and payload size."""
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded audio file is empty (0 bytes).")
    
    if len(file_bytes) > max_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValueError(f"Audio file size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.")

    ext = get_file_extension(filename)
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio format '{ext}'. Supported formats: {supported_str}")


def validate_video_file(filename: str, file_bytes: bytes, max_size_bytes: int = MAX_FILE_SIZE_BYTES) -> None:
    """Validate video file extension and payload size."""
    if not file_bytes or len(file_bytes) == 0:
        raise ValueError("Uploaded video file is empty (0 bytes).")
    
    if len(file_bytes) > max_size_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValueError(f"Video file size ({size_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.")

    ext = get_file_extension(filename)
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_VIDEO_EXTENSIONS))
        raise ValueError(f"Unsupported video format '{ext}'. Supported formats: {supported_str}")


def _resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample 1D float32 audio array to target sample rate."""
    if orig_sr == target_sr:
        return waveform

    try:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        resampled = resample_poly(waveform, up, down).astype(np.float32)
        return resampled
    except Exception:
        # Fallback linear interpolation
        num_target_samples = int(len(waveform) * target_sr / orig_sr)
        indices = np.linspace(0, len(waveform) - 1, num_target_samples)
        return np.interp(indices, np.arange(len(waveform)), waveform).astype(np.float32)


def process_audio_bytes(
    audio_bytes: bytes,
    filename: str,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """Safely decode audio file, resample to 16kHz mono, and compute acoustic features."""
    validate_audio_file(filename, audio_bytes)
    ext = get_file_extension(filename) or ".wav"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, dir=str(MEDIA_TEMP_DIR), delete=False) as tf:
            temp_path = tf.name
            tf.write(audio_bytes)

        # Decode audio
        data = None
        sr = target_sr

        # Try soundfile
        try:
            data, sr = sf.read(temp_path, dtype="float32")
        except Exception as sf_err:
            logger.debug(f"Soundfile read failed ({sf_err}), trying scipy/wave/torchaudio...")
            try:
                import scipy.io.wavfile as sc_wav
                sr, data = sc_wav.read(temp_path)
                if data.dtype != np.float32:
                    if np.issubdtype(data.dtype, np.integer):
                        max_val = np.iinfo(data.dtype).max
                        data = data.astype(np.float32) / max_val
                    else:
                        data = data.astype(np.float32)
            except Exception as sc_err:
                try:
                    import torchaudio
                    t_wav, sr = torchaudio.load(temp_path)
                    data = t_wav.numpy().T
                except Exception as ta_err:
                    raise ValueError(f"Unable to decode audio format '{ext}': {sf_err}; {sc_err}; {ta_err}")

        # Convert to mono if multi-channel
        if data.ndim > 1:
            data = np.mean(data, axis=-1)

        data = data.astype(np.float32)

        # Resample to target_sr
        if sr != target_sr:
            data = _resample_audio(data, sr, target_sr)
            sr = target_sr

        # Compute properties
        num_samples = len(data)
        duration = float(num_samples / max(sr, 1))
        rms_energy = float(np.sqrt(np.mean(data ** 2))) if num_samples > 0 else 0.0

        return {
            "waveform": data,
            "sample_rate": sr,
            "duration_seconds": round(duration, 3),
            "num_samples": num_samples,
            "rms_energy": round(rms_energy, 4),
            "filename": os.path.basename(filename),
            "is_valid": True,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp audio file {temp_path}: {e}")


def process_video_bytes(
    video_bytes: bytes,
    filename: str,
    num_sample_frames: int = 5,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """Safely decode video file, extract representative sampled frames, and extract audio track if present."""
    validate_video_file(filename, video_bytes)
    ext = get_file_extension(filename) or ".mp4"

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, dir=str(MEDIA_TEMP_DIR), delete=False) as tf:
            temp_path = tf.name
            tf.write(video_bytes)

        # 1. Video Frame Processing with OpenCV
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file '{filename}' for decoding.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = float(total_frames / max(fps, 1.0)) if total_frames > 0 else 0.0

        if total_frames <= 0:
            cap.release()
            raise ValueError(f"Video file '{filename}' contains 0 decodable frames.")

        # Representative frame sampling indices
        n_samples = max(1, min(num_sample_frames, total_frames))
        if total_frames <= n_samples:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = [int(idx) for idx in np.linspace(0, total_frames - 1, n_samples)]

        sampled_frames = []
        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Convert BGR -> RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to standard ViT resolution (224, 224)
                resized = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_AREA)
                normalized = resized.astype(np.float32) / 255.0
                sampled_frames.append(normalized)

        cap.release()

        if not sampled_frames:
            raise ValueError(f"Failed to extract any valid visual frames from video '{filename}'.")

        # 2. Video Audio Track Extraction
        has_audio_track = False
        audio_waveform = None
        audio_duration = 0.0
        audio_status = "No audio track detected in video file"

        # Method A: imageio_ffmpeg + soundfile (fast, robust, supports all container formats)
        try:
            import imageio_ffmpeg
            import subprocess
            import soundfile as sf
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            temp_wav_path = str(MEDIA_TEMP_DIR / f"temp_aud_{os.path.basename(temp_path)}.wav")
            
            cmd = [
                ffmpeg_exe, "-y", "-i", temp_path,
                "-vn", "-ac", "1", "-ar", str(target_sr),
                "-f", "wav", temp_wav_path
            ]
            run_res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if run_res.returncode == 0 and os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 100:
                audio_data, sr = sf.read(temp_wav_path, dtype="float32")
                if audio_data is not None and len(audio_data) > 0:
                    if audio_data.ndim > 1:
                        audio_waveform = np.mean(audio_data, axis=-1).astype(np.float32)
                    else:
                        audio_waveform = audio_data.astype(np.float32)
                    has_audio_track = True
                    audio_duration = float(len(audio_waveform) / target_sr)
                    audio_status = f"Extracted from video audio track ({audio_duration:.1f}s)"
                try:
                    os.remove(temp_wav_path)
                except Exception:
                    pass
        except Exception as ffmpeg_err:
            logger.debug(f"FFmpeg audio extraction failed: {ffmpeg_err}")

        # Method B: Fallback to MoviePy
        if not has_audio_track:
            try:
                try:
                    from moviepy import VideoFileClip
                except ImportError:
                    import moviepy.editor as mp
                    VideoFileClip = mp.VideoFileClip
                
                video_clip = VideoFileClip(temp_path)
                if video_clip.audio is not None:
                    has_audio_track = True
                    audio_soundarray = video_clip.audio.to_soundarray(fps=target_sr)
                    if audio_soundarray is not None and len(audio_soundarray) > 0:
                        if audio_soundarray.ndim > 1:
                            audio_waveform = np.mean(audio_soundarray, axis=-1).astype(np.float32)
                        else:
                            audio_waveform = audio_soundarray.astype(np.float32)
                        audio_duration = float(len(audio_waveform) / target_sr)
                        audio_status = f"Extracted from video audio track ({audio_duration:.1f}s)"
                video_clip.close()
            except Exception as aud_err:
                logger.debug(f"MoviePy audio fallback also failed: {aud_err}")
                if not has_audio_track:
                    has_audio_track = False
                    audio_status = "No audio track detected or audio stream unsupported"

        return {
            "frames": sampled_frames,
            "num_frames_sampled": len(sampled_frames),
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(duration, 3),
            "resolution": (width, height),
            "has_audio_track": has_audio_track,
            "audio_waveform": audio_waveform,
            "audio_duration_seconds": round(audio_duration, 3),
            "audio_status": audio_status,
            "filename": os.path.basename(filename),
            "is_valid": True,
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp video file {temp_path}: {e}")
