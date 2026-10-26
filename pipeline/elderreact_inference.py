"""
pipeline/elderreact_inference.py
==================================
ElderReact-trained multi-label emotion inference — integration layer.

DESIGN CONSTRAINTS (from integration plan)
-------------------------------------------
- Six INDEPENDENT emotion probabilities; never collapsed to a single argmax.
- Keeps MELD pipeline untouched; activated only when
  ``emotion_model = "elderreact"`` in the calling config or argument.
- Reliability policy is CONSERVATIVE:
    - High reliability requires per-emotion calibration AND cross-modal agreement.
    - High raw confidence on a poorly calibrated model does NOT trigger strong adaptation.
    - ECE for ER-A on ElderReact test is 0.212 (moderate), ER-B=0.261 (poor).
      Use ER-A as the default; ER-C as the optional richer variant.
- All thresholds and calibration constants are loaded from saved dev-tuned artifacts,
  never from test-set observations.
- Mock mode is always labelled clearly in output.

USAGE
-----
    from pipeline.elderreact_inference import ElderReactInferencePipeline
    pipe = ElderReactInferencePipeline(model_variant="ER-A")
    result = pipe.predict_from_features(audio_feat_148dim, video_feat_1418dim)

CONFIG FLAG
-----------
    emotion_model: "elderreact"   ← activates this module
    emotion_model: "meld"         ← uses existing MultimodalInferencePipeline (default)

CHECKPOINT LOCATION (read-only)
--------------------------------
    results/elderreact/official/checkpoints/er_a.pkl
    results/elderreact/official/thresholds.json
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Emotion labels (order is FIXED from official ElderReact annotations) ────
ELDERREACT_EMOTIONS: List[str] = [
    "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"
]
N_EMOTIONS = 6

# ── ECE reference values from test evaluation (used for calibration warning) ─
_ECE_REFERENCE = {"ER-A": 0.212, "ER-B": 0.261, "ER-C": 0.231}

# ── Default checkpoint directory (relative to project root) ──────────────────
_DEFAULT_CKPT_DIR = Path(__file__).parent.parent / "results" / "elderreact" / "official"


# ══════════════════════════════════════════════════════════════════════════════
# Output schema (dataclass — does not inherit from MELD schemas)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ElderReactEmotionOutput:
    """
    Multi-label emotion prediction output.  Never convert to single argmax.

    Fields
    ------
    probabilities : Dict[str, float]
        Independent sigmoid probability for each of the 6 emotions.
    active_emotions : List[str]
        Emotions whose probability meets or exceeds the dev-tuned threshold.
    thresholds : Dict[str, float]
        The dev-tuned per-emotion thresholds used for this prediction.
    model_variant : str
        Which checkpoint was used (ER-A, ER-B, ER-C).
    reliability_level : str
        Conservative assessment: "high" | "moderate" | "low".
    reliability_note : str
        Human-readable explanation of reliability assessment.
    is_mock : bool
        True when running without real features.
    raw_array : np.ndarray
        [6] array of float probabilities (for downstream reliability metrics).
    """
    probabilities: Dict[str, float]
    active_emotions: List[str]
    thresholds: Dict[str, float]
    model_variant: str
    reliability_level: str           # "high" | "moderate" | "low"
    reliability_note: str
    is_mock: bool = False
    raw_array: np.ndarray = field(default_factory=lambda: np.zeros(N_EMOTIONS))


# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction helpers (COVAREP + OpenFace subset)
# ══════════════════════════════════════════════════════════════════════════════

def extract_audio_subset(raw_148: np.ndarray) -> np.ndarray:
    """
    Extract the 72-dim COVAREP subset used during training.
    Input: raw 148-dim COVAREP feature vector.
    Returns: 72-dim array.
    """
    if raw_148.shape[-1] != 148:
        raise ValueError(f"Expected 148-dim audio feature, got {raw_148.shape[-1]}")
    return np.concatenate([raw_148[..., :36], raw_148[..., 74:110]], axis=-1)


def extract_video_subset(raw_1418: np.ndarray) -> np.ndarray:
    """
    Extract the 178-dim OpenFace subset used during training.
    Input: raw 1418-dim OpenFace feature vector.
    Returns: 178-dim array.
    """
    if raw_1418.shape[-1] != 1418:
        raise ValueError(f"Expected 1418-dim video feature, got {raw_1418.shape[-1]}")
    return np.concatenate([
        raw_1418[..., 0:8],
        raw_1418[..., 288:294],
        raw_1418[..., 634:717],
        raw_1418[..., 997:1003],
        raw_1418[..., 1343:1418],
    ], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Conservative reliability policy for multi-label predictions
# ══════════════════════════════════════════════════════════════════════════════

class ElderReactReliabilityPolicy:
    """
    Conservative reliability assessment for multi-label ElderReact predictions.

    Design rationale
    ----------------
    - The best ECE on ElderReact test is 0.212 (ER-A). This is moderate miscalibration.
      A raw sigmoid ≥ 0.70 should NOT be treated as reliably calibrated confidence.
    - High reliability requires: several active emotions agree with high probability,
      AND per-emotion spread (max-min) is not near the threshold, AND ECE is acceptable.
    - When only one emotion is marginally above threshold, output is LOW reliability.

    Parameters (all conservative defaults)
    ----------------------------------------
    high_min_prob      : float = 0.65 — minimum probability for "high" reliability
    high_min_margin    : float = 0.25 — probability must exceed threshold by this margin
    high_max_ece       : float = 0.25 — ECE must be below this for "high" reliability
    moderate_min_prob  : float = 0.45
    moderate_min_margin: float = 0.10
    """

    def __init__(
        self,
        high_min_prob:       float = 0.65,
        high_min_margin:     float = 0.25,
        high_max_ece:        float = 0.25,
        moderate_min_prob:   float = 0.45,
        moderate_min_margin: float = 0.10,
    ):
        self.high_min_prob       = high_min_prob
        self.high_min_margin     = high_min_margin
        self.high_max_ece        = high_max_ece
        self.moderate_min_prob   = moderate_min_prob
        self.moderate_min_margin = moderate_min_margin

    def assess(
        self,
        proba: np.ndarray,         # [6] sigmoid probabilities
        thresholds: np.ndarray,    # [6] dev-tuned thresholds
        model_variant: str,        # for ECE lookup
    ) -> Tuple[str, str]:
        """
        Return (level, note).  level ∈ {"high", "moderate", "low"}.

        NOTE: We deliberately do NOT call this "confident" merely because
        max(proba) is high. Calibration quality matters.
        """
        ece = _ECE_REFERENCE.get(model_variant, 0.30)
        margins = proba - thresholds           # [6] — positive = active
        active_mask = margins > 0
        n_active = active_mask.sum()

        # ── LOW: no active emotions, or all margins tiny, or ECE too high ─
        if n_active == 0:
            return "low", (
                "No emotion reached its dev-tuned threshold. "
                "Prediction is inconclusive; do not infer emotional state."
            )

        max_active_prob = float(proba[active_mask].max()) if n_active else 0.0
        max_margin      = float(margins[active_mask].max()) if n_active else 0.0

        if ece > self.high_max_ece:
            # Model is known to be moderately miscalibrated; cap at moderate
            if (max_active_prob >= self.moderate_min_prob
                    and max_margin >= self.moderate_min_margin):
                emotions_str = ", ".join(
                    e for e, m in zip(ELDERREACT_EMOTIONS, active_mask) if m
                )
                return "moderate", (
                    f"Active emotions: {emotions_str}. "
                    f"ECE={ece:.3f} (moderate miscalibration); "
                    "treat probabilities as indicative, not calibrated. "
                    "Use open-ended supportive language."
                )
            return "low", (
                f"ECE={ece:.3f} and low probability margins. "
                "Reliability is insufficient for emotional adaptation. "
                "Maintain neutral supportive presence."
            )

        # ECE is acceptable — allow high or moderate
        if (max_active_prob >= self.high_min_prob
                and max_margin >= self.high_min_margin):
            emotions_str = ", ".join(
                e for e, m in zip(ELDERREACT_EMOTIONS, active_mask) if m
            )
            return "high", (
                f"Active emotions: {emotions_str} with well-separated confidence. "
                "Acknowledge emotional tone warmly and empathetically, "
                "without medical or clinical assertions."
            )

        if (max_active_prob >= self.moderate_min_prob
                and max_margin >= self.moderate_min_margin):
            emotions_str = ", ".join(
                e for e, m in zip(ELDERREACT_EMOTIONS, active_mask) if m
            )
            return "moderate", (
                f"Possible emotions: {emotions_str} (moderate confidence). "
                "Use gentle, open-ended supportive language. "
                "Avoid strong assertions about emotional state."
            )

        return "low", (
            "Active emotion probabilities are close to threshold. "
            "Reliability is low; maintain neutral, warm, supportive tone."
        )


# ══════════════════════════════════════════════════════════════════════════════
# Main inference class
# ══════════════════════════════════════════════════════════════════════════════

class ElderReactInferencePipeline:
    """
    Loads a trained ElderReact checkpoint and produces multi-label
    emotion predictions from COVAREP+OpenFace features.

    Activated via config: ``emotion_model: "elderreact"``

    Parameters
    ----------
    model_variant : str
        One of "ER-A" (standard fusion, recommended), "ER-B" (disagreement-aware),
        "ER-C" (reliability-aware). Default: "ER-A" (best calibration on test set).
    ckpt_dir : str or Path
        Directory containing checkpoints and thresholds.json.
    mock_mode : bool
        If True, returns plausible random predictions without loading the checkpoint.
        Mock output is always labelled: reliability_level="low", is_mock=True.
    policy : ElderReactReliabilityPolicy or None
        Override the default conservative policy.
    """

    _VARIANT_MAP = {
        "ER-A": ("er_a.pkl", "ER_A"),
        "ER-B": ("er_b.pkl", "ER_B"),
        "ER-C": ("er_c.pkl", "ER_C"),
    }

    def __init__(
        self,
        model_variant: str = "ER-A",
        ckpt_dir: Optional[str] = None,
        mock_mode: bool = False,
        policy: Optional[ElderReactReliabilityPolicy] = None,
    ):
        if model_variant not in self._VARIANT_MAP:
            raise ValueError(f"model_variant must be one of {list(self._VARIANT_MAP)}")

        self.model_variant = model_variant
        self.mock_mode     = mock_mode
        self.policy        = policy or ElderReactReliabilityPolicy()

        ckpt_dir = Path(ckpt_dir) if ckpt_dir else _DEFAULT_CKPT_DIR
        pkl_name, thresh_key = self._VARIANT_MAP[model_variant]
        ckpt_path   = ckpt_dir / "checkpoints" / pkl_name
        thresh_path = ckpt_dir / "thresholds.json"

        if mock_mode:
            self._model      = None
            self._thresholds = np.full(N_EMOTIONS, 0.5)
            self._scaler_a   = None
            self._scaler_v   = None
            log.warning(
                "ElderReactInferencePipeline: mock_mode=True — "
                "predictions are random and clearly labelled as mock."
            )
            return

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                "Run experiment_elderreact_proper.py first."
            )

        # Load thresholds (dev-tuned, never test-tuned)
        thresh_data = json.loads(thresh_path.read_text())
        self._thresholds = np.array(thresh_data["thresholds"][thresh_key])
        log.info(f"Loaded {model_variant} thresholds (tuned on dev): "
                 f"{dict(zip(ELDERREACT_EMOTIONS, self._thresholds.round(3)))}")

        # Load scaler parameters — stored alongside checkpoint as metadata
        scaler_path = ckpt_dir / "checkpoints" / "scalers.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                scalers = pickle.load(f)
            self._scaler_a = scalers["audio"]
            self._scaler_v = scalers["video"]
        else:
            log.warning(
                "scalers.pkl not found — feature normalisation will be skipped. "
                "This may degrade prediction quality if raw features are provided. "
                "Run save_scalers() after experiment to persist scalers."
            )
            self._scaler_a = None
            self._scaler_v = None

        # Checkpoints were pickled with '__main__.MultiLabelMLP'.
        # Inject a compatible re-definition into __main__ before unpickling.
        import sys as _sys

        class _MultiLabelMLP:
            """Re-definition compatible with pickled checkpoints (saved as __main__.MultiLabelMLP)."""
            def __init__(self, hidden=(128, 64), seed=42):
                self.hidden = hidden; self.seed = seed; self.clfs = []
            def predict_proba(self, X):
                return np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])
            def predict(self, X, thresholds=None):
                p = self.predict_proba(X)
                t = thresholds if thresholds is not None else np.full(len(self.clfs), 0.5)
                return (p >= t).astype(int)

        # The pkl stores module='__main__', qualname='MultiLabelMLP'
        _main = _sys.modules.get("__main__")
        if _main is not None and not hasattr(_main, "MultiLabelMLP"):
            setattr(_main, "MultiLabelMLP", _MultiLabelMLP)
        # Also set locally in case __main__ lookup fails
        _sys.modules.setdefault(
            "__main__",
            type(_sys)("__main__")
        ).MultiLabelMLP = _MultiLabelMLP

        with open(ckpt_path, "rb") as f:
            self._model = pickle.load(f)

        log.info(f"Loaded {model_variant} checkpoint: "
                 f"{len(self._model.clfs)} classifiers from {ckpt_path}")

    # ── Public API ──────────────────────────────────────────────────────────

    def predict_from_features(
        self,
        audio_feat: np.ndarray,   # raw 148-dim COVAREP or 72-dim subset
        video_feat: np.ndarray,   # raw 1418-dim OpenFace or 178-dim subset
    ) -> ElderReactEmotionOutput:
        """
        Predict emotion probabilities from precomputed features.

        Parameters
        ----------
        audio_feat : array of shape (148,) or (72,)
        video_feat : array of shape (1418,) or (178,)

        Returns
        -------
        ElderReactEmotionOutput with 6 independent probabilities.
        """
        if self.mock_mode:
            return self._mock_output()

        # Extract subsets if raw features provided
        if audio_feat.shape[-1] == 148:
            audio_feat = extract_audio_subset(audio_feat)
        elif audio_feat.shape[-1] != 72:
            raise ValueError(f"audio_feat must be 148-dim (raw) or 72-dim (subset), got {audio_feat.shape[-1]}")

        if video_feat.shape[-1] == 1418:
            video_feat = extract_video_subset(video_feat)
        elif video_feat.shape[-1] != 178:
            raise ValueError(f"video_feat must be 1418-dim (raw) or 178-dim (subset), got {video_feat.shape[-1]}")

        # Normalise
        audio_feat = np.nan_to_num(audio_feat.flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        video_feat = np.nan_to_num(video_feat.flatten(), nan=0.0, posinf=0.0, neginf=0.0)

        if self._scaler_a is not None:
            audio_feat = self._scaler_a.transform(audio_feat.reshape(1, -1))[0]
        if self._scaler_v is not None:
            video_feat = self._scaler_v.transform(video_feat.reshape(1, -1))[0]

        X = np.concatenate([audio_feat, video_feat]).reshape(1, -1)
        proba = self._model.predict_proba(X)[0]   # [6] independent probabilities
        return self._build_output(proba)

    def predict_batch(
        self,
        audio_feats: np.ndarray,  # [N, 148] or [N, 72]
        video_feats: np.ndarray,  # [N, 1418] or [N, 178]
    ) -> List[ElderReactEmotionOutput]:
        return [
            self.predict_from_features(a, v)
            for a, v in zip(audio_feats, video_feats)
        ]

    def format_for_response_policy(self, output: ElderReactEmotionOutput) -> str:
        """
        Format a multi-label output as a directive string for the LLM prompt.
        Preserves all active emotions — does not collapse to single label.
        """
        level = output.reliability_level
        if output.is_mock:
            return (
                "[MOCK MODE — ElderReact pipeline not active]\n"
                "Do NOT use these predictions for emotional adaptation. "
                "Maintain a warm, neutral, supportive presence."
            )
        if not output.active_emotions:
            return (
                "[Affective Evidence: LOW RELIABILITY — No emotion detected above threshold]\n"
                "Maintain neutral, warm, supportive tone. Do not attribute emotional state."
            )
        emo_str = " | ".join(
            f"{e}:{output.probabilities[e]:.2f}" for e in output.active_emotions
        )
        return (
            f"[Affective Evidence: {level.upper()} RELIABILITY — ElderReact]\n"
            f"[Detected emotions: {emo_str}]\n"
            f"[Note: {output.reliability_note}]\n"
            "Respond naturally following these directives. "
            "Do not claim clinical or medical certainty about emotional state."
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_output(self, proba: np.ndarray) -> ElderReactEmotionOutput:
        assert proba.shape == (N_EMOTIONS,), f"Expected {N_EMOTIONS} probs, got {proba.shape}"
        proba_dict = dict(zip(ELDERREACT_EMOTIONS, proba.tolist()))
        thresh_dict = dict(zip(ELDERREACT_EMOTIONS, self._thresholds.tolist()))
        active = [e for e, p, t in zip(ELDERREACT_EMOTIONS, proba, self._thresholds) if p >= t]
        level, note = self.policy.assess(proba, self._thresholds, self.model_variant)
        return ElderReactEmotionOutput(
            probabilities=proba_dict,
            active_emotions=active,
            thresholds=thresh_dict,
            model_variant=self.model_variant,
            reliability_level=level,
            reliability_note=note,
            is_mock=False,
            raw_array=proba.copy(),
        )

    def _mock_output(self) -> ElderReactEmotionOutput:
        rng = np.random.default_rng()
        proba = rng.dirichlet(np.ones(N_EMOTIONS) * 0.5)
        proba_dict = dict(zip(ELDERREACT_EMOTIONS, proba.tolist()))
        thresh_dict = dict(zip(ELDERREACT_EMOTIONS, self._thresholds.tolist()))
        return ElderReactEmotionOutput(
            probabilities=proba_dict,
            active_emotions=[],
            thresholds=thresh_dict,
            model_variant=self.model_variant,
            reliability_level="low",
            reliability_note="MOCK MODE — predictions are random and not valid.",
            is_mock=True,
            raw_array=proba,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Scaler persistence utility (call once after training to enable inference)
# ══════════════════════════════════════════════════════════════════════════════

def save_scalers(scaler_a, scaler_v, ckpt_dir: Optional[str] = None):
    """
    Persist the train-fitted MinMaxScalers alongside the checkpoints.
    Call this once after running experiment_elderreact_proper.py.
    """
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else _DEFAULT_CKPT_DIR / "checkpoints"
    out = Path(ckpt_dir) / "scalers.pkl"
    with open(out, "wb") as f:
        pickle.dump({"audio": scaler_a, "video": scaler_v}, f)
    log.info(f"Scalers saved to {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Quick self-test (mock mode)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=== ElderReact Inference — Mock Self-Test ===")
    pipe = ElderReactInferencePipeline(mock_mode=True)
    a_feat = np.random.randn(148).astype(np.float32)
    v_feat = np.random.randn(1418).astype(np.float32)
    out = pipe.predict_from_features(a_feat, v_feat)
    print(f"  Model: {out.model_variant}")
    print(f"  Active emotions: {out.active_emotions}")
    print(f"  Reliability: {out.reliability_level}")
    print(f"  Is mock: {out.is_mock}")
    print(f"  Probabilities: {out.probabilities}")
    print(f"\nResponse policy directive:\n{pipe.format_for_response_policy(out)}")
    print("\nMock self-test PASSED")
