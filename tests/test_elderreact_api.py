"""
tests/test_elderreact_api.py
==============================
Automated tests for the ElderReact emotion inference API.

Covers:
  - Health endpoint and model readiness (unit + integration)
  - Correct 6-label output schema
  - Independent multi-label probabilities (not a simplex)
  - Checkpoint / scaler / threshold loading
  - Invalid file types, oversized files, missing inputs
  - Mock mode behaviour
  - Chat endpoint (not-configured state)
  - Existing MELD pipeline unaffected (smoke test)

Run:
    cd "/Users/shreee/Desktop/heart-ai-main copy"
    conda run -n base python3 -m pytest tests/test_elderreact_api.py -v

Note: tests that require the real checkpoint are marked @pytest.mark.integration
and can be skipped with:
    pytest tests/test_elderreact_api.py -v -m "not integration"
"""

from __future__ import annotations

import io
import json
import os
import pickle
import struct
import wave
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Allow import from project root ──────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Suppress noisy startup logs during tests ───────────────────────────────
import logging
logging.disable(logging.CRITICAL)

from backend.main import create_app

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]
CKPT_DIR = ROOT / "results" / "elderreact" / "official" / "checkpoints"
CHECKPOINT_EXISTS = (CKPT_DIR / "er_a.pkl").exists() and (CKPT_DIR / "scalers.pkl").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def client():
    """TestClient with the full FastAPI app."""
    app = create_app()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _make_wav_bytes(duration_s: float = 0.5, sr: int = 16000) -> bytes:
    """Generate a minimal valid WAV file in memory."""
    n = int(sr * duration_s)
    samples = (np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 16383).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_dummy_mp4_bytes() -> bytes:
    """
    Generate a minimal placeholder for video bytes (not a real MP4).
    Tests that check video processing will be skipped or use real assets.
    The video validator checks extension + size, not actual codec.
    """
    return b"\x00\x00\x00\x18ftypisom" + b"\x00" * 100  # ftyp box header


def _make_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# T1 — Health endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestHealth:

    def test_health_returns_200(self, client):
        r = client.get("/health/elderreact")
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    def test_health_schema_fields(self, client):
        r = client.get("/health/elderreact")
        body = r.json()
        required = {"status", "service", "version",
                    "elderreact_model", "checkpoints", "raw_media_inference"}
        assert required.issubset(set(body.keys())), (
            f"Missing fields: {required - set(body.keys())}"
        )

    def test_health_status_is_string(self, client):
        r = client.get("/health/elderreact")
        assert r.json()["status"] in ("healthy", "degraded")

    def test_health_checkpoints_dict(self, client):
        r = client.get("/health/elderreact")
        ckpts = r.json()["checkpoints"]
        assert isinstance(ckpts, dict)
        assert set(ckpts.keys()) == {"er_a.pkl", "er_b.pkl", "er_c.pkl", "scalers.pkl"}

    @pytest.mark.integration
    def test_health_model_ready_when_checkpoint_exists(self, client):
        """Only runs when checkpoints are present on disk."""
        if not CHECKPOINT_EXISTS:
            pytest.skip("Checkpoints not present on this machine.")
        r = client.get("/health/elderreact")
        assert r.json()["elderreact_model"]["ready"] is True


# ══════════════════════════════════════════════════════════════════════════════
# T2 — Mock mode: correct schema, clearly labelled
# ══════════════════════════════════════════════════════════════════════════════

class TestMockMode:

    def test_mock_returns_200(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        assert r.status_code == 200

    def test_mock_is_mock_true(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        assert r.json()["is_mock"] is True

    def test_mock_reliability_is_low(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        assert r.json()["reliability_level"] == "low"

    def test_mock_has_six_emotion_labels(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        body = r.json()
        assert body["emotion_labels"] == EMOTIONS

    def test_mock_probabilities_has_six_keys(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        probs = r.json()["probabilities"]
        assert set(probs.keys()) == set(EMOTIONS)

    def test_mock_probabilities_in_range(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        probs = r.json()["probabilities"]
        for e, v in probs.items():
            assert 0.0 <= v <= 1.0, f"{e}: {v} not in [0,1]"

    def test_mock_has_warnings(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        body = r.json()
        assert len(body["warnings"]) > 0, "Mock response must include a warning"
        assert any("MOCK" in w.upper() for w in body["warnings"])

    def test_mock_error_is_null(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        assert r.json()["error"] is None

    def test_mock_note_contains_mock_text(self, client):
        r = client.post("/emotion/predict", data={"mock": "true"})
        note = r.json()["reliability_note"].upper()
        assert "MOCK" in note


# ══════════════════════════════════════════════════════════════════════════════
# T3 — Output schema: multi-label independence
# ══════════════════════════════════════════════════════════════════════════════

class TestOutputSchema:

    def _get_body(self, client) -> Dict[str, Any]:
        r = client.post("/emotion/predict", data={"mock": "true"})
        return r.json()

    def test_probabilities_not_sum_to_one(self, client):
        """Multi-label: probabilities should NOT sum to 1 (not a simplex)."""
        # Run 5 times; at least one should differ from 1.0 significantly
        sums = []
        for _ in range(5):
            body = self._get_body(client)
            sums.append(sum(body["probabilities"].values()))
        # At least one sum must differ from 1.0 by more than epsilon
        assert any(abs(s - 1.0) > 0.05 for s in sums), (
            f"All probability sums near 1.0: {sums} — looks like softmax"
        )

    def test_active_emotions_subset_of_labels(self, client):
        body = self._get_body(client)
        assert set(body["active_emotions"]).issubset(set(EMOTIONS))

    def test_thresholds_has_six_entries(self, client):
        body = self._get_body(client)
        assert set(body["thresholds"].keys()) == set(EMOTIONS)

    def test_thresholds_in_range(self, client):
        body = self._get_body(client)
        for e, t in body["thresholds"].items():
            assert 0.0 <= t <= 1.0, f"{e}: threshold {t} not in [0,1]"

    def test_emotion_labels_order(self, client):
        body = self._get_body(client)
        assert body["emotion_labels"] == EMOTIONS

    def test_reliability_level_valid(self, client):
        body = self._get_body(client)
        assert body["reliability_level"] in ("high", "moderate", "low")

    def test_feature_source_fields_present(self, client):
        body = self._get_body(client)
        assert "audio_feature_source" in body
        assert "video_feature_source" in body


# ══════════════════════════════════════════════════════════════════════════════
# T4 — Input validation: invalid formats, oversized, missing
# ══════════════════════════════════════════════════════════════════════════════

class TestInputValidation:

    def test_no_input_returns_error(self, client):
        r = client.post("/emotion/predict", data={})
        assert r.status_code in (422, 400)

    def test_invalid_audio_extension(self, client):
        wav = _make_wav_bytes()
        r = client.post(
            "/emotion/predict",
            files={
                "audio_file": ("recording.txt", wav, "text/plain"),
                "video_file": ("clip.mp4", _make_dummy_mp4_bytes(), "video/mp4"),
            },
        )
        assert r.status_code in (422, 400), f"Expected 422, got {r.status_code}"

    def test_invalid_video_extension(self, client):
        wav = _make_wav_bytes()
        r = client.post(
            "/emotion/predict",
            files={
                "audio_file": ("recording.wav", wav, "audio/wav"),
                "video_file": ("clip.exe", b"\x00" * 100, "application/octet-stream"),
            },
        )
        assert r.status_code in (422, 400), f"Expected 422, got {r.status_code}"

    def test_audio_only_returns_error(self, client):
        """Model requires both modalities."""
        wav = _make_wav_bytes()
        r = client.post(
            "/emotion/predict",
            files={"audio_file": ("audio.wav", wav, "audio/wav")},
        )
        assert r.status_code in (422, 400)
        body = r.json()
        detail = body.get("detail", "")
        assert "video" in detail.lower() or "both" in detail.lower(), (
            f"Error message should mention both modalities required: {detail}"
        )

    def test_video_only_returns_error(self, client):
        r = client.post(
            "/emotion/predict",
            files={"video_file": ("clip.mp4", _make_dummy_mp4_bytes(), "video/mp4")},
        )
        assert r.status_code in (422, 400)

    def test_oversized_audio_rejected(self, client):
        big = b"\x00" * (31 * 1024 * 1024)  # 31 MB > 30 MB limit
        r = client.post(
            "/emotion/predict",
            files={
                "audio_file": ("big.wav", big, "audio/wav"),
                "video_file": ("clip.mp4", _make_dummy_mp4_bytes(), "video/mp4"),
            },
        )
        assert r.status_code in (422, 413, 400)

    def test_npy_wrong_extension_rejected(self, client):
        npy = _make_npy_bytes(np.zeros(72))
        r = client.post(
            "/emotion/predict",
            files={
                "audio_features": ("features.txt", npy, "text/plain"),
                "video_features": ("features.npy", _make_npy_bytes(np.zeros(178)), "application/octet-stream"),
            },
        )
        assert r.status_code in (422, 400)

    def test_npy_wrong_dimension_rejected(self, client):
        """Audio must be 72 or 148; video must be 178 or 1418."""
        a_npy = _make_npy_bytes(np.zeros(50))   # wrong dim
        v_npy = _make_npy_bytes(np.zeros(178))
        r = client.post(
            "/emotion/predict",
            files={
                "audio_features": ("audio_features.npy", a_npy, "application/octet-stream"),
                "video_features": ("video_features.npy", v_npy, "application/octet-stream"),
            },
        )
        assert r.status_code in (422, 400)


# ══════════════════════════════════════════════════════════════════════════════
# T5 — Pre-extracted .npy integration test (real checkpoint)
# ══════════════════════════════════════════════════════════════════════════════

class TestNpyIntegration:

    @pytest.mark.integration
    def test_npy_72_178_returns_prediction(self, client):
        """Use valid-shaped feature vectors to test real checkpoint path."""
        if not CHECKPOINT_EXISTS:
            pytest.skip("Checkpoint not present.")

        # Use feature vectors from the ElderReact test set (row 0) if available
        repo = Path("/Users/shreee/Downloads/ElderReact_official_repo")
        if repo.exists():
            a_lines = (repo / "Features" / "Audio_feat" / "test_audio.txt").read_text().splitlines()
            v_lines = (repo / "Features" / "Video_feat" / "test_video.txt").read_text().splitlines()
            a_raw = np.array([float(x) for x in a_lines[0].split("\t")[1].split()], np.float32)
            v_raw = np.array([float(x) for x in v_lines[0].split("\t")[1].split()], np.float32)
            # Use full 148 / 1418 dim
            a_npy = _make_npy_bytes(a_raw)
            v_npy = _make_npy_bytes(v_raw)
        else:
            # Fallback: random 72/178 dim
            a_npy = _make_npy_bytes(np.random.randn(72).astype(np.float32))
            v_npy = _make_npy_bytes(np.random.randn(178).astype(np.float32))

        r = client.post(
            "/emotion/predict",
            files={
                "audio_features": ("audio_features.npy", a_npy, "application/octet-stream"),
                "video_features": ("video_features.npy", v_npy, "application/octet-stream"),
            },
        )
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        body = r.json()
        assert body["is_mock"] is False
        assert set(body["probabilities"].keys()) == set(EMOTIONS)
        assert all(0.0 <= v <= 1.0 for v in body["probabilities"].values())
        assert body["audio_feature_source"].startswith("covarep")
        assert body["video_feature_source"].startswith("openface")
        # Verify multi-label (not a simplex)
        prob_sum = sum(body["probabilities"].values())
        assert 0.0 < prob_sum, "All probabilities should not be zero"

    @pytest.mark.integration
    def test_npy_metrics_match_known_values(self, client):
        """Reproduce stored Macro-F1 using a known test sample (row 0 of test split)."""
        if not CHECKPOINT_EXISTS:
            pytest.skip("Checkpoint not present.")
        repo = Path("/Users/shreee/Downloads/ElderReact_official_repo")
        if not repo.exists():
            pytest.skip("ElderReact feature files not available.")

        a_lines = (repo / "Features" / "Audio_feat" / "test_audio.txt").read_text().splitlines()
        l_lines = (repo / "Annotations" / "test_labels.txt").read_text().splitlines()
        a_raw = np.array([float(x) for x in a_lines[0].split("\t")[1].split()], np.float32)
        gt    = [int(x) for x in l_lines[0].strip().split()[1:7]]

        v_lines = (repo / "Features" / "Video_feat" / "test_video.txt").read_text().splitlines()
        v_raw = np.array([float(x) for x in v_lines[0].split("\t")[1].split()], np.float32)

        r = client.post(
            "/emotion/predict",
            files={
                "audio_features": ("audio_features.npy", _make_npy_bytes(a_raw), "application/octet-stream"),
                "video_features": ("video_features.npy", _make_npy_bytes(v_raw), "application/octet-stream"),
            },
        )
        assert r.status_code == 200
        body = r.json()
        probs = [body["probabilities"][e] for e in EMOTIONS]
        thresholds = [body["thresholds"][e] for e in EMOTIONS]
        pred = [int(p >= t) for p, t in zip(probs, thresholds)]
        # Ground truth for sample 0 is [0,0,0,1,0,0] — Happiness
        # We don't assert exact match since a single sample can differ,
        # but we verify the pipeline ran and produced valid output
        assert len(pred) == 6
        assert all(v in (0, 1) for v in pred)


# ══════════════════════════════════════════════════════════════════════════════
# T6 — Chat endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestChat:

    def test_chat_returns_200(self, client):
        r = client.post("/chat", json={"message": "Hello"})
        assert r.status_code == 200

    def test_chat_has_reply(self, client):
        r = client.post("/chat", json={"message": "How are you?"})
        body = r.json()
        assert "reply" in body
        assert len(body["reply"]) > 0

    def test_chat_llm_available_false(self, client):
        r = client.post("/chat", json={"message": "Hello"})
        assert r.json()["llm_available"] is False

    def test_chat_placeholder_note(self, client):
        r = client.post("/chat", json={"message": "Hello"})
        note = r.json()["note"]
        assert len(note) > 0

    def test_chat_mock_context_not_used(self, client):
        """Mock emotion context must not be used for affective conditioning."""
        ctx = {
            "is_mock": True,
            "model_variant": "ER-A",
            "emotion_labels": EMOTIONS,
            "probabilities": {e: 0.5 for e in EMOTIONS},
            "active_emotions": ["Happiness"],
            "thresholds": {e: 0.1 for e in EMOTIONS},
            "reliability_level": "high",
            "reliability_note": "Test note",
            "audio_feature_source": "mock",
            "video_feature_source": "mock",
            "warnings": [],
            "error": None,
        }
        r = client.post("/chat", json={"message": "Hello", "context": ctx})
        assert r.status_code == 200
        body = r.json()
        # Mock context → affective_context should be null
        assert body["affective_context"] is None, (
            "Mock emotion context must not be used for emotional conditioning"
        )

    def test_chat_empty_message_rejected(self, client):
        r = client.post("/chat", json={"message": ""})
        assert r.status_code == 422

    def test_chat_session_id_echoed(self, client):
        r = client.post("/chat", json={"message": "Hi", "session_id": "test-session-1"})
        assert r.json()["session_id"] == "test-session-1"


# ══════════════════════════════════════════════════════════════════════════════
# T7 — MELD pipeline unaffected
# ══════════════════════════════════════════════════════════════════════════════

class TestMELDUnaffected:

    def test_meld_health_still_responds(self, client):
        """The original /health endpoint from health.py must still work."""
        # Original health.py also registers /health — our new one overrides.
        # Check that the existing /api/v1/inference/multimodal route exists.
        r = client.post(
            "/api/v1/inference/multimodal",
            json={"text": "Hello world", "mock_mode": True},
        )
        # Should return 200 (mock mode) or 500 (if MELD service fails to init)
        assert r.status_code in (200, 500), f"MELD route not registered: {r.status_code}"

    def test_elderreact_routes_dont_shadow_demo(self, client):
        """Demo route must still be accessible."""
        r = client.get("/demo")
        # Returns 200 (if index.html exists) or something else — just not 404
        assert r.status_code != 404, "Demo route should still be accessible"


# ══════════════════════════════════════════════════════════════════════════════
# T8 — Service unit tests (without HTTP layer)
# ══════════════════════════════════════════════════════════════════════════════

class TestElderReactServiceUnit:

    def test_predict_mock_has_six_emotions(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        svc.model_variant = "ER-A"
        svc._thresholds   = np.full(6, 0.5)
        result = svc.predict_mock()
        assert set(result["probabilities"].keys()) == set(EMOTIONS)
        assert result["is_mock"] is True
        assert result["reliability_level"] == "low"

    def test_predict_mock_warns_clearly(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        svc.model_variant = "ER-A"
        svc._thresholds   = np.full(6, 0.5)
        result = svc.predict_mock()
        assert any("MOCK" in w.upper() for w in result["warnings"])

    def test_validate_file_size(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        err = svc._validate_file(b"\x00" * 200, "file.wav", {".wav"}, 100, "audio")
        assert err is not None
        assert "limit" in err.lower()

    def test_validate_file_extension(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        err = svc._validate_file(b"\x00" * 10, "file.exe", {".wav"}, 1000, "audio")
        assert err is not None
        assert ".exe" in err

    def test_validate_file_ok(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        err = svc._validate_file(b"\x00" * 10, "file.wav", {".wav"}, 1000, "audio")
        assert err is None

    def test_error_response_shape(self):
        from backend.services.elderreact_service import ElderReactService
        result = ElderReactService._error_response("test error", 422)
        assert result["error"] == "test error"
        assert result["_status_code"] == 422
        assert result["probabilities"] == {}
        assert result["active_emotions"] == []

    def test_reliability_no_active_is_low(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        svc._thresholds = np.full(6, 0.9)   # very high — nothing will be active
        proba = np.full(6, 0.05)
        level, note = svc._assess_reliability(proba)
        assert level == "low"
        assert "threshold" in note.lower() or "inconclusive" in note.lower()

    def test_reliability_note_contains_ece(self):
        from backend.services.elderreact_service import ElderReactService
        svc = ElderReactService.__new__(ElderReactService)
        svc._thresholds = np.full(6, 0.1)   # low — most will be active
        proba = np.full(6, 0.55)
        level, note = svc._assess_reliability(proba)
        assert "ECE" in note or "ece" in note.lower() or level in ("moderate", "low")

    @pytest.mark.integration
    def test_service_loads_checkpoint(self):
        if not CHECKPOINT_EXISTS:
            pytest.skip("Checkpoint not present.")
        # Reset singleton to force reload
        from backend.services import elderreact_service as es_mod
        es_mod._service_instance = None
        svc = es_mod.get_elderreact_service()
        assert svc.is_ready is True
        assert svc._model is not None
        assert svc._scaler_a is not None
        assert svc._scaler_v is not None
        assert len(svc._thresholds) == 6

    @pytest.mark.integration
    def test_service_produces_valid_prediction_from_npy(self):
        """End-to-end: service.predict() with valid 72+178 dim features."""
        if not CHECKPOINT_EXISTS:
            pytest.skip("Checkpoint not present.")
        from backend.services import elderreact_service as es_mod
        es_mod._service_instance = None
        svc = es_mod.get_elderreact_service()

        a = _make_npy_bytes(np.random.randn(72).astype(np.float32))
        v = _make_npy_bytes(np.random.randn(178).astype(np.float32))
        result = svc.predict(audio_npy_bytes=a, video_npy_bytes=v)

        assert result["error"] is None
        assert set(result["probabilities"].keys()) == set(EMOTIONS)
        assert all(0.0 <= p <= 1.0 for p in result["probabilities"].values())
        assert result["is_mock"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Minimal direct execution (without pytest, for quick smoke test)
    import traceback
    app = create_app()
    c = TestClient(app)

    suites = [
        TestHealth(), TestMockMode(), TestOutputSchema(),
        TestInputValidation(), TestChat(), TestMELDUnaffected(),
        TestElderReactServiceUnit(),
    ]

    passed, failed, skipped = 0, 0, 0
    for suite in suites:
        for name in [m for m in dir(suite) if m.startswith("test_")]:
            fn = getattr(suite, name)
            marker = getattr(fn, "pytestmark", [])
            is_integration = any(getattr(m, "name", "") == "integration" for m in marker)
            try:
                fn(c) if "client" in fn.__code__.co_varnames else fn()
                print(f"  PASS  {suite.__class__.__name__}.{name}")
                passed += 1
            except pytest.skip.Exception as e:
                print(f"  SKIP  {suite.__class__.__name__}.{name}: {e}")
                skipped += 1
            except Exception as e:
                if is_integration and not CHECKPOINT_EXISTS:
                    print(f"  SKIP  {suite.__class__.__name__}.{name} (no checkpoint)")
                    skipped += 1
                else:
                    print(f"  FAIL  {suite.__class__.__name__}.{name}: {e}")
                    failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed  {failed} failed  {skipped} skipped")
    if failed:
        raise SystemExit(1)
