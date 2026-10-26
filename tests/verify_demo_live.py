"""End-to-End Live Verification Script for H.E.A.R.T. AI Demonstrator."""

import json
import urllib.request

def run_live_verification():
    base_url = "http://127.0.0.1:8000"

    print("--- 1. Testing GET / (HTML Index) ---")
    with urllib.request.urlopen(f"{base_url}/") as response:
        assert response.getcode() == 200
        html = response.read().decode('utf-8')
        assert "H.E.A.R.T. AI" in html
        assert "user-utterance-input" in html
        assert "meld-sample-select" in html
        assert "results-idle" in html
        assert "gate-strong" in html
        print("[PASS] HTML index structure and all required element IDs verified.")

    print("\n--- 2. Testing GET /api/demo/meld_samples ---")
    with urllib.request.urlopen(f"{base_url}/api/demo/meld_samples?limit=10") as response:
        assert response.getcode() == 200
        samples = json.loads(response.read().decode('utf-8'))
        assert len(samples) >= 10
        print(f"[PASS] Retrieved {len(samples)} MELD test set samples (Sample 0: {samples[0]['sample_id']} - '{samples[0]['utterance'][:40]}...')")

    print("\n--- 3. Testing Mode 1: Custom User Utterance Analysis (Exact Text Transmission) ---")
    test_sentences = [
        ("Today was a beautiful day. I enjoyed talking to my grandchildren.", "joy", "strong"),
        ("I don't really want to talk to anyone today. I feel lonely.", "sadness", "cautious"),
        ("The weather is nice, but I am worried about tomorrow.", "fear", "cautious"),
    ]

    for sent, exp_emo, exp_adapt in test_sentences:
        payload = {"text": sent, "mode": "custom"}
        req = urllib.request.Request(
            f"{base_url}/api/demo/analyze",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.getcode() == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["analyzed_utterance"] == sent, f"Expected '{sent}', got '{data['analyzed_utterance']}'"
            assert data["mode"] == "custom"
            assert data["modalities"]["text"]["is_available"] is True
            assert data["modalities"]["audio"]["is_available"] is False
            assert data["modalities"]["visual"]["is_available"] is False
            print(f"[PASS] Analyzed exact sentence: '{sent[:45]}...' -> Emotion: {data['fusion']['emotion']}, Reliability: {data['reliability']['level']}, Adaptation: {data['response_policy']['adaptation']}")

    print("\n--- 4. Testing Mode 2: MELD Sample Selection Analysis ---")
    meld_payload = {
        "mode": "meld",
        "sample_id": "dia0_utt0",
        "text": "Why do all you’re coffee mugs have numbers on the bottom?"
    }
    req = urllib.request.Request(
        f"{base_url}/api/demo/analyze",
        data=json.dumps(meld_payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.getcode() == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert data["mode"] == "meld"
        assert data["sample_id"] == "dia0_utt0"
        assert "d1" in data["disagreement"]
        assert "d3" in data["disagreement"]
        print(f"[PASS] MELD sample analysis: {data['sample_id']} -> Fused Emotion: {data['fusion']['emotion']}, D1={data['disagreement']['d1']}, D3={data['disagreement']['d3']}")

    print("\n--- 5. Testing Mode 3: Controlled Demonstration Scenarios ---")
    scenarios = ["strong_agreement", "moderate_disagreement", "strong_conflict", "low_confidence"]
    for sc in scenarios:
        payload = {"mode": "demo", "scenario_key": sc}
        req = urllib.request.Request(
            f"{base_url}/api/demo/analyze",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.getcode() == 200
            data = json.loads(resp.read().decode('utf-8'))
            assert data["mode"] == "demo"
            assert "DEMONSTRATION SCENARIO" in data["demo_notice"]
            print(f"[PASS] Scenario '{sc}': Disagreement={data['disagreement']['level']}, Reliability={data['reliability']['level']}, Policy={data['response_policy']['adaptation']}")

    print("\n========================================================")
    print("ALL LIVE DEMONSTRATOR END-TO-END CHECKS PASSED PERFECTLY!")
    print("========================================================")

if __name__ == "__main__":
    run_live_verification()
