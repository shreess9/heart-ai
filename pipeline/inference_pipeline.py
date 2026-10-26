"""H.E.A.R.T. AI: Multimodal Inference & Reliability Pipeline.

Unifies Text, Audio, and Visual models with cross-modal disagreement,
predictive uncertainty, reliability-aware fusion, and conversational response policy.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch

from models.text.roberta import RobertaEmotionClassifier
from models.audio.wav2vec2 import Wav2Vec2EmotionClassifier
from models.visual.vit import ViTEmotionClassifier
from fusion.baseline import BaselineMultimodalMLP
from fusion.disagreement_aware import DisagreementAwareMLP
from fusion.reliability_aware import ReliabilityAwareFusion
from reliability.jsd import generalized_jsd_3, mean_pairwise_jsd_3
from reliability.categorical_disagreement import compute_categorical_disagreement_d3
from reliability.entropy import compute_predictive_entropy
from dialogue.schemas import (
    StructuredAffectiveEvidence,
    ModalityPredictionSchema,
    DisagreementSchema,
    UncertaintySchema,
    FusedPredictionSchema,
)
from dialogue.response_policy import AffectiveResponsePolicy
from dialogue.manager import DialogueManager

EMOTION_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]


class MultimodalInferencePipeline:
    """Unified multimodal inference & reliability analysis orchestrator."""

    def __init__(
        self,
        mock_mode: bool = True,
        fusion_model_type: str = "reliability_aware",
        device: str = "cpu",
    ):
        self.device = device
        self.mock_mode = mock_mode
        self.fusion_model_type = fusion_model_type

        # Initialize modality encoders
        self.text_model = RobertaEmotionClassifier(mock_mode=mock_mode).to(device)
        self.audio_model = Wav2Vec2EmotionClassifier(mock_mode=mock_mode).to(device)
        self.visual_model = ViTEmotionClassifier(mock_mode=mock_mode).to(device)

        # Initialize fusion architectures
        self.baseline_fusion = BaselineMultimodalMLP().to(device)
        self.disagreement_fusion = DisagreementAwareMLP().to(device)
        self.reliability_fusion = ReliabilityAwareFusion().to(device)

        # Response policy and dialogue manager
        self.policy = AffectiveResponsePolicy()
        self.dialogue_manager = DialogueManager(policy=self.policy)

    def process_utterance(
        self,
        text_input: str,
        audio_input: Optional[Any] = None,
        visual_input: Optional[Any] = None,
    ) -> StructuredAffectiveEvidence:
        """Execute full inference pipeline on a single multimodal turn."""
        if audio_input is None:
            # Default 1 second dummy waveform for test harness
            audio_input = np.random.randn(16000).astype(np.float32)
        if visual_input is None:
            # Default 224x224 RGB image for test harness
            visual_input = np.random.randn(3, 224, 224).astype(np.float32)

        # 1. Modality Feature Extraction
        t_feat = self.text_model.extract_features(text_input).to(self.device)
        a_feat = self.audio_model.extract_features(audio_input).to(self.device)
        v_feat = self.visual_model.extract_features(visual_input).to(self.device)

        # 2. Modality Probability Distributions
        t_probs = self.text_model.predict_distribution(t_feat)
        a_probs = self.audio_model.predict_distribution(a_feat)
        v_probs = self.visual_model.predict_distribution(v_feat)

        t_probs_np = t_probs[0].detach().cpu().numpy()
        a_probs_np = a_probs[0].detach().cpu().numpy()
        v_probs_np = v_probs[0].detach().cpu().numpy()

        t_id = int(np.argmax(t_probs_np))
        a_id = int(np.argmax(a_probs_np))
        v_id = int(np.argmax(v_probs_np))

        t_conf = float(t_probs_np[t_id])
        a_conf = float(a_probs_np[a_id])
        v_conf = float(v_probs_np[v_id])

        # 3. Disagreement Metrics
        d1_jsd = float(generalized_jsd_3(t_probs, a_probs, v_probs, normalized=True))
        d2_mean_jsd = float(mean_pairwise_jsd_3(t_probs, a_probs, v_probs, normalized=True))
        d3_cat = int(compute_categorical_disagreement_d3(t_id, a_id, v_id))

        # 4. Uncertainty / Entropy
        fused_probs_approx = (t_probs_np + a_probs_np + v_probs_np) / 3.0
        pred_entropy = float(compute_predictive_entropy(fused_probs_approx, normalized=False))
        norm_entropy = float(compute_predictive_entropy(fused_probs_approx, normalized=True))

        # 5. Multimodal Fusion
        if self.fusion_model_type == "baseline":
            fused_logits = self.baseline_fusion(t_feat, a_feat, v_feat)
            fused_probs = torch.softmax(fused_logits, dim=-1)[0].detach().cpu().numpy()
            cal_conf = float(np.max(fused_probs))
            model_name = self.baseline_fusion.name
        elif self.fusion_model_type == "disagreement_aware":
            fused_logits = self.disagreement_fusion(
                t_feat, a_feat, v_feat,
                text_probs=t_probs, audio_probs=a_probs, visual_probs=v_probs
            )
            fused_probs = torch.softmax(fused_logits, dim=-1)[0].detach().cpu().numpy()
            cal_conf = float(np.max(fused_probs))
            model_name = self.disagreement_fusion.name
        else: # reliability_aware
            raw_logits, cal_probs_t = self.reliability_fusion.forward_calibrated(
                t_feat, a_feat, v_feat,
                text_probs=t_probs, audio_probs=a_probs, visual_probs=v_probs
            )
            fused_probs = cal_probs_t[0].detach().cpu().numpy()
            cal_conf = float(np.max(fused_probs))
            model_name = self.reliability_fusion.name

        fused_id = int(np.argmax(fused_probs))
        fused_conf = float(fused_probs[fused_id])
        fused_emotion = EMOTION_LABELS[fused_id]

        # 6. Response Policy Evaluation
        rel_level = self.policy.evaluate_reliability_level(
            confidence=fused_conf,
            jsd=d1_jsd,
            normalized_entropy=norm_entropy,
            d3_categorical=d3_cat,
        )
        safety_guidance = self.policy.generate_safety_guidance(rel_level, fused_emotion)

        # 7. Assemble Structured Evidence
        evidence = StructuredAffectiveEvidence(
            utterance_text=text_input,
            text=ModalityPredictionSchema(
                modality="text",
                probabilities=t_probs_np.tolist(),
                predicted_emotion=EMOTION_LABELS[t_id],
                predicted_id=t_id,
                confidence=t_conf,
                is_mock=self.mock_mode,
            ),
            audio=ModalityPredictionSchema(
                modality="audio",
                probabilities=a_probs_np.tolist(),
                predicted_emotion=EMOTION_LABELS[a_id],
                predicted_id=a_id,
                confidence=a_conf,
                is_mock=self.mock_mode,
            ),
            visual=ModalityPredictionSchema(
                modality="visual",
                probabilities=v_probs_np.tolist(),
                predicted_emotion=EMOTION_LABELS[v_id],
                predicted_id=v_id,
                confidence=v_conf,
                is_mock=self.mock_mode,
            ),
            disagreement=DisagreementSchema(
                jsd=d1_jsd,
                jsd_pairwise_mean=d2_mean_jsd,
                categorical=d3_cat,
            ),
            uncertainty=UncertaintySchema(
                predictive_entropy=pred_entropy,
                normalized_entropy=norm_entropy,
            ),
            fused=FusedPredictionSchema(
                model_name=model_name,
                probabilities=fused_probs.tolist(),
                predicted_emotion=fused_emotion,
                predicted_id=fused_id,
                confidence=fused_conf,
                calibrated_confidence=cal_conf,
            ),
            reliability_level=rel_level,
            safety_guidance=safety_guidance,
            metadata={
                "mock_mode": self.mock_mode,
                "fusion_type": self.fusion_model_type,
            }
        )
        return evidence
