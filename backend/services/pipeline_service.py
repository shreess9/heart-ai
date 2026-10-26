"""Backend Service Layer wrapping Multimodal Pipeline and Dialogue Manager."""

from typing import Any, Dict, Optional
from pipeline.inference_pipeline import MultimodalInferencePipeline
from dialogue.schemas import StructuredAffectiveEvidence
from dialogue.manager import DialogueManager


class BackendPipelineService:
    """Service singleton managing inference pipelines and conversation states."""

    _instance = None

    def __init__(self):
        self.pipelines: Dict[str, MultimodalInferencePipeline] = {
            "mock_reliability_aware": MultimodalInferencePipeline(mock_mode=True, fusion_model_type="reliability_aware"),
            "mock_baseline": MultimodalInferencePipeline(mock_mode=True, fusion_model_type="baseline"),
            "mock_disagreement": MultimodalInferencePipeline(mock_mode=True, fusion_model_type="disagreement_aware"),
        }
        self.dialogue_manager = DialogueManager()

    @classmethod
    def get_instance(cls) -> "BackendPipelineService":
        if cls._instance is None:
            cls._instance = BackendPipelineService()
        return cls._instance

    def run_inference(
        self,
        text: str,
        audio_input: Optional[Any] = None,
        visual_input: Optional[Any] = None,
        fusion_type: str = "reliability_aware",
        mock_mode: bool = True,
    ) -> StructuredAffectiveEvidence:
        """Run multimodal inference and reliability analysis."""
        key = f"{'mock' if mock_mode else 'live'}_{fusion_type}"
        if key not in self.pipelines:
            self.pipelines[key] = MultimodalInferencePipeline(
                mock_mode=mock_mode, fusion_model_type=fusion_type
            )
        pipeline = self.pipelines[key]
        return pipeline.process_utterance(text, audio_input, visual_input)

    def generate_dialogue_reply(
        self,
        user_message: str,
        evidence: Optional[StructuredAffectiveEvidence] = None,
    ) -> Dict[str, Any]:
        """Generate conversational reply conditioned on affective evidence."""
        if evidence is None:
            evidence = self.run_inference(user_message, mock_mode=True)
        return self.dialogue_manager.generate_response(user_message, evidence)
