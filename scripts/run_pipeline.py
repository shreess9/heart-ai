#!/usr/bin/env python3
"""H.E.A.R.T. AI: Vertical Proof-of-Concept Multimodal Inference Pipeline CLI.

Demonstrates the complete end-to-end execution:
Text (RoBERTa) + Audio (Wav2Vec 2.0) + Visual (ViT)
-> Modality Emotion Distributions
-> JSD Disagreement (D1, D2) + Categorical Conflict (D3)
-> Predictive Uncertainty / Shannon Entropy
-> Reliability-Aware Fusion & Temperature Scaling
-> Safety Response Policy & Structured Affective Evidence JSON.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pipeline.inference_pipeline import MultimodalInferencePipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(description="H.E.A.R.T. AI Vertical Multimodal Inference Pipeline")
    parser.add_argument("--text", type=str, default="I'm feeling a little quiet today, but it's nice to chat with you.", help="Input text utterance")
    parser.add_argument("--mock", action="store_true", default=True, help="Run in mock/test-harness mode")
    parser.add_argument("--fusion", type=str, default="reliability_aware", choices=["baseline", "disagreement_aware", "reliability_aware"])
    parser.add_argument("--json-only", action="store_true", help="Print only raw JSON output")
    args = parser.parse_args()

    pipeline = MultimodalInferencePipeline(
        mock_mode=args.mock,
        fusion_model_type=args.fusion,
    )

    evidence = pipeline.process_utterance(text_input=args.text)
    evidence_dict = evidence.model_dump()

    if args.json_only:
        print(json.dumps(evidence_dict, indent=2))
        return

    # Rich formatted output
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]H.E.A.R.T. AI Multimodal Affective Evidence Pipeline[/bold cyan]\n"
        f"[dim]Mode: {'MOCK TEST HARNESS' if args.mock else 'LIVE MODELS'} | Fusion: {args.fusion.upper()}[/dim]",
        border_style="cyan"
    ))

    console.print(f"[bold]Utterance:[/bold] \"{args.text}\"\n")

    # Modality Table
    table = Table(title="Modality Predictions & Disagreement")
    table.add_column("Modality / Layer", style="bold cyan")
    table.add_column("Predicted Emotion", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Status / Details", style="dim")

    table.add_row("Text (RoBERTa)", evidence.text.predicted_emotion, f"{evidence.text.confidence:.3f}", "Mock Harness" if evidence.text.is_mock else "Live")
    table.add_row("Audio (Wav2Vec2)", evidence.audio.predicted_emotion, f"{evidence.audio.confidence:.3f}", "Mock Harness" if evidence.audio.is_mock else "Live")
    table.add_row("Visual (ViT)", evidence.visual.predicted_emotion, f"{evidence.visual.confidence:.3f}", "Mock Harness" if evidence.visual.is_mock else "Live")
    table.add_section()
    table.add_row(
        f"Fused ({evidence.fused.model_name})",
        f"[bold green]{evidence.fused.predicted_emotion}[/bold green]",
        f"[bold]{evidence.fused.calibrated_confidence:.3f}[/bold]",
        f"Reliability: [bold yellow]{evidence.reliability_level.upper()}[/bold yellow]"
    )
    console.print(table)
    console.print()

    # Reliability Summary
    rel_color = "green" if evidence.reliability_level == "high" else "yellow" if evidence.reliability_level == "moderate" else "red"
    console.print(Panel(
        f"[bold]D1 (Normalized JSD):[/bold] {evidence.disagreement.jsd:.4f}\n"
        f"[bold]D2 (Pairwise Mean JSD):[/bold] {evidence.disagreement.jsd_pairwise_mean:.4f}\n"
        f"[bold]D3 (Categorical Disagreement):[/bold] Level {evidence.disagreement.categorical}\n"
        f"[bold]Predictive Entropy:[/bold] {evidence.uncertainty.predictive_entropy:.4f} bits (Normalized: {evidence.uncertainty.normalized_entropy:.4f})\n"
        f"[bold]Assessed Reliability Level:[/bold] [{rel_color}]{evidence.reliability_level.upper()}[/{rel_color}]\n\n"
        f"[bold underline]Safety Guidance:[/bold underline]\n{evidence.safety_guidance}",
        title="[bold]Reliability & Safety Assessment[/bold]",
        border_style=rel_color
    ))

    # Dialogue Response Demo
    dialogue_resp = pipeline.dialogue_manager.generate_response(args.text, evidence)
    console.print(Panel(
        f"[bold green]Assistant Reply:[/bold green]\n\"{dialogue_resp['reply']}\"",
        title="[bold]Companion Dialogue Adaptation[/bold]",
        border_style="green"
    ))
    console.print()


if __name__ == "__main__":
    main()
