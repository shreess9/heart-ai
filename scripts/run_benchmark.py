#!/usr/bin/env python3
"""H.E.A.R.T. AI: Comparative Experiment Benchmark Runner.

Runs controlled side-by-side evaluation of:
- Model A: Baseline Multimodal MLP
- Model B: Disagreement-Aware MLP
- Model C: Reliability-Aware Attention + Calibration

Generates standardized metrics:
Accuracy | Macro-F1 | ECE | MCE | Brier | High-Conf Error | Bootstrap 95% CI
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

from experiments.runner import ExperimentBenchmarkRunner

console = Console()


def main():
    parser = argparse.ArgumentParser(description="H.E.A.R.T. AI Benchmark Comparison Runner")
    parser.add_argument("--mock", action="store_true", default=True, help="Run on simulated benchmark test split harness")
    parser.add_argument("--num-samples", type=int, default=500, help="Number of evaluation samples in test split")
    parser.add_argument("--bootstrap-iters", type=int, default=300, help="Bootstrap resampling iterations")
    parser.add_argument("--save-json", action="store_true", default=True, help="Save machine-readable results to results/metrics/")
    parser.add_argument("--output-filename", type=str, default="benchmark_comparison.json", help="Output filename")
    args = parser.parse_args()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]H.E.A.R.T. AI Controlled Experiment Benchmark[/bold cyan]\n"
        f"[dim]Comparing Model A (Baseline) vs Model B (Disagreement) vs Model C (Reliability)[/dim]\n"
        f"[dim]Samples: {args.num_samples} | Bootstrap Iterations: {args.bootstrap_iters} | Mode: {'MOCK TEST HARNESS' if args.mock else 'REAL MELD'}[/dim]",
        border_style="cyan"
    ))

    runner = ExperimentBenchmarkRunner()
    dataset = runner.generate_synthetic_benchmark_split(
        num_samples=args.num_samples,
        num_dialogues=max(10, args.num_samples // 10),
    )

    console.print("[dim]Running evaluation and cluster bootstrap resampling...[/dim]")
    results = runner.run_comparative_benchmark(dataset, bootstrap_iters=args.bootstrap_iters)

    # Comparison Table
    table = Table(title="Controlled Multimodal Fusion Benchmark Comparison")
    table.add_column("Model Architecture", style="bold cyan")
    table.add_column("Accuracy", justify="right")
    table.add_column("Macro-F1 (95% CI)", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("MCE", justify="right")
    table.add_column("Brier Score", justify="right")
    table.add_column("HC Error Rate", justify="right")

    m_a = results["model_a_baseline"]
    m_b = results["model_b_disagreement_aware"]
    m_c = results["model_c_reliability_aware"]

    table.add_row(
        "Model A (Baseline MLP)",
        f"{m_a['accuracy']:.4f}",
        f"{m_a['macro_f1']:.4f} [{m_a['macro_f1_ci'][0]:.3f}, {m_a['macro_f1_ci'][1]:.3f}]",
        f"{m_a['ece']:.4f}",
        f"{m_a['mce']:.4f}",
        f"{m_a['brier_score']:.4f}",
        f"{m_a['high_conf_error_rate']:.4f}",
    )
    table.add_row(
        "Model B (Disagreement-Aware)",
        f"{m_b['accuracy']:.4f}",
        f"{m_b['macro_f1']:.4f} [{m_b['macro_f1_ci'][0]:.3f}, {m_b['macro_f1_ci'][1]:.3f}]",
        f"{m_b['ece']:.4f}",
        f"{m_b['mce']:.4f}",
        f"{m_b['brier_score']:.4f}",
        f"{m_b['high_conf_error_rate']:.4f}",
    )
    table.add_row(
        "Model C (Reliability-Aware)",
        f"{m_c['accuracy']:.4f}",
        f"{m_c['macro_f1']:.4f} [{m_c['macro_f1_ci'][0]:.3f}, {m_c['macro_f1_ci'][1]:.3f}]",
        f"{m_c['ece']:.4f}",
        f"{m_c['mce']:.4f}",
        f"{m_c['brier_score']:.4f}",
        f"{m_c['high_conf_error_rate']:.4f}",
    )

    console.print()
    console.print(table)
    console.print()

    # Paired Significance Summary
    delta_ba = results["paired_deltas"]["b_vs_a"]
    delta_ca = results["paired_deltas"]["c_vs_a"]

    console.print(Panel(
        f"[bold]Model B vs Model A Delta (Macro-F1):[/bold] {delta_ba['point_delta']:+.4f} (95% CI: [{delta_ba['ci_lower']:+.4f}, {delta_ba['ci_upper']:+.4f}], p = {delta_ba['p_value']:.4f})\n"
        f"[bold]Model C vs Model A Delta (Macro-F1):[/bold] {delta_ca['point_delta']:+.4f} (95% CI: [{delta_ca['ci_lower']:+.4f}, {delta_ca['ci_upper']:+.4f}], p = {delta_ca['p_value']:.4f})\n\n"
        "[dim]Note: In mock harness mode, weights are randomly initialized. True empirical comparisons require fine-tuned checkpoints.[/dim]",
        title="[bold]Paired Statistical Significance[/bold]",
        border_style="yellow"
    ))

    # Save machine-readable JSON
    if args.save_json:
        out_dir = ROOT_DIR / "results" / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / args.output_filename
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        console.print(f"[dim green]Saved machine-readable results to: {out_path}[/dim green]")
    console.print()


if __name__ == "__main__":
    main()
