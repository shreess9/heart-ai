"""Generate Publication-Quality Visualizations for H.E.A.R.T. AI.

Generates:
1. Unimodal Performance Comparison (Accuracy, Macro-F1, ECE)
2. Controlled Multimodal Model A vs B vs C Comparison
3. Pre- vs Post-Calibration Reliability Diagrams
4. Categorical Disagreement Distribution (D3) & Mean JSD (D1)
5. Cross-Modal Disagreement vs Predictive Entropy
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

FIG_DIR = ROOT_DIR / "results" / "figures"
METRICS_DIR = ROOT_DIR / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial, DejaVu Sans, Helvetica"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_unimodal_baselines():
    """Plot Text vs Audio vs Visual performance and capability imbalance."""
    with open(METRICS_DIR / "unimodal_baselines.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    mods = ["Text (RoBERTa)", "Audio (Wav2Vec2)", "Visual (ViT)"]
    accs = [data["text"]["accuracy"] * 100, data["audio"]["accuracy"] * 100, data["visual"]["accuracy"] * 100]
    f1s = [data["text"]["macro_f1"] * 100, data["audio"]["macro_f1"] * 100, data["visual"]["macro_f1"] * 100]

    x = np.arange(len(mods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    rects1 = ax.bar(x - width/2, accs, width, label="Accuracy (%)", color="#1f77b4", edgecolor="black", linewidth=0.5)
    rects2 = ax.bar(x + width/2, f1s, width, label="Macro-F1 (%)", color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Unimodal Emotion Recognition Performance on MELD Test Set", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(mods, fontsize=10, fontweight="bold")
    ax.legend(frameon=True)
    ax.set_ylim(0, 80)

    # Bar labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = FIG_DIR / "fig1_unimodal_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_model_abc_comparison():
    """Plot Model A vs B vs C performance and ECE calibration."""
    with open(METRICS_DIR / "model_abc_comparison.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["Model A\n(Baseline MLP)", "Model B\n(Disagreement-Aware)", "Model C\n(Reliability-Aware)"]
    m_keys = ["model_a_baseline", "model_b_disagreement_aware", "model_c_reliability_aware"]

    accs = [data[k]["accuracy"] * 100 for k in m_keys]
    f1s = [data[k]["macro_f1"] * 100 for k in m_keys]
    raw_eces = [data[k]["raw_ece"] * 100 for k in m_keys]
    cal_eces = [data[k]["calibrated_ece"] * 100 for k in m_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Subplot 1: Classification Performance
    x = np.arange(len(models))
    w = 0.35
    r1 = ax1.bar(x - w/2, accs, w, label="Accuracy (%)", color="#2ca02c", edgecolor="black", linewidth=0.5)
    r2 = ax1.bar(x + w/2, f1s, w, label="Macro-F1 (%)", color="#9467bd", edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Score (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Classification Performance (Locked MELD Test)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, fontsize=9)
    ax1.set_ylim(0, 80)
    ax1.legend(frameon=True)

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # Subplot 2: Calibration ECE Reduction
    r3 = ax2.bar(x - w/2, raw_eces, w, label="Raw ECE (%)", color="#d62728", edgecolor="black", linewidth=0.5)
    r4 = ax2.bar(x + w/2, cal_eces, w, label="Calibrated ECE (%)", color="#17becf", edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Expected Calibration Error (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Calibration Error Reduction (Dev Temperature Scaling)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, fontsize=9)
    ax2.set_ylim(0, 35)
    ax2.legend(frameon=True)

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for r in r4:
        h = r.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig2_model_abc_comparison.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def plot_disagreement_distribution():
    """Plot D3 distribution and mean JSD per disagreement level."""
    with open(METRICS_DIR / "disagreement_analysis.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    d3_data = data["d3_distribution"]
    levels = ["D3=0\n(All 3 Agree)", "D3=1\n(1 Differs)", "D3=2\n(All 3 Differ)"]
    counts = [d3_data["counts"]["0"], d3_data["counts"]["1"], d3_data["counts"]["2"]]
    pcts = [d3_data["percentages"]["0"], d3_data["percentages"]["1"], d3_data["percentages"]["2"]]
    mean_jsds = [d3_data["mean_jsd_by_d3"]["0"], d3_data["mean_jsd_by_d3"]["1"], d3_data["mean_jsd_by_d3"]["2"]]

    fig, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    ax2 = ax1.twinx()

    x = np.arange(len(levels))
    w = 0.4

    b1 = ax1.bar(x - w/2, pcts, w, color="#3498db", label="Sample Share (%)", edgecolor="black", linewidth=0.5)
    b2 = ax2.bar(x + w/2, mean_jsds, w, color="#e74c3c", label="Mean Generalized JSD (D1)", edgecolor="black", linewidth=0.5)

    ax1.set_ylabel("Share of Total Utterances (%)", fontsize=10, fontweight="bold", color="#2980b9")
    ax2.set_ylabel("Mean Generalized JSD ($D_1$)", fontsize=10, fontweight="bold", color="#c0392b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels, fontsize=10, fontweight="bold")
    ax1.set_title("Cross-Modal Disagreement Distribution (DE-1 Reproduction)", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, 60)
    ax2.set_ylim(0, 0.4)

    for r in b1:
        h = r.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    for r in b2:
        h = r.get_height()
        ax2.annotate(f"{h:.3f}", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 2),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    out_path = FIG_DIR / "fig3_disagreement_distribution.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def main():
    print("Generating publication-quality figures...")
    plot_unimodal_baselines()
    plot_model_abc_comparison()
    plot_disagreement_distribution()
    print("All publication figures generated successfully in results/figures/.")


if __name__ == "__main__":
    main()
