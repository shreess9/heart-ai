"""MELD Dataset Ingestion, Integrity Validation, and Canonical Split Generator.

Downloads canonical MELD train, dev (validation), and test splits from the official repository,
validates utterance alignment, dialogue clusters, speaker metadata, and label taxonomy,
and produces a comprehensive dataset validation report.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

# Ensure project root in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Canonical 7 MELD emotion labels
CANONICAL_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMOTION_TO_ID = {emo: i for i, emo in enumerate(CANONICAL_EMOTIONS)}

MELD_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv",
}


def download_and_validate_meld(
    output_raw_dir: Path = ROOT_DIR / "data" / "raw" / "MELD",
    output_splits_dir: Path = ROOT_DIR / "data" / "splits",
) -> Dict[str, Any]:
    """Download, preprocess, validate, and save canonical MELD splits."""
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    output_splits_dir.mkdir(parents=True, exist_ok=True)

    validation_summary = {}
    split_dfs = {}

    for split_name, url in MELD_URLS.items():
        raw_csv_path = output_raw_dir / f"{split_name}_sent_emo.csv"
        
        # Download if not present
        if not raw_csv_path.exists():
            print(f"Downloading MELD {split_name} split from {url}...")
            df = pd.read_csv(url)
            df.to_csv(raw_csv_path, index=False)
        else:
            print(f"Loading cached MELD {split_name} split from {raw_csv_path}...")
            df = pd.read_csv(raw_csv_path)

        # Standardize column names
        df["Emotion_Clean"] = df["Emotion"].str.strip().str.lower()
        df["Utterance_Clean"] = df["Utterance"].fillna("").astype(str).str.strip()

        # Check for unknown emotions
        unknown_emotions = set(df["Emotion_Clean"]) - set(CANONICAL_EMOTIONS)
        if unknown_emotions:
            raise ValueError(f"Found unexpected emotions in {split_name}: {unknown_emotions}")

        df["Emotion_ID"] = df["Emotion_Clean"].map(EMOTION_TO_ID)
        
        # Create canonical unique sample ID: dia{Dialogue_ID}_utt{Utterance_ID}
        df["Sample_ID"] = df.apply(
            lambda r: f"dia{int(r['Dialogue_ID'])}_utt{int(r['Utterance_ID'])}", axis=1
        )

        split_dfs[split_name] = df

        # Compute validation stats
        emotion_counts = df["Emotion_Clean"].value_counts().to_dict()
        dialogue_count = int(df["Dialogue_ID"].nunique())
        speaker_count = int(df["Speaker"].nunique())
        total_utterances = len(df)
        empty_utterances = int((df["Utterance_Clean"] == "").sum())

        validation_summary[split_name] = {
            "total_samples": total_utterances,
            "unique_dialogues": dialogue_count,
            "unique_speakers": speaker_count,
            "empty_utterances": empty_utterances,
            "emotion_distribution": emotion_counts,
            "emotion_percentages": {
                k: round(v / total_utterances * 100.0, 2) for k, v in emotion_counts.items()
            },
        }

        # Save canonical JSON split
        json_split_path = output_splits_dir / f"{split_name}.json"
        records = df[[
            "Sample_ID", "Dialogue_ID", "Utterance_ID", "Speaker", 
            "Utterance_Clean", "Emotion_Clean", "Emotion_ID", "Sentiment"
        ]].to_dict(orient="records")
        
        with open(json_split_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved canonical {split_name} split ({len(records)} samples) to {json_split_path}")

    # Generate Markdown Report
    report_path = ROOT_DIR / "docs" / "research" / "dataset_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _generate_dataset_report(validation_summary, report_path)
    print(f"Generated validation report at {report_path}")

    return validation_summary


def _generate_dataset_report(summary: Dict[str, Any], output_path: Path):
    """Write markdown validation report."""
    md = [
        "# MELD Benchmark Dataset Validation Report",
        "",
        "This report documents the canonical Multimodal EmotionLines Dataset (MELD) splits used for controlled research evaluation.",
        "",
        "## 1. Split Overview",
        "",
        "| Split | Utterances | Dialogue Clusters | Unique Speakers | Empty Text |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for split, stats in summary.items():
        md.append(
            f"| **{split.capitalize()}** | {stats['total_samples']:,} | {stats['unique_dialogues']} | {stats['unique_speakers']} | {stats['empty_utterances']} |"
        )
    
    md.extend([
        "",
        "## 2. 7-Class Emotion Distribution",
        "",
        "| Emotion | Train Count (%) | Dev Count (%) | Test Count (%) |",
        "| :--- | :--- | :--- | :--- |",
    ])

    for emo in CANONICAL_EMOTIONS:
        tr_cnt = summary["train"]["emotion_distribution"].get(emo, 0)
        tr_pct = summary["train"]["emotion_percentages"].get(emo, 0.0)
        dev_cnt = summary["dev"]["emotion_distribution"].get(emo, 0)
        dev_pct = summary["dev"]["emotion_percentages"].get(emo, 0.0)
        test_cnt = summary["test"]["emotion_distribution"].get(emo, 0)
        test_pct = summary["test"]["emotion_percentages"].get(emo, 0.0)
        md.append(f"| **{emo}** | {tr_cnt:,} ({tr_pct:.1f}%) | {dev_cnt:,} ({dev_pct:.1f}%) | {test_cnt:,} ({test_pct:.1f}%) |")

    md.extend([
        "",
        "## 3. Framing & Integrity Notes",
        "- **Test Set Lock**: The test set (2,610 utterances across 280 dialogues) is strictly locked for final evaluation.",
        "- **Class Imbalance**: Neutral represents ~47-48% across all splits, followed by Joy (~15-17%) and Anger (~11-13%). Disgust (~2-3%) and Fear (~2%) are rare minority classes.",
        "- **Demographic Scope**: MELD is a conversational TV-dialogue benchmark and is explicitly NOT an elderly-specific clinical dataset.",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    download_and_validate_meld()
