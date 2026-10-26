# Data Management in H.E.A.R.T. AI

## Benchmark Dataset: MELD
The primary controlled multimodal benchmark is the **Multimodal EmotionLines Dataset (MELD)**:
- **Taxonomy**: 7 discrete emotion classes (`anger`, `disgust`, `fear`, `joy`, `neutral`, `sadness`, `surprise`).
- **Modalities**: Text transcripts, raw audio (.wav), and video clips (.mp4).
- **Dialogue Structure**: Multi-speaker conversation turns grouped by `Dialogue_ID` and `Utterance_ID`.

### Dataset Framing & Transfer Strategy
> [!IMPORTANT]
> **Scientific Integrity Principle**: MELD is used exclusively for *controlled evaluation of multimodal emotion-reliability behavior*. The elderly companion application is NOT claimed to be trained entirely on MELD. Elder-specific datasets (e.g., OASIS, elderly speech corpora) are considered for future domain-transfer validation and must be formally documented with source, licensing, and demographic details.

## Directory Structure
- `data/raw/`: Raw downloaded archives and media (gitignored).
- `data/processed/`: Extracted text, processed audio segments, and visual frames (gitignored).
- `data/splits/`: Canonical train, development, and test index splits (`train.json`, `dev.json`, `test.json`).

## Privacy & Handling
Raw audio and video files must never be committed to version control. All feature extraction and preprocessing pipelines process media ephemerally.
