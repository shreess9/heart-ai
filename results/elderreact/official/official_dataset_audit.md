# H.E.A.R.T. AI — ElderReact Official Dataset Audit

**Date:** 2026-10-26  **Commit:** 8a0ae96  **Clone path:** /Users/shreee/Downloads/ElderReact_official_repo/

---

## 1. Repository Source
- URL: https://github.com/Mayer123/ElderReact
- Commit: 8a0ae96 ("Update form link"), cloned 2026-10-26
- Paper: Ma et al., ICMI 2019 · DOI: 10.1145/3340555.3353747
- No Git LFS, no external links required — all files present.

## 2. Directory Structure
```
ElderReact_official_repo/
├── README.md, baseline.py
├── Annotations/
│   ├── train_labels.txt   (615 records)
│   ├── dev_labels.txt     (355 records)
│   └── test_labels.txt    (353 records)
└── Features/
    ├── Audio_feat/{train,dev,test}_audio.txt  148-dim COVAREP
    └── Video_feat/{train,dev,test}_video.txt  1418-dim OpenFace
```

## 3. Annotation Schema
Format per line: `{filename.mp4} {Anger} {Disgust} {Fear} {Happiness} {Sadness} {Surprise} {Gender} {Valence}`
- Labels: 6 binary integers [0,1] — confirmed from official README
- Label order: [Anger, Disgust, Fear, Happiness, Sadness, Surprise]
- Task: multi-label binary (not mutually exclusive)
- 3 AMT workers per clip; 1-4 Likert scale; threshold=2; majority vote

## 4. Sample Counts and Alignment
| Split  | Ann  | Audio | Video | Matched | Errors |
|--------|------|-------|-------|---------|--------|
| train  | 615  | 615   | 615   | 615/615 | 0      |
| dev    | 355  | 355   | 355   | 355/355 | 0      |
| test   | 353  | 353   | 353   | 353/353 | 0      |
| Total  | 1323 | 1323  | 1323  | 1323    | 0      |

Split exclusivity: train∩dev=0, dev∩test=0, train∩test=0 ✅

## 5. Label Statistics
| Emotion   | Train        | Dev          | Test         | Global       |
|-----------|-------------|-------------|-------------|-------------|
| Anger     | 187 (30.4%) | 87 (24.5%)  | 76 (21.5%)  | 350 (26.5%) |
| Disgust   | 146 (23.7%) | 69 (19.4%)  | 63 (17.8%)  | 278 (21.0%) |
| Fear      | 92 (15.0%)  | 31 (8.7%)   | 29 (8.2%)   | 152 (11.5%) |
| Happiness | 335 (54.5%) | 220 (62.0%) | 187 (53.0%) | 742 (56.1%) |
| Sadness   | 112 (18.2%) | 63 (17.7%)  | 66 (18.7%)  | 241 (18.2%) |
| Surprise  | 285 (46.3%) | 160 (45.1%) | 121 (34.3%) | 566 (42.8%) |

Labels per clip (zero/one/multi+): train 35/217/363 · dev 17/149/189 · test 40/151/162

## 6. Feature Formats
Audio: COVAREP, 10ms frames, mean+std concatenated → 148-dim. Subset used: [:36]+[74:110] = 72-dim.
Video: OpenFace 2.0, filtered frames, mean+std concatenated → 1418-dim. Subset: [0:8]+[288:294]+[634:717]+[997:1003]+[1343:1418] = 178-dim.
Row ordering: audio=video order (verified). Annotation has .mp4 suffix; features do not.

## 7. Data Quality Issues
| Issue                        | Split | Row | Sample       | Fix                     |
|------------------------------|-------|-----|--------------|-------------------------|
| NaN/Inf in audio features    | test  | 289 | rammstein_30 | Dropped before eval ✅  |

## 8. Missing Components
- Text transcripts: NOT available (audio-visual dataset only)
- Pretrained paper checkpoints: NOT in repo
- Subject identity metadata: inferred from filename prefix only

## 9. Reproducibility
```bash
cd "/Users/shreee/Desktop/heart-ai-main copy"
conda run -n base python3 research/elderreact/experiment_elderreact_proper.py
```
