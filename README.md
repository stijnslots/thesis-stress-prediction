# Predicting Stress Levels During Cognitive Tasks from PPG and GSR Signals and Facial Video

Master's thesis project (Data Science & Society, Tilburg University), investigating whether stress levels during cognitive tasks can be predicted from PPG and GSR signals, and facial video, using classical machine learning approaches.

## Project Status

This repository reflects work completed so far in the **exploratory data analysis, preprocessing, and literature review phase** of the thesis. As a result:

**Included:**
- Preprocessing pipelines for PPG, GSR, facial video, PsychoPy, and questionnaire (Qualtrics/PSS-10) data
- Exploratory data analysis and data quality validation scripts
- Documentation of methodological decisions, including deviations from the original research proposal and their justification

**Not yet included:**
- Model training and evaluation code (classical ML models, multimodal fusion), to follow as the thesis progresses into that phase
- Raw or processed participant data, excluded for privacy reasons; the dataset involves physiological and video recordings of human research participants and cannot be shared publicly

## Repository Structure

```
scripts/
  ppg/          # PPG signal exploration and NeuroKit2-based PRV feature extraction
  video/        # OpenFace 2.0-based facial feature extraction, blink-contamination handling, validation
  gsr/          # GSR/EDA feature extraction and SCR- validation
  labels/       # Stress label extraction from PsychoPy task data
  features/     # Feature merging across modalities into per-participant master tables
  qualtrics/    # PSS-10 questionnaire and demographics processing
  sync/         # Task-window extraction from PsychoPy logs, used to align PPG/video/GSR
  evaluation/   # Exploratory data analysis and validation scripts
  run_batch1_pipeline.py                  # Pipeline run for the first participant batch
  run_full_pipeline_all_participants.py   # Orchestrator for the full-cohort pipeline
notes/
  methodology_notes.md   # Methodological decision log
```

## Key Methodological Highlights

A few notable findings from the preprocessing/EDA phase:

- **PPG tool selection**: switched from the originally proposed HeartPy to NeuroKit2, after discovering HeartPy systematically misclassified the dicrotic notch as a separate heartbeat on the low-amplitude signal used in this study.
- **Facial video processing**: identified and corrected a blink-related contamination issue in OpenFace-derived Action Unit, gaze, and head pose features.
- **Data quality**: systematically documented and quantified sources of missing/unusable data (e.g., a hardware mismatch affecting PPG recording for a subset of participants).

## Tech Stack

- Python (pandas, NumPy, scikit-learn)
- NeuroKit2 (PPG/PRV and GSR processing)
- OpenFace 2.0 (facial behavior analysis)
- R
