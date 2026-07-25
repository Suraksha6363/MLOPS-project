# Wafer Fault Detection — MLOps Pipeline

An end-to-end MLOps pipeline demonstrating data versioning, reproducible ML
stages, and model training/evaluation, built with **Git + DVC + scikit-learn**.

## Project background

This project started from a cookiecutter-style MLOps scaffold intended for
wafer sensor fault detection. The raw wafer sensor batch files (590 sensor
readings per wafer) are included under `data/raw/`, but the repository did
not ship with any corresponding labels file, so there was no target to train
a classifier against.

To keep the full MLOps pipeline mechanics intact and demonstrable end-to-end,
this project instead uses the **scikit-learn Breast Cancer Wisconsin
dataset** — a fully-labeled binary classification dataset (30 numeric
features, malignant/benign target) that is structurally similar to the
original wafer sensor data. The pipeline architecture, tooling, and
reproducibility setup are identical to what a real wafer classifier would use.

## Pipeline overview

```
make_dataset.py  →  build_features.py  →  train_model.py  →  predict_model.py  →  visualize.py
   (load +             (scale               (train              (run                (confusion
   split data)          features)            classifier)         predictions)         matrix +
                                                                                       feature
                                                                                       importance)
```

| Stage | Script | Output |
|---|---|---|
| 1 | `src/data/make_dataset.py` | `data/processed/train.csv`, `test.csv` |
| 2 | `src/features/build_features.py` | `data/processed/train_features.csv`, `test_features.csv`, `models/scaler.pkl` |
| 3 | `src/models/train_model.py` | `models/model.pkl` |
| 4 | `src/models/predict_model.py` | `reports/predictions.csv` |
| 5 | `src/visualization/visualize.py` | `reports/figures/confusion_matrix.png`, `feature_importance.png` |

## Results

- **Model:** Random Forest Classifier (100 trees)
- **Test accuracy:** ~95.6%
- **Test set size:** 114 samples (455 training samples)

## Setup

```bash
python -m venv venv
.\venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Running the pipeline

Run the entire pipeline in one command with DVC:

```bash
dvc repro
```

DVC will only re-run stages whose inputs (code or data) have changed since
the last run, and skip the rest — this is what "Stage 'X' didn't change,
skipping" means when you see it in the terminal.

Or run each stage manually:

```bash
python src/data/make_dataset.py --output_dir data/processed --test_size 0.2
python src/features/build_features.py --input_dir data/processed --output_dir data/processed --model_dir models
python src/models/train_model.py --input_dir data/processed --model_dir models
python src/models/predict_model.py --input data/processed/test.csv --output reports/predictions.csv --model_dir models
python src/visualization/visualize.py --predictions reports/predictions.csv --model_dir models --output_dir reports/figures
```

## Data & model versioning

This project uses **DVC** to version large data files and trained models
separately from Git, so the Git history stays lightweight:

```bash
dvc push   # push data/model artifacts to the configured DVC remote
dvc pull   # fetch them back on another machine
```

## Project structure

```
├── data
│   ├── raw            <- Original wafer sensor batch files (unused; no labels available)
│   └── processed       <- Generated train/test splits and scaled features
├── models              <- Saved scaler.pkl and model.pkl
├── reports
│   └── figures         <- Confusion matrix and feature importance plots
├── src
│   ├── data             <- make_dataset.py
│   ├── features         <- build_features.py
│   ├── models            <- train_model.py, predict_model.py
│   └── visualization    <- visualize.py
├── dvc.yaml             <- Pipeline stage definitions
├── dvc.lock             <- Exact input/output hashes for reproducibility
└── requirements.txt
```