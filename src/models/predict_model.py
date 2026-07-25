"""
src/models/predict_model.py

Loads the trained model and fitted scaler, applies the same scaling used
during training to new input data, runs predictions, and writes the results
(including predicted class and confidence) to a CSV.

If the input CSV contains a 'target' column (e.g. you're running this on the
held-out test set to sanity-check things), it will be carried through
unchanged so you can compare predictions against ground truth. It is NOT
used as a feature.

Usage (from project root, with venv active):
    python src/models/predict_model.py --input data/processed/test.csv --output reports/predictions.csv --model_dir models
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "target"


def run_predictions(input_path: Path, output_path: Path, model_dir: Path):
    model_path = model_dir / "model.pkl"
    scaler_path = model_dir / "scaler.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run train_model.py first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Fitted scaler not found at {scaler_path}. Run build_features.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    df = pd.read_csv(input_path)
    logger.info("Loaded input data with shape %s", df.shape)

    has_target = TARGET_COL in df.columns
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    confidence = probabilities.max(axis=1)

    results = df.copy()
    results["predicted"] = predictions
    results["confidence"] = confidence

    if has_target:
        correct = (results["predicted"] == results[TARGET_COL]).sum()
        total = len(results)
        logger.info("Ground truth available: %d/%d correct (%.2f%%)", correct, total, 100 * correct / total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info("Wrote predictions to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Run predictions using the trained model.")
    parser.add_argument("--input", required=True, help="Path to input CSV (raw features, with or without a target column)")
    parser.add_argument("--output", required=True, help="Path to write predictions CSV")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.pkl and scaler.pkl")
    args = parser.parse_args()

    run_predictions(Path(args.input), Path(args.output), Path(args.model_dir))


if __name__ == "__main__":
    main()