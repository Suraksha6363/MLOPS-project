"""
src/features/build_features.py

Reads the train/test CSVs produced by make_dataset.py, separates features
from the target column, scales the features with StandardScaler, and writes
scaled versions back out plus the fitted scaler (so predict_model.py can
apply the exact same transform later).

Usage (from project root, with venv active):
    python src/features/build_features.py --input_dir data/processed --output_dir data/processed --model_dir models
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "target"


def build_features(input_dir: Path, output_dir: Path, model_dir: Path):
    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info("Loaded train %s and test %s", train_df.shape, test_df.shape)

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    train_out = X_train_scaled.copy()
    train_out[TARGET_COL] = y_train.values

    test_out = X_test_scaled.copy()
    test_out[TARGET_COL] = y_test.values

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_out_path = output_dir / "train_features.csv"
    test_out_path = output_dir / "test_features.csv"
    scaler_path = model_dir / "scaler.pkl"

    train_out.to_csv(train_out_path, index=False)
    test_out.to_csv(test_out_path, index=False)
    joblib.dump(scaler, scaler_path)

    logger.info("Wrote scaled train features to %s (shape %s)", train_out_path, train_out.shape)
    logger.info("Wrote scaled test features to %s (shape %s)", test_out_path, test_out.shape)
    logger.info("Saved fitted scaler to %s", scaler_path)


def main():
    parser = argparse.ArgumentParser(description="Scale features and prepare data for training.")
    parser.add_argument("--input_dir", required=True, help="Directory containing train.csv and test.csv")
    parser.add_argument("--output_dir", required=True, help="Directory to write scaled feature CSVs")
    parser.add_argument("--model_dir", required=True, help="Directory to save the fitted scaler")
    args = parser.parse_args()

    build_features(Path(args.input_dir), Path(args.output_dir), Path(args.model_dir))


if __name__ == "__main__":
    main()