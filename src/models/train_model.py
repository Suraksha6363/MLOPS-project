"""
src/models/train_model.py

Trains a classifier on the scaled training features produced by
build_features.py, evaluates it on the held-out test set, and saves the
trained model to models/.

Usage (from project root, with venv active):
    python src/models/train_model.py --input_dir data/processed --model_dir models
"""

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "target"


def train_and_evaluate(input_dir: Path, model_dir: Path, n_estimators: int, random_state: int):
    train_path = input_dir / "train_features.csv"
    test_path = input_dir / "test_features.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_COL]

    logger.info("Training RandomForestClassifier on %d samples, %d features", len(X_train), len(feature_cols))

    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    logger.info("Test accuracy: %.4f", acc)
    logger.info("Classification report:\n%s", classification_report(y_test, y_pred))
    logger.info("Confusion matrix:\n%s", confusion_matrix(y_test, y_pred))

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    joblib.dump(model, model_path)
    logger.info("Saved trained model to %s", model_path)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Train a classifier on the scaled features.")
    parser.add_argument("--input_dir", required=True, help="Directory containing train_features.csv and test_features.csv")
    parser.add_argument("--model_dir", required=True, help="Directory to save the trained model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the random forest")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_and_evaluate(Path(args.input_dir), Path(args.model_dir), args.n_estimators, args.random_state)


if __name__ == "__main__":
    main()