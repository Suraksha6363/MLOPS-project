"""
src/data/make_dataset.py

Loads the scikit-learn Breast Cancer Wisconsin dataset (30 numeric sensor-like
features + a binary target: 0 = malignant, 1 = benign) and writes a labeled
train/test split to data/processed/.

We use this in place of the original wafer sensor data because the raw wafer
batch files in this repo ship WITHOUT any labels file, so there is nothing to
train a classifier against. This dataset is structurally similar (many numeric
features, one binary target) so the rest of the pipeline (build_features.py,
train_model.py, predict_model.py) mirrors what a real wafer classifier would
look like.

Usage (from project root, with venv active):
    python src/data/make_dataset.py --output_dir data/processed --test_size 0.2
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_labeled_dataset():
    """Load the breast cancer dataset as a single labeled dataframe."""
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data.frame  # includes feature columns + a 'target' column
    logger.info("Loaded dataset with shape %s", df.shape)
    return df


def main():
    parser = argparse.ArgumentParser(description="Create labeled train/test datasets.")
    parser.add_argument("--output_dir", required=True, help="Directory to write train.csv and test.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to hold out for testing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_labeled_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["target"],
    )

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Wrote %s (shape %s)", train_path, train_df.shape)
    logger.info("Wrote %s (shape %s)", test_path, test_df.shape)
    logger.info("Target distribution in train set:\n%s", train_df["target"].value_counts())


if __name__ == "__main__":
    main()