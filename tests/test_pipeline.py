"""
tests/test_pipeline.py

Basic unit tests for the pipeline scripts. These test the core logic
(data loading, splitting, scaling, prediction shape) without needing to
retrain a full model each run, so they stay fast.

Run with (from project root, venv active):
    pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.make_dataset import load_labeled_dataset


def test_load_labeled_dataset_shape():
    """The loaded dataset should have 569 rows, 30 features + 1 target column."""
    df = load_labeled_dataset()
    assert df.shape == (569, 31)


def test_load_labeled_dataset_has_target_column():
    """The dataset must include a 'target' column for downstream training."""
    df = load_labeled_dataset()
    assert "target" in df.columns


def test_target_is_binary():
    """The target column should only contain 0 and 1 (malignant/benign)."""
    df = load_labeled_dataset()
    assert set(df["target"].unique()) == {0, 1}


def test_no_missing_values():
    """This dataset is clean; there should be no NaNs to worry about."""
    df = load_labeled_dataset()
    assert df.isna().sum().sum() == 0


@pytest.fixture
def sample_train_test_split(tmp_path):
    """Create a small train/test split in a temp directory for feature tests."""
    from sklearn.model_selection import train_test_split

    df = load_labeled_dataset()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return tmp_path


def test_build_features_output_shape(sample_train_test_split, tmp_path):
    """build_features.py should produce scaled outputs with the same row/col counts as input."""
    from src.features.build_features import build_features

    output_dir = tmp_path / "output"
    model_dir = tmp_path / "models"

    build_features(sample_train_test_split, output_dir, model_dir)

    train_features = pd.read_csv(output_dir / "train_features.csv")
    test_features = pd.read_csv(output_dir / "test_features.csv")

    original_train = pd.read_csv(sample_train_test_split / "train.csv")
    original_test = pd.read_csv(sample_train_test_split / "test.csv")

    assert train_features.shape == original_train.shape
    assert test_features.shape == original_test.shape
    assert (model_dir / "scaler.pkl").exists()


def test_build_features_scaling_centers_data(sample_train_test_split, tmp_path):
    """Scaled training features should have approximately mean 0 (StandardScaler property)."""
    from src.features.build_features import build_features

    output_dir = tmp_path / "output"
    model_dir = tmp_path / "models"

    build_features(sample_train_test_split, output_dir, model_dir)

    train_features = pd.read_csv(output_dir / "train_features.csv")
    feature_cols = [c for c in train_features.columns if c != "target"]

    means = train_features[feature_cols].mean()
    assert (means.abs() < 0.1).all(), "Scaled features should be roughly centered around 0"