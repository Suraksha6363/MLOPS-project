"""
src/visualization/visualize.py

Loads the trained model and the predictions produced by predict_model.py,
and generates two diagnostic plots into reports/figures/:
  1. confusion_matrix.png - how predictions compare to ground truth
  2. feature_importance.png - which features the model relies on most

Usage (from project root, with venv active):
    python src/visualization/visualize.py --predictions reports/predictions.csv --model_dir models --output_dir reports/figures
"""

import argparse
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # write to file, no GUI needed
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

TARGET_COL = "target"
PRED_COL = "predicted"


def plot_confusion_matrix(predictions_path: Path, output_dir: Path):
    df = pd.read_csv(predictions_path)

    if TARGET_COL not in df.columns or PRED_COL not in df.columns:
        logger.warning(
            "Skipping confusion matrix: need both '%s' and '%s' columns in %s",
            TARGET_COL, PRED_COL, predictions_path,
        )
        return

    cm = confusion_matrix(df[TARGET_COL], df[PRED_COL])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    out_path = output_dir / "confusion_matrix.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", out_path)


def plot_feature_importance(model_dir: Path, predictions_path: Path, output_dir: Path, top_n: int = 15):
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        logger.warning("Skipping feature importance: model not found at %s", model_path)
        return

    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        logger.warning("Skipping feature importance: model has no feature_importances_ attribute")
        return

    df = pd.read_csv(predictions_path)
    feature_cols = [c for c in df.columns if c not in (TARGET_COL, PRED_COL, "confidence")]

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")

    out_path = output_dir / "feature_importance.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for the trained model.")
    parser.add_argument("--predictions", required=True, help="Path to predictions CSV from predict_model.py")
    parser.add_argument("--model_dir", required=True, help="Directory containing model.pkl")
    parser.add_argument("--output_dir", required=True, help="Directory to write figures")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(predictions_path, output_dir)
    plot_feature_importance(model_dir, predictions_path, output_dir)


if __name__ == "__main__":
    main()