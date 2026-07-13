"""
src/data/make_dataset.py

Reads all raw wafer batch CSV files, validates that they share a common
schema, combines them into a single dataframe, and writes the result to
data/processed/.

Usage (from project root, with venv activated):
    python src/data/make_dataset.py --input data/raw/Training_Batch_Files --output data/processed/train.csv
    python src/data/make_dataset.py --input data/raw/Prediction_Batch_files --output data/processed/predict.csv
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_validate_csvs(input_dir: Path) -> pd.DataFrame:
    """Read every CSV in input_dir, check they share the same columns,
    and concatenate them into a single dataframe."""
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    logger.info("Found %d CSV files in %s", len(csv_files), input_dir)

    reference_columns = None
    frames = []
    skipped = []

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.warning("Skipping unreadable file %s: %s", csv_path.name, exc)
            skipped.append(csv_path.name)
            continue

        if reference_columns is None:
            reference_columns = list(df.columns)
        elif list(df.columns) != reference_columns:
            logger.warning(
                "Skipping %s: column mismatch (expected %d columns, got %d)",
                csv_path.name,
                len(reference_columns),
                len(df.columns),
            )
            skipped.append(csv_path.name)
            continue

        frames.append(df)

    if not frames:
        raise ValueError("No files passed schema validation; nothing to combine.")

    combined = pd.concat(frames, ignore_index=True)

    logger.info(
        "Combined %d files into a dataframe with shape %s (skipped %d)",
        len(frames),
        combined.shape,
        len(skipped),
    )
    if skipped:
        logger.info("Skipped files: %s", skipped)

    return combined


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared by train and prediction batches."""
    df = df.copy()

    # The wafer dataset uses "?" for missing values in some exports.
    df.replace("?", pd.NA, inplace=True)

    # First column is usually an unnamed index / wafer-name column.
    first_col = df.columns[0]
    if first_col.lower().startswith("unnamed"):
        df.rename(columns={first_col: "Wafer"}, inplace=True)

    # Drop fully-empty columns (common in these exports).
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        logger.info("Dropping %d fully-empty columns", len(empty_cols))
        df.drop(columns=empty_cols, inplace=True)

    # Drop exact duplicate rows.
    before = len(df)
    df.drop_duplicates(inplace=True)
    if len(df) != before:
        logger.info("Dropped %d duplicate rows", before - len(df))

    return df


def main():
    parser = argparse.ArgumentParser(description="Combine raw wafer batch CSVs into one processed file.")
    parser.add_argument("--input", required=True, help="Directory containing raw batch CSV files")
    parser.add_argument("--output", required=True, help="Path to write the combined CSV file")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = load_and_validate_csvs(input_dir)
    cleaned = clean_dataframe(combined)

    cleaned.to_csv(output_path, index=False)
    logger.info("Wrote processed data to %s (shape %s)", output_path, cleaned.shape)


if __name__ == "__main__":
    main()