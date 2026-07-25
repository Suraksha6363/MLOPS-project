"""
app.py

Streamlit UI for the MLOps pipeline. Two sections:
  1. Predict  - upload a CSV (or use a sample from the test set) and get
     predictions from the trained model.
  2. Dashboard - shows model accuracy, confusion matrix, and feature
     importance from the last training run.

Usage (from project root, with venv active):
    streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Wafer/Breast Cancer MLOps Pipeline", layout="wide")

MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
TARGET_COL = "target"


@st.cache_resource
def load_model_and_scaler():
    model_path = MODEL_DIR / "model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    if not model_path.exists() or not scaler_path.exists():
        return None, None
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def run_predictions(df: pd.DataFrame, model, scaler) -> pd.DataFrame:
    has_target = TARGET_COL in df.columns
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    X = df[feature_cols]
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    confidence = probabilities.max(axis=1)

    results = df.copy()
    results["predicted"] = predictions
    results["predicted_label"] = results["predicted"].map({0: "Malignant", 1: "Benign"})
    results["confidence"] = confidence.round(3)

    if has_target:
        results["correct"] = results["predicted"] == results[TARGET_COL]

    return results


st.title("🔬 Wafer/Breast Cancer MLOps Pipeline")
st.caption(
    "Trained via the make_dataset → build_features → train_model → predict_model → "
    "visualize pipeline. See README.md for background on the dataset choice."
)

tab_predict, tab_dashboard = st.tabs(["🔮 Predict", "📊 Results Dashboard"])

model, scaler = load_model_and_scaler()

# ---------------------------------------------------------------------------
# Tab 1: Predict
# ---------------------------------------------------------------------------
with tab_predict:
    if model is None or scaler is None:
        st.error(
            "No trained model found. Run `dvc repro` or the pipeline scripts first "
            "so `models/model.pkl` and `models/scaler.pkl` exist."
        )
    else:
        st.subheader("Run predictions")

        col_upload, col_sample = st.columns(2)

        with col_upload:
            uploaded_file = st.file_uploader("Upload a CSV with the same feature columns", type="csv")

        with col_sample:
            st.write("Or try it instantly:")
            use_sample = st.button("Use a sample from the test set")

        input_df = None

        if uploaded_file is not None:
            input_df = pd.read_csv(uploaded_file)
        elif use_sample:
            test_path = DATA_DIR / "test.csv"
            if test_path.exists():
                input_df = pd.read_csv(test_path).sample(10, random_state=None)
            else:
                st.warning("No test.csv found in data/processed. Run the pipeline first.")

        if input_df is not None:
            try:
                results = run_predictions(input_df, model, scaler)

                st.success(f"Ran predictions on {len(results)} rows.")

                display_cols = ["predicted_label", "confidence"]
                if TARGET_COL in results.columns:
                    display_cols = [TARGET_COL, "predicted", "predicted_label", "confidence", "correct"]

                st.dataframe(results[display_cols], use_container_width=True)

                if "correct" in results.columns:
                    acc = results["correct"].mean()
                    st.metric("Accuracy on this batch", f"{acc:.1%}")

                csv_out = results.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", csv_out, "predictions.csv", "text/csv")

            except Exception as e:
                st.error(f"Couldn't run predictions on this file: {e}")
                st.info(
                    "Make sure the uploaded CSV has the same feature columns the model "
                    "was trained on (see data/processed/train.csv for the expected format)."
                )

# ---------------------------------------------------------------------------
# Tab 2: Dashboard
# ---------------------------------------------------------------------------
with tab_dashboard:
    st.subheader("Model performance")

    predictions_path = REPORTS_DIR / "predictions.csv"
    if predictions_path.exists():
        preds_df = pd.read_csv(predictions_path)
        if TARGET_COL in preds_df.columns and "predicted" in preds_df.columns:
            acc = (preds_df[TARGET_COL] == preds_df["predicted"]).mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Test accuracy", f"{acc:.1%}")
            m2.metric("Test samples", len(preds_df))
            m3.metric(
                "Class balance",
                f"{(preds_df[TARGET_COL] == 1).sum()} benign / {(preds_df[TARGET_COL] == 0).sum()} malignant",
            )
    else:
        st.info("Run `predict_model.py` (or `dvc repro`) to generate reports/predictions.csv first.")

    st.divider()

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.write("**Confusion Matrix**")
        cm_path = FIGURES_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Run `visualize.py` to generate this plot.")

    with col_fi:
        st.write("**Feature Importance**")
        fi_path = FIGURES_DIR / "feature_importance.png"
        if fi_path.exists():
            st.image(str(fi_path), use_container_width=True)
        else:
            st.info("Run `visualize.py` to generate this plot.")

    st.divider()
    st.write("**Pipeline stages**")
    st.code(
        "make_dataset.py  →  build_features.py  →  train_model.py  →  "
        "predict_model.py  →  visualize.py",
        language=None,
    )