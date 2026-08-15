"""
app.py — Streamlit demo app for ML Assignment 2
Heart Disease Risk Classification

Features implemented (per assignment requirements):
  a. Dataset upload option (CSV)          -> st.file_uploader
  b. Model selection dropdown              -> st.selectbox
  c. Display of evaluation metrics         -> computed live on uploaded data
  d. Confusion matrix / classification report -> heatmap + sklearn report
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk — Classifier Comparison",
    page_icon="❤️",
    layout="wide",
)

MODEL_DIR = "model"
TARGET_COL = "has_heart_disease"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}


@st.cache_resource
def load_models():
    """Load all fitted pipelines once and cache them."""
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_data
def load_schema():
    with open(os.path.join(MODEL_DIR, "schema.json")) as f:
        return json.load(f)


@st.cache_data
def load_precomputed_metrics():
    path = os.path.join(MODEL_DIR, "metrics_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.title("❤️ Heart Disease Risk — Classification Model Demo")
st.markdown(
    """
    This app demonstrates **5 classification models** (Logistic Regression, Decision Tree,
    kNN, Naive Bayes, Random Forest) trained on a heart-disease risk dataset.
    Upload a CSV of **test data** (same columns as the training data, with or without the
    true label column `has_heart_disease`) to see predictions, live evaluation metrics,
    and the confusion matrix for the selected model.
    """
)

models = load_models()
schema = load_schema()
precomputed_metrics = load_precomputed_metrics()

if not models:
    st.error("No trained models found in the `model/` folder. Please run `model/train_models.py` "
             "or `model/train_models.ipynb` first to generate the .pkl files.")
    st.stop()

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data CSV", type=["csv"],
    help="Upload the provided test_data.csv, or any CSV with the same feature columns."
)

model_choice = st.sidebar.selectbox(
    "Select a model",
    options=list(models.keys()) + ["Compare all models"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: `test_data.csv` in this repo is a ready-made sample you can upload directly."
)

# -------------------------------------------------------------------
# Show precomputed training-time comparison table (always visible)
# -------------------------------------------------------------------
with st.expander("📊 Model comparison on the original held-out test split (from training)", expanded=(uploaded_file is None)):
    if precomputed_metrics is not None:
        st.dataframe(precomputed_metrics.set_index("Model").style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("metrics_comparison.csv not found — run the training script first.")

# -------------------------------------------------------------------
# Main logic: handle uploaded file
# -------------------------------------------------------------------
if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to run predictions and see live metrics below.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

st.subheader("🔍 Preview of uploaded data")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"Shape: {data.shape[0]} rows × {data.shape[1]} columns")

# Drop patient_id if present (not a model feature)
if "patient_id" in data.columns:
    data = data.drop(columns=["patient_id"])

# Cast boolean-like columns if they came in as True/False strings
for col in data.columns:
    if data[col].dtype == object and set(data[col].dropna().unique()).issubset({"True", "False", True, False}):
        data[col] = data[col].astype(str).map({"True": 1, "False": 0}).fillna(data[col])

has_labels = TARGET_COL in data.columns

if has_labels:
    y_true = data[TARGET_COL]
    X_input = data.drop(columns=[TARGET_COL])
else:
    y_true = None
    X_input = data

# Validate expected columns are present
expected_cols = schema["numeric_cols"] + schema["categorical_cols"]
missing_cols = [c for c in expected_cols if c not in X_input.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required columns: {missing_cols}")
    st.stop()

X_input = X_input[expected_cols]  # enforce correct column order


def run_single_model(name, pipe):
    y_pred = pipe.predict(X_input)
    y_proba = pipe.predict_proba(X_input)[:, 1]
    return y_pred, y_proba


# -------------------------------------------------------------------
# Single model view
# -------------------------------------------------------------------
if model_choice != "Compare all models":
    pipe = models[model_choice]
    y_pred, y_proba = run_single_model(model_choice, pipe)

    st.subheader(f"📈 Results — {model_choice}")

    result_df = X_input.copy()
    result_df["Predicted"] = y_pred
    result_df["Risk Probability"] = np.round(y_proba, 3)
    if has_labels:
        result_df["Actual"] = y_true.values
    st.dataframe(result_df.head(20), use_container_width=True)

    if has_labels:
        metrics = compute_metrics(y_true, y_pred, y_proba)
        st.markdown("#### Evaluation metrics (computed on uploaded data)")
        cols = st.columns(6)
        for col, (mname, mval) in zip(cols, metrics.items()):
            col.metric(mname, f"{mval:.4f}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Confusion Matrix")
            fig = plot_confusion_matrix(y_true, y_pred, model_choice)
            st.pyplot(fig)
        with c2:
            st.markdown("#### Classification Report")
            report = classification_report(y_true, y_pred, target_names=["No Disease", "Disease"])
            st.code(report)
    else:
        st.warning(
            f"Uploaded CSV has no `{TARGET_COL}` column, so evaluation metrics and the "
            "confusion matrix can't be computed — only predictions are shown above."
        )

# -------------------------------------------------------------------
# Compare all models view
# -------------------------------------------------------------------
else:
    st.subheader("📈 Comparing all 5 models on the uploaded data")

    if not has_labels:
        st.warning(
            f"Uploaded CSV has no `{TARGET_COL}` column, so metrics can't be computed. "
            "Showing predictions from each model instead."
        )
        preds_df = X_input.copy()
        for name, pipe in models.items():
            preds_df[f"{name} — Prediction"] = pipe.predict(X_input)
        st.dataframe(preds_df.head(20), use_container_width=True)
        st.stop()

    all_metrics = []
    cms = {}
    for name, pipe in models.items():
        y_pred, y_proba = run_single_model(name, pipe)
        m = compute_metrics(y_true, y_pred, y_proba)
        m["Model"] = name
        all_metrics.append(m)
        cms[name] = (y_true, y_pred)

    comp_df = pd.DataFrame(all_metrics).set_index("Model")
    comp_df = comp_df[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]]
    st.dataframe(comp_df.style.format("{:.4f}").highlight_max(axis=0, color="lightgreen"),
                 use_container_width=True)

    best_model = comp_df["F1"].idxmax()
    st.success(f"🏆 Best model on this data by F1 score: **{best_model}**")

    st.markdown("#### Confusion matrices")
    cols = st.columns(5)
    for col, (name, (yt, yp)) in zip(cols, cms.items()):
        with col:
            fig = plot_confusion_matrix(yt, yp, name)
            st.pyplot(fig)

st.markdown("---")
st.caption("BITS Pilani WILP — M.Tech (AIML/DSE) — Machine Learning — Assignment 2")
