"""
train_models.py
----------------
Trains 5 classification models (Logistic Regression, Decision Tree, KNN,
Gaussian Naive Bayes, Random Forest) on the Heart Disease Risk dataset,
evaluates each with 6 metrics, and saves:
  - one fitted sklearn Pipeline per model (preprocessing + classifier) as .pkl
  - the held-out test split as test_data.csv (raw, unprocessed columns)
  - a metrics comparison table as metrics_comparison.csv

Run this once (e.g. in Jupyter or via `python train_models.py`) to produce
everything the Streamlit app and README need.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
DATA_PATH = "../heart_disease_risk_2026.csv"   # adjust path if needed
MODEL_DIR = "."                                 # save models next to this script

# -----------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

TARGET = "has_heart_disease"
DROP_COLS = ["patient_id"]           # not a predictive feature, just an ID

df = df.drop(columns=DROP_COLS)

# Convert boolean columns to int (0/1) so they behave as numeric features
bool_cols = df.select_dtypes(include="bool").columns.tolist()
for c in bool_cols:
    df[c] = df[c].astype(int)

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = ["sex", "chest_pain_type", "smoker_status"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

print("Numeric features:", len(numeric_cols))
print("Categorical features:", len(categorical_cols))
print("Total features:", len(numeric_cols) + len(categorical_cols))

# -----------------------------------------------------------------
# 2. Train / test split (stratified, 80/20)
# -----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the RAW (unprocessed) test split -> this is the file used for the
# assignment's "test data" requirement and for the Streamlit app demo.
test_data = X_test.copy()
test_data[TARGET] = y_test.values
test_data.to_csv("../test_data.csv", index=False)
print("Saved ../test_data.csv with shape", test_data.shape)

# -----------------------------------------------------------------
# 3. Preprocessing pipeline (shared by every model)
# -----------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_cols),
    ]
)

# -----------------------------------------------------------------
# 4. Define models
# -----------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "kNN": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE
    ),
}

results = []

for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    # predict_proba for AUC (all 5 models support it)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    # Save the fitted pipeline (preprocessing + model together)
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(pipe, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))

# -----------------------------------------------------------------
# 5. Save metrics comparison table
# -----------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "metrics_comparison.csv"), index=False)
print("\nFinal comparison table:\n", results_df)

# Save the list of raw column names/dtypes/categories the app needs to
# validate an uploaded CSV against.
schema = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "categories": {c: sorted(X[c].unique().tolist()) for c in categorical_cols},
    "target": TARGET,
}
with open(os.path.join(MODEL_DIR, "schema.json"), "w") as f:
    json.dump(schema, f, indent=2)

print("\nAll models, metrics_comparison.csv and schema.json saved in model/")
