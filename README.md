# Heart Disease Risk Classification — ML Assignment 2

## a. Problem Statement

Cardiovascular disease is one of the leading causes of death worldwide, and early risk
identification from routine clinical and lifestyle measurements can help prioritize
preventive care. This project frames heart-disease risk prediction as a **binary
classification problem**: given a patient's clinical measurements (blood pressure,
cholesterol, blood sugar, heart rate, ECG-related indicators) and lifestyle/behavioral
data (smoking status, exercise, sleep, stress, diet), predict whether the patient
**has heart disease (1)** or **does not (0)**.

Five classification algorithms are trained and compared on the same dataset, and the
best-performing model is served through an interactive Streamlit web application.

## b. Dataset Description

- **Source:** `heart_disease_risk_2026.csv` (available on Kaggle)
- **Instances:** 9,000 patient records
- **Features:** 25 predictive features (after dropping the `patient_id` identifier)
- **Target variable:** `has_heart_disease` (0 = No disease, 1 = Disease) — approx. 30% positive class
- **Feature types:**
  - **Numeric (22):** age, resting_bp_systolic, resting_bp_diastolic, cholesterol_total, hdl,
    ldl, triglycerides, fasting_blood_sugar, hba1c, bmi, resting_heart_rate,
    max_heart_rate_achieved, exercise_induced_angina, st_depression, family_history,
    alcohol_units_per_week, exercise_minutes_per_week, sleep_hours, stress_score,
    wearable_owner, daily_steps, diet_quality_score
  - **Categorical (3):** sex (Male/Female), chest_pain_type (Asymptomatic / Non-Anginal Pain /
    Atypical Angina / Typical Angina), smoker_status (Never / Former / Current)
- **Missing values:** None
- **Preprocessing:** Boolean columns cast to 0/1; numeric features standardized with
  `StandardScaler`; categorical features one-hot encoded (`drop='first'`); 80/20
  stratified train-test split (random_state = 42).

## c. GitHub Repository Link

`https://github.com/lalitp30/ML-Assignment-2`

## d. Models Used

Five classifiers were trained on identical preprocessed data (train/test split, random_state=42) and evaluated on the same 20% held-out test set (1,800 records).

### Comparison Table

| ML Model Name             | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|----------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression        | 0.8978   | 0.9497 | 0.8632    | 0.7872 | 0.8234 | 0.7533 |
| Decision Tree               | 0.8394   | 0.8917 | 0.7520    | 0.7009 | 0.7255 | 0.6130 |
| kNN                          | 0.8606   | 0.9143 | 0.8973    | 0.6092 | 0.7257 | 0.6582 |
| Naive Bayes                  | 0.8700   | 0.9315 | 0.7686    | 0.8165 | 0.7918 | 0.6981 |
| Random Forest (Ensemble)     | 0.8783   | 0.9354 | 0.8655    | 0.7083 | 0.7790 | 0.7029 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer across almost every metric (highest Accuracy, AUC, F1, and MCC). The relationship between the clinical risk factors and heart disease in this dataset appears largely linear/additive, which favors a linear model. It also generalizes well and is the most interpretable of the five. |
| Decision Tree | Weakest performer overall. A single tree (depth capped at 6 to control overfitting) captures some interactions but loses accuracy compared to ensembling; it is also the most sensitive to small changes in the training data (high variance). |
| kNN | Highest **precision** (0.897) but by far the lowest **recall** (0.609) — it is conservative and only predicts "disease" when neighbors are strongly in agreement, so it misses many true positive cases. Performance is also sensitive to feature scaling (handled here via StandardScaler) and the choice of k. |
| Naive Bayes | Strong **recall** (0.817), the second-highest after Logistic Regression, meaning it catches the most actual disease cases, at some cost to precision. Its independence assumption between features is clearly violated here (e.g., cholesterol subtypes are correlated), yet it still performs respectably. |
| Random Forest (Ensemble) | Solid, well-balanced performance (2nd-best AUC and MCC) with less overfitting risk than a single Decision Tree, thanks to averaging over 300 trees. It underperforms Logistic Regression here, likely because the true decision boundary is close to linear, giving the extra model complexity little advantage. |
| **Overall Winner for your dataset?** | **Logistic Regression** — it achieves the best Accuracy (0.898), AUC (0.950), F1 (0.823), and MCC (0.753) of all five models, making it the most reliable classifier for this dataset. |


---

## Project Structure

```
project-folder/
│-- app.py                       # Streamlit web app
│-- requirements.txt             # Python dependencies
│-- README.md                    # This file
│-- heart_disease_risk_2026.csv  # Full original dataset
│-- test_data.csv                # Held-out test split (used for demo/upload)
│-- model/
│   │-- train_models.py          # Training script (script version)
│   │-- ML Assignment 2 - 2025da04315.ipynb       # Training notebook (run this in Jupyter)
│   │-- logistic_regression.pkl  # Fitted pipeline (preprocessing + model)
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- metrics_comparison.csv   # Saved metrics table
│   │-- schema.json              # Expected column names/types for the app
│   └-- confusion_matrices.png   # Confusion matrix grid from training
```


## Streamlit App Features

- **Dataset upload (CSV):** upload `test_data.csv` or any CSV with the same feature columns
- **Model selection dropdown:** choose any of the 5 models, or "Compare all models"
- **Live evaluation metrics:** Accuracy, AUC, Precision, Recall, F1, MCC computed on the uploaded data
- **Confusion matrix & classification report:** per-model heatmap + sklearn classification report

## Deployment

Deployed on **Streamlit Community Cloud**: `https://ml-assignment-2-lalitpetkule.streamlit.app/`

---
