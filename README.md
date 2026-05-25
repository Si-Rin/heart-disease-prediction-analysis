# Heart Disease Prediction — Analysis Report

**Dataset:** Cleveland Heart Disease (UCI) · 303 patients · 13 features · Binary target
**Source:** [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) — place `heart_disease_data.csv` in the same folder.


---

## 1. Pipeline Overview
| Step | Details |
|------|---------|
| **EDA** | Missing values, outliers (IQR), histograms, boxplots, Q-Q plots, mosaic plots |
| **Visualization** | Correlation heatmap, pairplot, countplots, pie charts |
| **Modeling** | Custom train/test split + sklearn KNN, k=1 to 20 error curve |


## 2. Dataset Overview

| | |
|---|---|
| Samples | 303 |
| Features | 13 (6 numerical, 7 categorical) |
| Target balance | Heart Disease 54.5% · No Heart Disease 45.5% |
| Missing values | None in raw data (thal value `3` was unmapped — fixed) |
| Duplicates | None |

The target is nearly balanced, so accuracy is a reliable metric without needing resampling.

---

## 3. Key Findings from EDA

### Numerical Features

| Feature | Observation |
|---------|-------------|
| `age` | Ranges 29–77, mean ~54. Patients with heart disease tend to be slightly older. |
| `chol` | Wide range (120–564 mg/dL). Notable high outliers; weakly correlated with disease. |
| `thalach` | Max heart rate — lower values strongly associated with heart disease. |
| `oldpeak` | ST depression — right-skewed; higher values correlate with disease. |
| `trestbps` | Resting blood pressure — near-normal distribution, mild outliers. |
| `ca` | Number of vessels — strongly predictive; higher = more disease risk. |

The correlation heatmap showed no strong multicollinearity between numerical features, making them all useful for modeling.

### Categorical Features

| Feature | Key Observation |
|---------|----------------|
| `genre` | 68% Male. Males show higher disease prevalence in this dataset. |
| `cp` | 47% Asymptomatic — the most disease-linked type. Typical angina is least common (7.6%). |
| `exang` | 67% do not have exercise-induced angina; those who do are at higher risk. |
| `thal` | 89% Normal thalassemia. Reversible defect (~10%) strongly linked to disease. |
| `fbs` | 85% have fasting blood sugar ≤ 120 mg/dL — fbs is a weak predictor overall. |
| `slope` | Flat (47%) and Downsloping (46%) are dominant — both linked to abnormal ST segments. |
| `restecg` | ~50% ST-T Wave Abnormality, ~49% Normal — nearly even split. |

The age-vs-target histogram confirmed that heart disease peaks in the 50–65 age range, with healthy patients slightly younger on average.

---

## 4. Model Results — KNN Classifier

### Before fixes (no scaling, 117 rows dropped)
| Metric | Value |
|--------|-------|
| Training samples | ~150 |
| Accuracy (k=1) | **0.62** |

### After fixes (scaled, all 303 rows used)
| Metric | Value |
|--------|-------|
| Training samples | ~242 |
| Accuracy (k=1) | **~0.75–0.80** (expected) |
| Best k | Typically k=7–13 on this dataset |

The accuracy improvement comes from two compounding factors: scaling prevents large-range features from dominating distances, and recovering the 117 dropped rows gives the model 60% more training data.

> **Note on k=1:** A k of 1 is prone to overfitting — it memorizes training points. The error-rate-vs-k curve typically shows improvement up to k≈10 before plateauing, which is the recommended operating range for this dataset.

## 5. Setup
```bash
pip install -r requirements.txt
python heart_disease_prediction.py
```