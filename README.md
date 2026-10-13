# SUPPORT2 Hospital Mortality Prediction

End-to-end data science project on the **SUPPORT2** clinical dataset: data cleaning and quality assessment, exploratory visualization, and a leakage-safe machine learning pipeline that predicts **180-day mortality** of seriously ill hospitalised patients from day-3 clinical data.

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-pipelines-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-notebook-F37626)

---

## Overview

SUPPORT2 (*Study to Understand Prognoses Preferences Outcomes and Risks of Treatment*) follows 9,105 seriously ill adults across 5 U.S. medical centres. Each patient is described by demographics, diagnosis, comorbidities and physiological measurements recorded at **day 3** of the hospital stay.

The goal of this project is to:

1. Clean and assess the quality of a messy real-world clinical dataset
2. Visualize how disease type, comorbidities, income and cancer status relate to mortality and hospital costs
3. Build and compare classification models for the target `death` (1 = died within 180 days, 0 = survived)

**Source:** [UCI Machine Learning Repository, SUPPORT2](https://archive.ics.uci.edu/dataset/880/support2)

---

## Project structure

```
.
├── main.ipynb            # Full analysis: cleaning, visualization, modeling
├── dataset.csv           # Raw dataset (9,255 rows, 25 columns)
├── README_provided.md    # Original dataset description and project goals
├── hints.md              # Dataset hints provided with the assignment
└── requirements.txt      # Python dependencies
```

---

## Pipeline

### 1. Data cleaning and tidying

| Step | Action | Result |
|---|---|---|
| Type conversion | `sex`, `ca` as unordered categoricals, `income` as ordered categorical | Correct dtypes and ordering |
| Unknown column | Dropped `sfdm2` (not documented in the specification) | -1 column |
| Duplicates | Detected and removed duplicated rows | 9,255 → **9,105** rows (150 removed) |
| Sentinel values | Replaced `-999` in `scoma` with missing | Hidden missingness exposed |
| Missing values | Dropped `glucose` (>50% missing) | Dataset completeness: **92%** |
| Range validation | Checked clinical ranges (e.g. `age` 0–120, `adlsc` 0–7, `prg6m` 0–1) and strictly positive vitals | Invalid rows removed → **9,001** rows |
| Suspicious column | Dropped `scoma` (distribution inconsistent with the documented 3–15 range) | -1 column |
| Outliers | **Kept**, since extreme values in a clinical setting often reflect real critical conditions | No clinical signal lost |
| Encoding | `dzgroup` converted into one binary flag per disease, so patients with multiple conditions are handled correctly | 9 disease flags |

### 2. Data quality assessment

Missing values are quantified per column and plotted before any imputation. Imputation (median for numeric, mode for categorical) is applied **only to the dataframe used for visualization**. The modeling dataframe is kept un-imputed to prevent data leakage, and imputation is done later inside the scikit-learn pipeline.

### 3. Data visualization

- **Comorbidities vs mortality by disease group.** Mean number of comorbidities (`num.co`) next to the died/survived proportion for each disease. *Finding:* more comorbidities do not mean higher mortality. CHF and COPD have the most comorbidities (~2.9 and ~2.5) but relatively low mortality (61% and 58%), while lung cancer and MOSF with malignancy have far fewer (~1.6) and the highest death rates (92% and 91%). The intrinsic lethality of the primary disease matters more than the comorbidity count.
- **Hospital charges vs income and cancer status.** Side-by-side box plots (plotnine). *Finding:* charges tend to increase with income, and metastatic cancer patients show lower charges than non-cancer patients.
- **Interactive physiological marker explorer.** An `ipywidgets` selector to compare the day-3 distribution of markers (blood pressure, heart rate, respiratory rate, temperature, creatinine, sodium, BUN, albumin) across disease groups.

### 4. Classification pipeline

**Feature selection**
- Kept: demographics, diagnosis flags, comorbidities, day-3 physiological markers, `income`, `prg6m`
- Removed `charges`: it is only known after the stay, so using it would be **data leakage**

**Preprocessing (inside `ColumnTransformer` + `Pipeline`)**
- Median / most-frequent imputation, fitted on training data only
- `log1p` transform on strongly skewed numeric features (|skew| ≥ 1)
- `StandardScaler` for the distance- and gradient-based models
- `OrdinalEncoder` for `income`, `OneHotEncoder` for `sex` and `ca`

**Modeling**
- Stratified 70/30 train/test split (6,300 / 2,701), `random_state=42`
- Three models: **Logistic Regression**, **Random Forest** (`class_weight="balanced_subsample"`) and **k-NN**
- Hyperparameter tuning with 5-fold `StratifiedKFold` + `GridSearchCV`
- Metric: **ROC-AUC**, because the target is imbalanced (~68% positive) and accuracy would be misleading
- Decision threshold set to **0.30** to favor recall on the positive class, since a missed high-risk patient is costlier than a false alarm in early triage

---

## Results

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| **Random Forest** | 0.8004 | **0.7976** |
| Logistic Regression | 0.7997 | 0.7966 |
| k-NN | 0.7555 | 0.7523 |

- **Random Forest** is the best candidate (best params: `max_depth=12`, `n_estimators=300`, `min_samples_leaf=2`).
- **Logistic Regression** is almost as good and is the best fallback when interpretability matters.
- **k-NN** performs worst, likely because distance-based similarity degrades in a 28-feature space with mixed feature types.
- At threshold 0.30, Logistic Regression correctly flags about 98% of patients who died, at the cost of low recall on survivors (0.21). This is the intended sensitivity-over-specificity trade-off.

---

## Key takeaways

- Real clinical data needs careful validation: duplicates, sentinel values and undocumented columns would have silently biased the results.
- Fitting imputers and scalers inside the pipeline, and removing post-stay variables like `charges`, keeps the evaluation honest.
- With imbalanced data, ROC-AUC and an explicit decision threshold are more informative than accuracy.

---

## How to run

```bash
git clone https://github.com/Cama-05/<repository-name>.git
cd <repository-name>

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter lab main.ipynb
```

Run the notebook top to bottom. The interactive widget requires a Jupyter environment (JupyterLab or Notebook).

---

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · plotnine · Plotly · ipywidgets · Jupyter

---

## Dataset citation

Harrell, F. SUPPORT2. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/880/support2
