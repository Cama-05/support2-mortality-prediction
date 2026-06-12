"""SUPPORT2 Mortality Prediction Project

This script performs data cleaning, quality assessment, visualization, and
classification modeling for the SUPPORT2 dataset provided in dataset.csv.
All code uses libraries and methods introduced in the course: pandas, NumPy,
plotnine, scikit-learn, ipywidgets, and matplotlib.
"""

import pandas as pd
import numpy as np

from plotnine import (
    ggplot, aes, geom_col, geom_boxplot, geom_histogram,
    facet_wrap, labs, theme_bw, coord_flip
)

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

import matplotlib.pyplot as plt


def load_and_basic_inspection(path: str = "dataset.csv") -> pd.DataFrame:
    """Load the dataset and print basic structure information."""
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nClass balance (death):")
    print(df["death"].value_counts(normalize=True))
    print("\nTop missing-value columns:")
    print(df.isna().sum().sort_values(ascending=False).head(15))
    print("\nNumber of duplicated rows:", df.duplicated().sum())
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps: fix non-standard missing values and drop duplicates."""
    # Fix non-standard missing values in income (string "nan" used instead of real NaN)
    if "income" in df.columns:
        print("\nIncome value counts before cleaning:")
        print(df["income"].value_counts(dropna=False))
        df["income"] = df["income"].replace("nan", np.nan)
        print("\nIncome value counts after cleaning:")
        print(df["income"].value_counts(dropna=False))

    # Drop exact duplicated rows
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    after = df.shape[0]
    print(f"\nDropped {before - after} duplicated rows. New shape: {df.shape}")
    return df


def describe_numeric(df: pd.DataFrame) -> None:
    """Print descriptive statistics for numeric columns."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric columns:", num_cols)
    print("\nNumeric describe:")
    print(df[num_cols].describe())


def plot_comorbidities_and_mortality(df: pd.DataFrame):
    """Create plotnine objects for mean num.co and death rate by dzgroup."""
    if "dzgroup" not in df.columns:
        print("Column 'dzgroup' not found.")
        return None, None

    group_stats = (
        df.groupby("dzgroup")
        .agg(
            mean_num_co=("num.co", "mean"),
            death_rate=("death", "mean"),
            n=("death", "size"),
        )
        .reset_index()
        .sort_values("death_rate", ascending=False)
    )

    print("\nComorbidity and mortality statistics by disease group:")
    print(group_stats)

    p_mean = (
        ggplot(group_stats, aes(x="dzgroup", y="mean_num_co"))
        + geom_col(fill="#4C72B0")
        + coord_flip()
        + theme_bw()
        + labs(
            title="Mean number of comorbidities by disease group",
            x="Disease group",
            y="Mean number of comorbidities",
        )
    )

    p_death = (
        ggplot(group_stats, aes(x="dzgroup", y="death_rate"))
        + geom_col(fill="#DD8452")
        + coord_flip()
        + theme_bw()
        + labs(
            title="Proportion of patients who died (death rate)",
            x="Disease group",
            y="Death rate",
        )
    )

    return p_mean, p_death


def plot_cost_vs_income_and_cancer(df: pd.DataFrame):
    """Create a boxplot of log10(charges) by income and cancer status."""
    required_cols = {"charges", "income", "ca"}
    if not required_cols.issubset(df.columns):
        print("Missing required columns for economic visualization.")
        return None

    econ = df.dropna(subset=["charges", "income", "ca"]).copy()
    econ["log_charges"] = np.log10(econ["charges"])

    p_cost = (
        ggplot(econ, aes(x="income", y="log_charges", fill="ca"))
        + geom_boxplot(outlier_alpha=0.3)
        + theme_bw()
        + labs(
            title="Hospital charges by income level and cancer status",
            x="Income category",
            y="log10(Charges in USD)",
            fill="Cancer status",
        )
    )

    return p_cost


def interactive_physio_widget(df: pd.DataFrame):
    """Build an ipywidgets-based interactive visualization for physiological markers.

    This function is meant to be used inside a Jupyter notebook. It sets up an
    interactive control that allows choosing a physiological marker, a disease
    group, and the plot type (histogram or boxplot).
    """
    physio_vars = [
        "meanbp",
        "hrt",
        "resp",
        "temp",
        "crea",
        "sod",
        "glucose",
        "bun",
        "alb",
    ]

    physio_vars = [v for v in physio_vars if v in df.columns]

    disease_options = ["All"]
    if "dzgroup" in df.columns:
        disease_options += sorted(df["dzgroup"].dropna().unique().tolist())

    kind_options = ["Histogram", "Boxplot"]

    def plot_physio(marker="meanbp", disease="All", kind="Histogram"):
        data = df.copy()
        if disease != "All" and "dzgroup" in data.columns:
            data = data[data["dzgroup"] == disease]

        data = data.dropna(subset=[marker])
        if data.empty:
            print("No data for this combination.")
            return

        if kind == "Histogram":
            p = (
                ggplot(data, aes(x=marker))
                + geom_histogram(bins=30, fill="#4C72B0", color="black", alpha=0.7)
                + theme_bw()
                + labs(
                    title=f"Distribution of {marker} (dzgroup = {disease})",
                    x=marker,
                    y="Count",
                )
            )
        else:
            if "dzgroup" not in data.columns:
                print("Column 'dzgroup' not available for boxplot.")
                return
            p = (
                ggplot(data, aes(x="dzgroup", y=marker))
                + geom_boxplot()
                + coord_flip()
                + theme_bw()
                + labs(
                    title=f"{marker} by disease group",
                    x="Disease group",
                    y=marker,
                )
            )

        display(p)

    interact(
        plot_physio,
        marker=widgets.Dropdown(options=physio_vars, value=physio_vars[0], description="Marker:"),
        disease=widgets.Dropdown(options=disease_options, value=disease_options[0], description="dzgroup:"),
        kind=widgets.ToggleButtons(options=kind_options, description="Plot type:"),
    )


def build_and_evaluate_model(df: pd.DataFrame):
    """Build a scikit-learn pipeline and evaluate a logistic regression classifier.

    The model predicts 180-day mortality (column 'death') from features available
    around day 3. Cost and late outcome variables are excluded to avoid leakage.
    """
    if "death" not in df.columns:
        print("Column 'death' not found.")
        return

    y = df["death"].astype(int)

    # Exclude leakage-prone columns
    leak_cols = ["charges", "sfdm2"]
    feature_cols = [c for c in df.columns if c not in ["death"] + leak_cols]

    X = df[feature_cols].copy()

    numeric_features = [c for c in feature_cols if X[c].dtype != "object"]
    categorical_features = [c for c in feature_cols if X[c].dtype == "object"]

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    print("\nTrain size:", X_train.shape[0], "Test size:", X_test.shape[0])
    print("Train class balance:")
    print(y_train.value_counts(normalize=True))

    # Baseline: always predict majority class 1
    y_pred_baseline = np.ones_like(y_test)
    print("\nBaseline confusion matrix (always predict death = 1):")
    print(confusion_matrix(y_test, y_pred_baseline))
    print("\nBaseline classification report:")
    print(classification_report(y_test, y_pred_baseline, zero_division=0))

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = LogisticRegression(max_iter=1000)

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nModel confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    roc_auc = roc_auc_score(y_test, y_prob)
    print("ROC-AUC:", roc_auc)

    return model


def main():
    df = load_and_basic_inspection("dataset.csv")
    df = clean_dataset(df)
    describe_numeric(df)

    # Create and optionally display visualizations when running in a notebook
    p_mean, p_death = plot_comorbidities_and_mortality(df)
    if p_mean is not None:
        print("\nTo see the comorbidity plots, display p_mean and p_death in a notebook.")

    p_cost = plot_cost_vs_income_and_cancer(df)
    if p_cost is not None:
        print("\nTo see the cost vs income and cancer plot, display p_cost in a notebook.")

    print(
        "\nFor the interactive physiological visualization, call "
        "interactive_physio_widget(df) inside a Jupyter notebook."
    )

    _ = build_and_evaluate_model(df)


if __name__ == "__main__":
    main()
