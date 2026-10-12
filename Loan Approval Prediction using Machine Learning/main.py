"""
Loan Approval Prediction.

Predicts whether a loan application will be approved from the applicant's
profile (gender, marital status, dependents, education, employment, income,
loan amount and term, credit history, property area). Explores the data,
builds a preprocessing pipeline (imputation, encoding, scaling) that is fitted
on training data only, and compares Random Forest, K-Nearest Neighbours,
SVM and Logistic Regression classifiers.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "LoanApprovalPrediction.csv"
    id_column: str = "Loan_ID"
    target_column: str = "Loan_Status"
    positive_label: str = "Y"                # "Y" = approved
    test_size: float = 0.4
    random_state: int = 1
    stratify: bool = True
    scale_features: bool = True              # KNN, SVM and Logistic Regression need scaled inputs
    knn_neighbors: int = 3
    rf_estimators: int = 7
    rf_criterion: str = "entropy"
    rf_random_state: int = 7
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and exploration
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path).drop(columns=[config.id_column])
    print(f"Loaded {len(df)} applications, {df.shape[1]} columns from {config.csv_path}")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum()[lambda s: s > 0].to_string())
    return df


def split_columns(df: pd.DataFrame, config: PipelineConfig):
    features = df.drop(columns=[config.target_column])
    categorical = features.select_dtypes(include="object").columns.tolist()
    numeric = features.select_dtypes(exclude="object").columns.tolist()
    print(f"\nCategorical features: {categorical}")
    print(f"Numeric features:     {numeric}")
    return categorical, numeric


def approval_rate(df: pd.DataFrame, config: PipelineConfig) -> float:
    rate = (df[config.target_column] == config.positive_label).mean()
    print(f"\nApproval rate: {rate:.2%}  (accuracy of always predicting 'approved')")
    return rate


# --------------------------------------------------------------------------- #
# Preprocessing and models
# --------------------------------------------------------------------------- #
def build_preprocessor(categorical: list, numeric: list, config: PipelineConfig) -> ColumnTransformer:
    """Most-frequent fill + encoding for categories; median fill (+ scaling) for numbers."""
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder()),
    ]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if config.scale_features:
        cat_steps.append(("scale", StandardScaler()))
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("categorical", Pipeline(cat_steps), categorical),
        ("numeric", Pipeline(num_steps), numeric),
    ])


def get_models(config: PipelineConfig) -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=config.rf_estimators, criterion=config.rf_criterion,
            random_state=config.rf_random_state,
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=config.knn_neighbors),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
    }


def split_data(df: pd.DataFrame, config: PipelineConfig):
    X = df.drop(columns=[config.target_column])
    y = (df[config.target_column] == config.positive_label).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state,
        stratify=y if config.stratify else None,
    )
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def train_and_compare(df: pd.DataFrame, config: PipelineConfig):
    """Fit each model inside the same preprocessing pipeline and compare accuracies."""
    categorical, numeric = split_columns(df, config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    rows, fitted = [], {}
    for name, model in get_models(config).items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(categorical, numeric, config)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Train accuracy (%)": 100 * accuracy_score(y_train, pipe.predict(X_train)),
            "Test accuracy (%)": 100 * accuracy_score(y_test, pipe.predict(X_test)),
        })

    results = pd.DataFrame(rows).sort_values("Test accuracy (%)", ascending=False).reset_index(drop=True)
    print("\nModel comparison:")
    print(results.round(2).to_string(index=False))

    best = results.loc[0, "Model"]
    print(f"\nClassification report for the best model ({best}):")
    print(classification_report(y_test, fitted[best].predict(X_test),
                                target_names=["Rejected", "Approved"]))
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_category_counts(df: pd.DataFrame, columns: list) -> None:
    n_cols = 3
    n_rows = -(-len(columns) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    for ax, col in zip(axes.flat, columns):
        counts = df[col].value_counts()
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    plt.tight_layout()


def plot_approval_by_category(df: pd.DataFrame, columns: list, config: PipelineConfig) -> None:
    """Share of approved applications within each category."""
    approved = (df[config.target_column] == config.positive_label).astype(int)
    n_cols = 3
    n_rows = -(-len(columns) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    for ax, col in zip(axes.flat, columns):
        rate = approved.groupby(df[col].astype(str)).mean()
        sns.barplot(x=rate.index, y=rate.values, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(f"Approval rate by {col}")
        ax.set_ylabel("Approval rate")
    for ax in list(axes.flat)[len(columns):]:
        ax.set_visible(False)
    plt.tight_layout()


def plot_correlations(df: pd.DataFrame, config: PipelineConfig) -> None:
    numeric = df.select_dtypes(exclude="object").copy()
    numeric[config.target_column] = (df[config.target_column] == config.positive_label).astype(int)
    plt.figure(figsize=(12, 6))
    sns.heatmap(numeric.corr(), cmap="BrBG", fmt=".2f", linewidths=2, annot=True)
    plt.title("Correlation of numeric features with loan approval")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    df = load_data(config)
    categorical, _ = split_columns(df, config)
    approval_rate(df, config)

    plot_category_counts(df, categorical + [config.target_column])
    plot_approval_by_category(df, categorical + ["Credit_History"], config)
    plot_correlations(df, config)

    results = train_and_compare(df, config)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())