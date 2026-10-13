"""
Loan Eligibility Prediction with an SVM.

Predicts whether a loan application is approved from four applicant details:
gender, marital status, applicant income and loan amount. Explores the data,
removes income outliers, rebalances the training set by oversampling the
minority class, standardizes features, trains an RBF-kernel Support Vector
Machine, and evaluates it with accuracy, ROC AUC, a classification report and
a confusion matrix.
"""

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "loan_data.csv"
    id_column: str = "Loan_ID"                  # identifier, never used as a feature
    target_column: str = "Loan_Status"
    positive_label: str = "Y"                   # "Y" = approved
    categorical_columns: list = field(default_factory=lambda: ["Gender", "Married"])
    numeric_columns: list = field(default_factory=lambda: ["ApplicantIncome", "LoanAmount"])
    max_applicant_income: float = 25000         # drop applicants earning this much or more
    max_loan_amount: Optional[float] = None     # LoanAmount is in thousands; None = no cap
    test_size: float = 0.2
    random_state: int = 10
    oversample: bool = True
    oversample_random_state: int = 0
    svm_kernel: str = "rbf"
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path)
    print(f"Loaded {len(df)} applications, {df.shape[1]} columns from {config.csv_path}")
    print(df.describe())
    print("\nMissing values per column:")
    print(df.isnull().sum()[lambda s: s > 0].to_string())
    return df


def remove_outliers_and_missing(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Drop rows with a missing loan amount, then rows above the income/loan caps."""
    start = len(df)
    df = df.dropna(subset=config.numeric_columns)
    after_missing = len(df)
    df = df[df["ApplicantIncome"] < config.max_applicant_income]
    if config.max_loan_amount is not None:
        df = df[df["LoanAmount"] < config.max_loan_amount]
    print(f"\nDropped {start - after_missing} rows with missing values "
          f"and {after_missing - len(df)} outliers -> {len(df)} rows left")
    return df


def encode_features(df: pd.DataFrame, config: PipelineConfig):
    """Label-encode the categorical columns and turn Loan_Status into 1/0."""
    X = df[config.categorical_columns + config.numeric_columns].copy()
    for col in config.categorical_columns:
        X[col] = LabelEncoder().fit_transform(X[col])
    y = (df[config.target_column] == config.positive_label).astype(int)
    print(f"Features used: {list(X.columns)}")
    return X, y


def split_data(X, y, config: PipelineConfig):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    print(f"Train: {len(X_train)} rows | Validation: {len(X_val)} rows")
    return X_train, X_val, y_train, y_val


def balance_training_set(X_train, y_train, config: PipelineConfig):
    """Duplicate random minority-class rows until both classes are the same size.
    Applied to the training set only, so the validation set stays realistic."""
    if not config.oversample:
        return X_train, y_train
    ros = RandomOverSampler(sampling_strategy="minority", random_state=config.oversample_random_state)
    X_bal, y_bal = ros.fit_resample(X_train, y_train)
    print(f"Oversampled training set: {len(X_train)} -> {len(X_bal)} rows "
          f"(class counts {y_bal.value_counts().sort_index().tolist()})")
    return X_bal, y_bal


def scale_features(X_train, X_val):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def train_svm(X_train, y_train, config: PipelineConfig) -> SVC:
    model = SVC(kernel=config.svm_kernel)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: SVC, X_train, y_train, X_val, y_val):
    """Accuracy from hard predictions; ROC AUC from the SVM's decision scores."""
    metrics = {}
    for name, X, y in [("Training", X_train, y_train), ("Validation", X_val, y_val)]:
        metrics[name] = {
            "Accuracy": accuracy_score(y, model.predict(X)),
            "ROC AUC": roc_auc_score(y, model.decision_function(X)),
        }
    table = pd.DataFrame(metrics).T
    print("\nModel performance:")
    print(table.round(4).to_string())
    print(f"Majority-class baseline accuracy (validation): {y_val.value_counts(normalize=True).max():.4f}")

    y_pred = model.predict(X_val)
    print("\nClassification report (validation):")
    print(classification_report(y_val, y_pred, target_names=["Rejected", "Approved"]))
    return y_pred, table


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_status_pie(df: pd.DataFrame, config: PipelineConfig) -> None:
    counts = df[config.target_column].value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    plt.title("Loan status")


def plot_categorical_vs_status(df: pd.DataFrame, config: PipelineConfig) -> None:
    cols = config.categorical_columns
    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5))
    for ax, col in zip(axes, cols):
        sns.countplot(data=df, x=col, hue=config.target_column, ax=ax)
        ax.set_title(f"{col} vs. loan status")
    plt.tight_layout()


def plot_numeric_distributions(df: pd.DataFrame, config: PipelineConfig) -> None:
    cols = config.numeric_columns
    fig, axes = plt.subplots(2, len(cols), figsize=(7 * len(cols), 9))
    for i, col in enumerate(cols):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0, i])
        axes[0, i].set_title(f"{col} distribution")
        sns.boxplot(x=df[col].dropna(), ax=axes[1, i])
        axes[1, i].set_title(f"{col} box plot")
    plt.tight_layout()


def print_loan_amount_by_group(df: pd.DataFrame) -> None:
    print("\nMean loan amount by gender:")
    print(df.groupby("Gender")["LoanAmount"].mean().round(1).to_string())
    print("\nMean loan amount by marital status and gender:")
    print(df.groupby(["Married", "Gender"])["LoanAmount"].mean().round(1).to_string())


def plot_confusion_matrix(y_val, y_pred) -> None:
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Rejected", "Approved"], yticklabels=["Rejected", "Approved"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    df = load_data(config)
    plot_status_pie(df, config)
    plot_categorical_vs_status(df, config)
    plot_numeric_distributions(df, config)

    df = remove_outliers_and_missing(df, config)
    print_loan_amount_by_group(df)

    X, y = encode_features(df, config)
    X_train, X_val, y_train, y_val = split_data(X, y, config)
    X_train, y_train = balance_training_set(X_train, y_train, config)
    X_train, X_val = scale_features(X_train, X_val)

    model = train_svm(X_train, y_train, config)
    y_pred, table = evaluate_model(model, X_train, y_train, X_val, y_val)
    plot_confusion_matrix(y_val, y_pred)

    if config.show_plots:
        plt.show()
    return table


if __name__ == "__main__":
    run_pipeline(PipelineConfig())
