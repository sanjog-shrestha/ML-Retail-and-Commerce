"""
Boston House Price Prediction with Linear Regression.

Predicts the median value of owner-occupied homes in Boston suburbs (in
$1000s) from 1970s census features such as crime rate, average number of
rooms, distance to employment centres and pupil-teacher ratio. Loads the
Boston Housing dataset from OpenML (or a local CSV), fits a Linear Regression
model, and evaluates it with MSE, RMSE, MAE and R^2, plus prediction and
residual plots and the learned coefficients.
"""

from dataclasses import dataclass, field
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    openml_name: str = "boston"
    openml_version: int = 1
    csv_path: Optional[str] = None           # use a local CSV instead of downloading
    target_column: str = "MEDV"              # median home value in $1000s
    # "B" is derived from the proportion of Black residents per town; it is excluded
    # by default (see README). Set drop_columns=[] to use every original feature.
    drop_columns: list = field(default_factory=lambda: ["B"])
    test_size: float = 0.2
    random_state: int = 0
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Return one DataFrame with all features plus the target column."""
    if config.csv_path:
        df = pd.read_csv(config.csv_path)
        source = config.csv_path
    else:
        boston = fetch_openml(name=config.openml_name, version=config.openml_version, as_frame=True)
        df = boston.frame
        source = f"OpenML '{config.openml_name}' v{config.openml_version}"
    # OpenML stores CHAS and RAD as categories; the model needs plain numbers.
    df = df.apply(pd.to_numeric)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {source}")
    print(df.head(10))
    return df


def split_data(df: pd.DataFrame, config: PipelineConfig):
    X = df.drop(columns=[config.target_column] + [c for c in config.drop_columns if c in df.columns])
    y = df[config.target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    print(f"\nFeatures used ({X.shape[1]}): {list(X.columns)}")
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def train_linear_regression(X_train, y_train) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X_test, y_test, y_train) -> dict:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R^2": r2_score(y_test, y_pred),
    }
    baseline_mae = mean_absolute_error(y_test, np.full(len(y_test), y_train.mean()))
    print("\nTest-set performance (prices in $1000s):")
    for name, value in metrics.items():
        print(f"  {name:<5}: {value:.4f}")
    print(f"  Baseline MAE (always predict the training mean): {baseline_mae:.4f}")
    return {"metrics": metrics, "y_pred": y_pred}


def coefficients(model: LinearRegression, feature_names) -> pd.Series:
    """Change in predicted price ($1000s) for a one-unit increase in each feature."""
    coefs = pd.Series(model.coef_, index=feature_names).sort_values()
    print("\nLinear Regression coefficients:")
    print(coefs.round(4).to_string())
    print(f"Intercept: {model.intercept_:.4f}")
    return coefs


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_true_vs_predicted(y_test, y_pred) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, c="green", alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    plt.xlabel("True price ($1000s)")
    plt.ylabel("Predicted price ($1000s)")
    plt.title("True vs. Predicted Value: Linear Regression")
    plt.legend()
    plt.tight_layout()


def plot_residuals(y_test, y_pred) -> None:
    residuals = y_test - y_pred
    plt.figure(figsize=(7, 5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Predicted price ($1000s)")
    plt.ylabel("Residual (true - predicted)")
    plt.title("Residuals")
    plt.tight_layout()


def plot_coefficients(coefs: pd.Series) -> None:
    plt.figure(figsize=(8, 5))
    coefs.plot(kind="barh", color=np.where(coefs > 0, "steelblue", "indianred"))
    plt.axvline(0, color="k", linewidth=0.8)
    plt.title("Linear Regression coefficients")
    plt.xlabel("Change in price ($1000s) per unit increase")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> dict:
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)

    model = train_linear_regression(X_train, y_train)
    result = evaluate_model(model, X_test, y_test, y_train)
    coefs = coefficients(model, X_train.columns)

    plot_true_vs_predicted(y_test, result["y_pred"])
    plot_residuals(y_test, result["y_pred"])
    plot_coefficients(coefs)

    if config.show_plots:
        plt.show()
    return result["metrics"]


if __name__ == "__main__":
    run_pipeline(PipelineConfig())