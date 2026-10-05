"""
Inventory Demand Forecasting with Machine Learning.

Predicts daily unit sales for each (store, item) pair from calendar features
alone — store, item, month, day, weekday/weekend, a holiday flag and a cyclical
encoding of the month. Explores the data with a few charts, removes extreme
sales outliers, then trains and compares four regressors (Linear Regression,
XGBoost, Lasso, Ridge) using Mean Absolute Error.
"""

from dataclasses import dataclass, field

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "train.csv"             # Store Item Demand Forecasting dataset
    target_column: str = "sales"
    holiday_country: str = "IN"             # country whose public holidays are flagged
    outlier_threshold: float = 140          # drop rows with sales >= this value
    drop_columns: list = field(default_factory=lambda: ["year"])  # excluded from features
    test_size: float = 0.05
    random_state: int = 22
    sma_window: int = 30                    # moving-average window for the trend chart
    sma_year: int = 2013
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Data loading and feature engineering
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, parse_dates=["date"])
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns from {config.csv_path}")
    print(f"Stores: {df['store'].nunique()} | Items: {df['item'].nunique()}")
    print(df.describe())
    return df


def add_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    """Split the date into year, month, day and weekday (Mon=0 ... Sun=6)."""
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def add_holiday_flag(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """1 if the date is a public holiday in the configured country, else 0."""
    df = df.copy()
    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    holiday_dates = holidays.country_holidays(config.holiday_country, years=years)
    df["holidays"] = df["date"].dt.date.isin(holiday_dates).astype(int)
    return df


def add_cyclical_month(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month on a circle so December (12) sits next to January (1)."""
    df = df.copy()
    df["m1"] = np.sin(df["month"] * (2 * np.pi / 12))
    df["m2"] = np.cos(df["month"] * (2 * np.pi / 12))
    return df


def engineer_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = add_date_parts(df)
    df = add_holiday_flag(df, config)
    df = add_cyclical_month(df)
    return df


def remove_outliers(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    kept = df[df[config.target_column] < config.outlier_threshold]
    print(f"Removed {len(df) - len(kept)} rows with sales >= {config.outlier_threshold}")
    return kept


def split_data(df: pd.DataFrame, config: PipelineConfig):
    features = df.drop(columns=["date", config.target_column] + config.drop_columns)
    target = df[config.target_column].values
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=config.test_size, random_state=config.random_state
    )
    print(f"Features used: {list(features.columns)}")
    print(f"Train: {X_train.shape} | Validation: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def scale_features(X_train, X_val):
    """Standardize features, fitting the scaler on training data only."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def get_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(),
        "Lasso": Lasso(),
        "Ridge": Ridge(),
    }


def train_and_compare(models: dict, X_train, X_val, y_train, y_val) -> pd.DataFrame:
    """Fit each model and report training and validation MAE."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append({
            "Model": name,
            "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
            "Validation MAE": mean_absolute_error(y_val, model.predict(X_val)),
        })
    results = pd.DataFrame(rows).sort_values("Validation MAE").reset_index(drop=True)
    print("\nModel comparison (Mean Absolute Error, units sold):")
    print(results.round(3).to_string(index=False))
    return results


# --------------------------------------------------------------------------- #
# Exploratory plots
# --------------------------------------------------------------------------- #
def plot_mean_sales_by_feature(df: pd.DataFrame, config: PipelineConfig) -> None:
    features = ["store", "year", "month", "weekday", "weekend", "holidays"]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, col in zip(axes.flat, features):
        df.groupby(col)[config.target_column].mean().plot.bar(ax=ax)
        ax.set_title(f"Mean sales by {col}")
    plt.tight_layout()


def plot_mean_sales_by_day(df: pd.DataFrame, config: PipelineConfig) -> None:
    plt.figure(figsize=(10, 5))
    df.groupby("day")[config.target_column].mean().plot()
    plt.title("Mean sales by day of month")
    plt.xlabel("Day of month")
    plt.ylabel("Mean sales")
    plt.tight_layout()


def plot_moving_average(df: pd.DataFrame, config: PipelineConfig) -> None:
    """Total daily sales for one year with a simple moving average on top."""
    daily = (df[df["year"] == config.sma_year]
             .groupby("date")[config.target_column].sum())
    sma = daily.rolling(config.sma_window).mean()
    plt.figure(figsize=(15, 6))
    plt.plot(daily.index, daily.values, label="Total daily sales", alpha=0.6)
    plt.plot(sma.index, sma.values, label=f"{config.sma_window}-day moving average", linewidth=2)
    plt.title(f"Daily sales in {config.sma_year}")
    plt.legend()
    plt.tight_layout()


def plot_sales_distribution(df: pd.DataFrame, config: PipelineConfig) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df[config.target_column], kde=True, ax=axes[0])
    axes[0].set_title("Sales distribution")
    sns.boxplot(x=df[config.target_column], ax=axes[1])
    axes[1].set_title("Sales box plot")
    plt.tight_layout()


def plot_high_correlations(df: pd.DataFrame) -> None:
    """Heatmap marking feature pairs with correlation above 0.8."""
    plt.figure(figsize=(10, 10))
    sns.heatmap(df.select_dtypes("number").corr() > 0.8, annot=True, cbar=False)
    plt.title("Feature pairs with correlation > 0.8")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    raw = load_data(config)
    data = engineer_features(raw, config)

    plot_mean_sales_by_feature(data, config)
    plot_mean_sales_by_day(data, config)
    plot_moving_average(data, config)
    plot_sales_distribution(data, config)
    plot_high_correlations(data)

    data = remove_outliers(data, config)
    X_train, X_val, y_train, y_val = split_data(data, config)
    X_train, X_val = scale_features(X_train, X_val)

    results = train_and_compare(get_models(), X_train, X_val, y_train, y_val)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())