"""
Sales Forecasting with XGBoost.

Turns daily sales into a supervised-learning problem by using the previous
N days' sales (lag features) to predict today's sales, trains an XGBoost
regressor on the earlier part of the history, and evaluates it on the most
recent part with RMSE and an actual-vs-predicted plot.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "train.csv"          # Superstore sales dataset
    date_column: str = "Order Date"
    date_format: str = "%d/%m/%Y"
    target_column: str = "Sales"
    aggregate_by_date: bool = True       # sum orders into one row per day before lagging
    n_lags: int = 5                      # number of previous days used as features
    test_size: float = 0.2               # last 20% of the timeline is held out
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 5
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Data loading and preparation
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    """Read the CSV and parse the order date."""
    df = pd.read_csv(config.csv_path)
    df[config.date_column] = pd.to_datetime(df[config.date_column], format=config.date_format)
    print(f"Loaded {len(df)} rows from {config.csv_path}")
    print(df.head())
    return df


def daily_sales(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Total sales per day, sorted chronologically."""
    return (
        df.groupby(config.date_column)[config.target_column]
        .sum()
        .reset_index()
        .sort_values(config.date_column)
        .reset_index(drop=True)
    )


def build_series(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """The series the model learns from: daily totals, or raw order rows."""
    if config.aggregate_by_date:
        return daily_sales(df, config)
    return df[[config.date_column, config.target_column]].reset_index(drop=True)


def create_lag_features(series: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Add lag_1 ... lag_N columns (sales 1..N steps earlier) and drop incomplete rows."""
    lagged = series.copy()
    for i in range(1, config.n_lags + 1):
        lagged[f"lag_{i}"] = lagged[config.target_column].shift(i)
    return lagged.dropna().reset_index(drop=True)


def split_chronologically(lagged: pd.DataFrame, config: PipelineConfig):
    """Earlier rows train, the most recent `test_size` fraction tests (no shuffling)."""
    feature_cols = [f"lag_{i}" for i in range(1, config.n_lags + 1)]
    split_idx = int(len(lagged) * (1 - config.test_size))

    train, test = lagged.iloc[:split_idx], lagged.iloc[split_idx:]
    X_train, y_train = train[feature_cols], train[config.target_column]
    X_test, y_test = test[feature_cols], test[config.target_column]
    test_dates = test[config.date_column]

    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return X_train, X_test, y_train, y_test, test_dates


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, config: PipelineConfig) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: xgb.XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    predictions = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    print(f"RMSE on test period: {rmse:.2f}")
    return predictions, rmse


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_sales_trend(sales_by_date: pd.DataFrame, config: PipelineConfig) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(sales_by_date[config.date_column], sales_by_date[config.target_column],
             label="Sales", color="red")
    plt.title("Sales Trend Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()


def plot_predictions(test_dates: pd.Series, y_test: pd.Series, predictions: np.ndarray) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates, y_test, label="Actual Sales", color="red")
    plt.plot(test_dates, predictions, label="Predicted Sales", color="green")
    plt.title("Sales Forecasting using XGBoost")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> float:
    raw = load_data(config)
    plot_sales_trend(daily_sales(raw, config), config)

    series = build_series(raw, config)
    lagged = create_lag_features(series, config)
    X_train, X_test, y_train, y_test, test_dates = split_chronologically(lagged, config)

    model = train_xgboost(X_train, y_train, config)
    predictions, rmse = evaluate_model(model, X_test, y_test)
    plot_predictions(test_dates, y_test, predictions)

    if config.show_plots:
        plt.show()
    return rmse


if __name__ == "__main__":
    run_pipeline(PipelineConfig())