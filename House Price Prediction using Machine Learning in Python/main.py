"""
House Price Prediction.

Predicts a house's sale price from its lot, building and zoning details
(lot area, building class and type, zoning, lot configuration, overall
condition, year built and remodelled, exterior material and basement size).
Explores the data, keeps only houses with a known sale price, one-hot encodes
the categorical features inside a preprocessing pipeline, and compares
Support Vector Regression, Random Forest and Linear Regression using Mean
Absolute Percentage Error (MAPE).
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    data_path: str = "HousePricePrediction.xlsx"
    id_column: str = "Id"
    target_column: str = "SalePrice"
    drop_missing_target: bool = True     # rows without a price can't be used for training
    test_size: float = 0.2
    random_state: int = 0
    scale_features: bool = True          # SVR needs scaled inputs
    scale_target_for_svr: bool = True    # ...and a scaled target
    rf_estimators: int = 10
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and exploration
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    if config.data_path.endswith((".xlsx", ".xls")):
        df = pd.read_excel(config.data_path)
    else:
        df = pd.read_csv(config.data_path)
    print(f"Loaded {df.shape[0]} houses, {df.shape[1]} columns from {config.data_path}")
    print(df.head())
    return df.drop(columns=[config.id_column], errors="ignore")


def describe_column_types(df: pd.DataFrame) -> list:
    categorical = df.select_dtypes(include="object").columns.tolist()
    integer = df.select_dtypes(include="integer").columns.tolist()
    floats = df.select_dtypes(include="float").columns.tolist()
    print(f"\nCategorical variables: {len(categorical)} {categorical}")
    print(f"Integer variables:     {len(integer)}")
    print(f"Float variables:       {len(floats)}")
    return categorical


def clean_data(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Drop houses with no sale price, then any row with other missing values."""
    start = len(df)
    print("\nMissing values per column:")
    print(df.isnull().sum()[lambda s: s > 0].to_string())

    if config.drop_missing_target:
        df = df.dropna(subset=[config.target_column])
    else:
        df = df.assign(**{config.target_column: df[config.target_column].fillna(df[config.target_column].mean())})
    without_target = start - len(df)
    df = df.dropna()
    print(f"Removed {without_target} rows without a sale price and "
          f"{start - without_target - len(df)} rows with other gaps -> {len(df)} rows left")
    return df


# --------------------------------------------------------------------------- #
# Preprocessing and models
# --------------------------------------------------------------------------- #
def build_preprocessor(categorical: list, numeric: list, config: PipelineConfig) -> ColumnTransformer:
    numeric_step = StandardScaler() if config.scale_features else "passthrough"
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("numeric", numeric_step, numeric),
    ])


def get_models(config: PipelineConfig) -> dict:
    # SVR's default C=1 and epsilon=0.1 assume a target of roughly unit scale; with
    # prices in the hundreds of thousands it barely moves from the median. Scaling the
    # target (and inverting predictions afterwards) fixes that.
    svr = SVR()
    if config.scale_target_for_svr:
        svr = TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler())
    return {
        "Support Vector Regression": svr,
        "Random Forest": RandomForestRegressor(n_estimators=config.rf_estimators,
                                               random_state=config.random_state),
        "Linear Regression": LinearRegression(),
    }


def split_data(df: pd.DataFrame, config: PipelineConfig):
    X = df.drop(columns=[config.target_column])
    y = df[config.target_column]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    print(f"\nTrain: {len(X_train)} rows | Validation: {len(X_valid)} rows")
    return X_train, X_valid, y_train, y_valid


def train_and_compare(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    categorical = df.drop(columns=[config.target_column]).select_dtypes(include="object").columns.tolist()
    numeric = [c for c in df.columns if c not in categorical + [config.target_column]]
    X_train, X_valid, y_train, y_valid = split_data(df, config)

    rows = []
    for name, model in get_models(config).items():
        pipe = Pipeline([
            ("preprocess", build_preprocessor(categorical, numeric, config)),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_valid)
        rows.append({
            "Model": name,
            "MAPE (%)": 100 * mean_absolute_percentage_error(y_valid, y_pred),
            "MAE": mean_absolute_error(y_valid, y_pred),
        })

    baseline = np.full(len(y_valid), y_train.median())
    rows.append({
        "Model": "Baseline (predict median price)",
        "MAPE (%)": 100 * mean_absolute_percentage_error(y_valid, baseline),
        "MAE": mean_absolute_error(y_valid, baseline),
    })

    results = pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)
    print("\nModel comparison on the validation set:")
    print(results.round(2).to_string(index=False))
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_correlations(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.select_dtypes(include="number").corr(), cmap="BrBG",
                fmt=".2f", linewidths=2, annot=True)
    plt.title("Correlation between numeric features")
    plt.tight_layout()


def plot_unique_counts(df: pd.DataFrame, categorical: list) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categorical, y=[df[c].nunique() for c in categorical])
    plt.title("Number of unique values per categorical feature")
    plt.xticks(rotation=90)
    plt.tight_layout()


def plot_category_distributions(df: pd.DataFrame, categorical: list) -> None:
    n_cols = 2
    n_rows = -(-len(categorical) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    for ax, col in zip(axes.flat, categorical):
        counts = df[col].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax)
        ax.set_title(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=90)
    for ax in list(axes.flat)[len(categorical):]:
        ax.set_visible(False)
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    df = load_data(config)
    categorical = describe_column_types(df)

    plot_correlations(df)
    plot_unique_counts(df, categorical)
    plot_category_distributions(df, categorical)

    df = clean_data(df, config)
    results = train_and_compare(df, config)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())