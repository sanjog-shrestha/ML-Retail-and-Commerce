"""
Box Office Revenue Prediction.

Predicts a film's domestic box-office revenue from its distributor, MPAA
rating, genres, number of opening theaters and days in release. Cleans the
numeric columns, log-transforms the skewed ones, turns genres into indicator
columns with a bag-of-words vectorizer, label-encodes the remaining categories,
then trains a Linear Regression baseline and an XGBoost regressor and compares
them with Mean Absolute Error.
"""

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class PipelineConfig:
    csv_path: str = "boxoffice.csv"
    encoding: str = "latin-1"
    title_column: str = "title"
    target_column: str = "domestic_revenue"
    # world/opening revenue would leak the answer; budget is left out as in the original analysis
    drop_columns: list = field(default_factory=lambda: ["world_revenue", "opening_revenue", "budget"])
    fill_mode_columns: list = field(default_factory=lambda: ["MPAA", "genres"])
    numeric_columns: list = field(default_factory=lambda: ["domestic_revenue", "opening_theaters", "release_days"])
    log_columns: list = field(default_factory=lambda: ["domestic_revenue", "opening_theaters", "release_days"])
    genre_column: str = "genres"
    sparse_genre_threshold: float = 0.95     # drop genre columns that are 0 for more than 95% of films
    categorical_columns: list = field(default_factory=lambda: ["distributor", "MPAA"])
    test_size: float = 0.1
    random_state: int = 22
    show_plots: bool = True


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_data(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path, encoding=config.encoding)
    print(f"Loaded {df.shape[0]} films, {df.shape[1]} columns from {config.csv_path}")
    print(df.describe().T)
    return df


def drop_unused_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    return df.drop(columns=[c for c in config.drop_columns if c in df.columns])


def handle_missing(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Fill missing ratings/genres with the most common value, then drop any other gaps."""
    df = df.copy()
    print("\nMissing values (% of rows):")
    print((df.isnull().mean() * 100).round(2).to_string())
    for col in config.fill_mode_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df.dropna()


def clean_numeric_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Strip '$' signs and thousands separators (if any) and convert to numbers."""
    df = df.copy()
    for col in config.numeric_columns:
        cleaned = df[col].astype(str).str.replace(r"[$,]", "", regex=True)
        df[col] = pd.to_numeric(cleaned, errors="coerce")
    before = len(df)
    df = df.dropna(subset=config.numeric_columns)
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with unreadable numbers")
    return df


def log_transform(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """log10 compresses long right tails so a few blockbusters don't dominate."""
    df = df.copy()
    for col in config.log_columns:
        df[col] = np.log10(df[col].clip(lower=1))
    return df


def encode_genres(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """One 0/1 column per genre word, dropping genres that almost never appear."""
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(df[config.genre_column]).toarray()
    genres = pd.DataFrame(matrix, columns=vectorizer.get_feature_names_out(), index=df.index)

    sparse = [g for g in genres.columns if (genres[g] == 0).mean() > config.sparse_genre_threshold]
    genres = genres.drop(columns=sparse)
    print(f"\nGenre columns kept: {list(genres.columns)} (removed {len(sparse)} rare)")
    return pd.concat([df.drop(columns=[config.genre_column]), genres], axis=1)


def encode_categoricals(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    for col in config.categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def split_data(df: pd.DataFrame, config: PipelineConfig):
    features = df.drop(columns=[config.title_column, config.target_column])
    target = df[config.target_column].values
    X_train, X_val, y_train, y_val = train_test_split(
        features, target, test_size=config.test_size, random_state=config.random_state
    )
    print(f"\nFeatures used: {list(features.columns)}")
    print(f"Train: {X_train.shape} | Validation: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def scale_features(X_train, X_val):
    """Standardize features, fitting the scaler on training data only."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val)


# --------------------------------------------------------------------------- #
# Modelling and evaluation
# --------------------------------------------------------------------------- #
def get_models(config: PipelineConfig) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "XGBoost": XGBRegressor(random_state=config.random_state),
    }


def train_and_compare(models: dict, X_train, X_val, y_train, y_val) -> pd.DataFrame:
    """MAE is in log10 units: 0.3 means predictions are off by a factor of about 2."""
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, model.predict(X_val))
        rows.append({
            "Model": name,
            "Train MAE": mean_absolute_error(y_train, model.predict(X_train)),
            "Validation MAE": val_mae,
            "Typical error factor": 10 ** val_mae,
        })
    results = pd.DataFrame(rows)
    baseline = mean_absolute_error(y_val, np.full_like(y_val, y_train.mean()))
    print("\nModel comparison (MAE on log10 revenue):")
    print(results.round(3).to_string(index=False))
    print(f"Baseline (always predict the training mean): validation MAE = {baseline:.3f}")
    return results


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_mpaa_counts(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.countplot(x="MPAA", data=df, order=sorted(df["MPAA"].unique()))
    plt.title("Films per MPAA rating")
    plt.tight_layout()


def print_revenue_by_mpaa(df: pd.DataFrame, config: PipelineConfig) -> None:
    print("\nMean domestic revenue by MPAA rating:")
    print(df.groupby("MPAA")[config.target_column].mean().round(0).to_string())


def plot_distributions(df: pd.DataFrame, columns: list, title_suffix: str) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    for ax, col in zip(axes, columns):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"{col} {title_suffix}")
    plt.tight_layout()


def plot_boxplots(df: pd.DataFrame, columns: list) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    for ax, col in zip(axes, columns):
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()


def plot_high_correlations(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr() > 0.8, annot=True, cbar=False)
    plt.title("Feature pairs with correlation > 0.8")
    plt.tight_layout()


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def run_pipeline(config: PipelineConfig = PipelineConfig()) -> pd.DataFrame:
    df = load_data(config)
    df = drop_unused_columns(df, config)
    df = handle_missing(df, config)
    df = clean_numeric_columns(df, config)

    plot_mpaa_counts(df)
    print_revenue_by_mpaa(df, config)
    plot_distributions(df, config.log_columns, "(raw)")
    plot_boxplots(df, config.log_columns)

    df = log_transform(df, config)
    plot_distributions(df, config.log_columns, "(log10)")

    df = encode_genres(df, config)
    df = encode_categoricals(df, config)
    plot_high_correlations(df)

    X_train, X_val, y_train, y_val = split_data(df, config)
    X_train, X_val = scale_features(X_train, X_val)
    results = train_and_compare(get_models(config), X_train, X_val, y_train, y_val)

    if config.show_plots:
        plt.show()
    return results


if __name__ == "__main__":
    run_pipeline(PipelineConfig())