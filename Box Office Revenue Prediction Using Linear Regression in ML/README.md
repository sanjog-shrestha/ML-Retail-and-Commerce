# Box Office Revenue Prediction

Predicts a film's **domestic box-office revenue** from its distributor, MPAA rating, genres, number of opening theaters and days in release. It compares a **Linear Regression** baseline with an **XGBoost** regressor, both trained on log-transformed revenue.

## What it does

1. Loads the film data (`boxoffice.csv`, 2,694 films): title, domestic and worldwide revenue, distributor, opening weekend revenue and theaters, budget, MPAA rating, genre and days in release.
2. Drops `world_revenue` and `opening_revenue`. Both are measured *after* release and are closely tied to the answer, so using them would be cheating. `budget` is also left out, as in the original analysis.
3. Fills missing MPAA ratings and genres with the most common value and drops any other incomplete rows.
4. Cleans the numeric columns (removes `$` signs and thousands separators where present) and converts them to numbers.
5. Plots films per MPAA rating, prints mean revenue per rating, and plots the distributions and box plots of revenue, opening theaters and release days.
6. Applies a **log10 transform** to those three skewed columns and re-plots their distributions.
7. Turns the genre text into 0/1 indicator columns with `CountVectorizer`, dropping genres that appear in fewer than 5% of films.
8. Label-encodes `distributor` and `MPAA` and plots a heatmap of highly correlated features.
9. Makes a random 90% / 10% train/validation split and standardizes the features.
10. Trains Linear Regression and XGBoost and compares their **Mean Absolute Error (MAE)** with a naive baseline that always predicts the average.

## Concepts covered

- **Avoiding target leakage** — worldwide and opening-weekend revenue are mostly *made up of* domestic revenue, so a model given them would look excellent and be useless for predicting a film before it earns anything. Dropping them keeps the task honest.
- **Cleaning numbers stored as text** — revenue figures often come as `"$1,234,567"`. Removing the `$` and `,` characters before `pd.to_numeric` turns them into real numbers; `errors="coerce"` turns anything still unreadable into `NaN` so it can be dropped.
- **Log transforms for skewed data** — box-office revenue has a long right tail: a few blockbusters earn hundreds of times more than a typical film. Taking `log10` makes the distribution more symmetric, so the model isn't dominated by a handful of outliers and errors become *relative* rather than absolute.
- **Bag-of-words for categories** — `CountVectorizer` splits each genre string into words and creates one column per word, which also handles films listed with several genres (e.g. `"Action Comedy"`).
- **Dropping rare features** — a genre that is 0 for 95%+ of films gives the model almost nothing to learn from and mainly adds noise.
- **Linear vs. tree-based models** — Linear Regression fits one straight-line relationship per feature; XGBoost builds many decision trees and can capture interactions (for example, a genre that only does well with a wide release). The comparison shows whether that extra flexibility helps.
- **Reading MAE on a log scale** — because the target is `log10(revenue)`, an MAE of 0.3 means predictions are typically off by a *factor* of 10^0.3 ≈ 2. The script prints this "typical error factor" next to each MAE.
- **Baselines** — a model is only useful if it beats simply predicting the average revenue for every film. The script prints that baseline next to the real models.

## Project structure

```
main.py      # full pipeline, organized into functions (config, loading,
              # leakage removal, missing values, numeric cleaning, log
              # transform, genre vectorizing, encoding, split, scaling,
              # model comparison, plotting)
README.md    # this file
```

The code is organized into small, named functions driven by a `PipelineConfig` dataclass: `load_data`, `drop_unused_columns`, `handle_missing`, `clean_numeric_columns`, `log_transform`, `encode_genres`, `encode_categoricals`, `split_data`, `scale_features`, `get_models`, `train_and_compare`, plus the plotting functions. `run_pipeline()` runs them in order.

Settings such as which columns to drop, the rare-genre threshold and the validation size live in `PipelineConfig`, so you can experiment without touching the pipeline code. For example, to put `budget` back in as a feature:

```python
from main import PipelineConfig, run_pipeline

run_pipeline(PipelineConfig(drop_columns=["world_revenue", "opening_revenue"]))
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

## How to run

Place `boxoffice.csv` in the same directory as `main.py`, then:

```bash
python main.py
```

## Sample output

Running the script prints dataset statistics, mean revenue by MPAA rating, the genre columns kept, the features used and a model comparison table, then displays the charts.

**Distributions after the log10 transform**

![Log-transformed distributions](log_distributions.png)

**Model comparison (MAE on log10 revenue)**

| Model | Train MAE | Validation MAE |
|-------|----------:|---------------:|
| Linear Regression | _run to fill in_ | _run to fill in_ |
| XGBoost | _run to fill in_ | _run to fill in_ |
| Baseline (predict the mean) | – | _run to fill in_ |

## A note on the results

The original notebook reported an XGBoost training MAE of 0.210 and a validation MAE of 0.636. Those figures came from a corrupted target. The cleaning step removed the first character of every revenue value on the assumption that it was a `$` sign. The values in this file are plain numbers, so it removed the leading *digit* instead: 6,026,491 became 26,491. This version only removes `$` and `,` characters, so the numbers will change when you rerun it.

Training error being far below validation error in that run (0.21 vs. 0.64) is also a sign that XGBoost was **overfitting**: it memorized the training films rather than learning patterns that carry over to new ones. Comparing against the Linear Regression and mean baselines shows how much real signal the features contain.

Keep expectations modest. The first rows of this dataset don't match real films (*Titanic* listed as rated G and distributed by Disney, *The Avengers* with $6 million domestic but $1.27 billion worldwide), and mean revenue was almost identical across every MPAA rating. That suggests the data is partly or fully synthetic, in which case no model will predict revenue much better than the baseline. A real dataset such as the [TMDB box-office data](https://www.kaggle.com/competitions/tmdb-box-office-prediction) would be a natural next step, ideally with budget, release date and cast/crew features.

## References

- [scikit-learn — CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [XGBoost documentation](https://xgboost.readthedocs.io/)
- [GeeksforGeeks — Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/), used as a general reference/inspiration while working on this project.

---
*This is a personal learning project. Predictions from this model should not be used for real business decisions.*
