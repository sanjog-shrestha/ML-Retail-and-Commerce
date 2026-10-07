# Automobile Data Analysis

Exploratory data analysis of the [UCI Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile): cleans and transforms the raw car specifications, groups prices into bands, explores how **price** relates to drive wheels, body style and engine size, and uses a **one-way ANOVA** to test whether two car makes differ in average price.

## What it does

1. Loads the car data (`output.csv`), drops the saved row-index column and assigns the 26 column names (make, fuel type, body style, dimensions, engine specs, fuel economy, price…).
2. Reports missing values. In this dataset they are written as `?`, not left blank.
3. Converts city fuel economy from **miles per gallon** to **litres per 100 km** (`235 / mpg`) and renames the column to `city-L/100km`.
4. Removes cars with an unknown price and converts `price` to an integer.
5. **Normalizes** `length`, `width` and `height` to a 0–1 scale by dividing each by its maximum.
6. **Bins** prices into three equal-width bands (Low / Medium / High) and plots how many cars fall in each.
7. Shows **one-hot encoding** of `fuel-type` into gas / diesel indicator columns.
8. Plots the price distribution, price by drive wheels, and engine size against price.
9. Builds a **pivot table** of mean price by drive wheels and body style, and shows it as a heatmap.
10. Runs a **one-way ANOVA** comparing Honda and Subaru prices.
11. Fits and plots a **linear regression line** of price against engine size.

## Concepts covered

- **Handling non-standard missing values** — `isnull()` only finds real `NaN`s. Here missing entries are the string `?`, so they must be looked for directly. Otherwise the whole column is read as text.
- **Unit conversion** — turning mpg into L/100km makes the data easier to read for metric users, and shows how to derive a new feature from an existing one. Note that the scale flips: a *higher* L/100km means a *less* efficient car.
- **Simple feature scaling** — dividing by the maximum puts length, width and height on the same 0–1 scale so they can be compared directly, without changing their shape.
- **Binning** — `pd.cut` turns a continuous variable (price) into categories. Equal-width bins split the price *range* evenly, so if prices are skewed most cars end up in the lowest bin.
- **Indicator (dummy) variables** — `pd.get_dummies` turns a text category into 0/1 columns that a model can use.
- **Group-by and pivot tables** — grouping by two categorical columns and pivoting gives a compact grid of mean prices, which a heatmap makes easy to scan.
- **One-way ANOVA** — the F-test compares the variation *between* groups with the variation *within* them. A small p-value (below 0.05) suggests the groups' means really differ; a large one means any difference could be due to chance.
- **Linear regression plot** — `sns.regplot` draws the best-fit line with a confidence band, giving a quick visual check of how strongly engine size predicts price.

## Project structure

```
main.py      # full pipeline, organized into functions (config, loading,
              # missing-value report, unit conversion, price cleaning,
              # normalization, binning, dummies, pivot table, ANOVA, plotting)
README.md    # this file
```

The code is organized into small, named functions driven by a `PipelineConfig` dataclass — `load_data`, `report_missing`, `convert_city_mpg`, `clean_price`, `normalize_dimensions`, `bin_price`, `show_fuel_type_dummies`, `price_by_drive_and_body`, `anova_by_make`, plus the plotting functions — tied together by `run_pipeline()`.

Settings such as the price bin labels and which makes to compare in the ANOVA live in `PipelineConfig`, so you can experiment without touching the pipeline code:

```python
from main import PipelineConfig, run_pipeline

run_pipeline(PipelineConfig(anova_makes=("toyota", "nissan", "mazda")))
```

## Requirements

```
numpy
pandas
matplotlib
seaborn
scipy
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scipy
```

## How to run

Place the dataset (`output.csv`: the UCI Automobile data saved with a leading index column) in the same directory as `main.py`, then:

```bash
python main.py
```

If your CSV has no index column, set `drop_first_column=False` in `PipelineConfig`.

## Sample output

Running the script prints the missing-value report, the number of cars in each price bin, the fuel-type dummies, summary statistics, the mean-price pivot table and the ANOVA result, then displays the charts.

**Mean price by drive wheels and body style**

![Price heatmap](price_heatmap.png)

**Engine size vs. price with linear fit**

![Engine size regression](engine_size_regression.png)

## A note on the results

Engine size shows one of the clearest relationships in the dataset: bigger engines go with higher prices, and the regression line fits the bulk of the cars well. Rear-wheel-drive cars are noticeably more expensive than front- and four-wheel-drive ones in the pivot table, because luxury and sports models in this dataset tend to be rear-wheel drive.

The ANOVA between Honda and Subaru is a useful example of a *non*-result: both makes sell mainly in the same budget segment, so their price distributions overlap heavily, and the test is not expected to find a significant difference. Comparing makes from different segments (for example Honda vs. Jaguar) would show what a significant result looks like. This project stops at exploration. A natural next step would be fitting a multiple linear regression on engine size, horsepower, curb weight and the other strongly correlated features to actually predict price.

## References

- [UCI Machine Learning Repository — Automobile dataset](https://archive.ics.uci.edu/dataset/10/automobile)
- [SciPy — `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
- [GeeksforGeeks — Machine Learning Projects](https://www.geeksforgeeks.org/machine-learning/machine-learning-projects/), used as a general reference/inspiration while working on this project.

---
*This is a personal learning project for practising exploratory data analysis.*
