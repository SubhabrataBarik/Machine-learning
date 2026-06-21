# Day 22 — Pandas Profiling

## Overview

**Pandas Profiling** (now called `ydata-profiling`) generates a comprehensive HTML EDA report from a DataFrame in a single function call. Instead of writing dozens of individual analysis lines, one call produces a complete interactive report covering distributions, correlations, missing values, duplicates, and more.

---

## Installation

```python
pip install pandas-profiling
# or newer:
pip install ydata-profiling
```

---

## Basic Usage

```python
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('train.csv')

prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```

This produces `output.html` — an interactive report you open in a browser. On the Titanic dataset (891 rows × 12 columns) it runs in seconds.

---

## What the Report Contains

### 1. Overview Tab
- Dataset shape (rows × columns)
- Total missing values and percentage
- Total duplicate rows
- Memory usage

### 2. Variables Tab (per column)
For each column the report shows:

**Numerical columns:**
- Min, max, mean, median, std
- IQR, skewness, kurtosis
- Histogram of distribution
- Count of zeros, missing, distinct values

**Categorical columns:**
- Value counts and bar chart
- Unique values and most frequent values
- Missing count and percentage

### 3. Correlations Tab
- Pearson correlation heatmap
- Spearman correlation heatmap
- Cramér's V (for categorical-categorical associations)

### 4. Missing Values Tab
- Matrix visualization showing where NaNs occur across rows/columns
- Bar chart of missing counts per column
- Heatmap of missing value co-occurrence

### 5. Duplicates Tab
- Lists all completely duplicate rows

### 6. Sample Tab
- First and last 10 rows of the dataset

---

## Titanic Dataset Example

```python
df = pd.read_csv('train.csv')
# Shape: (891, 12)

prof = ProfileReport(df)
prof.to_file(output_file='output.html')
```

Key findings the report reveals automatically:
- `Age`: 19.9% missing, skewness = 0.39 (right-skewed)
- `Cabin`: 77.1% missing — flagged as a high cardinality / missing column
- `Fare`: Heavy right skew, outliers at 512.33
- `Survived`: 38.4% positive class — mild imbalance
- Strong correlations: `Pclass` ↔ `Fare`, `SibSp` ↔ `Parch`

---

## Advanced Options

```python
# Minimal report (faster for large datasets)
prof = ProfileReport(df, minimal=True)

# Dark theme
prof = ProfileReport(df, dark_mode=True)

# Disable expensive correlations
prof = ProfileReport(df, correlations={"pearson": {"calculate": False}})

# Set title
prof = ProfileReport(df, title="Titanic EDA Report")

# Display inline in Jupyter
prof.to_notebook_iframe()
```

---

## When to Use Pandas Profiling

| Use Case | Benefit |
|----------|---------|
| First look at a new dataset | Instant overview without writing code |
| Sharing EDA with non-technical stakeholders | Clean HTML report they can open in browser |
| Quick missing value audit | Visual matrix shows pattern of missingness |
| Catching high-cardinality columns | Flagged automatically |
| Correlation analysis | All correlation methods in one tab |

---

## Limitations

- **Slow on large datasets** — > 100k rows can take minutes. Use `minimal=True` or sample first.
- **No automated decisions** — it shows you the data; you still decide what to do about skewness, outliers, and missing values.
- **Static snapshot** — the report reflects the data at generation time; you must regenerate after cleaning.

---

## Practical Workflow

```python
import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('data.csv')

# Full report saved to HTML
ProfileReport(df, title='EDA Report').to_file('eda_report.html')

# Quick inline view in Jupyter
ProfileReport(df, minimal=True).to_notebook_iframe()
```

After reviewing the report:
1. Handle columns flagged as high missing (> 40%)
2. Transform heavily skewed numerical columns
3. Encode high-cardinality categoricals carefully
4. Remove or investigate flagged duplicate rows
