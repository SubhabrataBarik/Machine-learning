# Day 38 — Missing Indicator & Random Sample Imputation

## Overview

This day covers two advanced missing data techniques: **Missing Indicator** (encoding the fact that data was missing as a feature) and **Random Sample Imputation** (filling missing values with randomly drawn observed values).

---

## Part 1: Missing Indicator

### What Is It?

For each column with missing values, create a companion binary column (0/1) flagging whether the original value was missing.

```
Age     →  Age (imputed)  +  Age_missing (0 or 1)
Cabin   →  Cabin (imputed) + Cabin_missing (0 or 1)
```

### Why Use It?

When data is **MNAR** (Missing Not At Random), the absence of a value carries information. Adding a missing indicator lets the model learn patterns from the missingness itself.

Example: In medical data, doctors may not record a test result if the patient appears healthy. Missing lab values might actually predict good outcomes.

### Implementation with sklearn

```python
from sklearn.impute import SimpleImputer, MissingIndicator

# Step 1: Create missing indicator
indicator = MissingIndicator()
indicator.fit(X_train)
missing_flags = indicator.transform(X_train)  # binary array

# Step 2: Impute the original column
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Step 3: Combine
import numpy as np
X_train_final = np.hstack([X_train_imputed, missing_flags])
```

### Shortcut with `add_indicator`

```python
imputer = SimpleImputer(strategy='mean', add_indicator=True)
X_train_final = imputer.fit_transform(X_train)
```

`add_indicator=True` automatically appends the binary missing-indicator columns to the imputed output — one column per feature that had any missing values.

---

## Part 2: Random Sample Imputation

### What Is It?

Replace missing values with **randomly drawn values from the observed (non-missing) portion** of the same column.

```python
def random_sample_impute(df, col):
    # Non-missing values in this column
    observed = df[col].dropna()
    # Randomly sample to fill the gaps
    df[col] = df[col].fillna(observed.sample(df[col].isnull().sum(), replace=True).values)
    return df
```

### Why Use It?

- Preserves the original distribution more faithfully than mean/median imputation
- Does not create an artificial spike at one value
- Introduces realistic variability into imputed values
- Good for visualizations and reports where the distribution must look realistic

### Limitation

- Introduces randomness — results differ each run unless you set a seed
- Can impute extreme outlier values
- Assumes MCAR — random values only make sense if missingness is random

### Setting a Seed for Reproducibility

```python
np.random.seed(42)
random_sample_impute(df, 'Age')
```

---

## Automatically Selecting Imputer Parameters

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('model', LogisticRegression())
])

param_grid = {
    'imputer__strategy': ['mean', 'median', 'most_frequent']
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_params_)
# {'imputer__strategy': 'median'}
```

Let cross-validation choose the best imputation strategy for your data — don't guess.

---

## Choosing the Right Imputation Strategy

| Strategy | Distribution preserved | MCAR required | Best for |
|----------|----------------------|---------------|----------|
| Mean | No (spike at mean) | Yes | Symmetric numerical, low missing rate |
| Median | No (spike at median) | Yes | Skewed numerical, low missing rate |
| Mode | No (spike at mode) | Yes | Categorical, low missing rate |
| Constant | No | No | Arbitrary sentinel or 'Missing' category |
| Random Sample | Yes | Yes | When distribution preservation matters |
| Missing Indicator | — | No | When missingness is informative |
| KNN | Approximately | No | When neighbor relationships are meaningful |
| Iterative | Approximately | No | Complex multivariate patterns |

---

## Practical Tips

- Always add a missing indicator when missing rate > 5% — it costs nothing and may improve model performance.
- `MissingIndicator(features='all')` creates indicators for all columns; `features='missing-only'` (default) only for columns with any missingness.
- Random sample imputation is rarely used in production (non-reproducible) — prefer it for exploratory analysis only.
- After any imputation, verify: `df.isnull().sum().sum() == 0`.
