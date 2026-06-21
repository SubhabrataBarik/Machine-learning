# Day 36 — Imputing Numerical Data

## Overview

When numerical columns have missing values, imputation replaces them with estimated values so the dataset can be used for modeling. This day covers two core strategies: **mean/median imputation** and **arbitrary value imputation**.

---

## Dataset: Titanic Toy (titanic_toy.csv)

Subset of the Titanic dataset with selected numerical and categorical columns demonstrating missing value patterns.

---

## Part 1: Mean and Median Imputation

### When to Use

| Statistic | Use when |
|-----------|----------|
| **Mean** | Distribution is approximately normal (symmetric) |
| **Median** | Distribution is skewed or has outliers |

Mean is sensitive to outliers — one extreme value shifts it significantly. Median is robust.

### sklearn SimpleImputer

```python
from sklearn.impute import SimpleImputer

# Mean imputation
imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train[['Age']])
X_train['Age'] = imputer.transform(X_train[['Age']])
X_test['Age']  = imputer.transform(X_test[['Age']])

# Median imputation
imputer = SimpleImputer(strategy='median')
```

`fit()` computes the statistic from training data only. `transform()` replaces NaN with that statistic — applied identically to test data.

### Checking the Learned Value

```python
imputer.statistics_
# array([29.36])  ← mean or median of 'Age' from training set
```

---

## Effect on Distribution

Mean/median imputation:
- Preserves the mean (if mean imputation)
- **Distorts the distribution** — creates a spike at the imputed value
- **Reduces variance** — spreads out the distribution artificially
- **Distorts correlations** with other variables

These distortions are acceptable when missing rate is low (< 5%). At high missing rates, use model-based imputation (KNN, Iterative).

---

## Part 2: Arbitrary Value Imputation

### What Is It?

Replace missing values with a specific value that is clearly outside the normal range — signaling that the value was missing.

Common choices:
- `-999` (for features that are always positive)
- `9999` (sentinel value)
- `0` (if zero is not a valid measurement)
- `-1` (for features that are always positive)

### When to Use

- When you suspect missingness is **MNAR** (Missing Not At Random) — the fact that a value is missing carries information
- When you want the model to learn that "was missing" is a distinct state
- Tree-based models can learn to split on the arbitrary sentinel value effectively

### Implementation

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='constant', fill_value=-999)
imputer.fit(X_train[['Age']])
X_train['Age'] = imputer.transform(X_train[['Age']])
X_test['Age']  = imputer.transform(X_test[['Age']])
```

Or with pandas:
```python
df['Age'].fillna(-999, inplace=True)
```

---

## Comparing Mean vs Median vs Arbitrary

```python
# Visualize the effect on the distribution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df['Age'].fillna(df['Age'].mean()).hist(ax=axes[0])
df['Age'].fillna(df['Age'].median()).hist(ax=axes[1])
df['Age'].fillna(-999).hist(ax=axes[2])
```

- Mean/median: spike at the imputed value within the normal range
- Arbitrary: a separate bar far from the distribution (visible outlier at -999)

---

## End-of-Tail Imputation

A variant of arbitrary imputation: replace NaN with a value at the far tail of the distribution (e.g., mean ± 3*std):

```python
mean = df['Age'].mean()
std  = df['Age'].std()
upper_tail = mean + 3 * std

df['Age'].fillna(upper_tail, inplace=True)
```

---

## Always Fit on Train Only

```python
# CORRECT
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test  = imputer.transform(X_test)

# WRONG — leaks test statistics into training
imputer.fit_transform(X_train)
imputer.fit_transform(X_test)   # fits again on test
```

---

## Practical Tips

- Add `add_indicator=True` to create a companion binary column flagging where imputation occurred: `SimpleImputer(strategy='mean', add_indicator=True)`.
- For highly skewed features (income, prices), median imputation is almost always better than mean.
- Arbitrary value imputation works best with tree models; linear models may treat the sentinel as a real value.
- After imputation, verify with `df.isnull().sum()` that no NaN remains.
