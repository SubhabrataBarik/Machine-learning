# Day 20 — Univariate Analysis

## Overview

**Univariate analysis** examines one variable at a time. It is the first step in understanding the distribution, spread, and shape of individual features in a dataset. This notebook uses the Titanic dataset to demonstrate both categorical and numerical univariate analysis techniques.

---

## Why Univariate Analysis?

Before building any model:
- Understand the distribution of each feature
- Detect outliers and skewness
- Decide which features need transformation (log, square root)
- Spot class imbalance in categorical features
- Understand the range of values for scaling decisions

---

## Categorical Univariate Analysis

### a. Count Plot

```python
import seaborn as sns
sns.countplot(df['Embarked'])
```

Shows the frequency of each category as bars. From the Titanic data:
- `S` (Southampton) — ~640 passengers
- `C` (Cherbourg) — ~168
- `Q` (Queenstown) — ~77

Reveals **class imbalance** — `S` dominates. If `Embarked` is a feature in a classifier, this imbalance might need to be considered.

Alternative:
```python
df['Survived'].value_counts().plot(kind='bar')
```

`value_counts()` returns counts in descending order. `.plot(kind='bar')` creates the bar chart without seaborn.

---

### b. Pie Chart

```python
df['Sex'].value_counts().plot(kind='pie', autopct='%.2f')
```

Output:
- Male: 64.76%
- Female: 35.24%

Pie charts are best for showing **proportions** when there are fewer than 5-6 categories. The `autopct='%.2f'` argument shows percentages with 2 decimal places on each slice.

**When to use**: pie chart for proportion storytelling; count plot for comparing raw frequencies.

---

## Numerical Univariate Analysis

### a. Histogram

```python
import matplotlib.pyplot as plt
plt.hist(df['Age'], bins=5)
```

Groups values into 5 equal-width bins and shows counts. Reveals:
- The shape of the distribution (normal, skewed, bimodal)
- The most common range of values
- Gaps in the data

With `bins=5` on Age (0.42–80):
- [0–16]: ~100 passengers
- [16–32]: ~346 — the most common age group
- [32–48]: ~188
- [48–64]: ~69
- [64–80]: ~11

**Choosing bins**: too few hides the shape; too many creates noise. A common heuristic is `bins=int(sqrt(n))`.

---

### b. Distribution Plot (KDE + Histogram)

```python
sns.distplot(df['Age'])
```

Overlays a **Kernel Density Estimate (KDE)** curve on a histogram. The KDE smooths the histogram into a continuous probability density function. Useful for:
- Comparing two distributions on the same plot
- Identifying bimodal distributions
- Seeing the approximate probability of any value range

---

### c. Box Plot

```python
sns.boxplot(df['Age'])
```

Visualizes the **five-number summary**:
- Minimum (excluding outliers)
- 25th percentile (Q1)
- Median (50th percentile)
- 75th percentile (Q3)
- Maximum (excluding outliers)
- Outliers: points beyond 1.5 × IQR from Q1/Q3

From Titanic Age:
```
min    0.42
Q1    20.12
median 28.00
Q3    38.00
max   80.00
```

Points beyond ~63.5 (Q3 + 1.5 × IQR = 38 + 1.5 × 17.88) appear as individual diamonds — outliers at ages 65, 70, 74, 80.

---

## Summary Statistics for Numerical Features

```python
df['Age'].min()    # 0.42
df['Age'].max()    # 80.0
df['Age'].mean()   # 29.70
df['Age'].skew()   # 0.389
```

**Skewness** measures asymmetry:
- `skew ≈ 0`: symmetric (approximately normal)
- `skew > 0`: right-skewed — long tail to the right, mean > median
- `skew < 0`: left-skewed — long tail to the left, mean < median

Age has `skew = 0.389` — slightly right-skewed (more young passengers than old, with a few elderly outliers pulling the mean up).

---

## Choosing the Right Plot

| Feature Type | Question | Plot |
|---|---|---|
| Categorical | How many in each category? | Count plot / bar chart |
| Categorical | What proportion is each category? | Pie chart |
| Numerical | What is the distribution shape? | Histogram / distplot |
| Numerical | What are the quartiles and outliers? | Box plot |
| Numerical | Quick statistical summary | `describe()` |

---

## What to Look For

| Pattern | Implication |
|---|---|
| Highly skewed distribution | Consider log or square-root transformation |
| Many outliers in box plot | Investigate — may need capping or removal |
| One category dominates (imbalance) | Address class imbalance before classification |
| Bimodal distribution | Feature may mix two populations |
| Narrow range (low variance) | Feature may have low predictive power |

---

## Libraries

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

Seaborn is built on top of Matplotlib and provides higher-level statistical plotting functions with better default styles.
