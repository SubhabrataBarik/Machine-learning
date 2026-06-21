# Day 42 — Outlier Removal Using Z-Score

## What is an Outlier?

An **outlier** is a data point that lies far from the bulk of the distribution. Outliers can arise from data entry errors, measurement errors, or rare but genuine extreme values.

Outliers distort model training — they inflate the loss for linear models, skew the mean, and can dominate coefficient estimates.

---

## Z-Score Method

The **Z-score** (standard score) measures how many standard deviations a data point is from the mean:

```
Z = (x - μ) / σ
```

By the empirical rule (68-95-99.7 rule):
- Z ∈ [-1, 1]: 68% of data
- Z ∈ [-2, 2]: 95% of data
- Z ∈ [-3, 3]: 99.7% of data

Points with `|Z| > 3` are considered outliers.

---

## Dataset

```python
df = pd.read_csv('placement.csv')
# 1000 rows: cgpa, placement_exam_marks, placed
```

Statistics for `cgpa`:
```
Mean:  6.9612
Std:   0.6159
Min:   4.89
Max:   9.12
```

---

## Method 1: Boundary Value Calculation

```python
upper_limit = df['cgpa'].mean() + 3 * df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3 * df['cgpa'].std()

print("Highest allowed:", upper_limit)  # 8.809
print("Lowest allowed:", lower_limit)   # 5.114
```

### Finding outliers

```python
df[(df['cgpa'] > 8.80) | (df['cgpa'] < 5.11)]
```

Result:
```
     cgpa  placement_exam_marks  placed
485  4.92                  44.0       1
995  8.87                  44.0       1
996  9.12                  65.0       1
997  4.89                  34.0       0
999  4.90                  10.0       1
```

5 outliers detected (0.5% of data).

---

## Method 2: Z-Score Column

```python
df['cgpa_zscore'] = (df['cgpa'] - df['cgpa'].mean()) / df['cgpa'].std()
```

Filter outliers by Z-score:
```python
df[df['cgpa_zscore'] > 3]   # 2 rows (high end)
df[df['cgpa_zscore'] < -3]  # 3 rows (low end)
df[(df['cgpa_zscore'] > 3) | (df['cgpa_zscore'] < -3)]  # 5 total
```

Mathematically equivalent to Method 1. The Z-score column is useful for auditing and visualization.

---

## Handling Outliers: Two Strategies

### Strategy 1: Trimming (Removal)

Delete the outlier rows entirely:

```python
new_df = df[(df['cgpa_zscore'] < 3) & (df['cgpa_zscore'] > -3)]
# 995 rows (5 removed)
```

Or using boundary values:
```python
new_df = df[(df['cgpa'] < 8.80) & (df['cgpa'] > 5.11)]
```

**Pros**: Simple, removes contamination completely.
**Cons**: Loses data; if outliers are genuine, removes real information.

---

### Strategy 2: Capping (Winsorization)

Instead of removing outliers, clip them to the boundary values:

```python
upper_limit = df['cgpa'].mean() + 3 * df['cgpa'].std()
lower_limit = df['cgpa'].mean() - 3 * df['cgpa'].std()

df['cgpa'] = np.where(
    df['cgpa'] > upper_limit,
    upper_limit,
    np.where(
        df['cgpa'] < lower_limit,
        lower_limit,
        df['cgpa']
    )
)
```

After capping:
```
count    1000.000000   (no rows removed)
mean        6.9615     (nearly unchanged)
std         0.6127     (slightly reduced)
min         5.1135     (was 4.89)
max         8.8089     (was 9.12)
```

**Pros**: No data loss; preserves extreme observations as "extreme but valid".
**Cons**: Artificially piles values at the boundary; distorts distribution near tails.

---

## Trimming vs Capping

| Aspect | Trimming | Capping |
|---|---|---|
| Data loss | Yes (rows deleted) | No |
| Distribution | Clean tails | Spikes at boundary |
| Use case | Outliers are errors | Outliers may be genuine |
| Sample size | Reduced | Preserved |

---

## When Z-Score Method Applies

Z-score assumes the data follows (approximately) a **normal distribution**. For skewed distributions:
- The mean is pulled toward the tail
- The standard deviation is inflated
- Z-score boundaries become asymmetric and unreliable

```python
df['placement_exam_marks'].skew()
# 0.8356  ← moderately right-skewed
```

For `placement_exam_marks` with skew 0.84, the Z-score method is less appropriate — use IQR (Day 43) or percentile method (Day 44) for skewed distributions.

---

## Z-Score Threshold Choices

| Threshold | % of data within bounds | Typical use |
|---|---|---|
| ±1σ | 68.3% | Very aggressive removal |
| ±2σ | 95.4% | Moderate |
| ±3σ | 99.7% | Standard (default) |
| ±4σ | 99.994% | Conservative |

---

## Full Code Pattern

```python
def remove_outliers_zscore(df, col, threshold=3):
    mean = df[col].mean()
    std  = df[col].std()
    return df[(df[col] >= mean - threshold*std) & (df[col] <= mean + threshold*std)]

def cap_outliers_zscore(df, col, threshold=3):
    mean = df[col].mean()
    std  = df[col].std()
    df[col] = np.clip(df[col], mean - threshold*std, mean + threshold*std)
    return df
```

Always compute mean and std from **training data** — never from the full dataset before train/test split, as that leaks test distribution statistics.

---

## Z-Score vs Other Outlier Methods

| Method | Based on | Robust to outliers | Assumes normality |
|---|---|---|---|
| Z-score | Mean, Std | No | Yes |
| IQR (Day 43) | Median, Quartiles | Yes | No |
| Percentile (Day 44) | Percentile cutoffs | Yes | No |

**Summary**: Use Z-score for normally distributed features; use IQR or percentile methods for skewed ones.
