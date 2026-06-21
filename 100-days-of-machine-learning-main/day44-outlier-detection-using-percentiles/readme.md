# Day 44 — Outlier Detection Using Percentiles

## What is Percentile-Based Outlier Detection?

Instead of using formulas based on mean/std (Z-score) or IQR, percentile-based detection **directly cuts at a chosen quantile**. Any value below the Nth percentile or above the (100-N)th percentile is treated as an outlier.

```python
upper_limit = df[col].quantile(0.99)   # 99th percentile
lower_limit = df[col].quantile(0.01)   # 1st percentile
```

This removes the top 1% and bottom 1% of values — exactly 2% of the data by definition, regardless of distribution shape.

---

## Dataset

```python
df = pd.read_csv('weight-height.csv')
# 10,000 rows: Gender, Height (inches), Weight (pounds)
```

Statistics for `Height`:
```
count    10000.000000
mean        66.367560
std          3.847528
min         54.263133
25%         63.505620
50%         66.318070
75%         69.174262
max         78.998742
```

There are extreme values at both tails (< 57 inches and > 77 inches).

---

## Setting Percentile Boundaries

```python
upper_limit = df['Height'].quantile(0.99)
lower_limit = df['Height'].quantile(0.01)

print(upper_limit)  # 74.786 inches (6'2")
print(lower_limit)  # 58.134 inches (4'10")
```

---

## Strategy 1: Trimming

```python
new_df = df[(df['Height'] <= 74.78) & (df['Height'] >= 58.13)]
new_df.shape  # (9799, 3) — 201 rows removed (2.01%)
```

201 rows removed — guaranteed by the choice of 1st and 99th percentile. Exactly 2% of data falls outside those bounds by definition.

### Comparison before and after

```
Before trimming:
  count    10000     std=3.848    min=54.26   max=78.99

After trimming:
  count     9799     std=3.644    min=58.13   max=74.77
```

Standard deviation drops from 3.848 to 3.644 — the tails are removed, distribution is tighter.

---

## Strategy 2: Capping (Winsorization)

```python
df['Height'] = np.where(
    df['Height'] >= upper_limit,
    upper_limit,
    np.where(
        df['Height'] <= lower_limit,
        lower_limit,
        df['Height']
    )
)
```

After capping:
```
count    10000       (no rows removed)
mean     66.366      (unchanged)
std       3.796      (slightly reduced)
min      58.134      (was 54.26)
max      74.786      (was 78.99)
```

All rows are preserved. Values below 58.13 are set to 58.13; values above 74.79 are set to 74.79.

---

## Choosing the Percentile Threshold

| Percentile cut | % removed | Use case |
|---|---|---|
| 1st / 99th | 2% | Standard; removes top/bottom 1% each |
| 2nd / 98th | 4% | More aggressive |
| 5th / 95th | 10% | Heavy cleaning of noisy data |
| 0.5th / 99.5th | 1% | Conservative — only extreme extremes |

The choice depends on domain knowledge and how much contamination you expect. For normally distributed data, the 1/99 percentile is very similar to the ±2.33σ boundary.

---

## Percentile vs Z-Score vs IQR

| Method | Assumption | % removed (default) | Best for |
|---|---|---|---|
| Z-score (±3σ) | Normal distribution | ~0.3% | Normally distributed features |
| IQR (1.5×) | None | ~0.7% | Skewed distributions |
| Percentile (1/99) | None | Exactly 2% | Any distribution; controllable removal |

The **percentile method** gives the most direct control: you decide exactly what fraction of data to remove, without reasoning about σ or IQR.

---

## Production Rule: Cap, Don't Trim Test Data

Trimming removes rows — you cannot do this for test data because you need predictions for every row.

```python
# Compute limits from training data
upper_limit = X_train['Height'].quantile(0.99)
lower_limit = X_train['Height'].quantile(0.01)

# Trim training data (acceptable to remove rows)
X_train = X_train[(X_train['Height'] >= lower_limit) & (X_train['Height'] <= upper_limit)]

# Cap test data (NEVER remove test rows)
X_test['Height'] = np.clip(X_test['Height'], lower_limit, upper_limit)
```

---

## Summary: Three Outlier Handling Methods Compared

| Day | Method | Formula | Robust? | Gaussian needed? |
|---|---|---|---|---|
| 42 | Z-Score | `μ ± 3σ` | No | Yes |
| 43 | IQR | `Q1/Q3 ± 1.5×IQR` | Yes | No |
| 44 | Percentile | `quantile(0.01/0.99)` | Yes | No |

**Decision guide:**
- Normal data → Z-score
- Skewed data → IQR
- Need exact control over removal % → Percentile
- Any method applied to test data → always **cap**, never trim

Always compute outlier boundaries on training data only — applying test-set statistics in preprocessing is data leakage.
