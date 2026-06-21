# Day 43 — Outlier Removal Using IQR Method

## Why IQR Instead of Z-Score?

The Z-score method (Day 42) assumes the data is normally distributed. Real-world features are often **skewed** — income, purchase amounts, exam scores. For skewed distributions:
- The mean is pulled toward the tail
- The standard deviation is inflated
- Z-score boundaries become asymmetric and unreliable

The **IQR (Interquartile Range) method** is **robust to skewness** because it is based on percentiles, not the mean and standard deviation. It makes no assumption about distribution shape.

---

## What is IQR?

```
IQR = Q3 - Q1
```

where:
- **Q1** (25th percentile) = value below which 25% of data falls
- **Q3** (75th percentile) = value below which 75% of data falls

**Outlier boundaries (Tukey's fences):**
```
upper_limit = Q3 + 1.5 × IQR
lower_limit = Q1 - 1.5 × IQR
```

Values beyond these limits are flagged as outliers.

---

## Dataset

```python
df = pd.read_csv('placement.csv')
# 1000 rows: cgpa, placement_exam_marks, placed
```

Statistics for `placement_exam_marks`:
```
count    1000.000000
mean       32.225
std        19.131
min         0.000
25%        17.000     ← Q1
50%        28.000
75%        44.000     ← Q3
max       100.000
```

---

## Computing IQR Boundaries

```python
percentile25 = df['placement_exam_marks'].quantile(0.25)  # 17.0
percentile75 = df['placement_exam_marks'].quantile(0.75)  # 44.0

iqr = percentile75 - percentile25  # 27.0

upper_limit = percentile75 + 1.5 * iqr  # 44 + 40.5 = 84.5
lower_limit = percentile25 - 1.5 * iqr  # 17 - 40.5 = -23.5

print("Upper limit", upper_limit)  # 84.5
print("Lower limit", lower_limit)  # -23.5
```

The lower limit is -23.5 — since exam marks cannot be negative, no observations fall below the lower bound.

---

## Finding Outliers

```python
df[df['placement_exam_marks'] > upper_limit]
```

Result: **15 rows** with marks above 84.5:
```
     cgpa  placement_exam_marks  placed
9    7.75                  94.0       1
40   6.60                  86.0       1
134  6.33                  93.0       0
162  7.80                  90.0       0
630  6.56                  96.0       1
846  6.99                  97.0       0
917  5.95                 100.0       0
...
```

```python
df[df['placement_exam_marks'] < lower_limit]
# Empty DataFrame — no observations below -23.5
```

---

## Handling Outliers

### Strategy 1: Trimming (Removal)

```python
new_df = df[df['placement_exam_marks'] < upper_limit]
new_df.shape  # (985, 3) — 15 rows removed
```

The distribution and box plot after removal show no extreme values beyond the whiskers.

### Strategy 2: Capping (Winsorization)

```python
new_df_cap = df.copy()
new_df_cap['placement_exam_marks'] = np.where(
    new_df_cap['placement_exam_marks'] > upper_limit,
    upper_limit,
    np.where(
        new_df_cap['placement_exam_marks'] < lower_limit,
        lower_limit,
        new_df_cap['placement_exam_marks']
    )
)

new_df_cap.shape  # (1000, 3) — no rows removed
```

Values above 84.5 are capped to 84.5. All rows are preserved.

---

## Visualizing Before and After

```python
plt.figure(figsize=(16, 8))

plt.subplot(2, 2, 1)
sns.distplot(df['placement_exam_marks'])
plt.title('Original Distribution')

plt.subplot(2, 2, 2)
sns.boxplot(df['placement_exam_marks'])
plt.title('Original Box Plot')

plt.subplot(2, 2, 3)
sns.distplot(new_df['placement_exam_marks'])
plt.title('After Trimming')

plt.subplot(2, 2, 4)
sns.boxplot(new_df['placement_exam_marks'])
plt.title('After Trimming Box Plot')
```

The box plot after trimming shows no points beyond the whiskers — outliers are gone.

---

## The 1.5 × IQR Rule (Tukey's Fences)

The multiplier `1.5` is the standard convention introduced by John Tukey (the inventor of the box plot). It flags approximately 0.7% of data as outliers in a normal distribution — comparable to the ±3σ rule.

Alternative multipliers:
- `1.5 × IQR`: mild outliers (standard "box plot" definition)
- `3.0 × IQR`: extreme outliers (only flag the most egregious values)

---

## IQR vs Z-Score: Which to Use?

| Situation | Z-Score | IQR |
|---|---|---|
| Normally distributed data | Preferred | Works |
| Skewed data | Unreliable | Preferred |
| Mean/std affected by outliers | Problematic | Robust |
| Need box-plot compatible method | Less intuitive | Yes |

**Rule of thumb**: Check `skewness` first:
- `|skew| < 0.5`: roughly symmetric → Z-score is fine
- `|skew| > 0.5`: skewed → prefer IQR method

---

## Full Pattern for Multiple Features

```python
def iqr_bounds(series, multiplier=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier*iqr, q3 + multiplier*iqr

for col in ['placement_exam_marks', 'cgpa']:
    lower, upper = iqr_bounds(df[col])
    df = df[(df[col] >= lower) & (df[col] <= upper)]
```

**Important**: When applying to multiple columns sequentially, each removal changes the distribution of remaining data. Apply column by column, not simultaneously.

Also: For test data, always **cap** (not trim) — you cannot remove test rows.
