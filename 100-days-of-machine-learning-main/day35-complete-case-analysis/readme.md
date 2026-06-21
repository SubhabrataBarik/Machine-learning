# Day 35 — Complete Case Analysis (CCA)

## What is Complete Case Analysis?

**Complete Case Analysis** (also called **listwise deletion**) is the simplest strategy for handling missing data: discard every row that has at least one missing value in any selected column, then train on what remains.

```python
new_df = df[cols].dropna()
```

No imputation is performed — only complete observations are used.

---

## Dataset

```python
df = pd.read_csv('data_science_job.csv')
# 19158 rows, 13 columns
```

Missing values per column:
```
city_development_index     2.50%
enrolled_university        2.01%
education_level            2.40%
major_discipline          14.68%
experience                 0.34%
company_size              30.99%
company_type              32.05%
gender                    23.53%
training_hours             4.00%
```

---

## Selecting Columns for CCA

Apply CCA only to columns where missingness is low (< 5%):

```python
cols = [var for var in df.columns 
        if df[var].isnull().mean() < 0.05 and df[var].isnull().mean() > 0]

# cols = ['city_development_index', 'enrolled_university', 
#          'education_level', 'experience', 'training_hours']
```

Applying CCA to `company_size` (31% missing) or `gender` (23.5% missing) would discard most of the dataset — impractical.

---

## Applying Complete Case Analysis

```python
new_df = df[cols].dropna()

len(new_df) / len(df)   # 0.8969 → 89.7% of data retained
df.shape, new_df.shape  # (19158, 13), (17182, 5)
```

17,182 rows are retained (1,976 rows dropped because at least one of the 5 columns was missing).

---

## Validating CCA: Distribution Preservation

CCA is only valid if the data is **MCAR (Missing Completely At Random)** — missing values occur randomly, not systematically. If data is MAR or MNAR, CCA introduces bias.

### Check 1: Numerical variable distributions

```python
fig = plt.figure()
ax = fig.add_subplot(111)

df['training_hours'].hist(bins=50, ax=ax, density=True, color='red')
new_df['training_hours'].hist(bins=50, ax=ax, color='green', density=True, alpha=0.8)
```

If the green (CCA) histogram overlaps closely with the red (original), distributions are preserved → CCA is appropriate.

```python
# KDE comparison
df['city_development_index'].plot.density(color='red')
new_df['city_development_index'].plot.density(color='green')
```

### Check 2: Categorical proportions

```python
temp = pd.concat([
    df['enrolled_university'].value_counts() / len(df),
    new_df['enrolled_university'].value_counts() / len(new_df)
], axis=1)
temp.columns = ['original', 'cca']
```

Result:
```
                     original    cca
no_enrollment        0.7212    0.7352
Full time course     0.1961    0.2007
Part time course     0.0625    0.0641
```

Proportions shift slightly after CCA. If shifts are large, CCA is biased and imputation should be preferred.

```python
# Education level proportions
                 original    cca
Graduate         0.6054    0.6198
Masters          0.2276    0.2341
High School      0.1053    0.1074
Phd              0.0216    0.0221
Primary School   0.0161    0.0166
```

---

## Missing Mechanisms

| Mechanism | Definition | CCA valid? |
|---|---|---|
| **MCAR** | Missingness is purely random — unrelated to any data | Yes |
| **MAR** | Missingness depends on other *observed* variables | Partially |
| **MNAR** | Missingness depends on the *missing value itself* | No |

Example of MNAR: older patients skip recording their age. Missing age values are systematically older — CCA biases the sample toward younger patients.

---

## When CCA is Appropriate

| Condition | CCA appropriate? |
|---|---|
| Missingness < 5% | Yes — minimal data loss |
| Missingness 5–15% | Possibly — check distribution shift |
| Missingness > 15% | No — too much data loss |
| Data is MCAR | Yes |
| Data is MAR or MNAR | No — introduces selection bias |
| Quick baseline model | Yes — use CCA first, then evaluate imputation |

---

## Pros and Cons

**Advantages:**
- Zero implementation complexity
- No leakage risk (unlike imputation which can leak test statistics into train)
- Preserves natural variability of observed data

**Disadvantages:**
- Loses potentially valuable rows
- Biased estimates if missingness is not MCAR
- Reduces effective sample size → lower statistical power

---

## CCA vs Imputation

| Aspect | CCA | Imputation |
|---|---|---|
| Code complexity | Minimal | Moderate |
| Data loss | Yes | No |
| Bias risk | High (if not MCAR) | Lower |
| Variance distortion | No | Yes (mean/median imputation reduces variance) |
| Preserves distribution | Yes | Partially |

---

## Practical Code Pattern

```python
# Select low-missingness columns
cols = [c for c in df.columns if 0 < df[c].isnull().mean() < 0.05]

# Drop rows with any null in those columns
df_cca = df[cols].dropna()

# Verify retention rate
print(f"Retained: {len(df_cca)/len(df)*100:.1f}%")

# Verify distribution preservation
for col in cols:
    print(f"{col}: original skew={df[col].skew():.3f}, cca skew={df_cca[col].skew():.3f}")
```

Always verify distribution preservation before committing to CCA. If the dropped rows are systematically different from the retained rows, your model learns from a biased sample and will not generalize to the full population.
