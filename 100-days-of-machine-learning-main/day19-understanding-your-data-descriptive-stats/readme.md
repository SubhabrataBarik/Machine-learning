# Day 19 — Understanding Your Data: Descriptive Statistics

## Overview

Before building any model, you must understand your data. This notebook covers the essential **Exploratory Data Analysis (EDA)** checklist using the Titanic dataset (891 rows × 12 columns). Every ML project should start with these seven questions.

---

## The EDA Checklist

### 1. How Big Is the Data?

```python
df.shape
# (891, 12)
```

Reveals the number of rows (samples) and columns (features). This tells you:
- Whether you have enough data for the model you plan to use
- Whether dimensionality reduction may be needed
- Rough compute requirements

---

### 2. How Does the Data Look?

```python
df.sample(5)
```

Random sample is better than `df.head(5)` because the first rows may not be representative — if the data is sorted by class or time, the head is biased.

Sample output (Titanic):
```
     PassengerId  Survived  Pclass  ...  Fare    Cabin  Embarked
498          499         0       1  ...  151.5  C22 C26        S
125          126         1       3  ...   11.2      NaN        C
604          605         1       1  ...   26.5      NaN        C
```

---

### 3. What Are the Column Data Types?

```python
df.info()
```

Output:
```
 #   Column       Non-Null Count  Dtype  
 0   PassengerId  891 non-null    int64  
 1   Survived     891 non-null    int64  
 ...
 5   Age          714 non-null    float64   ← has missing values
 10  Cabin        204 non-null    object    ← 77% missing
```

`df.info()` shows dtypes, non-null counts, and memory usage in one call. Key things to check:
- Numeric columns stored as `object` (string) — need type conversion
- Columns with fewer non-null entries than rows — have missing values
- Memory usage — tells you if you need to downcast types

---

### 4. Are There Missing Values?

```python
df.isnull().sum()
```

Output:
```
Age        177   # 19.9% missing
Cabin      687   # 77.1% missing — nearly useless
Embarked     2   # minor
```

Missing values require decisions: drop, impute, or encode as a separate category. The percentage matters:
- < 5% missing: safe to drop rows or use simple imputation
- 5–30% missing: imputation with mean/median/mode or model-based
- > 30% missing: consider dropping the column or using a missing indicator

---

### 5. How Does the Data Look Mathematically?

```python
df.describe()
```

Output:
```
          Age       Fare
count  714.0    891.000
mean    29.7     32.204
std     14.5     49.693
min      0.4      0.000
25%     20.1      7.910
50%     28.0     14.454
75%     38.0     31.000
max     80.0    512.329
```

`df.describe()` covers: count, mean, std, min, 25th/50th/75th percentiles, max.

Key patterns to look for:
- **Skewed distributions**: mean >> median (positive skew), mean << median (negative skew)
- **Outliers**: max >> 75th percentile by a large factor
- **Scale differences**: `Fare` ranges 0–512, `Age` ranges 0–80 — features need scaling for distance-based models
- **Suspicious values**: min = 0 for `Age` (passengers younger than 1 year)

---

### 6. Are There Duplicate Rows?

```python
df.duplicated().sum()
# 0
```

Duplicate rows bloat training data and can cause the model to overfit to repeated examples. Always check, especially after merging datasets.

To remove duplicates:
```python
df.drop_duplicates(inplace=True)
```

---

### 7. How Are Columns Correlated?

```python
df.corr()['Survived']
```

Output:
```
PassengerId   -0.005   (no relationship)
Pclass        -0.338   (higher class = more survived)
Age           -0.077   (slight negative)
SibSp         -0.035
Parch          0.082
Fare           0.257   (higher fare = more survived)
```

`df.corr()` computes Pearson correlation coefficients (−1 to +1). Features with high absolute correlation to the target are likely predictive. Watch out for:
- **Multicollinearity**: two features highly correlated with each other → redundant for linear models
- **Low correlation ≠ useless**: a feature may have nonlinear relationships not captured by Pearson

---

## Complete EDA Workflow

```python
import pandas as pd

df = pd.read_csv('train.csv')

# 1. Size
print(df.shape)

# 2. Sample
print(df.sample(5))

# 3. Types and nulls
df.info()

# 4. Missing values
print(df.isnull().sum())

# 5. Statistics
print(df.describe())

# 6. Duplicates
print(df.duplicated().sum())

# 7. Correlation
print(df.corr()['Survived'].sort_values())
```

---

## The Titanic Dataset

| Column | Type | Description |
|---|---|---|
| `PassengerId` | int | Unique identifier |
| `Survived` | int | Target: 0 = died, 1 = survived |
| `Pclass` | int | Ticket class (1st, 2nd, 3rd) |
| `Name` | str | Passenger name |
| `Sex` | str | Gender |
| `Age` | float | Age (177 missing) |
| `SibSp` | int | # siblings / spouses aboard |
| `Parch` | int | # parents / children aboard |
| `Fare` | float | Ticket price |
| `Cabin` | str | Cabin number (687 missing) |
| `Embarked` | str | Port of embarkation (S/C/Q) |

---

## Practical Tips

- Run `df.info()` and `df.describe()` as the very first step on every new dataset.
- Compare `mean` vs `median` (50%) in `describe()` — large difference indicates skewness and potential outliers.
- High missing rate (> 50%) usually means the column should be dropped unless the absence itself is informative.
- Correlation to the target guides feature selection; correlation between features reveals redundancy.
- Use `df.describe(include='all')` to include statistics for categorical (object) columns too.
