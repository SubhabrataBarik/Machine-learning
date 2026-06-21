# Day 37 — Handling Missing Categorical Data

## The Problem

Mean and median imputation only apply to numerical features. Categorical columns with missing values need different strategies:

1. **Frequent value imputation** — fill with the mode (most common category)
2. **Missing category imputation** — fill with a new category `"Missing"`

---

## Dataset

```python
df = pd.read_csv('train.csv', usecols=['GarageQual', 'FireplaceQu', 'SalePrice'])
# House Prices dataset (Kaggle), 1460 rows
```

Missing rates:
```
GarageQual     5.55%    (81 missing)
FireplaceQu   47.26%   (690 missing)
```

These represent very different situations — a 5% miss-rate vs 47%. The strategy must account for this.

---

## Strategy 1: Frequent Value Imputation (Mode)

Replace missing values with the **most frequent category** in the training data.

### Exploring the data first

```python
df['GarageQual'].value_counts()
# TA     1379
# Fa       78
# Gd       14

df['GarageQual'].mode()
# 0    TA    ← most frequent
```

**Before imputing**, check if houses with missing `GarageQual` have similar `SalePrice` distribution to houses with `TA` quality:

```python
fig = plt.figure()
ax = fig.add_subplot(111)

df[df['GarageQual'] == 'TA']['SalePrice'].plot(kind='kde', ax=ax)
df[df['GarageQual'].isnull()]['SalePrice'].plot(kind='kde', ax=ax, color='red')

ax.legend(['Houses with TA', 'Houses with NA'])
plt.title('GarageQual')
```

If the distributions overlap well, mode imputation is reasonable.

### Manual imputation

```python
df['GarageQual'] = df['GarageQual'].fillna('TA')
df['FireplaceQu'] = df['FireplaceQu'].fillna('Gd')  # mode for FireplaceQu
```

Note: `fillna(value, inplace=True)` raises a `ChainedAssignmentError` in modern pandas (Copy-on-Write). Use direct assignment instead.

### Using sklearn `SimpleImputer`

```python
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=['SalePrice']), df['SalePrice'], test_size=0.2
)

imputer = SimpleImputer(strategy='most_frequent')
X_train_trf = imputer.fit_transform(X_train)
X_test_trf  = imputer.transform(X_test)

imputer.statistics_
# array(['Gd', 'TA'], dtype=object) — modes learned from training data
```

---

## Problem with Mode Imputation for High-Missingness Columns

`FireplaceQu` has **47.3% missing**. Imputing 690 NaNs with `'Gd'` (the mode) massively over-represents that category:

Before imputation:
```
Gd    494   (of 770 non-null entries)
TA    412
Fa     41
```

After imputing 690 NaNs with `'Gd'`:
- `Gd` count jumps from 494 to 1184 — a **2.4× increase**
- The imputed distribution no longer reflects the true distribution

**Rule:** Mode imputation is only appropriate when missingness is low (< 10–15%). For high-missingness columns like `FireplaceQu`, the "Missing" category is a better choice.

---

## Strategy 2: Missing Category Imputation

Instead of guessing the missing value, **create a new category called `"Missing"`** that explicitly encodes the absence of information.

```python
df['GarageQual'] = df['GarageQual'].fillna('Missing')
```

After imputation, value counts show a new `"Missing"` category alongside the original categories.

### Using sklearn

```python
imputer = SimpleImputer(strategy='constant', fill_value='Missing')

X_train_trf = imputer.fit_transform(X_train)
X_test_trf  = imputer.transform(X_test)

imputer.statistics_
# array(['Missing', 'Missing'], dtype=object)
```

---

## When to Use Each Strategy

| Strategy | Best for | Risk |
|---|---|---|
| Mode (`most_frequent`) | Low missingness (< 10%), when missing is truly random | Over-represents dominant category; masks missingness |
| `"Missing"` category | High missingness, when absence is informative | Adds a new category that test data must handle |

### The "Missing" approach preserves information

For `FireplaceQu`, missing likely means the house has no fireplace — not that the quality is unknown. Encoding this as `"Missing"` lets the model learn that houses without fireplaces form a distinct group with different pricing patterns. This is a case of **MNAR** (Missing Not At Random) — the absence is meaningful.

---

## Handling the "Missing" Category at Test Time

When using OneHotEncoder downstream, ensure `handle_unknown='ignore'` so unseen categories in test data don't crash the encoder:

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore')
```

---

## Complete Pipeline Example

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, ['GarageQual', 'FireplaceQu'])
])

X_train_trf = preprocessor.fit_transform(X_train)
X_test_trf  = preprocessor.transform(X_test)
```

---

## Summary

| Column | Missing% | Mode | Recommendation |
|---|---|---|---|
| `GarageQual` | 5.5% | TA | Mode imputation safe |
| `FireplaceQu` | 47.3% | Gd | Use "Missing" category |

Always visualize the target distribution for missing vs non-missing observations before choosing a strategy — if they differ significantly, the "Missing" category is more honest than guessing the mode.
