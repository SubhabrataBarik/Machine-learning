# Day 27 — One-Hot Encoding

## What Is One-Hot Encoding?

One-Hot Encoding converts a categorical column with N unique values into **N binary (0/1) columns**, one per category. Each row gets a 1 in the column for its category and 0 everywhere else.

Example — `Color` column with values `{Red, Blue, Green}`:

```
Color        Color_Red  Color_Blue  Color_Green
Red      →       1          0           0
Blue     →       0          1           0
Green    →       0          0           1
```

---

## Why One-Hot Encoding?

Most ML algorithms require numeric input. For **nominal** (unordered) categories, integer encoding (1, 2, 3) implies a false ordering. One-hot encoding makes no such assumption — each category is independent.

---

## Dataset: Cars Data

```python
df = pd.read_csv('cars.csv')
```

Columns like `fuel_type` (Petrol/Diesel/CNG), `transmission` (Manual/Automatic), `brand`.

---

## pandas `get_dummies()`

```python
pd.get_dummies(df, columns=['fuel_type', 'transmission'])
```

Automatically creates binary columns for each category. Options:

```python
# Drop first category to avoid multicollinearity (dummy variable trap)
pd.get_dummies(df, columns=['fuel_type'], drop_first=True)

# Prefix control
pd.get_dummies(df, columns=['fuel_type'], prefix='fuel')
```

---

## sklearn OneHotEncoder

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop='first')
ohe.fit(df[['fuel_type']])
encoded = ohe.transform(df[['fuel_type']])
```

`drop='first'` drops the first category to avoid multicollinearity (the dummy variable trap).

`sparse=False` returns a dense NumPy array instead of a sparse matrix — needed when passing to pandas or when the encoded array must be dense.

---

## The Dummy Variable Trap

With N categories, you only need **N-1** binary columns. The Nth column is perfectly predictable from the others (if all others are 0, this one must be 1). Including all N columns creates **perfect multicollinearity**, which breaks linear models.

```python
# Petrol, Diesel, CNG → only need 2 columns
ohe = OneHotEncoder(drop='first')   # drops 'Petrol' column
# Remaining: is_Diesel, is_CNG
# If both are 0 → it's Petrol (implied)
```

---

## Handling High Cardinality

One-hot encoding a column with 1,000 unique values creates 1,000 new columns — the **curse of dimensionality**. Alternatives:

| Strategy | When to use |
|----------|-------------|
| One-hot | < ~20 categories |
| Ordinal encoding | Ordinal categories |
| Target encoding | High cardinality, regression/classification |
| Frequency encoding | When frequency correlates with target |
| Hash encoding | Very high cardinality, memory-constrained |
| Embedding (deep learning) | NLP, recommendation systems |

---

## Handling Unknown Categories at Test Time

```python
ohe = OneHotEncoder(handle_unknown='ignore')
```

If test data has a category not seen in training, `handle_unknown='ignore'` fills all its one-hot columns with 0 (the observation is treated as not belonging to any known category).

---

## OneHotEncoder vs get_dummies

| Feature | `pd.get_dummies()` | `OneHotEncoder` |
|---------|-------------------|-----------------|
| Sklearn Pipeline compatible | No | Yes |
| Handles unknown categories | No | Yes (`handle_unknown`) |
| Returns sparse matrix | No | Yes (efficient for high-cardinality) |
| Column names preserved | Yes | Needs `.get_feature_names_out()` |
| Fit/transform separation | No | Yes (avoids data leakage) |

**Prefer `OneHotEncoder` in production pipelines** — it correctly separates fit (on train) from transform (on test).

---

## Practical Tips

- Always use `drop='first'` for linear models (Logistic Regression, Linear Regression, SVM).
- Tree models (Random Forest, XGBoost) don't need the dummy drop.
- Use `ohe.get_feature_names_out(['col_name'])` to get descriptive column names after encoding.
- For pandas DataFrames: `pd.get_dummies()` is fine for quick EDA; use `OneHotEncoder` in pipelines.
