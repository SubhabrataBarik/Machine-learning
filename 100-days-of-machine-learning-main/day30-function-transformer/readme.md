# Day 30 — Function Transformer

## What Is FunctionTransformer?

`FunctionTransformer` wraps any Python function as an sklearn-compatible transformer. It lets you apply custom mathematical transformations inside a Pipeline using the standard `fit`/`transform` interface.

---

## Dataset: Titanic (train.csv)

```python
df = pd.read_csv('train.csv')
```

Used to demonstrate custom log transformation and column-specific transformations inside a pipeline.

---

## Why Use FunctionTransformer?

Sklearn's built-in transformers (StandardScaler, MinMaxScaler) cover standard transformations. But sometimes you need:
- Log transformation for right-skewed features
- Square root transformation
- Custom domain-specific feature engineering
- Any numpy/scipy function applied column-by-column

`FunctionTransformer` bridges custom functions with sklearn's Pipeline architecture.

---

## Basic Usage

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

# Apply log1p (log(1+x)) to handle zeros safely
log_transformer = FunctionTransformer(np.log1p)
log_transformer.fit_transform(df[['Fare']])
```

`np.log1p(x) = log(1 + x)` — preferred over `np.log(x)` because it handles `x=0` without producing `-inf`.

---

## Custom Functions

```python
def custom_transform(X):
    return X ** 0.5   # square root

sqrt_transformer = FunctionTransformer(custom_transform)
```

Any function that accepts a NumPy array and returns a NumPy array can be wrapped.

---

## Inside a Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

pipe = Pipeline([
    ('log', FunctionTransformer(np.log1p)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

The log transformation is applied first, then standardization, then the model — in order.

---

## Column-Specific Transformations in ColumnTransformer

```python
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('log_fare', FunctionTransformer(np.log1p), ['Fare']),
    ('scale_age', StandardScaler(), ['Age']),
])
```

Apply log only to `Fare` (which is right-skewed) and standardization only to `Age`.

---

## Why Log-Transform Right-Skewed Features?

Many features like `Fare`, `Income`, and `Price` have right-skewed distributions — most values cluster near zero with a long tail of high values. This causes:
- Linear model coefficients to be dominated by extreme values
- Residuals to be non-normal (violating linear regression assumptions)
- Distance-based algorithms to treat high-value outliers as significantly different

After log transform:
- Distribution becomes closer to normal
- Model focuses on multiplicative rather than additive relationships
- Residuals become more homoscedastic

---

## `validate` and `feature_names_out` Parameters

```python
FunctionTransformer(np.log1p, validate=True)
```

`validate=True` ensures input is a 2D array. `validate=False` (default) passes data as-is — useful for DataFrames.

```python
FunctionTransformer(np.log1p, feature_names_out='one-to-one')
```

Tells the transformer that output columns correspond 1:1 to input columns — preserves column names when used in a pipeline with `get_feature_names_out()`.

---

## Practical Tips

- Always use `np.log1p` instead of `np.log` to safely handle zero values.
- For negative values, consider `np.sign(x) * np.log1p(np.abs(x))` (signed log transform).
- `FunctionTransformer` has no learnable parameters — `fit()` is a no-op, which is correct for fixed transformations.
- Combine with `ColumnTransformer` for mixed transformations on different columns.
- Use `inverse_func` parameter to define the inverse transformation for interpretable predictions.
