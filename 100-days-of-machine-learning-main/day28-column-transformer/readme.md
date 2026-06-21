# Day 28 — Column Transformer

## What Is ColumnTransformer?

`ColumnTransformer` applies **different preprocessing steps to different columns** simultaneously, then concatenates the results into a single output matrix. It is the standard sklearn tool for building preprocessing pipelines on mixed-type datasets.

---

## Dataset: COVID Toy Dataset

```python
df = pd.read_csv('covid_toy.csv')
```

Typical structure: age (numeric), gender (binary categorical), fever (numeric), cough (ordinal — Mild/Strong), city (nominal categorical), has_covid (target).

---

## The Problem It Solves

Real datasets have mixed column types:
- Numerical columns → StandardScaler
- Ordinal columns → OrdinalEncoder
- Nominal columns → OneHotEncoder
- Binary columns → no transformation needed

Without `ColumnTransformer`, you must manually split, transform, and re-concatenate — error-prone and not pipeline-compatible.

---

## Basic Syntax

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

transformer = ColumnTransformer(transformers=[
    ('tnf1', StandardScaler(), ['age', 'fever']),
    ('tnf2', OrdinalEncoder(categories=[['Mild', 'Strong']]), ['cough']),
    ('tnf3', OneHotEncoder(sparse=False, drop='first'), ['gender', 'city']),
], remainder='passthrough')
```

Each tuple: `(name, transformer, columns)`.

- `remainder='passthrough'` — columns not mentioned are passed through unchanged.
- `remainder='drop'` — columns not mentioned are dropped.

---

## Fit and Transform

```python
from sklearn.model_selection import train_test_split

X = df.drop('has_covid', axis=1)
y = df['has_covid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

transformer.fit(X_train)
X_train_transformed = transformer.transform(X_train)
X_test_transformed  = transformer.transform(X_test)
```

The output is a 2D NumPy array with all transformations applied and concatenated.

---

## Column Selection Methods

```python
# By column name (list)
('scaler', StandardScaler(), ['age', 'fever'])

# By column index (list of ints)
('scaler', StandardScaler(), [0, 1])

# By dtype (make_column_selector)
from sklearn.compose import make_column_selector as selector

('num', StandardScaler(), selector(dtype_include=np.number)),
('cat', OneHotEncoder(), selector(dtype_include=object)),
```

`make_column_selector` automatically selects columns by dtype — useful when columns are many or dynamic.

---

## Output Column Names

```python
transformer.get_feature_names_out()
```

Returns the names of all output columns after transformation — useful for building a DataFrame from the result.

---

## Integration with Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('preprocessor', transformer),
    ('model', LogisticRegression())
])

pipe.fit(X_train, y_train)
pipe.predict(X_test)
```

This is the standard production pattern — the entire preprocessing + modeling is one object that can be fitted, saved, and deployed.

---

## Practical Tips

- Always put `ColumnTransformer` inside a `Pipeline` so cross-validation applies it correctly.
- Use `remainder='passthrough'` if you have ID or already-processed columns you want to keep.
- `get_feature_names_out()` is essential when you need to interpret model coefficients after transformation.
- For sparse OHE output: remove `sparse=False` and the output will be a sparse matrix — more memory-efficient for high-cardinality columns.
