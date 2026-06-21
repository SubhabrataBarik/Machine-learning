# Day 29 — sklearn Pipelines

## What Is a Pipeline?

A `Pipeline` chains multiple processing steps into a single object. Each step transforms the data and passes the result to the next step. The final step is typically an estimator (model).

```
Raw Data → Step 1 (Imputer) → Step 2 (Scaler) → Step 3 (Encoder) → Model
```

---

## Why Use Pipelines?

Without a Pipeline, you manually call `.fit()` and `.transform()` for each step, which leads to:
- **Data leakage**: accidentally fitting on test data
- **Error-prone code**: easy to forget a step during prediction
- **Cross-validation issues**: preprocessing must be inside CV folds

With a Pipeline, all these problems are eliminated — the pipeline handles the correct fit/transform flow automatically.

---

## Dataset: Titanic

The notebook demonstrates the same Titanic survival prediction task built two ways: with and without a Pipeline.

### Without Pipeline (Verbose and Error-Prone)

```python
# Separate fit/transform for each step — manual and fragile
imputer.fit(X_train[num_cols])
X_train[num_cols] = imputer.transform(X_train[num_cols])
X_test[num_cols]  = imputer.transform(X_test[num_cols])

scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# ... repeat for categorical columns
```

### With Pipeline (Clean and Safe)

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])

final_pipeline.fit(X_train, y_train)
final_pipeline.predict(X_test)
```

---

## How Pipeline.fit() Works

When you call `final_pipeline.fit(X_train, y_train)`:

1. For each step except the last: calls `.fit_transform(X, y)` and passes the output to the next step.
2. For the last step (estimator): calls `.fit(X_transformed, y)`.

When you call `final_pipeline.predict(X_test)`:

1. For each step except the last: calls `.transform(X)` — **no re-fitting**.
2. For the last step: calls `.predict(X_transformed)`.

This guarantees that scaler parameters learned on train are applied identically to test.

---

## Cross-Validation with Pipeline

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(final_pipeline, X, y, cv=5, scoring='accuracy')
```

Cross-validation with a Pipeline is correct:
- Each fold: the pipeline is **re-fitted from scratch** on the training portion
- Preprocessing stats (mean, std, imputation values) are computed only on training folds
- Test fold is transformed using training fold statistics only

Without a Pipeline, a common mistake is fitting scalers on the full dataset before CV, which leaks test information.

---

## Saving and Loading Pipelines

```python
import joblib

# Save
joblib.dump(final_pipeline, 'model_pipeline.pkl')

# Load and predict
pipeline = joblib.load('model_pipeline.pkl')
pipeline.predict(new_data)
```

The entire pipeline (preprocessing + model) is serialized together — you never need to separately save scaler, encoder, and model objects.

---

## Accessing Steps

```python
# By name
final_pipeline.named_steps['model']
final_pipeline.named_steps['preprocessor']

# By index
final_pipeline.steps[0]   # (name, transformer)
final_pipeline.steps[-1]  # (name, model)

# Get model coefficients
final_pipeline.named_steps['model'].coef_
```

---

## Practical Tips

- Name each step descriptively: `'imputer'`, `'scaler'`, `'ohe'`, `'model'` — easier to access later.
- Nested pipelines (num_pipeline, cat_pipeline inside ColumnTransformer inside final Pipeline) are the standard production pattern.
- `Pipeline` supports all sklearn estimators that implement `fit`/`transform`/`predict`.
- Use `set_params()` to tune pipeline hyperparameters: `pipeline.set_params(model__C=0.1)`.
- `GridSearchCV(pipeline, param_grid)` lets you tune both preprocessing and model parameters together.
