# Day 39 — KNN Imputer

## What is KNN Imputation?

**KNN Imputation** fills missing values using the values from the K-nearest neighbors of the row with the missing entry. Instead of using a global statistic (like mean), it uses local context — similar rows are likely to have similar values.

For a row with missing `Age`, the imputer finds the K rows most similar to it (based on the non-missing features), then fills `Age` with a weighted average of those neighbors' `Age` values.

---

## How It Works

1. For each row with a missing value, compute the distance to all other rows using the non-null features.
2. Identify the K nearest neighbors (rows with complete data for that feature).
3. Impute the missing value as a (weighted) average of the K neighbors' values.

With `weights='distance'`, closer neighbors contribute more:
```
imputed_value = Σ(neighbor_value / distance) / Σ(1 / distance)
```

With `weights='uniform'`, all K neighbors contribute equally.

---

## Dataset

```python
df = pd.read_csv('train.csv')[['Age', 'Pclass', 'Fare', 'Survived']]
# 891 rows; Age: 19.87% missing, Pclass/Fare: complete
```

The key insight: `Pclass` and `Fare` strongly correlate with `Age`. A 1st-class passenger paying high fare is likely older than a 3rd-class passenger — KNN uses this correlation.

---

## Applying KNN Imputer

```python
from sklearn.impute import KNNImputer

knn = KNNImputer(n_neighbors=3, weights='distance')

X_train_trf = knn.fit_transform(X_train)
X_test_trf  = knn.transform(X_test)
```

The imputer learns from training data (neighbor distances are computed from training rows). Test rows with missing `Age` are imputed by finding their nearest training-set neighbors.

---

## Model Evaluation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(X_train_trf, y_train)
y_pred = lr.predict(X_test_trf)

accuracy_score(y_test, y_pred)
# 0.7039
```

### Comparison with Simple Mean Imputer

```python
si = SimpleImputer()  # default strategy='mean'

X_train_trf2 = si.fit_transform(X_train)
X_test_trf2  = si.transform(X_test)

lr.fit(X_train_trf2, y_train)
accuracy_score(y_test, lr.predict(X_test_trf2))
# 0.6927
```

| Method | Accuracy |
|---|---|
| Mean imputation | 0.6927 |
| KNN imputation (k=3, distance) | 0.7039 |
| Improvement | +1.1% |

KNN outperforms mean imputation because it uses neighboring context — a 1st-class passenger with high `Fare` is imputed an `Age` drawn from other 1st-class wealthy passengers, not the global mean of all 714 passengers with known ages.

---

## Key Parameters

```python
KNNImputer(
    n_neighbors=3,           # number of nearest neighbors to use
    weights='distance',      # 'uniform' or 'distance'
    metric='nan_euclidean',  # distance metric (default handles NaN)
    add_indicator=False      # optionally add missing indicator columns
)
```

### `n_neighbors`

- Small K (1–3): very local, can be noisy, prone to outlier influence
- Large K (10–20): smoother, more like global mean, lower variance
- Tune with cross-validation

### `metric='nan_euclidean'`

The default distance metric handles NaN entries gracefully — it computes Euclidean distance only over the features that are non-null in both rows, then scales the distance to account for the missing dimensions.

---

## Always Scale Before KNN Imputation

KNN is distance-based — features on different scales distort the neighbor computation.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer(n_neighbors=5))
])

X_train_trf = pipe.fit_transform(X_train)
X_test_trf  = pipe.transform(X_test)
```

Without scaling, `Fare` (range 0–512) dominates the distance computation over `Pclass` (range 1–3), making class information nearly irrelevant to neighbor selection.

---

## KNN vs Mean Imputation: Why KNN Works Better

Mean imputation ignores all structure in the data — every missing Age becomes 29.78 (the training mean), regardless of how wealthy or poor-class the passenger was.

KNN preserves:
- **Class structure**: 1st-class passengers tend to be older
- **Fare correlations**: High-fare passengers cluster with other high-fare passengers
- **Local distribution**: The imputed value comes from a local neighborhood, not the global mean

---

## Limitations

| Limitation | Detail |
|---|---|
| Computationally expensive | O(n²) distance computation for large datasets |
| Sensitive to feature scaling | Features must be standardized first |
| Requires complete features for distance | Other missing features complicate neighbor finding |
| Memory-intensive | Entire training set stored at fit time |

---

## When to Use KNN Imputation

| Situation | Use KNN? |
|---|---|
| Small to medium dataset (< 100k rows) | Yes |
| Features are correlated with the missing feature | Yes (strong advantage) |
| Large dataset (> 1M rows) | Caution — slow |
| Missing feature is uncorrelated with others | No benefit over mean |
| Quick baseline | Start with mean, then try KNN |
