# Day 63 — Voting Ensemble

## What is a Voting Ensemble?

A **Voting Ensemble** combines predictions from multiple independent models and aggregates them into a single final prediction. Instead of relying on one model's judgment, it takes a "committee vote."

The key idea: even if individual models make different mistakes, those mistakes tend to cancel out when combined — resulting in a more robust prediction.

---

## Voting Classifier

### Dataset and Setup

```python
df = pd.read_csv('Iris.csv')
df = df.iloc[:, 1:]  # drop Id column

encoder = LabelEncoder()
df['Species'] = encoder.fit_transform(df['Species'])
# setosa=0, versicolor=1, virginica=2

X = df.iloc[:, 0:2]   # SepalLengthCm, SepalWidthCm only
y = df.iloc[:, -1]    # Species
```

Only 2 features (sepal dimensions) are used — intentionally limited to make the problem harder and demonstrate the ensemble benefit.

### Individual Model Performance

```python
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = KNeighborsClassifier()   # n_neighbors=5

estimators = [('lr', clf1), ('rf', clf2), ('knn', clf3)]

for estimator in estimators:
    x = cross_val_score(estimator[1], X, y, cv=10, scoring='accuracy')
    print(estimator[0], np.round(np.mean(x), 2))
```

```
lr  0.81
rf  0.72
knn 0.76
```

Logistic Regression performs best individually at 0.81.

---

## Hard Voting

Each model votes for a class. The class with the most votes wins (majority rule).

```python
from sklearn.ensemble import VotingClassifier

vc = VotingClassifier(estimators=estimators, voting='hard')
x = cross_val_score(vc, X, y, cv=10, scoring='accuracy')
print(np.round(np.mean(x), 2))
# 0.77
```

Hard voting result = **0.77** — lower than LR alone (0.81) because RF (0.72) pulls down the vote.

The weak link problem: when the majority of models agree on the wrong answer, hard voting is worse than the best individual model.

---

## Soft Voting

Each model outputs a **probability** for each class. Probabilities are averaged across models. The class with the highest average probability wins.

```python
vc1 = VotingClassifier(estimators=estimators, voting='soft')
x = cross_val_score(vc1, X, y, cv=10, scoring='accuracy')
print(np.round(np.mean(x), 2))
# 0.75
```

Soft voting result = **0.75**. Soft voting requires all estimators to support `predict_proba`. It gives more nuanced aggregation — a model that is 99% confident outweighs one that is 51% confident.

---

## Weighted Voting

Give stronger models more influence by assigning weights. A brute-force grid search over weight combinations [1,2,3]:

```python
for i in range(1, 4):       # lr weight
    for j in range(1, 4):   # rf weight
        for k in range(1, 4):  # knn weight
            vc = VotingClassifier(estimators=estimators, voting='soft',
                                  weights=[i, j, k])
            x = cross_val_score(vc, X, y, cv=10, scoring='accuracy')
            print(f"i={i},j={j},k={k}", np.round(np.mean(x), 2))
```

Best results (0.79) at weights **[3, 1, 1]**, **[3, 1, 2]**, **[3, 1, 3]** — all heavily favour LogisticRegression (the strongest individual model), downweighting RandomForest (the weakest).

| weights [lr, rf, knn] | Accuracy |
|---|---|
| [1, 1, 1] (equal) | 0.77 |
| [2, 1, 1] | 0.77 |
| [3, 1, 1] | **0.79** |
| [3, 1, 2] | **0.79** |
| [3, 2, 1] | 0.78 |
| [1, 3, 1] | 0.73 |

The takeaway: weighting by model quality can recover accuracy lost from combining with weaker models.

---

## Same-Algorithm Ensemble (SVMs with Different Kernels/Degrees)

Voting ensembles work even when all models use the **same algorithm** — just with different hyperparameters. Each variant learns a different decision surface.

```python
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=15, n_redundant=5,
                            random_state=2)

svm1 = SVC(probability=True, kernel='poly', degree=1)
svm2 = SVC(probability=True, kernel='poly', degree=2)
svm3 = SVC(probability=True, kernel='poly', degree=3)
svm4 = SVC(probability=True, kernel='poly', degree=4)
svm5 = SVC(probability=True, kernel='poly', degree=5)
```

`probability=True` is required for soft voting — it enables `predict_proba`.

Individual accuracies (cv=10):

```
svm1 (degree=1)  0.85
svm2 (degree=2)  0.85
svm3 (degree=3)  0.89   ← best single model
svm4 (degree=4)  0.81
svm5 (degree=5)  0.86
```

```python
estimators = [('svm1',svm1),('svm2',svm2),('svm3',svm3),('svm4',svm4),('svm5',svm5)]
vc1 = VotingClassifier(estimators=estimators, voting='soft')
x = cross_val_score(vc1, X, y, cv=10, scoring='accuracy')
print(np.round(np.mean(x), 2))
# 0.93
```

**Ensemble accuracy = 0.93** vs best individual = 0.89. A +4% gain just from combining the same algorithm at different complexities.

---

## Voting Regressor

For regression tasks, models output continuous predictions. `VotingRegressor` averages (or weighted-averages) these values.

### Dataset and Setup

```python
from sklearn.datasets import load_boston

X, y = load_boston(return_X_y=True)
# X.shape → (506, 13)
```

Boston Housing: 506 samples, 13 features, target = median house value.

### Individual Model R² (cv=10)

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

lr  = LinearRegression()
dt  = DecisionTreeRegressor()
svr = SVR()   # kernel='rbf', C=1.0

estimators = [('lr', lr), ('dt', dt), ('svr', svr)]

for estimator in estimators:
    scores = cross_val_score(estimator[1], X, y, scoring='r2', cv=10)
    print(estimator[0], np.round(np.mean(scores), 2))
```

```
lr   0.20
dt  -0.08
svr -0.41
```

All three models are weak on this dataset individually (especially SVR with default params).

### Equal-Weight VotingRegressor

```python
from sklearn.ensemble import VotingRegressor

vr = VotingRegressor(estimators)
scores = cross_val_score(vr, X, y, scoring='r2', cv=10)
print(np.round(np.mean(scores), 2))
# 0.41
```

**R² jumps from 0.20 to 0.41** — more than double, even though two of the three models had negative R² individually. Averaging smooths out their errors.

### Weighted VotingRegressor Grid Search

```python
for i in range(1, 4):       # lr weight
    for j in range(1, 4):   # dt weight
        for k in range(1, 4):  # svr weight
            vr = VotingRegressor(estimators, weights=[i, j, k])
            scores = cross_val_score(vr, X, y, scoring='r2', cv=10)
            print(f"i={i},j={j},k={k}", np.round(np.mean(scores), 2))
```

Best result: **R² = 0.47** at weights **[2, 1, 1]** — double weight on LinearRegression (the strongest), equal on DT and SVR.

| weights [lr, dt, svr] | R² |
|---|---|
| [1, 1, 1] equal | 0.41 |
| [1, 1, 2] | 0.37 |
| [1, 1, 3] | 0.26 |
| [2, 1, 1] | **0.47** |
| [3, 2, 1] | 0.46 |
| [3, 3, 3] | 0.46 |

Increasing SVR weight (k=2,3) consistently hurts — the weakest model drags predictions down when over-weighted.

### Same-Algorithm Ensemble: Decision Trees at Different Depths

```python
dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=3)
dt3 = DecisionTreeRegressor(max_depth=5)
dt4 = DecisionTreeRegressor(max_depth=7)
dt5 = DecisionTreeRegressor(max_depth=None)

estimators = [('dt1',dt1),('dt2',dt2),('dt3',dt3),('dt4',dt4),('dt5',dt5)]
```

Individual R² (cv=10):

```
dt1 (depth=1)    -0.85
dt2 (depth=3)    -0.13
dt3 (depth=5)    -0.04   ← best single
dt4 (depth=7)    -0.19
dt5 (depth=None) -0.10
```

All negative! Yet:

```python
vr = VotingRegressor(estimators)
scores = cross_val_score(vr, X, y, scoring='r2', cv=10)
print(np.round(np.mean(scores), 2))
# 0.17
```

Ensemble R² = **0.17** — positive, despite every individual model being negative. Averaging cancels systematic errors.

---

## Hard vs Soft Voting — When to Use Which

| | Hard Voting | Soft Voting |
|---|---|---|
| Prediction | Majority class vote | Highest average probability |
| Requires `predict_proba` | No | Yes |
| Confidence-aware | No | Yes |
| Generally better | — | Yes (when calibrated) |

Use hard voting when models don't support probability output (e.g., default SVC without `probability=True`). Use soft voting otherwise — it contains more information and typically performs better.

---

## Full Results Summary

**Voting Classifier (Iris, cv=10 accuracy):**

| Model | Score |
|---|---|
| LogisticRegression alone | 0.81 |
| RandomForest alone | 0.72 |
| KNN alone | 0.76 |
| Hard voting (lr+rf+knn) | 0.77 |
| Soft voting (lr+rf+knn) | 0.75 |
| Weighted soft [3,1,1] | **0.79** |
| Soft voting (5×SVC poly deg 1–5) | **0.93** |

**Voting Regressor (Boston Housing, cv=10 R²):**

| Model | Score |
|---|---|
| LinearRegression alone | 0.20 |
| DecisionTreeRegressor alone | -0.08 |
| SVR alone | -0.41 |
| Equal-weight VotingRegressor | 0.41 |
| Best weighted [2,1,1] | **0.47** |
| VotingRegressor (5×DecisionTree) | **0.17** |

---

## Key Takeaways

- Voting ensembles are the simplest ensemble method — no training interaction between models
- Work best when individual models are **diverse** (different algorithms or hyperparameters) and have **uncorrelated errors**
- Weighted voting lets stronger models dominate — find weights via grid search
- Even models with negative R² can contribute positively to an ensemble by averaging out each other's errors
- `VotingClassifier` and `VotingRegressor` are drop-in sklearn estimators compatible with `cross_val_score`, `GridSearchCV`, and `Pipeline`
