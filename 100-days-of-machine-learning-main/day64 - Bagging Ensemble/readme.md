# Day 64 — Bagging Ensemble

## What is Bagging?

**Bagging** (Bootstrap AGGregatING) trains multiple instances of the same base model on different random subsets of the training data, then aggregates their predictions (majority vote for classification, average for regression).

The core idea: individual models overfit to their specific training sample, but averaging many overfit models cancels out the variance — leaving only the signal.

```
Training set (N rows)
    ├─ Bootstrap sample 1 (N rows, with replacement) → Model 1
    ├─ Bootstrap sample 2 (N rows, with replacement) → Model 2
    └─ Bootstrap sample k (N rows, with replacement) → Model k
                                ↓
                    Aggregate predictions
                  (majority vote / average)
```

---

## The Four Variants

Bagging generalises across two axes of sampling: **rows** and **features**.

| Variant | Row sampling | Feature sampling | Row bootstrap | Feature bootstrap |
|---|---|---|---|---|
| **Bagging** | subset | all | True | False |
| **Pasting** | subset | all | False | False |
| **Random Subspaces** | all | subset | False | True |
| **Random Patches** | subset | subset | True | True |

---

## Part 1: Bagging Classifier

### Dataset

```python
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=10, n_informative=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Baseline: Single Decision Tree

```python
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
accuracy_score(y_test, dt.predict(X_test))
# 0.8755
```

### Bagging with Decision Trees

```python
from sklearn.ensemble import BaggingClassifier

bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.5,      # each tree sees 50% of training rows
    bootstrap=True,       # sample with replacement
    random_state=42
)
bag.fit(X_train, y_train)
accuracy_score(y_test, bag.predict(X_test))
# 0.9165
```

**Single DT: 0.8755 → Bagging 500 DTs: 0.9165** — a +4.1% improvement.

Each estimator is trained on 4,000 rows (50% of 8,000):
```python
bag.estimators_samples_[0].shape    # (4000,)
bag.estimators_features_[0].shape   # (10,)  ← all features used
```

### Bagging with SVM

```python
bag = BaggingClassifier(
    base_estimator=SVC(),
    n_estimators=500,
    max_samples=0.25,     # each SVM sees 25% of rows
    bootstrap=True,
    random_state=42
)
bag.fit(X_train, y_train)
accuracy_score(y_test, bag.predict(X_test))
# 0.901
```

Any sklearn estimator can be used as the base — SVMs, logistic regression, KNN, etc.

---

## Pasting (No Replacement)

Same as bagging but samples are drawn **without replacement** — no row can appear twice in the same bootstrap sample.

```python
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=False,      # ← no replacement = pasting
    random_state=42,
    n_jobs=-1             # use all CPU cores
)
bag.fit(X_train, y_train)
accuracy_score(y_test, bag.predict(X_test))
# 0.9165
```

Pasting gives each model a cleaner, non-repetitive sample. In practice, bagging tends to perform slightly better due to the randomness introduced by replacement, but pasting is faster to train.

The `n_jobs=-1` parameter parallelises training across all CPU cores — each tree is independent so they can be fit simultaneously.

---

## Random Subspaces (Feature Sampling)

Sample **features** instead of rows. Each tree sees all training rows but only a random subset of features.

```python
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=1.0,           # all rows
    bootstrap=False,
    max_features=0.5,          # each tree sees 50% of features
    bootstrap_features=True,   # ← feature sampling with replacement
    random_state=42
)
bag.fit(X_train, y_train)
accuracy_score(y_test, bag.predict(X_test))
# 0.911
```

```python
bag.estimators_samples_[0].shape    # (8000,)  ← all rows
bag.estimators_features_[0].shape   # (5,)     ← 5 of 10 features
```

Useful for **high-dimensional data** where many features are redundant or correlated. Forces each tree to specialise in different feature subsets.

---

## Random Patches (Row + Feature Sampling)

Sample **both rows and features** — the most aggressive diversity.

```python
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,          # 25% of rows, with replacement
    bootstrap=True,
    max_features=0.5,          # 50% of features, with replacement
    bootstrap_features=True,
    random_state=42
)
bag.fit(X_train, y_train)
accuracy_score(y_test, bag.predict(X_test))
# 0.909
```

---

## OOB Score — Free Validation

With bootstrap sampling, each tree is trained on ~63% of the data. The remaining ~37% — the **Out-Of-Bag (OOB)** samples — were never seen by that tree and can be used as a validation set for free.

```python
bag = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=500,
    max_samples=0.25,
    bootstrap=True,
    oob_score=True,    # ← enable OOB evaluation
    random_state=42
)
bag.fit(X_train, y_train)

bag.oob_score_                                        # 0.90425
accuracy_score(y_test, bag.predict(X_test))           # 0.9195
```

The OOB score (0.904) is a reliable approximation of test accuracy (0.920) — computed with no extra data or cross-validation overhead. This is one of bagging's key advantages.

---

## Classifier Results Summary

| Method | Config | Accuracy |
|---|---|---|
| Decision Tree (baseline) | single model | 0.8755 |
| Bagging (DT) | 500 est, 50% rows, bootstrap | **0.9165** |
| Bagging (SVM) | 500 est, 25% rows, bootstrap | 0.9010 |
| Pasting (DT) | 500 est, 25% rows, no bootstrap | 0.9165 |
| Random Subspaces (DT) | 500 est, all rows, 50% features | 0.9110 |
| Random Patches (DT) | 500 est, 25% rows, 50% features | 0.9090 |
| OOB Bagging (DT) | 500 est, 25% rows | OOB: 0.904, Test: **0.9195** |

---

## GridSearchCV on BaggingClassifier

```python
parameters = {
    'n_estimators': [50, 100, 500],
    'max_samples':  [0.1, 0.4, 0.7, 1.0],
    'bootstrap':    [True, False],
    'max_features': [0.1, 0.4, 0.7, 1.0]
}

search = GridSearchCV(BaggingClassifier(), parameters, cv=5)
search.fit(X_train, y_train)

search.best_score_    # 0.8986
search.best_params_
# {'bootstrap': True, 'max_features': 0.7, 'max_samples': 0.4, 'n_estimators': 500}
```

---

## Part 2: Bagging Regressor

### Dataset and Baselines

```python
# Boston Housing (506 rows, 13 features)
X_train, X_test, Y_train, Y_test = train_test_split(X_boston, Y_boston,
                                                     train_size=0.80,
                                                     test_size=0.20,
                                                     random_state=123)
```

Individual model R² on test set:

```
LinearRegression       R² = 0.659
DecisionTreeRegressor  R² = 0.438
KNeighborsRegressor    R² = 0.548
```

### Default BaggingRegressor

```python
from sklearn.ensemble import BaggingRegressor

bag_regressor = BaggingRegressor(random_state=1)
# defaults: n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True

bag_regressor.fit(X_train, Y_train)
bag_regressor.score(X_train, Y_train)   # 0.980  (training R²)
bag_regressor.score(X_test, Y_test)     # 0.812  (test R²)
```

**LR test R² = 0.659 → BaggingRegressor test R² = 0.812** — +15.3% improvement with just 10 trees and default settings.

### GridSearchCV on BaggingRegressor

```python
params = {
    'base_estimator':    [None, LinearRegression(), KNeighborsRegressor()],
    'n_estimators':      [20, 50, 100],
    'max_samples':       [0.5, 1.0],
    'max_features':      [0.5, 1.0],
    'bootstrap':         [True, False],
    'bootstrap_features':[True, False]
}

bagging_regressor_grid = GridSearchCV(BaggingRegressor(random_state=1, n_jobs=-1),
                                      param_grid=params, cv=3, n_jobs=-1, verbose=1)
bagging_regressor_grid.fit(X_train, Y_train)
# 144 candidates × 3 folds = 432 fits (~1 min)

print('Train R² : %.3f' % bagging_regressor_grid.best_estimator_.score(X_train, Y_train))
# 0.983
print('Test R²  : %.3f' % bagging_regressor_grid.best_estimator_.score(X_test, Y_test))
# 0.802
print('Best CV R² : %.3f' % bagging_regressor_grid.best_score_)
# 0.870
print('Best Params:', bagging_regressor_grid.best_params_)
# {'base_estimator': None, 'bootstrap': True, 'bootstrap_features': False,
#  'max_features': 1.0, 'max_samples': 1.0, 'n_estimators': 50}
```

`base_estimator=None` defaults to `DecisionTreeRegressor`. The best config uses all rows and all features — pure row bootstrap with 50 trees.

---

## Part 3: Step-by-Step — How Bootstrap Sampling Works

The learning tool notebook manually demonstrates bagging on Iris (classes 1 & 2 only, 2 features):

```python
# Three separate bootstrap samples, each with replacement
df_bag1 = df_train.sample(8, replace=True)  # rows can repeat
df_bag2 = df_train.sample(8, replace=True)
df_bag3 = df_train.sample(8, replace=True)
```

Each tree is trained on a different sample — some rows appear multiple times, others not at all.

**Individual predictions for one test point [SepalWidth=2.2, PetalLength=5.0]:**

```python
dt_bag1.predict([[2.2, 5.0]])   # [2]  Iris-virginica
dt_bag2.predict([[2.2, 5.0]])   # [1]  Iris-versicolor
dt_bag3.predict([[2.2, 5.0]])   # [2]  Iris-virginica
```

**Majority vote → class 2 (Iris-virginica)** ✓

This manual example shows the aggregation step: even when individual trees disagree, the majority is correct.

---

## Key Practical Tips

From the notebook's tips section:

1. **Bagging > Pasting** — replacement adds extra randomness that helps generalisation
2. **25%–50% row sampling** is the sweet spot for `max_samples`
3. **Random Patches / Subspaces** for high-dimensional data — forces feature diversity
4. **OOB score** replaces a separate validation set — use `oob_score=True` to get it for free
5. **`n_jobs=-1`** to parallelise across all CPU cores — trees are independent

---

## API Reference

```python
BaggingClassifier(
    base_estimator=None,      # default: DecisionTreeClassifier
    n_estimators=10,          # number of models in ensemble
    max_samples=1.0,          # fraction (or int) of rows per model
    max_features=1.0,         # fraction (or int) of features per model
    bootstrap=True,           # True=bagging, False=pasting
    bootstrap_features=False, # True=random subspaces/patches
    oob_score=False,          # compute OOB validation score
    n_jobs=None,              # parallelism (-1 = all cores)
    random_state=None
)
```

`BaggingRegressor` has the same parameters. It averages predictions instead of majority-voting.
