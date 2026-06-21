# Day 66 — AdaBoost

## What Is AdaBoost?

**AdaBoost** (Adaptive Boosting) is an ensemble method that trains a sequence of **weak learners** (typically shallow decision trees called "stumps"), each one focusing harder on the examples the previous learner got wrong.

Key ideas:
1. Each sample gets a **weight** — initially uniform
2. After each weak learner is trained, **misclassified samples get higher weight**
3. The next learner focuses more on those hard examples
4. Final prediction: **weighted majority vote** of all weak learners

---

## The AdaBoost Algorithm

**Step 1**: Initialize sample weights `wᵢ = 1/n` for all n samples.

**Step 2**: For each round t = 1, 2, ..., T:
1. Train a weak learner `hₜ` on the weighted data
2. Compute weighted error: `εₜ = Σ wᵢ × I(hₜ(xᵢ) ≠ yᵢ)`
3. Compute learner weight: `αₜ = 0.5 × log((1 − εₜ) / εₜ)`
4. Update sample weights: increase for misclassified, decrease for correct
5. Normalize weights to sum to 1

**Step 3**: Final prediction: `H(x) = sign(Σ αₜ × hₜ(x))`

---

## Implementation

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Default: 50 stumps (max_depth=1)
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
```

For regression:
```python
from sklearn.ensemble import AdaBoostRegressor
ada_reg = AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
```

---

## Key Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `n_estimators` | 50 | Number of weak learners |
| `learning_rate` | 1.0 | Shrinks each learner's contribution — lower = more regularization |
| `estimator` | DecisionTree(depth=1) | Weak learner type |
| `algorithm` | 'SAMME.R' | 'SAMME.R' uses probability estimates; 'SAMME' uses class labels |

---

## Decision Stumps

A decision stump is a depth-1 decision tree — it makes exactly one binary split. It is a weak learner (slightly better than random), but AdaBoost combines many stumps into a strong ensemble.

```python
stump = DecisionTreeClassifier(max_depth=1)
stump.fit(X_train, y_train)
print("Stump accuracy:", stump.score(X_test, y_test))
# e.g., 0.64

ada = AdaBoostClassifier(n_estimators=200)
ada.fit(X_train, y_train)
print("AdaBoost accuracy:", ada.score(X_test, y_test))
# e.g., 0.86
```

---

## Learning Rate vs. n_estimators Trade-off

```python
# Staged predictions allow plotting the learning curve
staged_scores = [
    accuracy_score(y_test, y_pred)
    for y_pred in ada.staged_predict(X_test)
]

plt.plot(staged_scores)
plt.xlabel('Number of estimators')
plt.ylabel('Test accuracy')
plt.title('AdaBoost: Accuracy vs. n_estimators')
```

Key insight: `learning_rate` and `n_estimators` are coupled:
- Low `learning_rate` (e.g., 0.1) → need more `n_estimators` (e.g., 500) → more regularized
- High `learning_rate` (e.g., 1.0) → fewer estimators needed → may overfit

---

## Why AdaBoost Can Overfit

Unlike Bagging (Random Forest), AdaBoost focuses on hard examples. If the dataset has noisy or mislabeled samples, these get increasingly high weights, causing the model to overfit to noise.

```python
# Mitigation: limit n_estimators and use learning_rate < 1
ada = AdaBoostClassifier(n_estimators=100, learning_rate=0.5)
```

---

## AdaBoost vs. Random Forest

| Property | Random Forest | AdaBoost |
|----------|--------------|---------|
| Strategy | Bagging (parallel) | Boosting (sequential) |
| Base learner | Deep trees | Shallow stumps |
| Variance reduction | Primary strength | Less emphasis |
| Bias reduction | Moderate | Primary strength |
| Noise sensitivity | Robust | Sensitive (noisy labels hurt) |
| Speed | Parallelizable | Sequential (slower) |
| Typical performance | Strong | Strong, often better than RF |

---

## Feature Importance

```python
importances = pd.Series(ada.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title('AdaBoost Feature Importances')
```

AdaBoost's feature importance measures how often each feature was used for splits, weighted by the learner weight αₜ.

---

## Practical Tips

- Depth-1 stumps are the most common base learner, but depth-2 or 3 trees often improve performance.
- Tune `learning_rate` and `n_estimators` together: try `(lr=0.1, n=500)` vs. `(lr=1.0, n=50)`.
- AdaBoost does NOT handle missing values — impute before training.
- Use `staged_predict` to find the optimal `n_estimators` without re-training.
- For noisy data with many outliers, prefer Random Forest or Gradient Boosting over AdaBoost.
