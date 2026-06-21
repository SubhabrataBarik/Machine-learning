# Day 68 — Stacking and Blending

## What Are Stacking and Blending?

Both are **meta-ensemble** techniques where predictions from multiple base models ("level-0") are used as features to train a final "meta-learner" ("level-1"). They capture complementary strengths of different model types.

---

## Stacking vs. Blending

| Property | Stacking | Blending |
|----------|---------|---------|
| How level-0 predictions are made | Out-of-fold cross-validation | Hold-out set |
| Data leakage risk | Low (CV prevents leakage) | Higher (hold-out may overfit) |
| Training time | Higher (k-fold) | Lower (single split) |
| Data efficiency | High (uses all data) | Lower (hold-out is wasted) |
| Implementation complexity | Higher | Simpler |

---

## Stacking: How It Works

```
Training Data
    ↓
┌───────────────────────────────────────┐
│  Level-0: Train k base models         │
│  • LinearRegression                   │
│  • RandomForestRegressor              │
│  • SVR                                │
│                                       │
│  Each uses out-of-fold CV to generate │
│  predictions without data leakage     │
└───────────────────────────────────────┘
    ↓ (base model OOF predictions become new features)
┌───────────────────────────────────────┐
│  Level-1: Meta-learner                │
│  • Ridge / LogisticRegression         │
│  • Trained on OOF predictions         │
└───────────────────────────────────────┘
    ↓
Final predictions on test set
```

---

## sklearn StackingClassifier

```python
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Level-0 base learners
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
]

# Level-1 meta-learner
meta = LogisticRegression()

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=meta,
    cv=5,           # 5-fold cross-validation for OOF predictions
    stack_method='predict_proba',  # use probability estimates as features
    n_jobs=-1
)

stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)
print("Stacking Accuracy:", accuracy_score(y_test, y_pred))
```

---

## StackingRegressor

```python
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gbm', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('svr', SVR(kernel='rbf')),
]

stacking = StackingRegressor(
    estimators=estimators,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

stacking.fit(X_train, y_train)
print("Stacking R²:", stacking.score(X_test, y_test))
```

---

## Blending: Manual Implementation

Blending uses a hold-out validation set to generate level-0 predictions:

```python
from sklearn.model_selection import train_test_split
import numpy as np

# 3-way split: train → base model training, val → blending, test → evaluation
X_train_b, X_val, y_train_b, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Level-0: Train base models on train_b, predict on val
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

base_models = [
    LogisticRegression(C=1.0, max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(probability=True, random_state=42)
]

val_preds   = np.zeros((len(X_val), len(base_models)))
test_preds  = np.zeros((len(X_test), len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train_b, y_train_b)
    val_preds[:, i]  = model.predict_proba(X_val)[:, 1]
    test_preds[:, i] = model.predict_proba(X_test)[:, 1]

# Level-1: Train meta-learner on val_preds
meta = LogisticRegression()
meta.fit(val_preds, y_val)

# Final predictions
y_final = meta.predict(test_preds)
print("Blending Accuracy:", accuracy_score(y_test, y_final))
```

---

## Why Diverse Base Models Work Best

Stacking/blending benefit most when base models are:
- **Different algorithms** (linear, tree-based, kernel-based)
- **Low correlation of errors** — they fail on different examples
- **All individually competent** — weak models contribute noise

```python
# Check correlation between OOF predictions — lower is better
import pandas as pd
oof_df = pd.DataFrame(val_preds, columns=['LR', 'RF', 'SVC'])
print(oof_df.corr())
```

Highly correlated models (e.g., two Random Forests with similar params) add little value.

---

## Multi-Level Stacking

The meta-learner's output can itself be stacked with another level:

```
Level-0: [RF, SVC, KNN] → OOF predictions
Level-1: [GBM, Logistic] trained on OOF → OOF predictions
Level-2: Ridge meta-learner → final prediction
```

In practice, more than 2 levels rarely helps and increases risk of overfitting on small datasets.

---

## Stacking vs. Other Ensembles

| Method | How models combine | Best for |
|--------|------------------|---------|
| Bagging (RF) | Parallel, same algorithm | High-variance models |
| Boosting (AdaBoost, GBM) | Sequential, same algorithm | High-bias models |
| Stacking | Sequential, different algorithms | Maximum accuracy |

Stacking is the most powerful but most expensive ensemble method.

---

## Practical Tips

- Use `stack_method='predict_proba'` — probabilities carry more information than hard class labels.
- The meta-learner should be simple (Ridge, Logistic) — complex meta-learners overfit.
- Use at least 3 diverse base models: one linear, one tree-based, one kernel/distance-based.
- For Kaggle: stacking + blending is the standard approach for top leaderboard positions.
- Validate the meta-learner on OOF predictions from training, not on the test set.
