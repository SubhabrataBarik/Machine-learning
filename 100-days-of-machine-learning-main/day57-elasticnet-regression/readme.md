# Day 57 — ElasticNet Regression

## What Is ElasticNet?

**ElasticNet** combines L1 (Lasso) and L2 (Ridge) penalties in a single regularizer:

```
Cost = MSE + α × [l1_ratio × Σ|βᵢ| + (1 - l1_ratio) × Σβᵢ²]
```

Where:
- `α` controls overall regularization strength
- `l1_ratio` (ρ) controls the mix between L1 and L2
  - `l1_ratio=1.0` → pure Lasso
  - `l1_ratio=0.0` → pure Ridge
  - `l1_ratio=0.5` → equal mix

---

## Why ElasticNet?

Lasso has two weaknesses:
1. **Correlated features**: Lasso arbitrarily selects one and zeros the rest — inconsistent behavior.
2. **p > n**: When features exceed samples, Lasso selects at most n features.

ElasticNet fixes both by keeping the group-selection property of Ridge while still producing sparse solutions via the L1 component.

---

## Implementation

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('enet', ElasticNet(alpha=0.1, l1_ratio=0.5))
])

pipe.fit(X_train, y_train)
print("R²:", pipe.score(X_test, y_test))
print("Non-zero coefs:", np.sum(pipe.named_steps['enet'].coef_ != 0))
```

---

## Choosing Hyperparameters with CV

```python
from sklearn.linear_model import ElasticNetCV

enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    alphas=np.logspace(-4, 2, 50),
    cv=5
)
enet_cv.fit(X_train_scaled, y_train)

print("Best alpha:", enet_cv.alpha_)
print("Best l1_ratio:", enet_cv.l1_ratio_)
print("Non-zero coefs:", np.sum(enet_cv.coef_ != 0))
```

---

## Comparing Ridge, Lasso, and ElasticNet

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet

models = {
    'Ridge':     Ridge(alpha=1.0),
    'Lasso':     Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    r2 = model.score(X_test_scaled, y_test)
    nz = np.sum(model.coef_ != 0)
    print(f"{name}: R²={r2:.3f}, Non-zero coefs={nz}")
```

---

## Regularization Comparison Table

| Property | Ridge | Lasso | ElasticNet |
|----------|-------|-------|-----------|
| Penalty | L2 (β²) | L1 (|β|) | L1 + L2 |
| Sparsity | No | Yes | Yes |
| Handles correlated features | Yes | Unstable | Yes |
| Group selection | No | No | Yes |
| Hyperparameters | α | α | α, l1_ratio |
| Works when p > n | Yes | Limited (max n nonzero) | Yes |

---

## l1_ratio Effect

```python
# Sweep l1_ratio from 0 to 1
for ratio in [0.0, 0.2, 0.5, 0.8, 1.0]:
    enet = ElasticNet(alpha=0.1, l1_ratio=ratio)
    enet.fit(X_train_scaled, y_train)
    nz = np.sum(enet.coef_ != 0)
    r2 = enet.score(X_test_scaled, y_test)
    print(f"l1_ratio={ratio}: non-zero={nz}, R²={r2:.3f}")
```

Higher l1_ratio → more sparsity. Lower → more shrinkage without zeroing.

---

## Practical Tips

- ElasticNet is the **default regularizer to try first** in high-dimensional settings.
- `l1_ratio=0.5` is a reasonable starting point — then tune via `ElasticNetCV`.
- Scale features first — both L1 and L2 penalties are scale-sensitive.
- Use ElasticNet when you have many correlated features and want feature selection.
- For pure interpretability (maximally sparse), Lasso is simpler; for pure stability, Ridge is simpler; ElasticNet is the balanced choice.
