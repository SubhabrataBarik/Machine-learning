# Day 56 — Lasso Regression

## What Is Lasso Regression?

**Lasso** (Least Absolute Shrinkage and Selection Operator) adds an **L1 penalty** (sum of absolute values of coefficients) to the MSE loss:

```
Cost = MSE + α × Σ|βᵢ|
```

The key property: Lasso drives some coefficients to **exactly zero** — it performs automatic **feature selection**.

---

## Lasso vs. Ridge: The Core Difference

| Property | Ridge (L2) | Lasso (L1) |
|----------|-----------|-----------|
| Penalty | Σβᵢ² | Σ|βᵢ| |
| Coefficients reach exactly zero? | No | Yes |
| Feature selection | No | Yes |
| Solution shape | Smooth (differentiable) | Non-smooth (requires coordinate descent) |
| Preferred when | Many small effects | Few large effects (sparse model) |

---

## Implementation

```python
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

pipe.fit(X_train, y_train)
print("R²:", pipe.score(X_test, y_test))

# Check which features survived (non-zero coefficients)
lasso = pipe.named_steps['lasso']
print("Non-zero features:", np.sum(lasso.coef_ != 0))
print("Zeroed features:", np.sum(lasso.coef_ == 0))
```

---

## Automatic Feature Selection

```python
# Before Lasso: 20 features
# After Lasso (alpha=1.0): only 8 features have non-zero coefficients

feature_names = X.columns
selected = feature_names[lasso.coef_ != 0]
print("Selected features:", selected.tolist())
```

This is Lasso's main advantage over Ridge: the resulting model is interpretable and deployable with fewer features.

---

## Choosing Alpha with Cross-Validation

```python
lasso_cv = LassoCV(alphas=np.logspace(-4, 2, 100), cv=5)
lasso_cv.fit(X_train_scaled, y_train)

print("Best alpha:", lasso_cv.alpha_)
print("Non-zero coefs:", np.sum(lasso_cv.coef_ != 0))
```

`LassoCV` uses a coordinate descent path algorithm — much faster than a grid search.

---

## Coefficient Paths

```python
from sklearn.linear_model import lasso_path
import numpy as np

alphas, coefs, _ = lasso_path(X_train_scaled, y_train)

plt.plot(alphas, coefs.T)
plt.xscale('log')
plt.xlabel('Alpha (decreasing regularization →)')
plt.ylabel('Coefficients')
plt.title('Lasso coefficient paths')
```

As alpha decreases (moving right), more features enter the model. Each line is one feature. The order in which features enter reveals their relative importance.

---

## Why L1 Creates Sparsity (Geometric Intuition)

Ridge constraint region: a **sphere** (smooth, no corners)
Lasso constraint region: a **diamond** (sharp corners at axes)

The OLS solution (ellipse contours of MSE) is more likely to touch the diamond at a corner — where one or more coordinates are exactly zero. This is why Lasso produces sparse solutions and Ridge does not.

---

## When to Use Lasso

- When you suspect only a subset of features are truly relevant (sparse ground truth)
- When interpretability matters — fewer features is easier to explain
- For feature selection before passing to another model
- When p >> n (more features than samples)

---

## Practical Tips

- Scale features before Lasso — L1 penalty is scale-sensitive.
- Lasso can be unstable with correlated features — it arbitrarily selects one from a correlated group and zeros the rest.
- For correlated features, use **ElasticNet** instead (combination of L1 and L2).
- `max_iter` default is 1000 — increase to 10000 for convergence on large datasets.
- `LassoCV` is preferred over manual grid search — it uses efficient warm-start paths.
