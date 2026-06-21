# Day 55 — Regularized Linear Models (Ridge Regression)

## What Is Regularization?

**Regularization** adds a penalty term to the loss function to constrain coefficient magnitude. This prevents overfitting — especially when:
- There are many features relative to samples
- Features are highly correlated (multicollinearity)
- Polynomial features cause coefficient explosion

Without regularization, ordinary least squares (OLS) can produce enormous coefficients that fit training data perfectly but generalize poorly.

---

## Ridge Regression (L2 Regularization)

Ridge adds the **sum of squared coefficients** to the MSE loss:

```
Cost = MSE + α × Σβᵢ²
```

Where:
- `α` (alpha) is the regularization strength hyperparameter
- Higher α → stronger shrinkage → smaller coefficients → simpler model
- α = 0 → identical to ordinary linear regression

---

## Implementation

```python
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])

pipe.fit(X_train, y_train)
print("R²:", pipe.score(X_test, y_test))
```

**Always scale features before Ridge.** The penalty treats all coefficients equally, so unscaled features with different magnitudes are penalized disproportionately.

---

## Choosing Alpha with Cross-Validation

```python
from sklearn.linear_model import RidgeCV

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='r2')
ridge_cv.fit(X_train_scaled, y_train)

print("Best alpha:", ridge_cv.alpha_)
print("R²:", ridge_cv.score(X_test_scaled, y_test))
```

`RidgeCV` uses Leave-One-Out CV by default, or k-fold with `cv=k`. The best alpha minimizes validation error.

---

## Effect of Alpha on Coefficients

```python
import matplotlib.pyplot as plt

coefs = []
alphas = np.logspace(-3, 5, 100)

for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train_scaled, y_train)
    coefs.append(ridge.coef_)

plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient values')
plt.title('Ridge coefficient paths')
```

As alpha increases, all coefficients shrink toward zero — but never reach exactly zero. This is the key difference from Lasso.

---

## Ridge vs. OLS: When Ridge Wins

| Scenario | OLS | Ridge |
|----------|-----|-------|
| Low-dimensional, independent features | Good | Slightly worse |
| High multicollinearity | Unstable | Stable |
| More features than samples (p > n) | Fails | Works |
| After PolynomialFeatures (degree > 3) | Overfit | Controlled |

---

## Mathematical Intuition

OLS solution: `β = (XᵀX)⁻¹ Xᵀy`

Ridge solution: `β = (XᵀX + αI)⁻¹ Xᵀy`

Adding `αI` ensures the matrix is always invertible (even with multicollinearity) and shrinks β toward zero. The ridge "regularizes" the inversion.

---

## Bias-Variance Tradeoff

- **α = 0**: High variance, low bias (standard OLS, can overfit)
- **α → ∞**: High bias, low variance (coefficients → 0, underfitting)
- **Optimal α**: Balances bias and variance via cross-validation

```
Total error = Bias² + Variance + Irreducible noise
```

Ridge reduces variance at the cost of introducing some bias — the right tradeoff when variance is the dominant error source.

---

## Practical Tips

- Scale features before Ridge — L2 penalty is scale-sensitive.
- Use `RidgeCV` instead of manual grid search — it's more efficient.
- Ridge keeps ALL features (no selection) — use Lasso if you want sparsity.
- For polynomial regression, Ridge is the standard regularizer to control degree explosion.
- Start with `alpha=1.0` as a baseline and tune logarithmically (0.001 to 1000).
