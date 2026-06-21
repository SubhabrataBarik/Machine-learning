# Day 53 — Polynomial Regression

## What is Polynomial Regression?

**Polynomial Regression** extends linear regression to fit non-linear relationships by adding polynomial terms (X², X³, X·Y, ...) as additional features. The model is still **linear in the parameters** — it remains a linear regression applied to the expanded feature set.

```
y = β₀ + β₁x + β₂x²   (degree 2, quadratic)
y = β₀ + β₁x + β₂x² + β₃x³  (degree 3, cubic)
```

---

## The Problem: Linear Regression on Curved Data

The notebook uses a synthetic quadratic dataset:

```python
# True relationship: y = 0.8x² + 0.9x + 2 + noise
X = 6 * np.random.rand(200, 1) - 3        # X ∈ [-3, 3]
y = 0.8 * X**2 + 0.9 * X + 2 + np.random.randn(200, 1)
```

Applying linear regression to this curved data:

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
r2_score(y_test, lr.predict(X_test))
# -0.003   ← negative R²! Worse than predicting the mean
```

A straight line cannot capture a U-shaped curve. Negative R² confirms the model is useless on this data.

---

## `PolynomialFeatures`: Expanding the Input

```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=True)
X_train_trans = poly.fit_transform(X_train)
X_test_trans  = poly.transform(X_test)
```

What `PolynomialFeatures(degree=2)` does to a single input feature:

```python
print(X_train[0])       # [-1.66904319]
print(X_train_trans[0]) # [ 1.         -1.66904319  2.78570518]
#                              ^            ^             ^
#                          bias=1          x            x²
```

The single feature `x` becomes `[1, x, x²]`. LinearRegression then learns:
```
y = β₀ × 1 + β₁ × x + β₂ × x²
```

`include_bias=True` prepends the constant 1 column. Set `include_bias=False` when using inside a Pipeline, since LinearRegression already adds its own intercept.

---

## Fitting Polynomial Regression

```python
lr = LinearRegression()
lr.fit(X_train_trans, y_train)

y_pred = lr.predict(X_test_trans)
r2_score(y_test, y_pred)
# 0.8653  ← dramatic improvement from -0.003
```

### Recovered Coefficients

```python
print(lr.coef_)      # [0.         0.95033992  0.80522584]
print(lr.intercept_) # [2.04244747]
```

The fitted equation:
```
y ≈ 2.042 + 0.950x + 0.805x²
```

True equation used to generate data:
```
y = 2 + 0.9x + 0.8x²
```

The model nearly perfectly recovers the true coefficients — demonstrating that polynomial regression can identify the underlying mathematical structure.

---

## Examining Feature Powers

```python
poly.powers_
# array([[0],
#        [1],
#        [2]])
# → features: x⁰=1, x¹=x, x²
```

---

## Polynomial Regression via Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def polynomial_regression(degree):
    pipe = Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=False)),
        ("std_scaler",    StandardScaler()),
        ("lin_reg",       LinearRegression()),
    ])
    pipe.fit(X, y)
    return pipe
```

This is the recommended pattern:
1. `PolynomialFeatures` expands features (set `include_bias=False`)
2. `StandardScaler` normalizes the expanded features — important for higher degrees where x¹⁰ can be astronomically large
3. `LinearRegression` fits the model

---

## SGDRegressor with Polynomial Features

```python
from sklearn.linear_model import SGDRegressor

poly = PolynomialFeatures(degree=2)
X_train_trans = poly.fit_transform(X_train)
X_test_trans  = poly.transform(X_test)

sgd = SGDRegressor(max_iter=100)
sgd.fit(X_train_trans, y_train)
r2_score(y_test, sgd.predict(X_test_trans))
```

Gradient descent works equally well on the polynomial-expanded feature set. `SGDRegressor` is preferred over `LinearRegression` when the dataset is large, since it avoids the expensive normal equation computation.

---

## 3D Polynomial Regression

For two input features, polynomial features include cross-product interaction terms:

```python
# True relationship: z = x² + y² + 0.2x + 0.2y + 0.1xy + 2
x = 7 * np.random.rand(100, 1) - 2.8
y = 7 * np.random.rand(100, 1) - 2.8
z = x**2 + y**2 + 0.2*x + 0.2*y + 0.1*x*y + 2 + np.random.randn(100, 1)
```

```python
poly = PolynomialFeatures(degree=2)
X_multi = np.array([x, y]).reshape(100, 2)
X_multi_trans = poly.fit_transform(X_multi)

print("Input features:", poly.n_features_in_)    # 2
print("Output features:", poly.n_output_features_)   # 6
print("Powers:\n", poly.powers_)
```

Output powers:
```
[[0 0]   → 1      (bias)
 [1 0]   → x
 [0 1]   → y
 [2 0]   → x²
 [1 1]   → x·y   (interaction term)
 [0 2]]  → y²
```

Two input features expand to 6 features: `[1, x, y, x², xy, y²]`. LinearRegression fits:
```
z = β₀ + β₁x + β₂y + β₃x² + β₄xy + β₅y²
```

The interaction term `xy` captures how the effect of `x` depends on the value of `y` — something pure polynomial powers miss.

---

## Overfitting with High Degree

| Degree | Behavior |
|---|---|
| 1 | Underfitting — cannot fit curves |
| 2 | Quadratic — good for smooth U-shapes |
| 3–5 | Can capture inflections and S-curves |
| 10+ | Severe overfitting — wildly oscillates between data points (Runge's phenomenon) |

For high degrees, add regularization:
```python
from sklearn.linear_model import Ridge

pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=5)),
    ('ridge', Ridge(alpha=10.0))
])
```

Ridge's L2 penalty shrinks large coefficients that overfitting produces.

---

## Feature Count Growth

For `n` input features and degree `d`, the output feature count is `C(n+d, d)`:

| Input features | Degree | Output features |
|---|---|---|
| 1 | 2 | 3 |
| 2 | 2 | 6 |
| 2 | 3 | 10 |
| 15 | 2 | 136 |

At high degrees and many input features, `PolynomialFeatures` creates an enormous number of terms. Use `interaction_only=True` to suppress pure power terms when only interactions matter:

```python
# Only cross-products, no x², y²:
PolynomialFeatures(degree=2, interaction_only=True)
# [1, x, y, x·y]  ← 4 features instead of 6
```

---

## Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `degree` | 2 | Maximum degree of polynomial terms |
| `include_bias` | True | Add column of ones (x⁰) |
| `interaction_only` | False | Only cross-terms (xy), no pure powers (x², y²) |

---

## Summary

| | Linear Regression | Polynomial Regression |
|---|---|---|
| Relationship modeled | Linear | Non-linear |
| Feature expansion | None | X → X, X², X³, ... |
| Model class | Linear in parameters | Linear in parameters |
| Risk | Underfitting on curves | Overfitting at high degree |
| R² on quadratic data | -0.003 (useless) | 0.865 (good fit) |

Polynomial regression is a form of **basis function expansion** — the same idea underlying splines, RBF kernels, and neural network hidden layers. The difference is that the basis functions (x², x³, xy) are fixed here, whereas in neural networks they are learned from data.
