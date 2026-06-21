# Day 58 — Logistic Regression

## What Is Logistic Regression?

Despite the name, **Logistic Regression** is a **classification** algorithm. It models the probability that a sample belongs to a class using the **sigmoid function**, then classifies based on a threshold (default 0.5).

For binary classification:
```
P(y=1 | X) = σ(Xβ) = 1 / (1 + e^(−Xβ))
```

Where σ is the sigmoid function that squashes any real number to [0, 1].

---

## The Sigmoid Function

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 100)
sigma = 1 / (1 + np.exp(-z))

plt.plot(z, sigma)
plt.xlabel('z = Xβ')
plt.ylabel('P(y=1)')
plt.axhline(0.5, linestyle='--')
plt.title('Sigmoid Function')
```

Properties:
- σ(0) = 0.5 → decision boundary at z = 0
- σ(∞) = 1, σ(−∞) = 0
- Smooth, differentiable → gradient descent friendly

---

## Loss Function: Binary Cross-Entropy (Log Loss)

Logistic regression does NOT minimize MSE. It maximizes the **log-likelihood** of the observed classes:

```
Loss = −(1/n) × Σ [yᵢ log(ŷᵢ) + (1 − yᵢ) log(1 − ŷᵢ)]
```

- If yᵢ = 1: Loss = −log(ŷᵢ) → penalizes low predicted probability for positives
- If yᵢ = 0: Loss = −log(1 − ŷᵢ) → penalizes high predicted probability for negatives

---

## sklearn Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Binary classification (e.g., Titanic survived/not survived)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

lr = LogisticRegression(C=1.0, max_iter=1000)
lr.fit(X_train_s, y_train)

y_pred = lr.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

---

## Key Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `C` | 1.0 | Inverse regularization strength (1/α). Higher C → less regularization |
| `penalty` | 'l2' | Regularization type: 'l1', 'l2', 'elasticnet', None |
| `solver` | 'lbfgs' | Optimization algorithm |
| `max_iter` | 100 | Maximum iterations for convergence |
| `multi_class` | 'auto' | 'ovr' or 'multinomial' for multi-class |

---

## Probabilities vs Predictions

```python
# Predicted class (0 or 1)
y_pred = lr.predict(X_test_s)

# Predicted probability of each class
y_prob = lr.predict_proba(X_test_s)
# Shape: (n_samples, 2) — columns: P(y=0), P(y=1)

# Probability of positive class
y_prob_pos = y_prob[:, 1]
```

Probabilities allow you to tune the classification threshold:

```python
threshold = 0.3  # More sensitive (catches more positives)
y_pred_custom = (y_prob_pos >= threshold).astype(int)
```

---

## Decision Boundary

Logistic regression is a **linear classifier** — the decision boundary is a hyperplane:

```
Xβ = 0  →  P(y=1) = 0.5
```

For 2D features, the boundary is a line. Non-linear boundaries require polynomial features or a non-linear model.

---

## Multi-Class Logistic Regression

```python
lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0)
lr_multi.fit(X_train_s, y_train)

# lr_multi.coef_.shape → (n_classes, n_features)
# lr_multi.intercept_.shape → (n_classes,)
```

Multinomial logistic regression uses **softmax** instead of sigmoid — each class gets its own probability, and they sum to 1.

---

## Coefficients and Odds Ratios

```python
import pandas as pd
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coef': lr.coef_[0],
    'odds_ratio': np.exp(lr.coef_[0])
})
```

The **odds ratio** `exp(β)` is the multiplicative change in odds for a one-unit increase in that feature:
- OR > 1: feature increases probability of class 1
- OR < 1: feature decreases probability of class 1
- OR = 1: no effect

---

## Solvers by Use Case

| Solver | Supports L1? | Large datasets | Sparse data |
|--------|-------------|----------------|-------------|
| 'lbfgs' | No | No | No |
| 'liblinear' | Yes | No | Yes |
| 'saga' | Yes | Yes | Yes |
| 'newton-cg' | No | Moderate | No |

Use `'saga'` for large datasets or when L1 regularization is needed.

---

## Practical Tips

- Scale features — the L2 penalty (default) is scale-sensitive.
- Increase `max_iter` to 1000+ if you get `ConvergenceWarning`.
- Use `C` for regularization, not `alpha` (C = 1/alpha — confusingly inverted vs. Ridge/Lasso).
- For imbalanced classes, use `class_weight='balanced'`.
- Always check `predict_proba` — not just `predict` — for threshold tuning.
