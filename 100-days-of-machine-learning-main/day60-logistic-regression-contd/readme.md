# Day 60 — Logistic Regression (Continued)

## Topics Covered in This Session

Day 58 introduced the logistic regression model, sigmoid function, and binary cross-entropy loss. This notebook extends that foundation with:
- Multi-class logistic regression (softmax)
- Regularization (C parameter vs. Lasso/Ridge alpha)
- Practical implementation on real datasets
- Decision boundaries in 2D
- Comparison with other classifiers

---

## Multi-Class: One-vs-Rest (OvR)

OvR trains one binary classifier per class. The class with the highest probability wins.

```python
from sklearn.linear_model import LogisticRegression

# OvR (default for most solvers)
lr_ovr = LogisticRegression(multi_class='ovr', solver='lbfgs', C=1.0)
lr_ovr.fit(X_train, y_train)

# lr_ovr.coef_.shape → (n_classes, n_features)
# Each row is the coefficient vector for one class vs. all others
```

For a 3-class problem (e.g., Iris): 3 binary classifiers are trained.

---

## Multi-Class: Multinomial (Softmax)

Softmax regression jointly trains all classes, optimizing the full log-likelihood:

```python
lr_multi = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1.0, max_iter=1000)
lr_multi.fit(X_train, y_train)

# Probabilities sum to 1 across classes
proba = lr_multi.predict_proba(X_test)
print(proba[0])  # e.g., [0.03, 0.91, 0.06] — strongly class 1
```

**Softmax formula** for class k:
```
P(y=k | X) = exp(Xβₖ) / Σⱼ exp(Xβⱼ)
```

Multinomial is preferred for multi-class problems when using 'lbfgs', 'saga', or 'newton-cg' solvers.

---

## Regularization: C Parameter

Logistic regression in sklearn uses `C` (inverse of regularization strength):

```
C = 1 / α   →   small C = strong regularization = simpler model
```

```python
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
    lr = LogisticRegression(C=C, max_iter=1000)
    lr.fit(X_train, y_train)
    print(f"C={C}: train={lr.score(X_train, y_train):.3f}, test={lr.score(X_test, y_test):.3f}")
```

Finding the optimal C via cross-validation:

```python
from sklearn.linear_model import LogisticRegressionCV

lr_cv = LogisticRegressionCV(Cs=10, cv=5, scoring='accuracy')
lr_cv.fit(X_train, y_train)
print("Best C:", lr_cv.C_[0])
```

---

## Decision Boundary Visualization (2D)

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#FFAAAA', '#AAFFAA']))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title('Logistic Regression Decision Boundary')

# Use 2 features for visualization
lr = LogisticRegression()
lr.fit(X_train[:, :2], y_train)
plot_decision_boundary(lr, X_train[:, :2], y_train)
```

Logistic regression produces a **linear** decision boundary. Non-linearly separable data requires polynomial features or a non-linear classifier.

---

## Handling Class Imbalance

```python
# Check class distribution
print(y.value_counts())
# 0    765  (majority)
# 1    268  (minority)

# Balanced class weights
lr = LogisticRegression(class_weight='balanced', C=1.0)
lr.fit(X_train, y_train)

# Or manual weights
lr = LogisticRegression(class_weight={0: 1, 1: 3}, C=1.0)
```

`class_weight='balanced'` adjusts the loss to give more weight to the minority class — equivalent to oversampling.

---

## Feature Importance from Coefficients

```python
coef = pd.Series(lr.coef_[0], index=feature_names).sort_values()

# Most positive → strongest predictor of class 1
# Most negative → strongest predictor of class 0
coef.plot(kind='barh')
plt.title('Logistic Regression Coefficients')
```

After standardizing features, the magnitude of coefficients directly compares feature importance.

---

## Logistic Regression Assumptions

| Assumption | Note |
|-----------|------|
| Linear decision boundary | Non-linear relationships need feature engineering |
| No severe multicollinearity | Use ridge (L2) regularization to handle it |
| Large enough sample size | Rule of thumb: 10 events per predictor variable |
| No extreme outliers | Outliers distort the sigmoid boundary |

---

## When Logistic Regression Excels

- Baseline model before trying complex models
- When interpretability is required (coefficients = odds ratios)
- Low-dimensional, linearly separable data
- Probability calibration — logistic regression outputs are well-calibrated probabilities
- Online learning (with SGD solver)

---

## Practical Tips

- Always scale features: use `StandardScaler` in a pipeline.
- `max_iter=1000` avoids most convergence warnings.
- For multi-class: `multinomial + lbfgs` is more accurate than `ovr`.
- For imbalanced datasets: `class_weight='balanced'` + recall-focused metrics.
- Check both `predict` and `predict_proba` — the threshold matters as much as the model.
