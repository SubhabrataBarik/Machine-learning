# Day 50 — Multiple Linear Regression

## What Is Multiple Linear Regression?

**Multiple Linear Regression** extends simple linear regression to predict a continuous output from **multiple input features**:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

Where β₀ is the intercept and β₁...βₙ are the coefficients for each feature.

---

## Dataset: 50 Startups

```python
df = pd.read_csv('50_Startups.csv')
# Features: R&D Spend, Administration, Marketing Spend, State
# Target: Profit
```

---

## sklearn Implementation

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

X = df.drop('Profit', axis=1)
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

preprocessor = ColumnTransformer([
    ('ohe', OneHotEncoder(drop='first'), ['State'])
], remainder='passthrough')

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
print("R²:", r2_score(y_test, pipe.predict(X_test)))
```

---

## The Normal Equation (Analytical Solution)

```
β = (XᵀX)⁻¹ Xᵀy
```

Closed-form solution for the optimal coefficients. In matrix notation:
- X: design matrix (n × p+1, with column of 1s for intercept)
- y: target vector (n × 1)
- β: coefficient vector (p+1 × 1)

```python
import numpy as np

# Add intercept column
X_b = np.c_[np.ones((len(X_train), 1)), X_train]

# Normal equation
beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y_train
```

The Normal Equation fails when XᵀX is singular (non-invertible) — occurs with:
- Perfect multicollinearity (one feature is a linear combination of others)
- More features than samples (p > n)

In these cases, use gradient descent or regularization.

---

## From-Scratch Implementation (Gradient Descent)

```python
import numpy as np

class MultipleLinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, p = X.shape
        self.coef_ = np.zeros(p)
        self.intercept_ = 0

        for _ in range(self.epochs):
            y_pred = X @ self.coef_ + self.intercept_
            residuals = y_pred - y

            self.coef_    -= self.lr * (2/n) * X.T @ residuals
            self.intercept_ -= self.lr * (2/n) * residuals.sum()

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
```

---

## Interpreting Coefficients

```python
coef = pd.Series(model.coef_, index=feature_names)
print(coef)
```

Example output:
```
R&D Spend       0.806    ← $1 more R&D → $0.81 more profit (holding others constant)
Administration -0.028    ← slight negative effect
Marketing Spend 0.027
State_Florida  -0.015    ← vs. reference category (e.g., California)
```

Each coefficient represents the **marginal effect** of that feature, holding all others constant. This is the Ceteris Paribus interpretation.

---

## Assumptions

| Assumption | Check with |
|------------|-----------|
| Linearity | Residual vs fitted plot |
| Independence | Domain knowledge |
| Homoscedasticity | Residual vs fitted plot |
| Normality of residuals | Q-Q plot |
| No multicollinearity | VIF (Variance Inflation Factor) |

### Checking Multicollinearity with VIF

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame({
    'Feature': X.columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
# VIF > 10 → high multicollinearity
```

---

## Normal Equation vs Gradient Descent

| Property | Normal Equation | Gradient Descent |
|----------|----------------|-----------------|
| Solution type | Exact | Approximate (iterative) |
| Scales with features | O(p³) | O(n × p × epochs) |
| Preferred when | p < ~10,000 | p is very large |
| Requires learning rate | No | Yes |
| Handles regularization | With modification | Naturally |

---

## Practical Tips

- Standardize features when comparing coefficient magnitudes — raw coefficients are scale-dependent.
- Always check VIF for multicollinearity in multiple regression.
- The `R²` for multiple regression always increases when you add features — use **Adjusted R²** for model comparison.
- For high-dimensional data (many features), use Ridge or Lasso regression instead of plain OLS.
