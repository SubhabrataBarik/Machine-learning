# Day 48 — Simple Linear Regression

## What Is Simple Linear Regression?

**Simple Linear Regression** models the relationship between a single input feature (X) and a continuous output variable (y) as a straight line:

```
y = mx + b
```

Where:
- `m` = slope (coefficient) — how much y changes per unit change in X
- `b` = intercept — value of y when X = 0

The goal is to find the values of `m` and `b` that best fit the training data.

---

## Dataset: Placement Data (placement.csv)

```python
df = pd.read_csv('placement.csv')
# cgpa → placement_exam_marks
```

Predicting placement exam marks from CGPA.

---

## The Loss Function: Mean Squared Error

We find the "best fit" line by minimizing the **Mean Squared Error (MSE)**:

```
MSE = (1/n) × Σ(y_actual - y_predicted)²
```

The squared error penalizes large mistakes more than small ones. Minimizing MSE gives us the optimal slope and intercept via the **Normal Equation** (closed-form solution).

---

## Normal Equation (Analytical Solution)

```
m = Σ((xᵢ - x̄)(yᵢ - ȳ)) / Σ(xᵢ - x̄)²
b = ȳ - m × x̄
```

This finds the global optimum directly — no iteration needed. Computationally feasible when the number of features is small.

---

## sklearn Implementation

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

df = pd.read_csv('placement.csv')

X = df[['cgpa']]
y = df['placement_exam_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

lr = LinearRegression()
lr.fit(X_train, y_train)

print("Slope (m):", lr.coef_[0])
print("Intercept (b):", lr.intercept_)

y_pred = lr.predict(X_test)
print("R² score:", r2_score(y_test, y_pred))
```

---

## Interpreting the Coefficients

```python
# Example output:
lr.coef_[0]    # e.g., 10.3  → each 1-point increase in CGPA → 10.3 more exam marks
lr.intercept_  # e.g., -45.2 → predicted mark when CGPA = 0 (extrapolation, may not be meaningful)
```

---

## Visualizing the Regression Line

```python
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('CGPA')
plt.ylabel('Placement Exam Marks')
plt.legend()
plt.show()
```

---

## From-Scratch Implementation

```python
class LinearRegressionScratch:
    def fit(self, X, y):
        x_mean = X.mean()
        y_mean = y.mean()
        numerator   = ((X - x_mean) * (y - y_mean)).sum()
        denominator = ((X - x_mean) ** 2).sum()
        self.m = numerator / denominator
        self.b = y_mean - self.m * x_mean

    def predict(self, X):
        return self.m * X + self.b
```

This is mathematically equivalent to sklearn's `LinearRegression` for the simple case.

---

## Assumptions of Linear Regression

| Assumption | Description | Violation consequence |
|------------|-------------|----------------------|
| Linearity | X and y have a linear relationship | Biased predictions |
| Independence | Observations are independent | Wrong standard errors |
| Homoscedasticity | Residuals have constant variance | Inefficient estimates |
| Normality of residuals | Residuals are normally distributed | Invalid inference |
| No multicollinearity | (N/A for simple regression) | — |

---

## R² — Coefficient of Determination

```
R² = 1 - (SS_residual / SS_total)
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
```

- R² = 1: perfect fit — model explains all variance
- R² = 0: model explains no more variance than just predicting the mean
- R² < 0: model is worse than predicting the mean (overfitting or wrong model)

---

## Practical Tips

- Check the scatter plot before fitting — if the relationship is non-linear, use Polynomial Regression.
- R² alone is not enough — check residual plots for systematic patterns.
- Outliers have a large influence on the regression line — always check for them.
- Simple linear regression is rarely used alone in ML — it is the foundation for understanding multiple regression, regularization, and gradient descent.
