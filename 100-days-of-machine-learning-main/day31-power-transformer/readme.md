# Day 31 — Power Transformer

## What Is a Power Transformer?

A Power Transformer applies a **mathematical power transformation** to make numerical features more normally distributed. It is especially useful for right-skewed or left-skewed data where linear models and distance-based algorithms perform poorly.

Two methods are available in sklearn:

| Method | Formula | Handles zeros | Handles negatives |
|--------|---------|---------------|-------------------|
| **Box-Cox** | Varies by λ | No (x must be > 0) | No |
| **Yeo-Johnson** | Varies by λ | Yes | Yes |

---

## Dataset: Concrete Data

```python
df = pd.read_csv('concrete_data.csv')
```

Concrete compressive strength dataset with features like cement, water, age, and strength. Many features are right-skewed.

---

## Why Normalize Distributions?

Many statistical models assume input features are **normally (Gaussian) distributed**:
- Linear/Logistic Regression: assumptions of normality in residuals
- LDA: assumes Gaussian class-conditional distributions
- Bayesian models: Gaussian priors

Non-normal distributions cause model assumptions to be violated, leading to biased coefficients and poor generalization.

---

## Applying PowerTransformer

```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')  # works for all values
pt.fit(df)
df_transformed = pt.transform(df)
```

After transformation, each feature's distribution is closer to a standard normal (mean ≈ 0, std ≈ 1, shape approximately bell-curve).

`method='box-cox'` is used when all values are strictly positive. `method='yeo-johnson'` is the safer default.

---

## How Box-Cox Works

The Box-Cox transformation is parameterized by λ (lambda):

```
y(λ) = (x^λ - 1) / λ     if λ ≠ 0
y(λ) = log(x)             if λ = 0
```

Sklearn finds the optimal λ for each feature by **maximum likelihood estimation** — the λ that makes the transformed feature most normal.

Special cases:
- λ = 1: no transformation (x itself)
- λ = 0: log transformation
- λ = 0.5: square root transformation
- λ = -1: reciprocal transformation

---

## How Yeo-Johnson Works

Extends Box-Cox to support zero and negative values:

```
For x ≥ 0:  ((x+1)^λ - 1) / λ        if λ ≠ 0
            log(x + 1)                 if λ = 0
For x < 0:  -((-x+1)^(2-λ) - 1) / (2-λ)  if λ ≠ 2
            -log(-x + 1)               if λ = 2
```

Same maximum likelihood optimization for λ.

---

## Visualizing the Effect

Before transformation: features like cement and water are right-skewed (long tail).
After PowerTransformer: distributions become approximately normal bell curves.

This can be checked with:
```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['cement'], ax=axes[0], kde=True)
sns.histplot(df_transformed['cement'], ax=axes[1], kde=True)
```

---

## Checking Learned Lambdas

```python
pt.lambdas_
# array([0.12, -0.3, 1.1, ...])  — one λ per feature
```

λ ≈ 0 means log transformation was optimal; λ ≈ 1 means no transformation was needed.

---

## Comparison with Other Scalers

| Transformer | Normalizes shape | Handles negatives | Handles zeros |
|-------------|-----------------|-------------------|---------------|
| StandardScaler | No | Yes | Yes |
| MinMaxScaler | No | Yes | Yes |
| PowerTransformer (Yeo-Johnson) | Yes | Yes | Yes |
| PowerTransformer (Box-Cox) | Yes | No | No |
| QuantileTransformer | Yes (any dist) | Yes | Yes |

---

## Practical Tips

- Use `method='yeo-johnson'` by default — it works for all values and is generally robust.
- After PowerTransformer, features are approximately normal AND zero-mean/unit-variance (because `standardize=True` by default).
- Run a normality test (Shapiro-Wilk, Q-Q plot) before and after to verify the transformation helped.
- Like all transformers, fit only on training data: `pt.fit(X_train)`, then `pt.transform(X_test)`.
- PowerTransformer is not a substitute for outlier removal — extreme outliers still exist post-transformation, just compressed.
