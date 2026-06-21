# Day 52 — Types of Gradient Descent

## What Is Gradient Descent?

**Gradient Descent** is an iterative optimization algorithm that minimizes a loss function by moving parameters in the direction of the steepest negative gradient. The full algorithm is covered in Day 51; this notebook focuses on the **three variants** that differ in how much data is used to compute each gradient update.

---

## The Three Variants

| Variant | Data per update | Speed | Noise | Memory |
|---------|----------------|-------|-------|--------|
| Batch GD | All n samples | Slow | Low | High |
| Stochastic GD | 1 sample | Fast | High | Low |
| Mini-batch GD | k samples | Medium | Medium | Medium |

---

## 1. Batch Gradient Descent (BGD)

Computes the gradient using the **entire training set** in each update step.

```python
class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            # Use ALL samples
            y_hat = np.dot(X_train, self.coef_) + self.intercept_
            intercept_der = -2 * np.mean(y_train - y_hat)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]
            self.intercept_ -= self.lr * intercept_der
            self.coef_      -= self.lr * coef_der

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

Applied to the diabetes dataset:
- `epochs=100, learning_rate=0.5`
- Final R² ≈ **0.453**
- sklearn LinearRegression R²: **0.4399**

**Pros**: Stable convergence, smooth loss curve.
**Cons**: Very slow for large datasets — must process all n samples per step.

---

## 2. Stochastic Gradient Descent (SGD)

Updates parameters using **one randomly selected sample** at each step.

```python
class SGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(X_train.shape[0]):
                # Pick ONE sample
                idx = np.random.randint(0, X_train.shape[0])
                y_hat = np.dot(X_train[idx], self.coef_) + self.intercept_
                intercept_der = -2 * (y_train[idx] - y_hat)
                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.intercept_ -= self.lr * intercept_der
                self.coef_      -= self.lr * coef_der

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

**Pros**: Fast updates, can escape local minima due to noisy gradients, works for online learning.
**Cons**: Noisy loss curve, may never truly converge (oscillates around minimum).

---

## 3. Mini-Batch Gradient Descent

Updates parameters using a **small batch of k samples** — the middle ground between Batch and SGD.

```python
class MBGDRegressor:
    def __init__(self, batch_size=10, learning_rate=0.01, epochs=100):
        self.coef_ = None
        self.intercept_ = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X_train, y_train):
        self.intercept_ = 0
        self.coef_ = np.ones(X_train.shape[1])

        for i in range(self.epochs):
            for j in range(0, X_train.shape[0], self.batch_size):
                # Slice a batch
                X_batch = X_train[j : j + self.batch_size]
                y_batch = y_train[j : j + self.batch_size]

                y_hat = np.dot(X_batch, self.coef_) + self.intercept_
                intercept_der = -2 * np.mean(y_batch - y_hat)
                coef_der = -2 * np.dot((y_batch - y_hat), X_batch) / X_batch.shape[0]
                self.intercept_ -= self.lr * intercept_der
                self.coef_      -= self.lr * coef_der

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

**Pros**: Efficient (vectorized batch ops), less noisy than SGD, works well with hardware (GPU batches).
**Cons**: Requires tuning batch_size; not as stable as full Batch GD.

---

## When Does Each Epoch End?

| Variant | # of weight updates per epoch |
|---------|-------------------------------|
| Batch GD | 1 (using all n samples) |
| SGD | n (one per sample) |
| Mini-batch | ceil(n / batch_size) |

With 354 training samples and batch_size=10: **36 updates per epoch** for Mini-batch.

---

## Loss Curve Comparison

```python
# Batch GD: smooth, monotonically decreasing
# SGD: jagged, noisy — oscillates around minimum
# Mini-batch: moderate noise, faster than Batch, more stable than SGD
```

Visualizing the loss curve at each epoch is the key diagnostic tool for choosing the right variant.

---

## sklearn's SGD Implementation

```python
from sklearn.linear_model import SGDRegressor

reg = SGDRegressor(max_iter=1000, eta0=0.01, learning_rate='constant')
reg.fit(X_train, y_train)
```

sklearn's `SGDRegressor` actually implements Mini-batch SGD internally with `max_iter` controlling passes over the data. It includes adaptive learning rate schedules (`learning_rate='invscaling'`, `'adaptive'`).

---

## Practical Decision

- **Small data (< 10K samples)**: Batch GD or Normal Equation (exact solution)
- **Large data (> 100K samples)**: Mini-batch GD (the default in deep learning)
- **Online / streaming data**: Stochastic GD (update as new data arrives)
- **Deep learning**: Always Mini-batch (batch_size typically 32, 64, 128, or 256)
