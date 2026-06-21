# Day 51 — Gradient Descent

## What Is Gradient Descent?

**Gradient Descent** is an iterative optimization algorithm that finds the minimum of a function (the loss) by repeatedly moving in the direction of the **steepest descent** (negative gradient).

In ML, it is used to find the model parameters (weights) that minimize the loss function — the core of training almost every ML model.

---

## The Intuition

Imagine you are standing on a hilly landscape in the dark. You want to reach the valley (minimum). Your strategy:
1. Feel the slope under your feet (compute the gradient)
2. Take a step downhill (update parameters)
3. Repeat until you stop descending

The **learning rate** controls the step size. Too large → you overshoot and oscillate. Too small → convergence is very slow.

---

## The Loss Surface (Cost Function)

For Linear Regression with a single feature, the cost function is **convex** (bowl-shaped) with a single global minimum:

```
J(m, b) = (1/n) × Σ(y - (mx + b))²
```

This is MSE as a function of m (slope) and b (intercept). Gradient descent finds the (m, b) at the bottom of this bowl.

---

## The Update Rules

At each iteration, update parameters by subtracting a fraction of the gradient:

```
m ← m - α × ∂J/∂m
b ← b - α × ∂J/∂b
```

Where:
- α (alpha) = learning rate
- ∂J/∂m = gradient of loss with respect to slope
- ∂J/∂b = gradient of loss with respect to intercept

### Gradient Formulas for MSE

```
∂J/∂m = (-2/n) × Σ xᵢ(yᵢ - ŷᵢ)
∂J/∂b = (-2/n) × Σ (yᵢ - ŷᵢ)
```

---

## From-Scratch Implementation

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
            y_hat = np.dot(X_train, self.coef_) + self.intercept_

            # Gradients
            intercept_der = -2 * np.mean(y_train - y_hat)
            coef_der = -2 * np.dot((y_train - y_hat), X_train) / X_train.shape[0]

            # Updates
            self.intercept_ -= self.lr * intercept_der
            self.coef_      -= self.lr * coef_der

    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_
```

Applied to the diabetes dataset (442 samples, 10 features):
- sklearn LinearRegression R²: **0.4399**
- GDRegressor (1000 epochs, lr=0.5) R²: **0.4535**

---

## Learning Rate Effect

| Learning Rate | Behavior |
|---------------|----------|
| Too large (e.g., 1.0) | Oscillates, may diverge |
| Too small (e.g., 0.0001) | Converges very slowly |
| Just right (e.g., 0.1–0.5) | Smooth convergence |

Common learning rates to try: 0.001, 0.01, 0.1, 0.5

---

## Convergence Check

Monitor the loss at each epoch:

```python
losses = []
for i in range(epochs):
    y_hat = ...
    loss = np.mean((y_train - y_hat) ** 2)
    losses.append(loss)

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
```

If the loss plot is decreasing and leveling off → converging correctly.
If it oscillates or increases → learning rate is too large.

---

## Gradient Descent in 3D (m and b simultaneously)

The notebook includes animated visualizations (`animation.gif`) showing:
- The cost surface as a 3D bowl
- The path of gradient descent converging toward the minimum
- Effect of different learning rates on the convergence path

---

## Why Gradient Descent Over Normal Equation?

| Scenario | Normal Equation | Gradient Descent |
|----------|----------------|-----------------|
| Features < 10,000 | Preferred (exact) | Works |
| Features > 10,000 | Expensive (O(p³)) | Preferred |
| Deep learning | Not applicable | Required |
| Online learning | Not applicable | Stochastic GD |

---

## Practical Tips

- Standardize features before gradient descent — unequal scales make the cost surface elongated, causing slow convergence.
- Learning rate scheduling (decaying lr over epochs) helps: large steps early, fine-tuning later.
- The diabetes dataset used here is from sklearn: `load_diabetes()` — 442 samples, 10 features, all pre-scaled.
- Gradient descent is the foundation for SGD, Adam, RMSProp, and all deep learning optimizers.
