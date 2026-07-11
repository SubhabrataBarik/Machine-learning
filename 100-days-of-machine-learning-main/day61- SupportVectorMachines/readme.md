# Day 61 — Support Vector Machines

## What is SVM?

A **Support Vector Machine (SVM)** finds the hyperplane that maximizes the **margin** — the gap between the two classes. Unlike logistic regression which minimizes log-loss, SVM directly optimizes geometry: it wants the boundary to be as far from both classes as possible.

The points that sit exactly on the margin boundaries are the **support vectors**. Everything else is irrelevant — you can add or remove non-support-vector points and the boundary doesn't change.

---

## Part 1: Linear SVM on Perfectly Separable Data

```python
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

model = SVC(kernel='linear', C=1)
model.fit(X, y)
```

The `plot_svc_decision_function` helper draws the boundary and margins:

```python
def plot_svc_decision_function(model, ax=None, plot_support=True):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # levels=[-1, 0, 1]: dashed margin lines, solid decision boundary
    ax.contour(X, Y, P, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])

    # circle the support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=300, linewidth=1, facecolors='none')
```

The `decision_function` output is the signed distance from the hyperplane:
- `+1` → on the positive margin boundary
- `0` → exactly on the decision boundary  
- `-1` → on the negative margin boundary

---

## Only Support Vectors Matter

The key insight: **the decision boundary is determined only by support vectors**, not the total number of training points.

```python
def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2, random_state=0, cluster_std=0.60)
    X = X[:N]  # use only first N points
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)  # hard margin (no violations)
    model.fit(X, y)
    ...

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
```

With `C=1E10` (effectively a hard margin), the boundary with N=60 and N=120 is **identical** — because the support vectors don't change. Non-support-vector points have zero influence on the hyperplane.

---

## The `C` Parameter: Hard vs Soft Margin

When classes overlap, a perfect separation is impossible. The `C` parameter controls the tradeoff between margin width and training errors:

```python
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=0.8)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
for axi, C in zip(ax, [100.0, 0.01]):
    model = SVC(kernel='linear', C=C).fit(X, y)
    ...
    axi.set_title('C = {0:.1f}'.format(C))
```

| C value | Margin width | Misclassifications | Behavior |
|---|---|---|---|
| `C=100.0` (high) | Narrow | Few | Hard-ish margin, fits training data closely |
| `C=0.01` (low) | Wide | More allowed | Soft margin, prioritizes generalization |

- **High C** → smaller slack, tries to classify every point correctly → overfitting risk
- **Low C** → large slack, accepts more margin violations → better generalization

---

## Part 2: The Kernel Trick

### The Problem: Non-Linearly Separable Data

```python
from sklearn.datasets import make_circles

X, y = make_circles(100, factor=0.1, noise=0.1)
```

Two concentric rings — inner ring is class 0, outer ring is class 1. No straight line can separate them.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = SVC(kernel="linear")
classifier.fit(X_train, y_train.ravel())
accuracy_score(y_test, classifier.predict(X_test))
# 0.55   ← barely better than random guessing
```

### The Intuition: Lift to Higher Dimensions

The key insight: data that is not linearly separable in 2D **might be linearly separable in 3D**.

```python
def plot_3d_plot(X, y):
    r = np.exp(-(X ** 2).sum(1))   # radial basis: large near origin, small far away
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=100, cmap='bwr')
```

Adding `r = exp(-(x² + y²))` as a third dimension:
- Points near the origin (inner ring, class 1) → `r ≈ 1.0` (high up)
- Points on the outer ring (class 0) → `r ≈ 0.0` (flat)

In 3D, the two classes are now separable by a **horizontal plane**. A linear classifier in 3D corresponds to a **curved boundary** in 2D.

### The Kernel Trick

Computing the actual transformation `φ(x)` and then the dot product `φ(x)·φ(z)` is expensive. The **kernel trick** computes `K(x, z) = φ(x)·φ(z)` directly without ever computing `φ`:

```
K_rbf(x, z) = exp(-γ ||x - z||²)
K_poly(x, z) = (γ·x·z + r)^d
K_linear(x, z) = x·z
```

The SVM only needs these dot products — never the explicit high-dimensional feature vectors.

---

## RBF Kernel

```python
rbf_classifier = SVC(kernel="rbf")
rbf_classifier.fit(X_train, y_train)
accuracy_score(y_test, rbf_classifier.predict(X_test))
# 1.0   ← perfect
```

The RBF (Radial Basis Function) kernel implicitly maps data to **infinite-dimensional space**. For concentric circles, it finds a perfect circular boundary in 2D.

The `gamma` parameter controls the width of influence of each support vector:
- **High gamma** → each support vector only affects nearby points → complex, tight boundary → overfitting risk
- **Low gamma** → each support vector influences a wider area → smoother boundary

---

## Polynomial Kernel

```python
poly_classifier = SVC(kernel="poly", degree=2)
poly_classifier.fit(X_train, y_train)
accuracy_score(y_test, poly_classifier.predict(X_test))
# 1.0   ← also perfect
```

The polynomial kernel of degree 2 also perfectly separates the concentric circles.

---

## Manual Feature Transformation (What the Kernel Computes)

You can manually see what the kernel is doing by computing `exp(-x²)` explicitly:

```python
np.exp(-(X**2)).sum(1)
# array([1.92474932, 1.30551477, 1.33451789, ...])
# Values range ~1.0 to ~2.0 (element-wise exp, then sum)

X_new = np.exp(-(X**2))
plt.scatter(X_new[:, 0], X_new[:, 1], c=y, s=50, cmap='bwr')
```

In the transformed space `X_new`, the two concentric-ring classes form two separate clusters — linearly separable.

---

## Kernel Comparison

| Kernel | Accuracy on `make_circles` | When to use |
|---|---|---|
| `linear` | 0.55 | Linearly separable data, high-dimensional text |
| `rbf` | **1.0** | General purpose — start here |
| `poly` (degree=2) | **1.0** | Known polynomial relationship |

---

## Key Parameters Summary

| Parameter | Effect |
|---|---|
| `kernel` | `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'` |
| `C` | Regularization: high C = hard margin, low C = soft margin |
| `gamma` | RBF/poly: reach of each support vector. `'scale'` (default) = `1/(n_features * X.var())` |
| `degree` | Polynomial kernel degree |

---

## Practical Notes

- Always scale features before SVM — distances and dot products are sensitive to feature scale. Use `StandardScaler`.
- Default `gamma='scale'` since sklearn 0.22 — use this over `'auto'`.
- SVM is memory-intensive for large datasets (n > 100,000). Consider `LinearSVC` for linear kernels on large data — it uses liblinear instead of libsvm.
- SVMs are naturally binary classifiers. sklearn uses one-vs-rest (OVR) by default for multiclass: `decision_function_shape='ovr'`.
