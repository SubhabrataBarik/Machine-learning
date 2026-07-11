# Day 64 — Decision Trees

## What is a Decision Tree?

A **Decision Tree** recursively splits the dataset by asking binary yes/no questions about feature values. Each internal node is a split condition, each leaf is a prediction. The result is a flowchart-like structure that humans can read and interpret.

```
Is petal length ≤ 2.45?
├── Yes → Setosa (class 0)
└── No → Is petal width ≤ 1.75?
         ├── Yes → Versicolor (class 1)
         └── No → Virginica (class 2)
```

At each node, the algorithm picks the feature and threshold that best separates the classes.

---

## Part 1: Classification Tree on Iris

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data   # 4 features: sepal length/width, petal length/width
y = iris.target # 3 classes: setosa=0, versicolor=1, virginica=2
```

Dataset: 150 samples, 3 classes (50 each), 4 features.

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train.shape → (120, 4)
# X_test.shape  → (30, 4)

clf = DecisionTreeClassifier()  # default: criterion='gini', no depth limit
clf.fit(X_train, y_train)
accuracy_score(y_test, clf.predict(X_test))
# 0.90   ← 90% accuracy on Iris test set
```

### Visualizing the Tree

```python
from sklearn.tree import plot_tree
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 80, 50
plot_tree(clf)
```

The fully-grown tree (no `max_depth`) can get quite deep. Use `max_depth=3` for readable plots.

---

## How Does the Tree Split? — Gini Impurity

The default criterion `gini` measures the probability that a random sample would be misclassified if labeled randomly:

```
Gini(node) = 1 - Σ pᵢ²
```

Where `pᵢ` is the proportion of class `i` at the node.

- Pure node (all one class): `Gini = 0`
- Maximally impure (50/50 split): `Gini = 0.5`

At each potential split, the algorithm computes:
```
Gini gain = Gini(parent) - [weighted avg of Gini(left child) + Gini(right child)]
```

The split with the highest Gini gain is chosen.

### Alternative: Entropy (Information Gain)

```python
clf = DecisionTreeClassifier(criterion='entropy')
```

```
Entropy(node) = -Σ pᵢ log₂(pᵢ)
```

Gini is slightly faster to compute (no log). Both usually give similar trees.

---

## Part 2: Depth-Limited Tree on Social Network Ads

```python
data = pd.read_csv('Social_Network_Ads.csv')
# Columns: User ID, Gender, Age, EstimatedSalary, Purchased

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
X = data.iloc[:, 1:4].values   # Gender, Age, EstimatedSalary
y = data.iloc[:, -1].values    # Purchased (binary)
# X.shape → (400, 3)

clf1 = DecisionTreeClassifier(max_depth=3)
clf1.fit(X, y)
```

Setting `max_depth=3` limits the tree to 3 levels of splits, producing a much more readable model and reducing overfitting risk.

---

## Part 3: Regression Tree on Housing Data

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['MedHouseVal'] = housing.target

X = df.iloc[:, 0:8]
y = df.iloc[:, 8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Fitting the Regression Tree

```python
rt = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
rt.fit(X_train, y_train)

r2_score(y_test, rt.predict(X_test))
# 0.8834   ← explains 88.3% of variance
```

For regression trees, the split criterion is `squared_error` (MSE) — the algorithm picks the split that most reduces the mean squared error of predictions within each child node.

### How Regression Trees Predict

A regression tree assigns each leaf the **mean** of training samples that fall into it. So predictions are step functions — constant within each leaf region.

---

## Feature Importance

```python
for importance, name in sorted(zip(rt.feature_importances_, X_train.columns), reverse=True):
    print(name, importance)
```

Output (Boston Housing cached result — feature importances sum to 1.0):
```
RM       0.6345   ← avg rooms per dwelling (63.4%)
LSTAT    0.1943   ← % lower-status population (19.4%)
CRIM     0.0740   ← crime rate
DIS      0.0674   ← distance to employment centers
B        0.0119
AGE      0.0062
PTRATIO  0.0044
NOX      0.0036
INDUS    0.0026
RAD      0.0012
ZN       0.0000   ← zero contribution
TAX      0.0000
CHAS     0.0000
```

Feature importance = total Gini/MSE reduction brought by that feature across all splits, normalized to sum to 1.

---

## Hyperparameter Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth':         [2, 4, 8, 10, None],
    'criterion':         ['squared_error', 'absolute_error'],
    'max_features':      [0.25, 0.5, 1.0],
    'min_samples_split': [0.25, 0.5, 1.0]
}

reg = GridSearchCV(DecisionTreeRegressor(), param_grid=param_grid)
reg.fit(X_train, y_train)

reg.best_score_    # 0.6452  (cross-validated R²)
reg.best_params_
# {'criterion': 'mse', 'max_depth': None, 'max_features': 0.5, 'min_samples_split': 0.25}
```

The best CV score (0.645) is lower than the manually tuned test R² (0.883) because cross-validation averages across folds and isn't trained on all data.

---

## Key Parameters

| Parameter | Effect |
|---|---|
| `max_depth` | Limits tree depth. `None` = grow until pure leaves (overfit risk) |
| `criterion` | Split quality measure: `'gini'`/`'entropy'` for classification; `'squared_error'`/`'absolute_error'` for regression |
| `min_samples_split` | Minimum samples to split a node (int or fraction). Higher = more pruning |
| `min_samples_leaf` | Minimum samples in a leaf. Higher = smoother predictions |
| `max_features` | How many features to consider at each split (fraction or `'sqrt'`, `'log2'`) |
| `random_state` | Controls randomness in tie-breaking |

---

## Overfitting: The Main Risk

An unconstrained decision tree will memorize the training data (every leaf = one sample, `Gini=0`):

```python
# Overfit tree
clf_deep = DecisionTreeClassifier()        # no max_depth
clf_deep.fit(X_train, y_train)
# train accuracy ≈ 1.0, test accuracy much lower

# Constrained tree
clf_shallow = DecisionTreeClassifier(max_depth=3)
clf_shallow.fit(X_train, y_train)
# lower train accuracy, better test accuracy
```

Controls for overfitting:
- `max_depth` — hard depth limit (most common)
- `min_samples_split` — require a minimum number of samples to split
- `min_samples_leaf` — require a minimum number in each leaf
- `max_leaf_nodes` — limit total number of leaves
- `ccp_alpha` — cost-complexity pruning (post-pruning)

---

## Advantages and Limitations

**Advantages:**
- Interpretable — can be visualized and explained to non-technical stakeholders
- No feature scaling required (splits are threshold-based)
- Handles mixed types (numerical and categorical)
- Non-linear decision boundaries

**Limitations:**
- High variance — small changes in data can produce very different trees
- Prone to overfitting without constraints
- Biased toward features with many distinct values
- Worse than ensembles (Random Forest, Gradient Boosting) in practice

Decision trees are the **building block** for Random Forest, AdaBoost, and Gradient Boosting — understanding them well is essential for understanding ensemble methods.
