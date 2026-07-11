# End-to-End Machine Learning Pipeline

## Overview

This notebook walks through the complete machine learning workflow — from raw data to a deployed model. It covers every step in the standard ML pipeline using a binary classification task (predicting placement outcomes from academic metrics).

---

## Dataset

```python
df = pd.read_csv('placement.csv')
df.head()
```

```
   cgpa     iq  placement
0   6.8  123.0          1
1   5.9  106.0          0
2   5.3  121.0          0
3   7.4  132.0          1
4   5.8  142.0          0
```

- 100 samples
- 2 features: `cgpa` (CGPA on a 10-point scale), `iq` (IQ score)
- Target: `placement` (1 = placed, 0 = not placed)

```python
df = df.iloc[:, 1:]  # drop the unnamed index column
```

---

## The 6-Step ML Pipeline

```python
# Steps:
# 0. Preprocess + EDA + Feature Selection
# 1. Extract input and output cols
# 2. Scale the values
# 3. Train test split
# 4. Train the model
# 5. Evaluate the model / model selection
# 6. Deploy the model
```

---

## Step 0: EDA

```python
plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
```

A scatter plot of CGPA vs IQ colored by placement. Helps visually check if the classes are separable. High CGPA + high IQ tends toward placement=1.

---

## Step 1: Extract Features and Target

```python
X = df.iloc[:, 0:2]  # cgpa, iq
y = df.iloc[:, -1]   # placement
# X.shape → (100, 2)
# y.shape → (100,)
```

---

## Step 2 & 3: Train/Test Split and Scaling

**Split first, then scale** — a critical ordering rule. Fitting the scaler on the full dataset leaks test set statistics into training.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# 90 training samples, 10 test samples
```

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit on train only
X_test  = scaler.transform(X_test)       # transform test with train statistics
```

After scaling, `X_train` values center around 0 with unit variance. Example scaled test set:

```
array([[ 1.315,  -1.580],
       [ 0.041,   0.214],
       [ 0.041,  -1.074],
       [-2.507,   1.502],
       ...])
```

---

## Step 4: Train the Model

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X_train, y_train)
```

Default LogisticRegression: `C=1.0`, `solver='lbfgs'`, `max_iter=100`.

---

## Step 5: Evaluate

```python
from sklearn.metrics import accuracy_score

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
# 0.9   ← 90% accuracy on 10-sample test set
```

Test set ground truth vs predictions:
```
y_test: [1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
```

The small test set (10 samples) makes the accuracy metric noisy — a single misclassification changes accuracy by 10%. For real projects, use cross-validation or a larger test set.

### Visualizing the Decision Boundary

```python
from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train, y_train.values, clf=clf, legend=2)
```

This plots the logistic regression decision boundary over the scaled training data. With 2 features, the boundary is a straight line in the standardized feature space.

---

## Step 6: Deploy — Save the Model

```python
import pickle

pickle.dump(clf, open('model.pkl', 'wb'))
```

The trained model is serialized to `model.pkl`. To reload and use in production:

```python
model = pickle.load(open('model.pkl', 'rb'))
model.predict(scaler.transform([[7.0, 130.0]]))  # predict for a new student
```

**Important:** Save the `scaler` too, not just the model. New inputs must be scaled with the same `StandardScaler` fitted on the training data.

```python
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```

---

## Why This Order Matters

| Step | Why |
|---|---|
| Split before scaling | Prevents data leakage: scaler must not see test set during fit |
| Fit scaler on train only | Using test statistics during training gives optimistic (inflated) metrics |
| Use `transform` (not `fit_transform`) on test | Applies the training mean/std to test data |
| Save scaler with model | Prediction at serve time requires the same transformation |

---

## Full Pipeline Code

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load and clean
df = pd.read_csv('placement.csv')
df = df.iloc[:, 1:]

# Split
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Train
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
print(accuracy_score(y_test, clf.predict(X_test)))  # 0.9

# Deploy
pickle.dump(clf, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
```
