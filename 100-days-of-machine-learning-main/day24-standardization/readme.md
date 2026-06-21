# Day 24 — Standardization (Z-score Scaling)

## What Is Standardization?

Standardization transforms each feature so that it has **mean = 0** and **standard deviation = 1**:

```
z = (x - mean) / std
```

Also called **Z-score normalization**. After transformation the feature values are in units of standard deviations from the mean.

---

## Why Standardize?

Many ML algorithms are sensitive to feature scale:

| Algorithm | Scale-sensitive? | Reason |
|-----------|-----------------|--------|
| Logistic Regression | Yes | Gradient descent converges faster with equal scales |
| KNN | Yes | Distance metrics treat all features equally |
| SVM | Yes | Margin maximization is scale-dependent |
| PCA | Yes | Variance computation is scale-dependent |
| Neural Networks | Yes | Weight updates assume similar feature ranges |
| Decision Tree | No | Splits use thresholds, not distances |
| Random Forest | No | Ensemble of trees |

---

## Dataset: Social Network Ads

```python
df = pd.read_csv('Social_Network_Ads.csv')
df = df.iloc[:, 2:]  # keep Age, EstimatedSalary, Purchased
```

```
     Age  EstimatedSalary  Purchased
208   40           142000          1
367   46            88000          1
157   29            75000          0
```

- `Age`: range 18–60
- `EstimatedSalary`: range 15,000–150,000

These two features are on completely different scales. Without standardization, salary dominates any distance calculation.

---

## Train-Test Split First, Then Scale

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1), df['Purchased'],
    test_size=0.3, random_state=0
)
# X_train: (280, 2)   X_test: (120, 2)
```

**Critical rule**: always split before scaling. Fitting the scaler on the full dataset leaks test statistics (mean, std) into training — a form of data leakage.

---

## Applying StandardScaler

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X_train)          # learn mean and std from TRAIN only
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # use TRAIN statistics on test
```

`scaler.mean_` → `[37.86, 69807.14]`  (means of Age and Salary in training set)

### Before vs After

```
Before scaling:
         Age  EstimatedSalary
mean    37.9          69807.1
std     10.2          34641.2
min     18.0          15000.0
max     60.0         150000.0

After StandardScaler:
         Age  EstimatedSalary
mean     0.0              0.0
std      1.0              1.0
min     -1.9             -1.6
max      2.2              2.3
```

Both features now have identical scale. The shape of the distribution is preserved — StandardScaler does not change the distribution, only shifts and scales it.

---

## Effect on Model Accuracy

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(accuracy_score(y_test, lr.predict(X_test)))          # 0.658

lr_scaled = LogisticRegression()
lr_scaled.fit(X_train_scaled, y_train)
print(accuracy_score(y_test, lr_scaled.predict(X_test_scaled)))  # 0.867
```

Logistic Regression accuracy jumps from **65.8% → 86.7%** with standardization. This is because gradient descent can't converge properly when features have vastly different scales — it oscillates and gets stuck.

```python
from sklearn.tree import DecisionTreeClassifier

dt.fit(X_train, y_train)       # 0.875
dt_scaled.fit(X_train_scaled, y_train)  # 0.875
```

Decision Tree accuracy is identical with or without scaling — it makes no distance calculations.

---

## Effect of Outliers

StandardScaler uses mean and std — both are sensitive to outliers. Adding outliers (Age=5,90,95; Salary=1000,250000,350000) shifts the mean and inflates std, causing all "normal" values to compress toward 0.

For outlier-robust scaling, use `RobustScaler` (uses median and IQR instead of mean and std).

---

## StandardScaler vs Other Scalers

| Scaler | Formula | Output Range | Outlier-robust |
|--------|---------|--------------|----------------|
| `StandardScaler` | `(x - mean) / std` | Unbounded (~-3 to +3) | No |
| `MinMaxScaler` | `(x - min) / (max - min)` | [0, 1] | No |
| `RobustScaler` | `(x - median) / IQR` | Unbounded | Yes |
| `MaxAbsScaler` | `x / max(abs(x))` | [-1, 1] | No |

---

## Always Fit on Train, Transform Both

```python
# CORRECT
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# WRONG — data leakage
scaler.fit(X_train)
X_test_scaled = scaler.fit_transform(X_test)  # learns test stats
```

Using `fit_transform()` on test data leaks test distribution information and gives overly optimistic evaluation metrics.

---

## Practical Tips

- StandardScaler is the default choice for most gradient-based models.
- Convert scaled arrays back to DataFrame to keep column names: `pd.DataFrame(X_train_scaled, columns=X_train.columns)`.
- In sklearn Pipelines, the scaler is automatically applied only to training data during cross-validation.
- Always save the fitted scaler (`pickle` or `joblib`) and use it on new inference data.
