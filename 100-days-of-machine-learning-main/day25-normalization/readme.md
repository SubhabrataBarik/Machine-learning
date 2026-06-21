# Day 25 — Normalization (Min-Max Scaling)

## What Is Normalization?

Normalization (Min-Max Scaling) rescales features to a fixed range, typically **[0, 1]**:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

After scaling: the minimum value maps to 0, the maximum maps to 1, all others fall in between.

---

## Dataset: Wine Data

```python
df = pd.read_csv('wine_data.csv', header=None, usecols=[0,1,2])
df.columns = ['Class label', 'Alcohol', 'Malic acid']
```

```
     Class label  Alcohol  Malic acid
0              1    14.23        1.71
1              1    13.20        1.78
...
[178 rows x 3 columns]
```

- `Alcohol`: range ~11–14.8
- `Malic acid`: range ~0.9–5.65

Both are in different units and scales. Normalization brings them to [0,1].

---

## Applying MinMaxScaler

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Class label', axis=1), df['Class label'],
    test_size=0.3, random_state=0
)
# X_train: (124, 2)   X_test: (54, 2)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

### Before vs After

```
Before MinMaxScaler:
       Alcohol  Malic acid
mean      13.0         2.4
std        0.8         1.1
min       11.0         0.9
max       14.8         5.6

After MinMaxScaler:
       Alcohol  Malic acid
mean       0.5         0.3
std        0.2         0.2
min        0.0         0.0
max        1.0         1.0
```

Both features now live in [0, 1]. The shape of the distribution is preserved.

---

## Standardization vs Normalization

| Property | StandardScaler | MinMaxScaler |
|----------|---------------|--------------|
| Formula | (x - mean) / std | (x - min) / (max - min) |
| Output range | Unbounded (~−3 to +3) | Fixed [0, 1] |
| Mean after | 0 | Not guaranteed |
| Std after | 1 | Not guaranteed |
| Preserves distribution shape | Yes | Yes |
| Sensitive to outliers | Yes (mean/std shift) | Yes (min/max shift) |
| Use when | Algorithm needs zero mean | Algorithm needs bounded input |

### When to Use MinMaxScaler

- **Neural networks** with sigmoid/tanh activations that expect input in [0, 1]
- **Image pixel values** (normalize 0–255 → 0–1)
- **KNN / K-means** when all features should have equal weight within [0,1]
- When you want to preserve zero values (MinMaxScaler keeps zeros at 0)

### When to Use StandardScaler

- **Linear models** (Logistic Regression, Linear SVM, Ridge/Lasso)
- **PCA** (requires zero-mean features)
- When data has meaningful outliers that shouldn't define the range
- Most general-purpose use cases

---

## Sensitivity to Outliers

MinMaxScaler is highly sensitive to outliers because the range is defined by min and max. One extreme outlier compresses all other values into a tiny slice of [0, 1]:

```
Normal values: [10, 12, 14] → scaled: [0.0, 0.5, 1.0]
With outlier:  [10, 12, 14, 1000] → normal values → [0.0, 0.002, 0.004]
```

If your data has outliers, use `RobustScaler` (IQR-based) or clip outliers before scaling.

---

## Custom Range: [a, b]

```python
scaler = MinMaxScaler(feature_range=(-1, 1))
```

Scales to any [a, b] range. Common choices:
- `(0, 1)` — default, sigmoid activation
- `(-1, 1)` — tanh activation, symmetric around zero

---

## Always Fit on Train Only

```python
scaler.fit(X_train)                        # learn min/max from train
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # apply train's min/max to test
```

If test data falls outside the training range, scaled values will be outside [0, 1] — this is expected and acceptable.

---

## Practical Tips

- For image data: simply divide by 255.0 instead of fitting a scaler.
- Keep the fitted scaler and use it on production inference data.
- Check for outliers before applying MinMaxScaler — they compress the useful range.
- Use `scaler.data_min_` and `scaler.data_max_` to inspect what range was learned.
