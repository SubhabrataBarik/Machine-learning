# Day 32 — Binning and Binarization

## What Is Binning?

**Binning** (also called discretization) converts a continuous numerical feature into discrete bins/intervals. Instead of the exact value, each observation is assigned a bin label.

Example: Age 25 → "Young Adult" bin (20–35).

---

## Dataset: Titanic (train.csv)

```python
df = pd.read_csv('train.csv')
```

Used to demonstrate binning the continuous `Age` and `Fare` features.

---

## Why Bin Continuous Features?

- **Non-linear relationships**: if survival rate is high for children AND elderly but low for middle-aged, binning captures this U-shape that linear models cannot.
- **Handling outliers**: extreme values fall into the top/bottom bin and lose their outlier influence.
- **Interpretability**: bins have meaningful labels (Child, Adult, Senior) instead of raw numbers.
- **Tree-like behavior in linear models**: discretized features let linear models approximate step functions.

---

## Uniform Binning with `KBinsDiscretizer`

```python
from sklearn.preprocessing import KBinsDiscretizer

kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
kbins.fit(df[['Age']])
df['Age_binned'] = kbins.transform(df[['Age']])
```

`strategy='uniform'`: equal-width bins. Age 0–80 with 5 bins → each bin is 16 years wide.

`encode='ordinal'`: returns integer bin labels (0, 1, 2, 3, 4).
`encode='onehot'`: returns sparse binary columns (one per bin).

---

## Quantile Binning (Equal-Frequency)

```python
kbins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
```

`strategy='quantile'`: each bin contains the same number of observations. Handles skewed distributions better than uniform binning.

---

## k-means Binning

```python
kbins = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans')
```

`strategy='kmeans'`: bins are defined by k-means cluster centers — finds natural groupings in the data.

---

## Custom Binning with `pd.cut()`

```python
bins = [0, 12, 18, 35, 60, 80]
labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

Full control over bin edges and labels. `pd.cut()` is for equal-width bins; `pd.qcut()` is for quantile bins.

---

## Binarization

**Binarization** is a special case of binning with only two bins: values above a threshold become 1, values below become 0.

```python
from sklearn.preprocessing import Binarizer

bn = Binarizer(threshold=50)
df['High_Fare'] = bn.fit_transform(df[['Fare']])
# Fare > 50 → 1 (high fare passenger)
# Fare ≤ 50 → 0 (standard fare)
```

Useful for:
- Converting probability outputs to binary labels
- Creating presence/absence features
- Simplifying a continuous feature into a meaningful flag

---

## Binning vs Original Feature

| Aspect | Raw continuous | Binned |
|--------|---------------|--------|
| Model type | Any | Especially linear models |
| Captures non-linearity | With polynomial features | Naturally |
| Loses information | No | Yes (within-bin variation lost) |
| Interpretability | Lower | Higher |
| Handles outliers | Needs separate treatment | Automatically (top/bottom bin) |

---

## Practical Tips

- Binning is most useful for **linear models** — it lets them capture step-function relationships.
- Tree-based models (Decision Trees, Random Forests) do not benefit from binning — they find their own splits.
- Use `strategy='quantile'` for skewed features to ensure equal representation in each bin.
- Never bin the target variable — only input features.
- After binning with `encode='ordinal'`, consider whether the integer labels imply an incorrect ordering for nominal bins.
