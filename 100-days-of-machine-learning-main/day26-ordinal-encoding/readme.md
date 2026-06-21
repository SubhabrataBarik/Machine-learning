# Day 26 — Ordinal Encoding

## What Is Ordinal Encoding?

Ordinal encoding converts categorical text values into integers while **preserving the natural order** (rank) of the categories. It is used when a categorical feature has a meaningful ranking (e.g., Low < Medium < High).

---

## Dataset: Customer Data

```python
df = pd.read_csv('customer.csv')
```

Typical columns: `review` (Good/Average/Poor), `education` (School/UG/PG), `purchased` (Yes/No).

---

## Why Not Use Label Encoding for Ordinal Data?

`LabelEncoder` assigns integers alphabetically — no guarantee of order:
- `Average → 0, Good → 1, Poor → 2`

This says `Poor > Good > Average` which is wrong. **Ordinal encoding lets you specify the correct order explicitly.**

---

## sklearn OrdinalEncoder

```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[
    ['Poor', 'Average', 'Good'],    # review column order
    ['School', 'UG', 'PG']          # education column order
])

oe.fit(df[['review', 'education']])
df[['review', 'education']] = oe.transform(df[['review', 'education']])
```

Result:
- `Poor → 0, Average → 1, Good → 2`
- `School → 0, UG → 1, PG → 2`

The integers now correctly reflect the ordering.

---

## Ordinal vs Nominal Encoding

| Feature Type | Has Order? | Encoding |
|---|---|---|
| Ordinal | Yes (Low < Med < High) | OrdinalEncoder with explicit order |
| Nominal | No (Red, Blue, Green) | OneHotEncoder |

**Never apply ordinal encoding to nominal data** — it implies a false ordering that models will learn and exploit incorrectly. E.g., encoding `Red=0, Blue=1, Green=2` implies Green > Blue > Red, which is meaningless.

---

## Handling Unknown Categories

```python
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
```

If test data contains a category not seen during training, `handle_unknown='use_encoded_value'` maps it to `unknown_value` (-1 by default) instead of raising an error.

---

## When the Order Is Natural but Not Explicit

For binary categories (Yes/No, True/False):

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['purchased'] = le.fit_transform(df['purchased'])
# No → 0, Yes → 1
```

`LabelEncoder` is fine for binary columns since there are only two values.

---

## Practical Tips

- Always specify the `categories` order explicitly — never rely on alphabetical defaults for ordinal features.
- After encoding, verify the mapping with `oe.categories_`.
- Ordinal encoding preserves the column as a single integer column — much lower dimensionality than one-hot encoding.
- Tree-based models (Decision Trees, Random Forests) can learn non-linear relationships even from integer-encoded ordinals, making ordinal encoding fine for them too.
- For linear models, consider whether the integer spacing is appropriate (e.g., does 2× the integer mean 2× the effect?).
