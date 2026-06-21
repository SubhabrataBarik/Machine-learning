# Day 45 — Feature Construction and Feature Splitting

## Overview

**Feature Engineering** is the process of creating new, more informative features from existing ones. This day covers two key techniques: **Feature Construction** (building new features by combining existing ones) and **Feature Splitting** (extracting sub-components from a single feature).

---

## Dataset: Titanic (train.csv)

```python
df = pd.read_csv('train.csv')
```

---

## Part 1: Feature Construction

### What Is It?

Creating new features by combining, transforming, or aggregating existing features in ways that add predictive power beyond the originals.

### Examples from Titanic

#### Family Size
```python
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
```
`SibSp` (siblings/spouses) and `Parch` (parents/children) individually have limited signal. Their sum (family size) is more interpretable and predictive — larger families had different survival dynamics.

#### Is Alone
```python
df['Is_Alone'] = (df['Family_Size'] == 1).astype(int)
```
Binary feature derived from family size. Passengers traveling alone had different survival rates than those with family.

#### Fare Per Person
```python
df['Fare_Per_Person'] = df['Fare'] / df['Family_Size']
```
Divides fare by family size to get a more accurate per-person cost (families may share a ticket).

#### Age Group
```python
df['Age_Group'] = pd.cut(df['Age'],
                          bins=[0, 12, 18, 35, 60, 80],
                          labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
```

---

## Part 2: Feature Splitting

### What Is It?

Decomposing a single feature that encodes multiple pieces of information into separate, usable columns.

### Name → Title

```python
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Mrs → Mrs, Mr → Mr, Miss → Miss, Dr → Dr, ...
```

The `Name` column contains a social title. Extracting it gives a categorical feature that correlates with age, gender, and class — all without adding any external data.

#### Map Rare Titles to Standard Groups
```python
df['Title'] = df['Title'].replace(
    ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
     'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare'
)
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
```

#### Ticket Prefix
```python
df['Ticket_Prefix'] = df['Ticket'].str.extract(r'^([A-Za-z/. ]+)', expand=False)
df['Ticket_Prefix'] = df['Ticket_Prefix'].str.strip()
```

Some tickets have an alphabetic prefix that encodes booking class or origin.

#### Cabin Deck
```python
df['Cabin_Deck'] = df['Cabin'].str.extract(r'([A-Za-z])', expand=False)
```

The deck letter is the most useful part of the Cabin column.

---

## Why Feature Engineering Matters

| Original Feature | Issue | Engineered Feature | Benefit |
|------------------|-------|--------------------|---------|
| `SibSp`, `Parch` | Separate signals | `Family_Size` | Unified, interpretable |
| `Name` | Too many unique strings | `Title` | 5 categories, predictive |
| `Cabin` | 77% missing | `Cabin_Deck` | Less missing, still useful |
| `Ticket` | Alphanumeric mess | `Ticket_Prefix` | Structured category |
| `Age` | Continuous | `Age_Group` | Non-linear relationships captured |

---

## Practical Tips

- Domain knowledge drives good feature construction — understand what each variable represents.
- Always compare model performance with and without engineered features.
- Avoid creating features that leak the target (e.g., creating a feature using the target variable).
- Feature splitting is especially powerful for text, ID, and code columns that encode structured information.
- After feature engineering, re-run EDA to verify the new features have the expected distribution and correlation with the target.
