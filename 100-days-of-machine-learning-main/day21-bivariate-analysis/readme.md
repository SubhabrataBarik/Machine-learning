# Day 21 — Bivariate Analysis

## Overview

**Bivariate analysis** examines relationships between two variables at a time. While univariate analysis describes individual features, bivariate analysis reveals how features interact — which is what actually matters for predictive modeling. This notebook uses the Titanic, Tips, Flights, and Iris datasets to cover all major variable-type combinations.

---

## Why Bivariate Analysis?

- Identify which features are predictive of the target
- Detect multicollinearity between features (redundant predictors)
- Understand how features interact
- Guide feature engineering decisions

---

## Datasets Used

```python
tips = sns.load_dataset('tips')       # restaurant bills and tips
titanic = pd.read_csv('train.csv')    # Titanic survival data
flights = sns.load_dataset('flights') # monthly airline passengers 1949–1960
iris = sns.load_dataset('iris')       # flower measurements by species
```

---

## 1. Scatter Plot — Numerical vs Numerical

```python
sns.scatterplot(
    tips['total_bill'], tips['tip'],
    hue=df['sex'], style=df['smoker'], size=df['size']
)
```

Plots two numerical variables on X and Y axes. Each point is one observation. Parameters:
- `hue` — colors points by a categorical variable (e.g., `sex`)
- `style` — changes point shape by a categorical variable (e.g., `smoker`)
- `size` — scales point size by a variable (e.g., `size` of party)

**What to look for**: positive/negative slope (linear relationship), clusters (subgroups), outliers, fan shapes (heteroscedasticity).

---

## 2. Bar Plot — Numerical vs Categorical

```python
sns.barplot(titanic['Pclass'], titanic['Age'], hue=titanic['Sex'])
```

Shows the **mean** (with confidence interval error bars) of a numerical variable grouped by a categorical variable. Adding `hue` splits each group by a second categorical variable.

From Titanic:
- Class 1: average age ~41 (male), ~35 (female)
- Class 2: ~31 / ~29
- Class 3: ~26 / ~23

Higher class passengers were consistently older. The confidence intervals (error bars) show statistical uncertainty.

**Note**: Seaborn's `barplot` shows mean ± CI, not sum. For sums, use `df.groupby().sum().plot(kind='bar')`.

---

## 3. Box Plot — Numerical vs Categorical

```python
sns.boxplot(titanic['Sex'], titanic['Age'], hue=titanic['Survived'])
```

Shows the five-number summary of a numerical variable split by a categorical variable. Adding `hue` creates side-by-side boxes for each category value.

From Titanic:
- Males who survived had a slightly lower median age than those who died
- Females who survived were older on average than those who died
- The spread (IQR) is similar across groups

Box plots are excellent for spotting whether the distribution of a numerical variable differs across classes — a key check for classification features.

---

## 4. Distribution Plot — Numerical vs Categorical

```python
sns.distplot(titanic[titanic['Survived'] == 0]['Age'], hist=False)
sns.distplot(titanic[titanic['Survived'] == 1]['Age'], hist=False)
```

Overlays two KDE curves: one for survivors, one for non-survivors. `hist=False` shows only the density curve.

This is more informative than two separate histograms because the curves share the same scale, making comparison easier. Look for regions where the curves separate — those age ranges have higher/lower survival probability.

---

## 5. Heat Map — Categorical vs Categorical

```python
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived']))
```

`pd.crosstab()` builds a frequency table: rows = Pclass, columns = Survived, values = count. `sns.heatmap()` color-encodes the counts — darker = higher frequency.

From Titanic:
- Class 3, Survived=0: very dark (~372 passengers) — most deaths in 3rd class
- Class 1, Survived=1: moderate count — 1st class had the best survival

For proportions instead of counts:
```python
pd.crosstab(titanic['Pclass'], titanic['Survived'], normalize='index')
# → shows survival rate per class
```

---

## 6. Cluster Map — Categorical vs Categorical

```python
sns.clustermap(pd.crosstab(titanic['Parch'], titanic['Survived']))
```

Like a heatmap, but **hierarchically clusters** both rows and columns using dendrogram linkage. Groups similar rows/columns together automatically. Useful for discovering unexpected groupings.

---

## 7. Pair Plot — All Numerical vs All Numerical

```python
sns.pairplot(iris, hue='species')
```

Creates a grid of scatter plots for every pair of numerical features. The diagonal shows the distribution of each individual feature (histogram or KDE). `hue='species'` colors points by class.

On the Iris dataset, this immediately reveals:
- `petal_length` and `petal_width` strongly separate the three species
- `sepal_width` overlaps heavily between species

Pair plots are the fastest way to survey all pairwise relationships at once for datasets with fewer than ~15 features.

---

## 8. Line Plot — Numerical vs Numerical (Time Series)

```python
new = flights.groupby('year').sum().reset_index()
sns.lineplot(new['year'], new['passengers'])
```

Line plots connect sequential observations — ideal for time series. The flights dataset shows total annual passengers growing from ~1,500 in 1949 to ~5,700 in 1960.

For month-by-year patterns:
```python
sns.clustermap(flights.pivot_table(values='passengers', index='month', columns='year'))
```

---

## Summary: Which Plot for Which Variable Types?

| Variable X | Variable Y | Best Plot |
|---|---|---|
| Numerical | Numerical | Scatter plot, Line plot |
| Categorical | Numerical | Bar plot (means), Box plot (distribution) |
| Numerical | Categorical | Distribution plot (overlaid KDE) |
| Categorical | Categorical | Heatmap, Crosstab |
| All numerical | All numerical | Pair plot |

---

## Correlation: The Numerical Summary of Bivariate Relationships

```python
titanic.groupby('Embarked').mean()['Survived'] * 100
# Embarked C: 55.4%  Q: 39.0%  S: 33.7%
```

Numerical aggregations like `.groupby().mean()` give the exact relationship in numbers. Combine with plots for full understanding — plots show shape, numbers give precise values.
