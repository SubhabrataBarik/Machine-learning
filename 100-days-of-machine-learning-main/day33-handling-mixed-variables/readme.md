# Day 33 — Handling Mixed Variables

## What Are Mixed Variables?

A **mixed variable** is a single column that contains both numerical and categorical information encoded in the same string. Real datasets frequently contain these — they are not errors, just information-dense encodings.

Examples from the Titanic dataset:
- `Cabin`: `"C85"` → letter `C` (deck/section) + number `85` (cabin number)
- `Ticket`: `"A/5 21171"` → prefix `A/5` (ticket type code) + number `21171`
- `number` column: values like `5`, `3`, `6`, `A` → numeric counts mixed with letter codes

Mixed variables cannot be used directly by any ML algorithm — they must be split into component parts first.

---

## Dataset

```python
df = pd.read_csv('titanic.csv')
# Columns: Cabin, Ticket, number, Survived
```

---

## Handling the `number` Column (Count + Letter Code)

The `number` column contains values like `5`, `3`, `6`, `A`, `2`. Most are digits (passenger group size), but some are letter codes.

```python
df['number'].unique()
# ['5', '3', '6', 'A', '2', '1', '4']
```

### Step 1: Extract the numerical part

```python
df['number_numerical'] = pd.to_numeric(df["number"], errors='coerce', downcast='integer')
```

`errors='coerce'` converts non-numeric strings to `NaN`. Integer values become numbers; letter values become NaN.

### Step 2: Extract the categorical part

```python
df['number_categorical'] = np.where(df['number_numerical'].isnull(), df['number'], np.nan)
```

Where the numerical part is NaN (it was a letter), retain the original string. Otherwise NaN.

Result:
```
number  number_numerical  number_categorical
5             5.0               NaN
A             NaN               A
```

---

## Handling the `Cabin` Column (Deck Letter + Cabin Number)

`Cabin` values like `C85`, `C123`, `B78`, `C23 C25 C27`. The letter prefix encodes the ship deck; the number encodes the specific cabin.

```python
df['cabin_num'] = df['Cabin'].str.extract('(\d+)')   # first digit sequence
df['cabin_cat'] = df['Cabin'].str[0]                  # first character (deck letter)
```

`str.extract('(\d+)')` uses regex to pull the first sequence of digits. `str[0]` retrieves the first character.

Result:
```
Cabin   cabin_num  cabin_cat
C85     85         C
C123    123        C
NaN     NaN        NaN
G6      6          G
```

---

## Handling the `Ticket` Column (Category Code + Number)

`Ticket` values like `A/5 21171`, `PC 17599`, `113803`. The last space-separated token is the number; the first (if non-numeric) is the category code.

### Extract ticket number (last token)

```python
df['ticket_num'] = df['Ticket'].apply(lambda s: s.split()[-1])
df['ticket_num'] = pd.to_numeric(df['ticket_num'], errors='coerce', downcast='integer')
```

`split()[-1]` takes the last whitespace-separated token:
- `A/5 21171` → `21171`
- `113803` → `113803`
- `STON/O2. 3101282` → `3101282`

### Extract ticket category (first non-numeric token)

```python
df['ticket_cat'] = df['Ticket'].apply(lambda s: s.split()[0])
df['ticket_cat'] = np.where(df['ticket_cat'].str.isdigit(), np.nan, df['ticket_cat'])
```

Takes the first token. If it's purely numeric (no prefix code), set to NaN. Result: `A/5`, `PC`, `SOTON/OQ`, `NaN`, etc.

Unique ticket categories found: 44 distinct codes (`A/5`, `PC`, `STON/O2.`, `PP`, `C.A.`, `SC/Paris`, etc.).

### Full result

```
Ticket          ticket_num  ticket_cat
A/5 21171        21171.0     A/5
PC 17599         17599.0     PC
STON/O2. 3101282 3101282.0   STON/O2.
113803           113803.0    NaN
373450           373450.0    NaN
```

---

## Why Split Mixed Variables?

| Raw mixed column | Usable by ML? | After splitting |
|---|---|---|
| `"C85"` | No | `cabin_cat="C"` (encode), `cabin_num=85` (scale) |
| `"A/5 21171"` | No | `ticket_cat="A/5"` (encode), `ticket_num=21171` (scale) |
| `"A"` (letter count) | No | `number_categorical="A"` (encode) |

After splitting:
- **Categorical parts** → OrdinalEncoder or OneHotEncoder
- **Numerical parts** → use directly or normalize

---

## Key Functions Summary

| Function | Purpose |
|---|---|
| `pd.to_numeric(col, errors='coerce')` | Convert to number, NaN on failure |
| `series.str.extract('(\d+)')` | Regex extract first digit sequence |
| `series.str[0]` | First character of each string |
| `series.apply(lambda s: s.split()[-1])` | Last whitespace-split token |
| `series.str.isdigit()` | True if all characters are digits |
| `np.where(condition, true_val, false_val)` | Vectorized if-else |

---

## Practical Tips

- Always visualize the mixed column's unique values first to understand the pattern.
- `str.extract()` with named groups (e.g., `'(?P<deck>[A-Z])(?P<num>\d+)'`) can split in one step.
- After splitting, drop the original mixed column — it contains no additional information.
- Missing cabin (`NaN`) in Titanic correlates with 3rd-class passengers — missing itself is informative. Consider a `has_cabin` binary indicator rather than imputing the deck letter.
- The deck letter in a ship cabin is genuinely meaningful — higher decks (A, B, C) are first-class; lower decks (E, F, G) are third-class.
