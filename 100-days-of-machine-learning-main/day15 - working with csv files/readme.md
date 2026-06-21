# Day 15 — Working with CSV Files

## What is a CSV File?

CSV (Comma-Separated Values) is the most common format for storing and exchanging tabular data. Each line is a row and values within a row are separated by a delimiter (usually a comma). Pandas' `read_csv()` is the primary tool for loading CSV data into a DataFrame, and it exposes dozens of parameters that control parsing behavior.

---

## Why This Matters

Almost every real-world ML project begins with loading data from a CSV file. Understanding `read_csv()` deeply means you can handle:
- Files with non-standard delimiters (TSV, pipe-separated)
- Huge datasets that don't fit in RAM
- Files with encoding issues (non-English text)
- Columns that need type coercion on load
- Custom missing value representations

---

## Parameters Covered

### 1. Basic Load

```python
df = pd.read_csv('aug_train.csv')
```

Loads a 19,158-row HR analytics dataset. Pandas infers column types automatically — numeric columns become `int64`/`float64`, text becomes `object`.

---

### 2. Loading from a URL

```python
import requests
from io import StringIO

url = "https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
headers = {"User-Agent": "Mozilla/5.0 ..."}
req = requests.get(url, headers=headers)
data = StringIO(req.text)
pd.read_csv(data)
```

`StringIO` wraps a string as a file-like object so it can be passed directly to `read_csv()`. The `User-Agent` header avoids 403 blocks from servers that reject Python's default agent.

---

### 3. `sep` — Custom Delimiter

```python
pd.read_csv('movie_titles_metadata.tsv', sep='\t',
            names=['sno','name','release_year','rating','votes','genres'])
```

The file is tab-separated (`.tsv`). Setting `sep='\t'` tells the parser to split on tabs. The `names` parameter assigns column headers when the file has none.

| Separator | `sep` value |
|-----------|-------------|
| Comma     | `','` (default) |
| Tab       | `'\t'` |
| Semicolon | `';'` |
| Pipe      | `'\|'` |

---

### 4. `index_col` — Set Index Column

```python
pd.read_csv('aug_train.csv', index_col='enrollee_id')
```

Makes `enrollee_id` the DataFrame index instead of a regular column. Avoids creating a redundant integer index when the CSV already has a meaningful row identifier.

---

### 5. `header` — Which Row is the Header

```python
pd.read_csv('test.csv', header=1)
```

`header=1` means row index 1 (second row) is the column header — row 0 is skipped. Use `header=None` when there is no header row at all, combined with `names` to provide column labels.

---

### 6. `usecols` — Load Only Specific Columns

```python
pd.read_csv('aug_train.csv', usecols=['enrollee_id', 'gender', 'education_level'])
```

Only listed columns are read from disk. Critical for memory efficiency on wide datasets — columns you don't need are never loaded into RAM.

---

### 7. `nrows` and `skiprows` — Partial Reads

```python
pd.read_csv('aug_train.csv', nrows=100)       # first 100 rows only
pd.read_csv('aug_train.csv', skiprows=[1, 2]) # skip specific rows
```

`nrows` is essential when exploring a large file — load a small slice first to understand the schema before committing to a full read.

---

### 8. `encoding` — Handle Non-ASCII Files

```python
pd.read_csv('zomato.csv', encoding='latin-1')
```

The default encoding is `utf-8`. Files with accented or non-Latin characters often use `latin-1` (ISO-8859-1). Without the right encoding you get `UnicodeDecodeError`.

| Encoding | Use case |
|----------|----------|
| `utf-8`  | Default, modern files |
| `latin-1` | Western European characters |
| `cp1252` | Windows-generated files |
| `utf-16` | Some Excel exports |

---

### 9. `error_bad_lines=False` — Skip Malformed Rows

```python
pd.read_csv('BX-Books.csv', sep=';', encoding='latin-1', error_bad_lines=False)
```

The BX-Books dataset has rows with more fields than the header expects. Setting `error_bad_lines=False` skips those rows and logs a warning instead of crashing. 271,360 rows loaded successfully.

> In newer pandas: use `on_bad_lines='skip'` instead.

---

### 10. `dtype` — Force Column Types on Load

```python
pd.read_csv('aug_train.csv', dtype={'target': int})
```

By default `target` is inferred as `float64` because of NaNs. Specifying `dtype={'target': int}` avoids silent type mismatches downstream (e.g., passing float labels to a classifier expecting integers).

---

### 11. `parse_dates` — Auto-parse Date Columns

```python
pd.read_csv('IPL Matches 2008-2020.csv', parse_dates=['date'])
```

Without this, `date` loads as `object` (string). With `parse_dates=['date']` it becomes `datetime64[ns]`, enabling time-based filtering, `.dt.year`/`.dt.month` extraction, and resampling.

---

### 12. `converters` — Apply Functions During Load

```python
def rename(name):
    if name == "Royal Challengers Bangalore":
        return "RCB"
    return name

pd.read_csv('IPL Matches 2008-2020.csv', converters={'team1': rename})
```

`converters` applies a callable to a column during parsing — before type inference. More efficient than loading then calling `.apply()`.

---

### 13. `na_values` — Custom Missing Value Markers

```python
pd.read_csv('aug_train.csv', na_values=['Male'])
```

Extends pandas' default NaN list. Useful when a domain-specific sentinel (`"-1"`, `"N/A"`, `"unknown"`) isn't in pandas' defaults.

---

### 14. `chunksize` — Load in Chunks

```python
dfs = pd.read_csv('aug_train.csv', chunksize=5000)

for chunk in dfs:
    print(chunk.shape)
# (5000, 14)
# (5000, 14)
# (5000, 14)
# (4158, 14)
```

Returns a `TextFileReader` iterator. Each iteration yields a DataFrame of at most `chunksize` rows. Standard pattern for files larger than RAM — aggregate chunk-by-chunk without holding everything in memory at once.

---

## Practical Tips

- Always call `df.info()` after loading — confirms dtypes, non-null counts, and memory usage.
- Use `nrows=5` first on unknown files to inspect structure.
- Combine `usecols` + `dtype` to minimize RAM: only load what you need in the type you need.
- `parse_dates` is more efficient than calling `pd.to_datetime()` post-load.
- If you get `UnicodeDecodeError`, try `encoding='latin-1'` then `cp1252`.

---

## Common Pitfalls

| Mistake | Fix |
|---------|-----|
| Loading all columns when you need 3 | Use `usecols` |
| Wrong separator → single-column DataFrame | Inspect file, set correct `sep` |
| Date column stays as string | Add `parse_dates=['col']` |
| Float target when model expects int | Add `dtype={'target': int}` |
| Crash on malformed rows | Use `on_bad_lines='skip'` |
| Encoding error on international data | Try `encoding='latin-1'` |
| RAM exhaustion on large files | Use `chunksize` |

---

## Datasets Used

| File | Description |
|------|-------------|
| `aug_train.csv` | HR analytics — job change prediction (19,158 rows) |
| `movie_titles_metadata.tsv` | Tab-separated movie metadata (617 rows) |
| `IPL Matches 2008-2020.csv` | IPL cricket match results (816 rows) |
| `zomato.csv` | Restaurant data with international characters (9,551 rows) |
| `BX-Books.csv` | Book-crossing dataset with malformed rows (271K+ rows) |
