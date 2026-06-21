# Day 16 — Working with JSON and SQL in Pandas

## Overview

Real-world data rarely arrives as clean CSV. Two extremely common formats are:

- **JSON** — the default for REST APIs, web services, and NoSQL databases
- **SQL** — the backbone of virtually every production data system

Pandas reads both natively, making it the single tool for ingesting data from almost any source.

---

## Part 1 — Working with JSON

### What is JSON?

JSON (JavaScript Object Notation) is a hierarchical key-value text format supporting nested objects and arrays. More expressive than CSV but potentially harder to flatten into tabular form.

### Loading a Local JSON File

```python
import pandas as pd

df = pd.read_json('train.json')
# → 39,774 rows × 3 columns
```

The `train.json` file is a recipe dataset — each record has an `id`, `cuisine` label, and `ingredients` list:

```
      id      cuisine                                        ingredients
0  10259        greek  [romaine lettuce, black olives, ...]
1  25693  southern_us  [plain flour, ground pepper, ...]
```

Pandas automatically converts the JSON array of objects into a DataFrame. List-valued columns (like `ingredients`) are preserved as Python lists.

### Loading JSON from an API URL

```python
pd.read_json('https://api.exchangerate-api.com/v4/latest/INR')
```

`pd.read_json()` accepts URLs directly. The exchange rate API returns a nested JSON with currency codes as keys and exchange rates as values. Pandas unpacks it into rows automatically — no explicit `requests.get()` + `json.loads()` needed.

### The `orient` Parameter

Different JSON structures require different `orient` values:

| `orient` | JSON structure |
|---|---|
| `'records'` | `[{"col1": v1, "col2": v2}, ...]` (default for most APIs) |
| `'index'` | `{"row0": {"col1": v1}, ...}` |
| `'columns'` | `{"col1": {"row0": v1}, ...}` |
| `'split'` | `{"columns": [...], "data": [[...]]}` |

### Handling Deeply Nested JSON

When JSON has nested objects, use `pd.json_normalize()`:

```python
from pandas import json_normalize

# response.json() = {"results": [{"user": {"name": "Alice", "city": "NY"}}], "page": 1}
df = json_normalize(response.json(), record_path=['results'], meta=['page'])
```

---

## Part 2 — Working with SQL

### Setup: MySQL via `mysql.connector`

```python
import mysql.connector

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='',
    database='world'
)
```

This connects to a local MySQL instance (via XAMPP) running the `world` sample database — a classic SQL teaching dataset with countries, cities, and languages.

### `pd.read_sql_query()`

```python
df = pd.read_sql_query("SELECT * FROM countrylanguage", conn)
```

Output:
```
    CountryCode    Language IsOfficial  Percentage
0           ABW       Dutch          T         5.3
1           ABW     English          F         9.5
...
[984 rows × 4 columns]
```

`pd.read_sql_query()` takes any valid SQL string and a connection object, executes the query, and returns the result as a DataFrame.

### Push Filtering to SQL

```python
df = pd.read_sql_query(
    "SELECT CountryCode, Language FROM countrylanguage WHERE IsOfficial='T'",
    conn
)
```

Always filter and aggregate in SQL when possible — the database engine handles these far more efficiently than Pandas on large tables. Bring only what you need into memory.

### Close the Connection

```python
conn.close()
```

Always close the connection after loading data. Leaving connections open wastes database resources (connection pool slots).

### Parameterized Queries (Safe for User Input)

```python
pd.read_sql_query(
    "SELECT * FROM countrylanguage WHERE CountryCode = %s",
    conn,
    params=('USA',)
)
```

Never format user input directly into SQL strings — this opens SQL injection vulnerabilities. Use parameterized queries instead.

### Modern Approach: SQLAlchemy

```python
from sqlalchemy import create_engine

engine = create_engine('mysql+mysqlconnector://root:password@localhost/world')
df = pd.read_sql_query("SELECT * FROM countrylanguage", engine)
```

SQLAlchemy is the preferred modern approach — it works with all major databases (PostgreSQL, SQLite, SQL Server, Oracle) via a unified connection interface and is officially recommended by Pandas.

---

## Comparison: CSV vs JSON vs SQL

| Property | CSV | JSON | SQL |
|---|---|---|---|
| Structure | Flat, tabular | Hierarchical, nested | Relational, multi-table |
| Human readable | Yes | Yes | Via query |
| Handles nesting | No | Yes | Via JOINs |
| Best for | Static data exchange | APIs, configs | Production systems |
| Pandas function | `read_csv()` | `read_json()` | `read_sql_query()` |
| Requires connection | No | No (for local files) | Yes |

---

## Practical Tips

- Use `pd.read_json(url)` for quick API pulls; switch to `requests` + explicit handling when you need error handling or authentication headers.
- For **very large SQL tables**, add `chunksize` to `read_sql_query()` — works the same as in `read_csv()`.
- `pd.read_sql_table()` reads an entire table without needing a SQL string — but only works with SQLAlchemy engines, not raw connections.
- The `world` database used here is a standard MySQL sample dataset available from MySQL's official downloads.
