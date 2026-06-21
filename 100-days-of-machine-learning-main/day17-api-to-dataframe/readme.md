# Day 17 — API to DataFrame

## Overview

Most modern data sources expose data through **REST APIs** — HTTP endpoints that return JSON. This notebook demonstrates fetching paginated API data and building a Pandas DataFrame from it. The example uses the TMDB (The Movie Database) API to collect 8,551 top-rated movies across 428 pages.

---

## What is a REST API?

A REST API is a web service responding to HTTP requests with structured data (typically JSON). You control the response via:

- **Query parameters** — `?api_key=...&page=1`
- **Path parameters** — `/movie/{id}`
- **HTTP methods** — GET (read), POST (write), PUT (update), DELETE

---

## The TMDB API Endpoint

```
GET https://api.themoviedb.org/3/movie/top_rated?api_key=KEY&language=en-US&page=1
```

Returns up to 20 top-rated movies per page. Total pages: 428 → 8,551 movies.

---

## Step 1 — Single Request

```python
import pandas as pd
import requests

response = requests.get(
    'https://api.themoviedb.org/3/movie/top_rated?api_key=...&language=en-US&page=1'
)

temp_df = pd.DataFrame(response.json()['results'])[
    ['id','title','overview','release_date','popularity','vote_average','vote_count']
]
```

What happens:
1. `requests.get(url)` — sends the HTTP GET request
2. `response.json()` — parses the response body into a Python dict
3. `['results']` — extracts the list of movie records (each record is a dict)
4. `pd.DataFrame(list_of_dicts)` — converts to DataFrame automatically
5. Column selection keeps only the 7 relevant fields

---

## Step 2 — Paginated Loop

```python
df = pd.DataFrame()

for i in range(1, 429):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/top_rated?api_key=...&language=en-US&page={i}'
    )
    temp_df = pd.DataFrame(response.json()['results'])[
        ['id','title','overview','release_date','popularity','vote_average','vote_count']
    ]
    df = df.append(temp_df, ignore_index=True)
```

Final result: **8,551 rows × 7 columns**.

### Modern Version (Pandas 2.0+)

`df.append()` was deprecated in Pandas 2.0. Use `pd.concat()`:

```python
chunks = []
for i in range(1, 429):
    response = requests.get(f'...&page={i}')
    temp_df = pd.DataFrame(response.json()['results'])[...]
    chunks.append(temp_df)

df = pd.concat(chunks, ignore_index=True)
```

`pd.concat()` is significantly more memory-efficient because it does not copy the growing DataFrame on every iteration.

---

## Step 3 — Save to Disk

```python
df.to_csv('movies.csv')
```

Saves to CSV so the 428 API calls don't need to be repeated.

---

## API Authentication Methods

| Auth Method | How it works | Example |
|---|---|---|
| API Key (query param) | `?api_key=KEY` | TMDB, OpenWeather |
| API Key (header) | `headers={'Authorization': 'Bearer KEY'}` | Stripe |
| OAuth 2.0 | Token exchange flow | Twitter, GitHub |
| Basic Auth | `requests.get(url, auth=('user', 'pass'))` | Some internal APIs |

**Never hardcode API keys.** Use environment variables:

```python
import os
api_key = os.environ['TMDB_API_KEY']
```

---

## Handling API Errors

```python
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
elif response.status_code == 429:
    time.sleep(60)  # Rate limited — back off and retry
elif response.status_code == 401:
    raise ValueError("Invalid API key")
elif response.status_code == 404:
    print("Endpoint not found")
```

Common status codes:
- `200` — OK
- `401` — Unauthorized (bad API key)
- `404` — Not found
- `429` — Rate limit exceeded
- `500` — Server error

---

## Rate Limiting

APIs cap requests per minute/hour. TMDB's free tier allows ~40 requests per 10 seconds. Add delays if hitting limits:

```python
import time

for i in range(1, 429):
    response = requests.get(...)
    ...
    time.sleep(0.25)  # 4 requests/second
```

For production scrapers, implement **exponential backoff**:

```python
import time

def fetch_with_retry(url, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            time.sleep(2 ** attempt)  # 1, 2, 4, 8, 16 seconds
    raise RuntimeError("Max retries exceeded")
```

---

## Output Dataset Schema

| Column | Type | Description |
|---|---|---|
| `id` | int | TMDB movie ID |
| `title` | str | Movie title |
| `overview` | str | Plot summary |
| `release_date` | str | Release date (YYYY-MM-DD) |
| `popularity` | float | TMDB popularity score |
| `vote_average` | float | Average user rating (0–10) |
| `vote_count` | int | Number of votes |

Final shape: **(8551, 7)** — saved to `movies.csv`.

---

## Useful Tools

- **RapidAPI** — marketplace for hundreds of free APIs
- **JSON Viewer** (jsonviewer.stack.hu) — visual tree view for exploring API responses
- **Postman** — GUI for testing API endpoints before writing code
- **TMDB API docs** — https://developers.themoviedb.org/
