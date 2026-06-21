# Day 18 — Pandas DataFrame Using Web Scraping

## Overview

Web scraping extracts data from websites programmatically. When no API is available, it is often the only way to collect the data. This notebook demonstrates scraping company data from AmbitionBox using **BeautifulSoup** and building a structured Pandas DataFrame.

---

## Libraries

```python
import requests               # HTTP requests
from bs4 import BeautifulSoup # HTML parsing
import pandas as pd           # DataFrame construction
import numpy as np            # NaN for missing values
```

- `requests` — sends HTTP GET requests and retrieves raw HTML
- `BeautifulSoup` with `lxml` backend — parses HTML into a navigable tree
- `lxml` must be installed separately: `pip install lxml`

---

## How Web Scraping Works

```
URL → requests.get() → raw HTML → BeautifulSoup → DOM navigation → extracted data → DataFrame
```

---

## Step 1 — Fetch the Page

```python
webpage = requests.get('https://www.ambitionbox.com/list-of-companies?page=1').text
soup = BeautifulSoup(webpage, 'lxml')
```

`requests.get(url).text` returns the raw HTML as a string. `BeautifulSoup(html, 'lxml')` parses it into a tree.

### Handling 403 Access Denied

Many sites block Python's default user agent:

```python
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/80.0.3987.162 Safari/537.36'
}
webpage = requests.get(url, headers=headers).text
```

The `User-Agent` header makes the request look like a real browser.

---

## Step 2 — Inspect the HTML Structure

```python
print(soup.prettify())        # View formatted HTML tree
soup.find_all('h1')           # Find all <h1> elements
soup.find_all('h2')           # Find all <h2> elements
soup.find_all('p', class_='rating')        # <p class="rating">
soup.find_all('a', class_='review-count') # <a class="review-count">
```

Before writing scraping code, always inspect the page's HTML in browser DevTools (F12 → Elements) to identify the CSS classes and tag structure for the data you want.

---

## Step 3 — Extract Data from Elements

```python
company = soup.find_all('div', class_='company-content-wrapper')

for i in company:
    name.append(i.find('h2').text.strip())
    rating.append(i.find('p', class_='rating').text.strip())
    reviews.append(i.find('a', class_='review-count').text.strip())
    ctype.append(i.find_all('p', class_='infoEntity')[0].text.strip())
    hq.append(i.find_all('p', class_='infoEntity')[1].text.strip())
    how_old.append(i.find_all('p', class_='infoEntity')[2].text.strip())
    no_of_employee.append(i.find_all('p', class_='infoEntity')[3].text.strip())
```

Each `company-content-wrapper` div is one company card:
- `h2` → company name
- `p.rating` → rating
- `a.review-count` → number of reviews
- `p.infoEntity[0..3]` → type, HQ, age, employee count

`.text` extracts the visible text. `.strip()` removes whitespace.

---

## Step 4 — Handle Missing Fields with try/except

Real websites have inconsistent HTML — some cards may omit fields. Wrap each extraction:

```python
for i in company:
    try:
        name.append(i.find('h2').text.strip())
    except:
        name.append(np.nan)

    try:
        rating.append(i.find('p', class_='rating').text.strip())
    except:
        rating.append(np.nan)
    # ... etc.
```

`np.nan` keeps all lists the same length so the DataFrame can be built without errors.

---

## Step 5 — Multi-Page Scraping (1,000 Pages)

```python
final = pd.DataFrame()

for j in range(1, 1001):
    webpage = requests.get(
        f'https://www.ambitionbox.com/list-of-companies?page={j}'
    ).text
    soup = BeautifulSoup(webpage, 'lxml')
    company = soup.find_all('div', class_='company-content-wrapper')

    # ... extract with try/except ...

    df = pd.DataFrame({
        'name': name, 'rating': rating, 'reviews': reviews,
        'company_type': ctype, 'Head_Quarters': hq,
        'Company_Age': how_old, 'No_of_Employee': no_of_employee,
    })

    final = pd.concat([final, df], ignore_index=True)

# final.shape → (30000, 7)
```

Each page yields ~30 companies. 1,000 pages → ~30,000 companies.

**Note**: Use `pd.concat([final, df])` — not `final.append(df)`. The `.append()` method was removed in Pandas 2.0.

---

## Output DataFrame Sample

```
                       name  rating  company_type  Head_Quarters  Company_Age    No_of_Employee
Bank of America         4.3     MNC  Charlotte, NC  22 years old  10000+ employees
R.R. Donnelley          4.1     MNC    Chicago, IL 156 years old  10000+ employees
Itc Infotech India      3.4     ...          ...         ...              ...
```

---

## BeautifulSoup Quick Reference

| Method | Returns | Use case |
|---|---|---|
| `soup.find(tag)` | First matching element | When only one exists |
| `soup.find_all(tag)` | List of all matches | For multiple elements |
| `soup.find(tag, class_='x')` | First element with CSS class | Targeting styled elements |
| `element.text` | Inner text content | Extracting visible text |
| `element.get('href')` | Attribute value | Extracting links, src |
| `element.find_all('p')[0]` | Indexed sub-element | Positional extraction |

---

## Legal and Ethical Considerations

- **Check `robots.txt`** — `https://site.com/robots.txt` lists paths that bots should not scrape.
- **Rate limit yourself** — add `time.sleep(0.5)` between requests to avoid overloading servers.
- **Terms of Service** — some sites explicitly prohibit scraping.
- **Use APIs first** — if the site offers an API, always prefer it.

---

## Common Issues

| Issue | Cause | Fix |
|---|---|---|
| `Access Denied` / 403 | Missing User-Agent | Add browser User-Agent header |
| `AttributeError: 'NoneType'` | Element not found | Use `try/except`, append `np.nan` |
| Empty results, correct HTML | JavaScript-rendered content | Use `selenium` or `playwright` |
| IP blocked | Too many rapid requests | Add `time.sleep()` between requests |
| `AttributeError: DataFrame has no attribute 'append'` | Pandas 2.0+ | Use `pd.concat()` |
