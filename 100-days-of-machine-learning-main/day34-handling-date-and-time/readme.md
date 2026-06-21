# Day 34 — Handling Date and Time Features

## Why Date/Time Features Need Special Handling

Datetime columns are stored as strings or datetime objects — neither can be directly used by ML models. But they carry rich information:

- **Order delivery date** → can derive days since order, day of week, month, year
- **Transaction timestamp** → can derive hour of day, weekend flag, time since last transaction
- **Customer registration date** → can derive account age, tenure

The goal is to extract numerical or categorical features from datetime columns.

---

## Datasets Used

```python
orders = pd.read_csv('orders.csv')
messages = pd.read_csv('messages.csv')
```

---

## Step 1: Parse Datetime Strings

```python
# At load time
orders = pd.read_csv('orders.csv', parse_dates=['order_date'])

# Or post-load
orders['order_date'] = pd.to_datetime(orders['order_date'])
```

`parse_dates` or `pd.to_datetime()` converts string columns to `datetime64[ns]` dtype.

---

## Step 2: Extract Components with `.dt` Accessor

Once a column is `datetime64`, the `.dt` accessor unlocks all datetime components:

```python
df['year']        = df['order_date'].dt.year
df['month']       = df['order_date'].dt.month         # 1–12
df['day']         = df['order_date'].dt.day           # 1–31
df['day_of_week'] = df['order_date'].dt.dayofweek     # 0=Monday, 6=Sunday
df['day_name']    = df['order_date'].dt.day_name()    # 'Monday', 'Tuesday'...
df['quarter']     = df['order_date'].dt.quarter       # 1–4
df['week']        = df['order_date'].dt.isocalendar().week
df['hour']        = df['order_date'].dt.hour
df['minute']      = df['order_date'].dt.minute
df['is_weekend']  = df['order_date'].dt.dayofweek >= 5  # True/False
```

---

## Step 3: Compute Time Differences (Duration Features)

```python
# Days between order date and delivery date
df['delivery_days'] = (df['delivery_date'] - df['order_date']).dt.days

# Days since a fixed reference point
reference = pd.Timestamp('2020-01-01')
df['days_since_launch'] = (df['order_date'] - reference).dt.days

# Account age in months
df['account_age_months'] = (
    (df['current_date'] - df['signup_date']) / pd.Timedelta(days=30)
).astype(int)
```

---

## Step 4: Time-Based Flags

```python
df['is_holiday_season'] = df['month'].isin([11, 12])
df['is_business_hours'] = df['hour'].between(9, 17)
df['is_morning_rush']   = df['hour'].between(7, 9)
```

---

## Common `.dt` Properties Reference

| Property | Description | Example |
|----------|-------------|---------|
| `.year` | Year | 2021 |
| `.month` | Month number | 3 |
| `.day` | Day of month | 15 |
| `.dayofweek` | 0=Monday, 6=Sunday | 2 |
| `.dayofyear` | Day 1–365 | 74 |
| `.quarter` | Quarter 1–4 | 1 |
| `.hour` | Hour 0–23 | 14 |
| `.minute` | Minute 0–59 | 30 |
| `.is_month_start` | True on first day | True/False |
| `.is_month_end` | True on last day | True/False |
| `.day_name()` | Day name string | 'Wednesday' |

---

## Cyclical Encoding for Periodic Features

Month, day-of-week, and hour are **cyclical** — December (12) is close to January (1) but encoding them as integers makes December 12× larger than January. Use sine/cosine encoding:

```python
import numpy as np

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
```

This places months/hours on a circle so December and January are numerically close.

---

## Handling Timezones

```python
# Localize to a timezone
df['ts'] = pd.to_datetime(df['timestamp'], utc=True)

# Convert to local timezone
df['ts_local'] = df['ts'].dt.tz_convert('Asia/Kolkata')
```

---

## Practical Tips

- Always parse datetime columns at load time with `parse_dates` — avoids forgetting later.
- Time differences always use `.dt.days`, `.dt.seconds`, `.dt.total_seconds()` to extract scalar values.
- For tree models, extracting year/month/day/hour as integers is sufficient.
- For linear models, use cyclical encoding for periodic features.
- Drop the original raw datetime column after feature extraction — it cannot be used as-is by ML models.
