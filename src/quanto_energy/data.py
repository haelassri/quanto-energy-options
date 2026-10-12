"""Data loaders for daily temperature and day-ahead power prices.

- Temperature: Open-Meteo historical archive (free, no API key).
- Power: ENTSO-E Transparency Platform via the optional `entsoe-py` package
  (free API key: register at transparency.entsoe.eu and request API access).
- Any CSV with a date column and a value column.

Downloaded data is cached under data/raw/ so notebooks re-run offline.
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Representative weather points for the main power bidding zones.
LOCATIONS = {
    "paris": (48.8566, 2.3522),
    "berlin": (52.5200, 13.4050),
    "london": (51.5074, -0.1278),
    "amsterdam": (52.3676, 4.9041),
    "madrid": (40.4168, -3.7038),
}

# ENTSO-E bidding-zone codes understood by entsoe-py.
BIDDING_ZONES = {"france": "FR", "germany": "DE_LU", "netherlands": "NL", "spain": "ES", "belgium": "BE"}

CACHE_DIR = Path(os.environ.get("QUANTO_DATA_DIR", "data/raw"))


def parse_open_meteo(payload: dict) -> pd.Series:
    """Turn an Open-Meteo archive JSON response into a daily temperature Series (deg C)."""
    daily = payload["daily"]
    s = pd.Series(daily["temperature_2m_mean"], index=pd.to_datetime(daily["time"]), name="temperature", dtype=float)
    s.index.name = "date"
    return s


def fetch_temperature(location: str | tuple[float, float], start: str, end: str, cache: bool = True) -> pd.Series:
    """Daily mean 2m temperature from the Open-Meteo archive (ERA5-based)."""
    lat, lon = LOCATIONS[location.lower()] if isinstance(location, str) else location
    tag = location.lower() if isinstance(location, str) else f"{lat:.2f}_{lon:.2f}"
    path = CACHE_DIR / f"temperature_{tag}_{start}_{end}.csv"
    if cache and path.exists():
        return load_csv(path, "date", "temperature")

    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_mean", "timezone": "Europe/Paris",
    })
    with urllib.request.urlopen(f"{OPEN_METEO_URL}?{query}", timeout=60) as resp:
        series = parse_open_meteo(json.load(resp))
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(path)
    return series


def fetch_day_ahead_prices(zone: str, start: str, end: str, api_key: str | None = None, cache: bool = True) -> pd.Series:
    """Daily average day-ahead price (EUR/MWh) for a bidding zone from ENTSO-E."""
    code = BIDDING_ZONES.get(zone.lower(), zone)
    path = CACHE_DIR / f"dayahead_{code}_{start}_{end}.csv"
    if cache and path.exists():
        return load_csv(path, "date", "price")

    try:
        from entsoe import EntsoePandasClient
    except ImportError as exc:
        raise ImportError("pip install entsoe-py (or install this package with the [data] extra)") from exc
    api_key = api_key or os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError("set ENTSOE_API_KEY (free key from transparency.entsoe.eu)")

    client = EntsoePandasClient(api_key=api_key)
    tz = "Europe/Brussels"
    raw = client.query_day_ahead_prices(code, start=pd.Timestamp(start, tz=tz), end=pd.Timestamp(end, tz=tz))
    series = daily_average(raw).rename("price")
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.to_csv(path)
    return series


def daily_average(intraday: pd.Series) -> pd.Series:
    """Average hourly/15-min prices to calendar days (local time), dropping the timezone."""
    s = intraday.copy()
    if s.index.tz is not None:
        s.index = s.index.tz_convert("Europe/Brussels").tz_localize(None)
    out = s.resample("D").mean().dropna()
    out.index.name = "date"
    return out


def load_csv(path, date_col: str = "date", value_col: str | None = None) -> pd.Series:
    """Load a daily series from CSV (first non-date column if value_col is None)."""
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col).sort_index()
    col = value_col or df.columns[0]
    return df[col].astype(float).rename(col)
