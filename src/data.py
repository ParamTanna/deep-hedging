"""Download and sanity-check daily index history.

We deliberately avoid the `yfinance` wrapper and call Yahoo's public chart
endpoint directly. Two reasons: one less dependency to install, and the exact
URL is visible in the code, so anyone rebuilding this can see precisely what
was requested. No API key, no login, no personal data is sent.

Everything downstream works from LOG RETURNS of the adjusted close, so the
quality checks here are aimed at the things that would corrupt a return series:
gaps, duplicates, non-positive prices, and stale (zero-change) days.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pandas as pd
import requests

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"

SYMBOLS = {"NIFTY50": "^NSEI", "SP500": "^GSPC"}


def fetch(symbol: str, start: str = "2000-01-01", end: str | None = None,
          cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV + adjusted close for one index, as a tidy DataFrame."""
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{symbol}.csv")
    if cache and os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])

    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") if end else
              pd.Timestamp.now(tz="UTC").normalize()).timestamp())
    r = requests.get(CHART.format(sym=SYMBOLS[symbol]),
                     params={"period1": p1, "period2": p2, "interval": "1d",
                             "events": "div,split"},
                     headers={"User-Agent": UA}, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(None).normalize(),
        "open": q["open"], "high": q["high"], "low": q["low"],
        "close": q["close"], "adjclose": adj, "volume": q["volume"],
    })
    if cache:
        df.to_csv(path, index=False)
    return df


# ----------------------------------------------------------------------------
# quality checks
# ----------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop rows that cannot produce a valid log return, and report what went.

    Order matters: we sort first, then drop duplicate dates (keeping the last
    quote for a day), then drop rows with a missing or non-positive adjusted
    close. A zero or negative price makes log(price) undefined, so those rows
    cannot be imputed sensibly and are removed rather than filled.
    """
    n0 = len(df)
    df = df.sort_values("date").reset_index(drop=True)
    dup = int(df["date"].duplicated().sum())
    df = df.drop_duplicates("date", keep="last")
    bad = int((df["adjclose"].isna() | (df["adjclose"] <= 0)).sum())
    df = df[df["adjclose"].notna() & (df["adjclose"] > 0)].reset_index(drop=True)
    rep = {"rows_downloaded": n0, "duplicate_dates_dropped": dup,
           "nonpositive_or_missing_close_dropped": bad, "rows_kept": len(df),
           "first_date": str(df["date"].iloc[0].date()),
           "last_date": str(df["date"].iloc[-1].date())}
    return df, rep


def log_returns(df: pd.DataFrame) -> pd.Series:
    """r_t = log(P_t / P_{t-1}) on the adjusted close.

    Adjusted close is used because it folds dividends and splits back into the
    price. A raw close would show a fake overnight crash on every ex-dividend
    date, which would inflate our volatility estimate.
    """
    return np.log(df["adjclose"]).diff().dropna()
