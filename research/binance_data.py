from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]
CACHE_ROOT = Path(os.environ.get("BINANCE_CACHE_DIR", "research/cache/binance"))


def month_starts(start: str, end: str):
    for p in pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M"):
        yield p.strftime("%Y-%m")


def normalise_ts(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    ms = np.where(x > 100_000_000_000_000, x / 1000.0, x)
    return pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")


def fetch_month(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    cache = CACHE_ROOT / symbol / interval / f"{symbol}-{interval}-{month}.zip"
    raw = None
    if cache.exists():
        raw = cache.read_bytes()
    else:
        url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
        try:
            req = Request(url, headers={"User-Agent": "prebreakout-research/2.0"})
            with urlopen(req, timeout=45) as r:
                raw = r.read()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(raw)
        except (HTTPError, URLError, TimeoutError):
            return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                return pd.read_csv(fh, header=None, names=COLS)
    except (zipfile.BadZipFile, IndexError, pd.errors.EmptyDataError):
        if cache.exists():
            cache.unlink(missing_ok=True)
        return None


def load_symbol(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
    frames = []
    for month in month_starts(start, end):
        df = fetch_month(symbol, interval, month)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLS)
    df = pd.concat(frames, ignore_index=True)
    df["time"] = normalise_ts(df["open_time"])
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time")
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return df[(df.time >= lo) & (df.time < hi)].drop_duplicates("time").reset_index(drop=True)
