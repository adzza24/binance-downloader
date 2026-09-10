from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SCHEMA_VERSION = "2"
HOURLY_KEEP = 800
DAILY_KEEP = 240
CACHE_PATH = Path(os.environ.get("LIVE_MARKET_CACHE", "live/cache/market.sqlite"))
OUT_DIR = Path(os.environ.get("LIVE_MARKET_OUTPUT", "live/output"))
EXCLUSIONS_PATH = Path(os.environ.get("LIVE_MARKET_EXCLUSIONS", "live/excluded_symbols.json"))

BASE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api-gcp.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
    "https://api4.binance.com",
]

# Objective universe exclusions only. Liquidity and other strategy judgements remain
# inputs for the live Strategy Specification / Watch, not rules in this pipeline.
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "USDS", "USD1",
    "EUR", "EURI", "AEUR", "TRY", "BRL", "GBP", "AUD", "BIDR", "IDRT",
}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

# Explicitly identified tokenised-equity instruments. Keep this list explicit rather
# than relying on a suffix heuristic, which could exclude unrelated crypto assets.
TOKENISED_EQUITY_BASES = {
    "AAOIB", "AAPLB", "DELLB", "MRNAB", "MRVLB", "MSFTB", "MSTRB", "MUUB", "WDCB",
}


def load_account_exclusions() -> set[str]:
    if not EXCLUSIONS_PATH.exists():
        return set()
    data = json.loads(EXCLUSIONS_PATH.read_text())
    values = data.get("account_unavailable_symbols", [])
    return {str(x).upper().strip() for x in values if str(x).strip()}


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def expected_open(interval: str, now_ms: int) -> int:
    dt = pd.Timestamp(now_ms, unit="ms", tz="UTC")
    if interval == "1h":
        return int((dt.floor("h") - pd.Timedelta(hours=1)).timestamp() * 1000)
    if interval == "1d":
        return int((dt.floor("D") - pd.Timedelta(days=1)).timestamp() * 1000)
    raise ValueError(interval)


def interval_ms(interval: str) -> int:
    return 3_600_000 if interval == "1h" else 86_400_000


def iso_ms(value: int | float | None) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return pd.Timestamp(int(value), unit="ms", tz="UTC").isoformat()


def finite(value):
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def public_get(path: str, params: dict | None = None, attempts: int = 4):
    last_error = None
    for attempt in range(attempts):
        for base in BASE_URLS:
            try:
                r = requests.get(
                    f"{base}{path}", params=params, timeout=20,
                    headers={"User-Agent": "crypto-live-market-snapshot/1.0"},
                )
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (418, 429):
                    retry = float(r.headers.get("Retry-After", "1"))
                    time.sleep(min(max(retry, 1), 30))
                    last_error = RuntimeError(f"{base}{path}: HTTP {r.status_code}")
                    continue
                last_error = RuntimeError(f"{base}{path}: HTTP {r.status_code} {r.text[:160]}")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"Binance request failed for {path}: {last_error}")


def connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_PATH, timeout=60)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS klines (
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER NOT NULL,
            quote_volume REAL,
            trades INTEGER,
            taker_buy_base REAL,
            PRIMARY KEY(symbol, interval, open_time)
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_klines_lookup ON klines(symbol, interval, open_time)")
    db.commit()
    return db


def current_universe() -> list[dict]:
    info = public_get("/api/v3/exchangeInfo")
    account_exclusions = load_account_exclusions()
    out = []
    for s in info.get("symbols", []):
        symbol = s.get("symbol", "")
        base = s.get("baseAsset", "")
        if s.get("quoteAsset") != "USDT":
            continue
        if s.get("status") != "TRADING" or not s.get("isSpotTradingAllowed", False):
            continue
        if symbol in account_exclusions:
            continue
        if base in STABLE_BASES or base.endswith(LEVERAGED_SUFFIXES):
            continue
        if base in TOKENISED_EQUITY_BASES:
            continue
        out.append({
            "symbol": symbol,
            "base_asset": base,
            "quote_asset": s["quoteAsset"],
            "trading_status": s["status"],
        })
    return sorted(out, key=lambda x: x["symbol"])


def ticker_map() -> dict[str, dict]:
    try:
        rows = public_get("/api/v3/ticker/24hr")
        return {r["symbol"]: r for r in rows if isinstance(r, dict) and "symbol" in r}
    except Exception as exc:
        print(f"WARN ticker/24hr unavailable: {exc}", flush=True)
        return {}


def cached_max(db: sqlite3.Connection, symbol: str, interval: str) -> int | None:
    row = db.execute(
        "SELECT MAX(open_time) FROM klines WHERE symbol=? AND interval=?", (symbol, interval)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def fetch_klines(symbol: str, interval: str, start: int | None, end_open: int, limit: int) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    span = interval_ms(interval)
    params["endTime"] = end_open + span - 1
    if start is not None:
        params["startTime"] = start
    return public_get("/api/v3/klines", params=params)


def parse_rows(symbol: str, interval: str, rows: list, now_ms: int) -> list[tuple]:
    out = []
    for r in rows:
        if len(r) < 11:
            continue
        if int(r[6]) >= now_ms:
            continue
        out.append((
            symbol, interval, int(r[0]), float(r[1]), float(r[2]), float(r[3]),
            float(r[4]), float(r[5]), int(r[6]), float(r[7]), int(r[8]), float(r[9]),
        ))
    return out


def update_interval(db_path: Path, symbol: str, interval: str, keep: int, expected: int, now_ms: int) -> tuple[int, str | None]:
    db = sqlite3.connect(db_path, timeout=60)
    try:
        last = cached_max(db, symbol, interval)
        if last is not None and last >= expected:
            count = db.execute(
                "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?", (symbol, interval)
            ).fetchone()[0]
            return count, None

        span = interval_ms(interval)
        if last is None:
            rows = fetch_klines(symbol, interval, None, expected, keep)
            parsed = parse_rows(symbol, interval, rows, now_ms)
        else:
            parsed = []
            start = last + span
            while start <= expected:
                rows = fetch_klines(symbol, interval, start, expected, 1000)
                batch = parse_rows(symbol, interval, rows, now_ms)
                if not batch:
                    break
                parsed.extend(batch)
                newest = max(x[2] for x in batch)
                if newest < start:
                    break
                start = newest + span

        if parsed:
            db.executemany(
                """
                INSERT OR REPLACE INTO klines
                (symbol, interval, open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_base)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, parsed,
            )
        db.execute(
            """
            DELETE FROM klines
            WHERE symbol=? AND interval=? AND open_time NOT IN (
                SELECT open_time FROM klines WHERE symbol=? AND interval=?
                ORDER BY open_time DESC LIMIT ?
            )
            """, (symbol, interval, symbol, interval, keep),
        )
        db.commit()
        count = db.execute(
            "SELECT COUNT(*) FROM klines WHERE symbol=? AND interval=?", (symbol, interval)
        ).fetchone()[0]
        return count, None
    except Exception as exc:
        return 0, f"{interval}: {exc}"
    finally:
        db.close()


def load_frame(db: sqlite3.Connection, symbol: str, interval: str) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT open_time, open, high, low, close, volume, close_time,
               quote_volume, trades, taker_buy_base
        FROM klines WHERE symbol=? AND interval=? ORDER BY open_time
        """, db, params=(symbol, interval),
    )


def calc_snapshot_row(meta: dict, h: pd.DataFrame, d: pd.DataFrame, btc: pd.DataFrame, ticker: dict, error: str | None) -> dict:
    controlled_complete = len(h) >= 800
    coin200_complete = len(d) >= 200
    row = {
        **meta,
        "snapshot_time": datetime.now(timezone.utc).isoformat(),
        "hourly_candles": len(h),
        "daily_candles": len(d),
        "controlled_activity_history_complete": controlled_complete,
        "coin200_history_complete": coin200_complete,
        "history_complete": bool(controlled_complete and coin200_complete),
        "data_error": error or "",
        "latest_hourly_open": iso_ms(h.open_time.iloc[-1]) if len(h) else "",
        "latest_daily_open": iso_ms(d.open_time.iloc[-1]) if len(d) else "",
        "last_price": finite(ticker.get("lastPrice")) if ticker else (finite(h.close.iloc[-1]) if len(h) else None),
        "quote_volume_24h": finite(ticker.get("quoteVolume")) if ticker else None,
        "prior_impulse": None,
        "base_low_120h": None,
        "base_high_120h": None,
        "base_range_120h": None,
        "resistance_distance": None,
        "volume_contraction": None,
        "trigger_volume_ratio": None,
        "trigger_trade_ratio": None,
        "taker_buy_ratio": None,
        "return_72h": None,
        "btc_return_72h": None,
        "btc_relative_strength_72h": None,
        "trigger_open": None,
        "trigger_close": None,
        "completed_daily_close": None,
        "ma_200d": None,
        "ma_200d_20d_ago": None,
    }

    if len(h):
        cur = h.iloc[-1]
        row["trigger_open"] = finite(cur.open)
        row["trigger_close"] = finite(cur.close)
        if len(h) >= 49:
            volbase = h.volume.iloc[-49:-1].mean()
            trbase = h.trades.iloc[-49:-1].mean()
            row["trigger_volume_ratio"] = finite(cur.volume / volbase) if volbase else None
            row["trigger_trade_ratio"] = finite(cur.trades / trbase) if trbase else None
        row["taker_buy_ratio"] = finite(cur.taker_buy_base / cur.volume) if cur.volume else None
        if len(h) >= 73:
            row["return_72h"] = finite(cur.close / h.close.iloc[-73] - 1)
        if len(h) >= 121:
            base = h.iloc[-121:-1]
            lo, hi = base.low.min(), base.high.max()
            row["base_low_120h"] = finite(lo)
            row["base_high_120h"] = finite(hi)
            row["base_range_120h"] = finite(hi / lo - 1) if lo else None
            row["resistance_distance"] = finite((hi - cur.close) / hi) if hi else None
            recent = h.volume.iloc[-61:-1].mean()
            prior = h.volume.iloc[-121:-61].mean()
            row["volume_contraction"] = finite(recent / prior) if prior else None
        if len(h) >= 720:
            ref_close = h.close.iloc[-121]
            prior_low = h.low.iloc[-720:-120].min()
            row["prior_impulse"] = finite(ref_close / prior_low - 1) if prior_low else None

    if len(btc) >= 73 and len(h) >= 73:
        b = btc.set_index("open_time")["close"]
        cur_t = int(h.open_time.iloc[-1])
        old_t = int(h.open_time.iloc[-73])
        if cur_t in b.index and old_t in b.index:
            btc72 = b.loc[cur_t] / b.loc[old_t] - 1
            row["btc_return_72h"] = finite(btc72)
            if row["return_72h"] is not None:
                row["btc_relative_strength_72h"] = finite(row["return_72h"] - btc72)

    if len(d):
        row["completed_daily_close"] = finite(d.close.iloc[-1])
        ma = d.close.rolling(200, min_periods=180).mean()
        if len(d) >= 180:
            row["ma_200d"] = finite(ma.iloc[-1])
        if len(d) >= 200:
            row["ma_200d_20d_ago"] = finite(ma.iloc[-21])
    return row


def main() -> None:
    now_ms = utc_now_ms()
    expected_h = expected_open("1h", now_ms)
    expected_d = expected_open("1d", now_ms)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    db = connect()

    universe = current_universe()
    symbols = [x["symbol"] for x in universe]
    account_exclusions = load_account_exclusions()
    print(
        f"Universe: {len(symbols)} active eligible Binance USDT spot symbols "
        f"({len(account_exclusions)} account exclusions; {len(TOKENISED_EQUITY_BASES)} tokenised-equity exclusions)",
        flush=True,
    )

    if symbols:
        placeholders = ",".join("?" for _ in symbols)
        db.execute(f"DELETE FROM klines WHERE symbol NOT IN ({placeholders})", symbols)
        db.commit()
    db.close()

    failures: dict[str, list[str]] = {}
    work = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for symbol in symbols:
            work.append((symbol, "1h", ex.submit(update_interval, CACHE_PATH, symbol, "1h", HOURLY_KEEP, expected_h, now_ms)))
            work.append((symbol, "1d", ex.submit(update_interval, CACHE_PATH, symbol, "1d", DAILY_KEEP, expected_d, now_ms)))
        for symbol, interval, future in work:
            _, err = future.result()
            if err:
                failures.setdefault(symbol, []).append(err)
                print(f"WARN {symbol} {err}", flush=True)

    tickers = ticker_map()
    db = connect()
    btc = load_frame(db, "BTCUSDT", "1h")
    rows = []
    latest_h = []
    latest_d = []
    for meta in universe:
        symbol = meta["symbol"]
        h = load_frame(db, symbol, "1h")
        d = load_frame(db, symbol, "1d")
        if len(h): latest_h.append(int(h.open_time.iloc[-1]))
        if len(d): latest_d.append(int(d.open_time.iloc[-1]))
        error = "; ".join(failures.get(symbol, [])) or None
        rows.append(calc_snapshot_row(meta, h, d, btc, tickers.get(symbol, {}), error))
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()

    snapshot = pd.DataFrame(rows)
    snapshot.to_csv(OUT_DIR / "current_snapshot.csv", index=False)

    incomplete = int((~snapshot.history_complete).sum()) if len(snapshot) else 0
    controlled_incomplete = int((~snapshot.controlled_activity_history_complete).sum()) if len(snapshot) else 0
    coin200_incomplete = int((~snapshot.coin200_history_complete).sum()) if len(snapshot) else 0
    failed_symbols = sorted(failures)
    pipeline_status = "FAILED" if not len(snapshot) else ("PARTIAL" if failed_symbols else "OK")
    meta = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_status": pipeline_status,
        "expected_latest_completed_1h_open": iso_ms(expected_h),
        "expected_latest_completed_1d_open": iso_ms(expected_d),
        "min_latest_1h_open": iso_ms(min(latest_h)) if latest_h else "",
        "max_latest_1h_open": iso_ms(max(latest_h)) if latest_h else "",
        "min_latest_1d_open": iso_ms(min(latest_d)) if latest_d else "",
        "max_latest_1d_open": iso_ms(max(latest_d)) if latest_d else "",
        "universe_count": len(universe),
        "snapshot_count": len(snapshot),
        "account_exclusion_count": len(account_exclusions),
        "tokenised_equity_exclusion_count": len(TOKENISED_EQUITY_BASES),
        "incomplete_count": incomplete,
        "controlled_activity_incomplete_count": controlled_incomplete,
        "coin200_incomplete_count": coin200_incomplete,
        "failed_count": len(failed_symbols),
        "failed_symbols": failed_symbols,
        "retention": {"1h": HOURLY_KEEP, "1d": DAILY_KEEP},
    }
    (OUT_DIR / "snapshot_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)

    if pipeline_status == "FAILED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
