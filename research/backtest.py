from __future__ import annotations

import argparse
import io
import json
import math
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


def month_starts(start: str, end: str):
    a = pd.Timestamp(start).to_period("M")
    b = pd.Timestamp(end).to_period("M")
    for p in pd.period_range(a, b, freq="M"):
        yield p.strftime("%Y-%m")


def normalise_ts(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    # Binance public archives switched newer files to microsecond timestamps.
    ms = np.where(x > 100_000_000_000_000, x / 1000.0, x)
    return pd.to_datetime(ms, unit="ms", utc=True, errors="coerce")


def fetch_month(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "prebreakout-research/1.0"})
        with urlopen(req, timeout=45) as r:
            raw = r.read()
    except (HTTPError, URLError, TimeoutError):
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                df = pd.read_csv(fh, header=None, names=COLS)
    except (zipfile.BadZipFile, IndexError, pd.errors.EmptyDataError):
        return None
    return df


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


def add_features(df: pd.DataFrame, btc: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = cfg["trigger"]
    b = int(t["base_hours"])
    imp = int(t["impulse_lookback_hours"])
    exc = int(t["impulse_exclude_recent_hours"])

    x = df.copy()
    x["base_low"] = x.low.shift(1).rolling(b).min()
    x["base_high"] = x.high.shift(1).rolling(b).max()
    x["base_range_pct"] = (x.base_high - x.base_low) / x.base_low
    x["resistance_distance"] = (x.base_high - x.close) / x.base_high

    recent_vol = x.volume.shift(1).rolling(max(24, b // 2)).mean()
    prior_vol = x.volume.shift(1 + b // 2).rolling(max(24, b // 2)).mean()
    x["volume_contraction"] = recent_vol / prior_vol
    x["trigger_volume_ratio"] = x.volume / x.volume.shift(1).rolling(48).mean()
    x["trade_count_ratio"] = x.trades / x.trades.shift(1).rolling(48).mean()
    x["taker_buy_ratio"] = x.taker_buy_base / x.volume.replace(0, np.nan)

    # Prior impulse must be fully visible before the recent consolidation window.
    ref_close = x.close.shift(exc)
    prior_low = x.low.shift(exc).rolling(max(24, imp - exc)).min()
    x["prior_impulse_pct"] = ref_close / prior_low - 1

    btc72 = btc.set_index("time").close.pct_change(72).rename("btc_72h")
    x["asset_72h"] = x.close.pct_change(72)
    x = x.join(btc72, on="time")
    x["rs_72h"] = x.asset_72h - x.btc_72h
    return x


def detect_signals(symbol: str, df: pd.DataFrame, btc: pd.DataFrame, cfg: dict) -> list[dict]:
    if len(df) < 800:
        return []
    x = add_features(df, btc, cfg)
    t = cfg["trigger"]
    cond = (
        (x.prior_impulse_pct >= t["min_prior_impulse_pct"])
        & (x.base_range_pct <= t["max_base_range_pct"])
        & (x.resistance_distance.between(-0.01, t["max_distance_to_resistance_pct"]))
        & (x.volume_contraction <= t["min_volume_contraction_ratio"])
        & (x.trigger_volume_ratio >= t["min_trigger_volume_ratio"])
        & (x.trade_count_ratio >= t["min_trade_count_ratio"])
        & (x.taker_buy_ratio >= t["min_taker_buy_ratio"])
        & (x.rs_72h >= t["min_relative_strength_72h"])
        & (x.close > x.open)
    )
    idxs = np.flatnonzero(cond.fillna(False).to_numpy())
    out, last_i = [], -10**9
    cooldown = int(t["cooldown_hours"])
    for i in idxs:
        if i - last_i < cooldown or i + 1 >= len(x):
            continue
        entry_i = i + 1
        entry = float(x.open.iloc[entry_i]) * (1 + cfg["slippage_rate"])
        structural = float(x.base_low.iloc[i]) * 0.997
        overhead = float(x.high.iloc[max(0, i - 720):i].max())
        if not np.isfinite(structural) or structural >= entry:
            continue
        risk = entry - structural
        dynamic_rr = (overhead - entry) / risk if overhead > entry else np.nan
        out.append({
            "signal_id": f"{symbol}-{x.time.iloc[i].strftime('%Y%m%dT%H%M%SZ')}",
            "symbol": symbol,
            "signal_time": x.time.iloc[i],
            "entry_time": x.time.iloc[entry_i],
            "entry_index": int(entry_i),
            "entry_price": entry,
            "structural_stop": structural,
            "dynamic_target": overhead if np.isfinite(dynamic_rr) and dynamic_rr >= 2 else np.nan,
            "dynamic_rr": dynamic_rr,
            "prior_impulse_pct": float(x.prior_impulse_pct.iloc[i]),
            "base_range_pct": float(x.base_range_pct.iloc[i]),
            "trigger_volume_ratio": float(x.trigger_volume_ratio.iloc[i]),
            "trade_count_ratio": float(x.trade_count_ratio.iloc[i]),
            "taker_buy_ratio": float(x.taker_buy_ratio.iloc[i]),
            "rs_72h": float(x.rs_72h.iloc[i]),
        })
        last_i = i
    return out


def net_pnl(entry: float, exits: list[tuple[float, float]], cfg: dict) -> float:
    notional = float(cfg["position_usdt"])
    fee = float(cfg["fee_rate"])
    qty = notional / entry
    pnl = -notional * fee
    for fraction, raw_price in exits:
        price = raw_price * (1 - cfg["slippage_rate"])
        proceeds = qty * fraction * price
        cost_basis = notional * fraction
        pnl += proceeds - cost_basis - proceeds * fee
    return pnl


def simulate_method(df: pd.DataFrame, sig: dict, method: str, cfg: dict) -> dict:
    start = sig["entry_index"]
    end = min(len(df), start + int(cfg["max_holding_hours"]) + 1)
    entry = sig["entry_price"]
    structural = sig["structural_stop"]
    fixed_target = entry * 1.05
    fixed_stop = entry * 0.99
    dynamic = sig["dynamic_target"]

    if method == "C" and not np.isfinite(dynamic):
        return {"status": "SKIP_RR", "pnl_usdt": 0.0, "return_pct": 0.0, "ambiguous_bars": 0}

    half_taken = False
    ambiguous = 0
    runner_stop = structural
    for j in range(start, end):
        bar = df.iloc[j]
        low, high = float(bar.low), float(bar.high)

        if method == "A":
            stop, target = structural, fixed_target
            hit_s, hit_t = low <= stop, high >= target
            if hit_s and hit_t:
                ambiguous += 1
            if hit_s:  # conservative when order is unknowable inside an hourly candle
                pnl = net_pnl(entry, [(1.0, stop)], cfg)
                return finish(method, sig, bar.time, "STOP", pnl, ambiguous)
            if hit_t:
                pnl = net_pnl(entry, [(1.0, target)], cfg)
                return finish(method, sig, bar.time, "TARGET", pnl, ambiguous)

        elif method == "D":
            hit_s, hit_t = low <= fixed_stop, high >= fixed_target
            if hit_s and hit_t:
                ambiguous += 1
            if hit_s:
                pnl = net_pnl(entry, [(1.0, fixed_stop)], cfg)
                return finish(method, sig, bar.time, "STOP", pnl, ambiguous)
            if hit_t:
                pnl = net_pnl(entry, [(1.0, fixed_target)], cfg)
                return finish(method, sig, bar.time, "TARGET", pnl, ambiguous)

        elif method == "C":
            hit_s, hit_t = low <= structural, high >= dynamic
            if hit_s and hit_t:
                ambiguous += 1
            if hit_s:
                pnl = net_pnl(entry, [(1.0, structural)], cfg)
                return finish(method, sig, bar.time, "STOP", pnl, ambiguous)
            if hit_t:
                pnl = net_pnl(entry, [(1.0, dynamic)], cfg)
                return finish(method, sig, bar.time, "TARGET", pnl, ambiguous)

        elif method == "B":
            if not half_taken:
                hit_s, hit_t = low <= structural, high >= fixed_target
                if hit_s and hit_t:
                    ambiguous += 1
                if hit_s:
                    pnl = net_pnl(entry, [(1.0, structural)], cfg)
                    return finish(method, sig, bar.time, "STOP", pnl, ambiguous)
                if hit_t:
                    half_taken = True
                    # Conservative: do not allow the same candle to stop the runner.
                    runner_stop = entry
                    continue
            else:
                if low <= runner_stop:
                    pnl = net_pnl(entry, [(0.5, fixed_target), (0.5, runner_stop)], cfg)
                    return finish(method, sig, bar.time, "RUNNER_STOP", pnl, ambiguous)
                # Trail from completed bars only, never from the current bar.
                left = max(start, j - 48)
                if j > left:
                    support = float(df.low.iloc[left:j].min()) * 0.997
                    runner_stop = max(entry, runner_stop, support)

    last = df.iloc[end - 1]
    if method == "B" and half_taken:
        pnl = net_pnl(entry, [(0.5, fixed_target), (0.5, float(last.close))], cfg)
        reason = "TIME_RUNNER"
    else:
        pnl = net_pnl(entry, [(1.0, float(last.close))], cfg)
        reason = "TIME"
    return finish(method, sig, last.time, reason, pnl, ambiguous)


def finish(method: str, sig: dict, exit_time, reason: str, pnl: float, ambiguous: int) -> dict:
    return {
        "signal_id": sig["signal_id"], "symbol": sig["symbol"], "method": method,
        "entry_time": sig["entry_time"], "exit_time": exit_time, "exit_reason": reason,
        "pnl_usdt": pnl, "return_pct": pnl / 300.0, "ambiguous_bars": ambiguous, "status": "TRADE",
    }


def summarise(trades: pd.DataFrame, starting_equity: float = 1750.0) -> pd.DataFrame:
    rows = []
    for method, g in trades[trades.status == "TRADE"].sort_values("exit_time").groupby("method"):
        pnl = g.pnl_usdt.to_numpy(float)
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        equity = starting_equity + np.cumsum(pnl)
        peaks = np.maximum.accumulate(np.r_[starting_equity, equity])[:-1]
        dd = (equity - peaks) / peaks
        streak = cur = 0
        for v in pnl:
            cur = cur + 1 if v < 0 else 0
            streak = max(streak, cur)
        rows.append({
            "method": method, "trades": len(g), "net_pnl_usdt": pnl.sum(),
            "expectancy_usdt": pnl.mean(), "median_pnl_usdt": np.median(pnl),
            "win_rate": (pnl > 0).mean(), "loss_rate": (pnl < 0).mean(),
            "avg_loss_usdt": losses.mean() if len(losses) else 0,
            "worst_loss_usdt": losses.min() if len(losses) else 0,
            "max_drawdown_pct": dd.min() if len(dd) else 0,
            "profit_factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() else math.inf,
            "max_losing_streak": streak,
            "ambiguous_bars": int(g.ambiguous_bars.sum()),
        })
    return pd.DataFrame(rows).sort_values(["max_drawdown_pct", "net_pnl_usdt"], ascending=[False, False])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="research/config.json")
    ap.add_argument("--out", default="research/results")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    if btc.empty:
        raise SystemExit("Unable to load BTCUSDT archive data")

    signals, trades = [], []
    for symbol in cfg["symbols"]:
        print(f"Processing {symbol}", flush=True)
        df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
        if df.empty:
            continue
        ss = detect_signals(symbol, df, btc, cfg)
        signals.extend(ss)
        for sig in ss:
            for method in "ABCD":
                trades.append(simulate_method(df, sig, method, cfg))

    s = pd.DataFrame(signals)
    t = pd.DataFrame(trades)
    s.to_csv(out / "historical_signals.csv", index=False)
    t.to_csv(out / "historical_trades.csv", index=False)
    if not t.empty:
        summary = summarise(t)
        summary.to_csv(out / "backtest_summary.csv", index=False)
        print(summary.to_string(index=False))
    else:
        pd.DataFrame().to_csv(out / "backtest_summary.csv", index=False)
        print("No qualifying signals found")


if __name__ == "__main__":
    main()
