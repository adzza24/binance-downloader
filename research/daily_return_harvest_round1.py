from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from binance_data import load_symbol
from early_breakout_round1 import TASK_EXCLUSIONS

OUT = Path("research/results/daily_return_harvest/round1")
START = "2021-01-01"
END = "2026-09-30"
INTERVAL = "1h"
UNIVERSE_SIZE = 120
MIN_QUOTE_VOL_24H = 1_000_000.0
COOLDOWN_HOURS = 6
MAX_FORWARD_HOURS = 48
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
TARGETS = [0.01, 0.02, 0.03, 0.05, 0.075]
STOPS = [0.01, 0.015, 0.02, 0.03, 0.05]
HORIZONS = [1, 3, 6, 12, 24, 48]

STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR",
    "TRY", "BRL", "GBP", "AUD", "UAH", "RUB", "BIDR", "IDRT", "NGN", "ZAR",
    "VAI", "UST", "USTC", "USD1", "RLUSD", "XUSD", "USDE", "USDS", "PYUSD",
    "EURC", "BFUSD",
}
TOKENISED_EQUITY_BASES = {"MSTRB", "CRCLB", "SPCXB"}
RESEARCH_EXCLUSIONS = {
    "USUALUSDT", "CFGUSDT", "COWUSDT", "WUSDT", "PYTHUSDT", "STRKUSDT",
    "GPSUSDT", "BBUSDT", "KMNOUSDT", "JTOUSDT", "RESOLVUSDT", "TNSRUSDT",
}
ALL_EXCLUSIONS = set(TASK_EXCLUSIONS) | RESEARCH_EXCLUSIONS


def get_json(url: str):
    req = Request(url, headers={"User-Agent": "daily-return-harvest-research/1.0"})
    with urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def current_liquid_universe() -> pd.DataFrame:
    info = get_json("https://data-api.binance.vision/api/v3/exchangeInfo")
    tick = get_json("https://data-api.binance.vision/api/v3/ticker/24hr")
    eligible = {}
    for s in info["symbols"]:
        sym = s.get("symbol", "")
        base = s.get("baseAsset", "")
        if s.get("quoteAsset") != "USDT" or s.get("status") != "TRADING":
            continue
        if not s.get("isSpotTradingAllowed", True):
            continue
        if base in STABLE_BASES or base in TOKENISED_EQUITY_BASES or sym in ALL_EXCLUSIONS:
            continue
        if re.search(r"(UP|DOWN|BULL|BEAR)USDT$", sym):
            continue
        if not sym.isascii():
            continue
        eligible[sym] = base
    vols = {
        x["symbol"]: float(x.get("quoteVolume", 0) or 0)
        for x in tick if x.get("symbol") in eligible
    }
    ranked = sorted(vols.items(), key=lambda kv: kv[1], reverse=True)[:UNIVERSE_SIZE]
    return pd.DataFrame([
        {
            "symbol": sym,
            "current_quote_volume_24h": vol,
            "current_liquidity_rank": i + 1,
            "base_asset": eligible[sym],
        }
        for i, (sym, vol) in enumerate(ranked)
    ])


def split_name(ts: pd.Timestamp) -> str:
    y = int(pd.Timestamp(ts).year)
    if y <= 2023:
        return "DISCOVERY_2021_2023"
    if y <= 2025:
        return "VALIDATION_2024_2025"
    return "HOLDOUT_2026"


def clip01(v):
    return np.clip(v, 0.0, 1.0)


def build_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("time")
    b = btc.copy().set_index("time").close.reindex(x.index)
    c, h, l, o = x.close, x.high, x.low, x.open

    for n in [1, 3, 6, 12, 24, 72, 168]:
        x[f"ret_{n}h"] = c.pct_change(n)
    for n in [6, 24]:
        x[f"rs_{n}h"] = x[f"ret_{n}h"] - b.pct_change(n)

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr24_pct"] = tr.rolling(24).mean() / c
    x["atr168_pct"] = tr.rolling(168).mean() / c
    x["atr24_vs_7d"] = x.atr24_pct / x.atr168_pct

    for span in [6, 12, 24]:
        x[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()
        x[f"close_vs_ema{span}"] = c / x[f"ema{span}"] - 1

    prev12h_high = h.shift(1).rolling(12).max()
    prev24h_high = h.shift(1).rolling(24).max()
    x["drawdown_from_12h_high"] = c / prev12h_high - 1
    x["dist_prev24h_high"] = c / prev24h_high - 1
    x["rebound_6h_low"] = c / l.rolling(6).min() - 1
    x["rebound_12h_low"] = c / l.rolling(12).min() - 1

    vol_base = x.volume.shift(1).rolling(72).mean()
    trade_base = x.trades.shift(1).rolling(72).mean()
    x["volume_ratio_current"] = x.volume / vol_base
    x["volume_ratio_3h"] = x.volume.rolling(3).mean() / vol_base
    x["trade_ratio_current"] = x.trades / trade_base
    x["trade_ratio_3h"] = x.trades.rolling(3).mean() / trade_base
    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker_buy_current"] = taker
    x["taker_buy_3h"] = taker.rolling(3).mean()
    x["quote_volume_24h"] = x.quote_volume.rolling(24).sum()
    x["green_body_pct"] = c / o - 1
    candle_range = (h-l).replace(0, np.nan)
    x["close_location"] = (c-l) / candle_range

    return x.reset_index()


def btc_context(btc: pd.DataFrame) -> pd.DataFrame:
    x = btc.copy().set_index("time")
    c, h, l = x.close, x.high, x.low
    x["btc_ret_6h"] = c.pct_change(6)
    x["btc_ret_24h"] = c.pct_change(24)
    x["btc_ret_168h"] = c.pct_change(168)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["btc_atr24_pct"] = tr.rolling(24).mean() / c
    ema168 = c.ewm(span=168, adjust=False).mean()
    x["btc_vs_ema168"] = c / ema168 - 1

    def regime(row):
        if row.btc_ret_24h <= -0.02 and row.btc_vs_ema168 < 0:
            return "DOWN"
        if row.btc_ret_24h >= 0.02 and row.btc_vs_ema168 > 0:
            return "UP"
        return "MIXED"

    z = x.reset_index()[[
        "time", "btc_ret_6h", "btc_ret_24h", "btc_ret_168h",
        "btc_atr24_pct", "btc_vs_ema168",
    ]]
    z["btc_regime"] = z.apply(regime, axis=1)
    return z


def family_masks(ff: pd.DataFrame) -> dict[str, pd.Series]:
    liquid = ff.quote_volume_24h >= MIN_QUOTE_VOL_24H
    history = ff[[
        "ret_1h", "ret_3h", "ret_6h", "ret_24h", "ret_72h",
        "rs_6h", "rs_24h", "atr24_pct", "atr168_pct",
        "volume_ratio_current", "volume_ratio_3h",
        "trade_ratio_current", "taker_buy_current",
    ]].notna().all(axis=1)

    momentum = (
        liquid & history
        & (ff.ret_3h >= 0.007)
        & (ff.ret_6h >= 0.012)
        & (ff.ret_24h >= 0.020)
        & (ff.rs_6h >= 0.004)
        & (ff.rs_24h >= 0.008)
        & (ff.volume_ratio_3h >= 1.05)
        & (ff.close_vs_ema24 > 0)
        & (ff.ret_6h <= 0.08)
        & (ff.ret_24h <= 0.18)
    )

    pullback = (
        liquid & history
        & (ff.ret_24h >= 0.025)
        & (ff.ret_72h >= 0.030)
        & (ff.rs_24h >= 0.008)
        & ff.drawdown_from_12h_high.between(-0.060, -0.010)
        & (ff.rebound_6h_low >= 0.008)
        & (ff.close_vs_ema6 >= 0)
        & (ff.green_body_pct > 0)
        & (ff.volume_ratio_current >= 0.75)
        & (ff.ret_1h <= 0.04)
    )

    expansion = (
        liquid & history
        & (ff.atr24_vs_7d <= 0.80)
        & ff.dist_prev24h_high.between(-0.025, 0.010)
        & ff.ret_1h.between(0.003, 0.040)
        & (ff.volume_ratio_current >= 1.30)
        & (ff.trade_ratio_current >= 1.15)
        & (ff.taker_buy_current >= 0.51)
        & (ff.rs_6h >= 0)
        & (ff.ret_24h <= 0.12)
    )
    return {"MOMENTUM": momentum, "PULLBACK_RECLAIM": pullback, "FRESH_EXPANSION": expansion}


def family_score(r: pd.Series, family: str) -> float:
    if family == "MOMENTUM":
        parts = [
            clip01((r.ret_6h - 0.012) / 0.05),
            clip01((r.rs_24h - 0.008) / 0.07),
            clip01((r.volume_ratio_3h - 1.05) / 1.25),
            clip01(r.close_vs_ema24 / 0.05),
        ]
    elif family == "PULLBACK_RECLAIM":
        ideal_pullback = clip01(1 - abs(r.drawdown_from_12h_high + 0.03) / 0.03)
        parts = [
            clip01((r.ret_24h - 0.025) / 0.10),
            clip01((r.rs_24h - 0.008) / 0.07),
            ideal_pullback,
            clip01((r.rebound_6h_low - 0.008) / 0.04),
            clip01((r.volume_ratio_current - 0.75) / 1.0),
        ]
    else:
        parts = [
            clip01((0.80 - r.atr24_vs_7d) / 0.35),
            clip01((r.dist_prev24h_high + 0.025) / 0.035),
            clip01((r.volume_ratio_current - 1.30) / 2.0),
            clip01((r.trade_ratio_current - 1.15) / 1.5),
            clip01((r.taker_buy_current - 0.51) / 0.10),
        ]
    return float(np.nanmean(parts))


def net_return(raw_entry: float, exit_price: float) -> float:
    entry_cash = raw_entry * (1 + SLIPPAGE_RATE) * (1 + FEE_RATE)
    exit_cash = exit_price * (1 - SLIPPAGE_RATE) * (1 - FEE_RATE)
    return float(exit_cash / entry_cash - 1)


def forward_path(df: pd.DataFrame, entry_i: int, raw_entry: float) -> dict:
    entry = raw_entry * (1 + SLIPPAGE_RATE)
    end = min(len(df), entry_i + MAX_FORWARD_HOURS)
    if entry_i >= len(df) or end <= entry_i:
        return {}
    z = df.iloc[entry_i:end]
    out = {}
    for hrs in HORIZONS:
        w = df.iloc[entry_i:min(len(df), entry_i + hrs)]
        if len(w) < hrs:
            out[f"h{hrs}_observed"] = False
            out[f"h{hrs}_mfe"] = np.nan
            out[f"h{hrs}_mae"] = np.nan
            out[f"h{hrs}_end_return"] = np.nan
            out[f"h{hrs}_net_end_return"] = np.nan
            continue
        out[f"h{hrs}_observed"] = True
        out[f"h{hrs}_mfe"] = float(w.high.max() / entry - 1)
        out[f"h{hrs}_mae"] = float(w.low.min() / entry - 1)
        out[f"h{hrs}_end_return"] = float(w.close.iloc[-1] / entry - 1)
        out[f"h{hrs}_net_end_return"] = net_return(raw_entry, float(w.close.iloc[-1]))

    target_hits = {t: None for t in TARGETS}
    stop_hits = {s: None for s in STOPS}
    for k, (_, bar) in enumerate(z.iterrows()):
        hi = float(bar.high)
        lo = float(bar.low)
        for t in TARGETS:
            if target_hits[t] is None and hi >= entry * (1 + t):
                target_hits[t] = k
        for s in STOPS:
            if stop_hits[s] is None and lo <= entry * (1 - s):
                stop_hits[s] = k

    for t, k in target_hits.items():
        p = int(round(t * 1000))
        out[f"target_{p}bp_hour"] = np.nan if k is None else int(k)
    for s, k in stop_hits.items():
        p = int(round(s * 1000))
        out[f"stop_{p}bp_hour"] = np.nan if k is None else int(k)
    return out


def dedupe_candidates(symbol: str, df: pd.DataFrame, ff: pd.DataFrame, family: str, mask: pd.Series) -> list[dict]:
    rows = []
    last_i = -10_000
    valid = np.flatnonzero(mask.fillna(False).to_numpy())
    for i in valid:
        if i - last_i < COOLDOWN_HOURS or i + 1 >= len(df):
            continue
        raw_entry = float(df.open.iloc[i + 1])
        if not np.isfinite(raw_entry) or raw_entry <= 0:
            continue
        r = ff.iloc[i]
        path = forward_path(df, i + 1, raw_entry)
        if not path:
            continue
        row = {
            "signal_id": f"DRH1:{family}:{symbol}:{pd.Timestamp(r.time).isoformat()}",
            "symbol": symbol,
            "family": family,
            "decision_time": pd.Timestamp(r.time),
            "entry_time": pd.Timestamp(df.time.iloc[i + 1]),
            "raw_entry_open": raw_entry,
            "entry_price": raw_entry * (1 + SLIPPAGE_RATE),
            "split": split_name(pd.Timestamp(r.time)),
            "family_score": family_score(r, family),
        }
        for f in [
            "ret_1h", "ret_3h", "ret_6h", "ret_12h", "ret_24h", "ret_72h",
            "rs_6h", "rs_24h", "atr24_pct", "atr168_pct", "atr24_vs_7d",
            "close_vs_ema6", "close_vs_ema12", "close_vs_ema24",
            "drawdown_from_12h_high", "dist_prev24h_high",
            "rebound_6h_low", "rebound_12h_low",
            "volume_ratio_current", "volume_ratio_3h",
            "trade_ratio_current", "trade_ratio_3h",
            "taker_buy_current", "taker_buy_3h",
            "quote_volume_24h", "green_body_pct", "close_location",
        ]:
            v = r.get(f, np.nan)
            row[f] = float(v) if pd.notna(v) else np.nan
        row.update(path)
        rows.append(row)
        last_i = i
    return rows


def process_symbol(meta: pd.Series, btc: pd.DataFrame):
    symbol = meta.symbol
    df = load_symbol(symbol, INTERVAL, START, END)
    if len(df) < 240:
        return symbol, [], None, f"insufficient_history:{len(df)}"
    ff = build_features(df, btc)
    masks = family_masks(ff)
    rows = []
    for family, mask in masks.items():
        rows.extend(dedupe_candidates(symbol, df, ff, family, mask))

    liquid = (ff.quote_volume_24h >= MIN_QUOTE_VOL_24H).fillna(False).to_numpy()
    times = pd.to_datetime(ff.time, utc=True).astype("int64").to_numpy()
    breadth = {
        "times": times[liquid],
        "pos6": (ff.ret_6h.to_numpy()[liquid] > 0).astype(np.int16),
        "pos24": (ff.ret_24h.to_numpy()[liquid] > 0).astype(np.int16),
    }
    return symbol, rows, breadth, None


def q(s: pd.Series, p: float) -> float:
    z = pd.to_numeric(s, errors="coerce").dropna()
    return float(z.quantile(p)) if len(z) else np.nan


def pf(vals) -> float:
    x = np.asarray(vals, float)
    pos = x[x > 0].sum()
    neg = x[x < 0].sum()
    return float(pos / abs(neg)) if neg < 0 else math.inf


def add_breadth(signals: pd.DataFrame, breadth_parts: list[dict]) -> pd.DataFrame:
    if signals.empty:
        return signals
    accum = {}
    for b in breadth_parts:
        if not b:
            continue
        for t, p6, p24 in zip(b["times"], b["pos6"], b["pos24"]):
            rec = accum.setdefault(int(t), [0, 0, 0])
            rec[0] += 1
            rec[1] += int(p6)
            rec[2] += int(p24)
    rows = [
        {
            "decision_time_ns": t,
            "breadth_eligible": v[0],
            "breadth_positive_6h": v[1] / v[0] if v[0] else np.nan,
            "breadth_positive_24h": v[2] / v[0] if v[0] else np.nan,
        }
        for t, v in accum.items()
    ]
    br = pd.DataFrame(rows)
    sig = signals.copy()
    sig["decision_time_ns"] = pd.to_datetime(sig.decision_time, utc=True).astype("int64")
    sig = sig.merge(br, on="decision_time_ns", how="left").drop(columns="decision_time_ns")
    return sig


def family_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", "ALL", signals)]
    for split in sorted(signals.split.unique()):
        groups.append((split, "ALL", signals[signals.split == split]))
        for family in sorted(signals.family.unique()):
            groups.append((split, family, signals[(signals.split == split) & (signals.family == family)]))
    for split, family, g in groups:
        if g.empty:
            continue
        dates = pd.to_datetime(g.decision_time, utc=True).dt.date
        cal_days = max(1, (max(dates) - min(dates)).days + 1)
        rows.append({
            "split": split,
            "family": family,
            "signals": len(g),
            "calendar_days_spanned": cal_days,
            "signals_per_calendar_day": len(g) / cal_days,
            "unique_signal_days": dates.nunique(),
            "median_family_score": q(g.family_score, .5),
            "median_24h_mfe": q(g.h24_mfe, .5),
            "p75_24h_mfe": q(g.h24_mfe, .75),
            "median_24h_mae": q(g.h24_mae, .5),
            "p10_24h_mae": q(g.h24_mae, .10),
            "median_24h_net_end_return": q(g.h24_net_end_return, .5),
            "mean_24h_net_end_return": float(g.h24_net_end_return.mean()),
            "positive_24h_net_rate": float((g.h24_net_end_return > 0).mean()),
            "hit_2pct_24h": float((g.h24_mfe >= .02).mean()),
            "hit_3pct_24h": float((g.h24_mfe >= .03).mean()),
            "hit_5pct_24h": float((g.h24_mfe >= .05).mean()),
            "hit_5pct_48h": float((g.h48_mfe >= .05).mean()),
        })
    return pd.DataFrame(rows)


def horizon_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in sorted(signals.split.unique()):
        for family in ["ALL"] + sorted(signals.family.unique()):
            g = signals[signals.split == split]
            if family != "ALL":
                g = g[g.family == family]
            for h in HORIZONS:
                z = g[g[f"h{h}_observed"] == True]
                if z.empty:
                    continue
                rows.append({
                    "split": split,
                    "family": family,
                    "hours": h,
                    "signals": len(z),
                    "median_mfe": q(z[f"h{h}_mfe"], .5),
                    "p75_mfe": q(z[f"h{h}_mfe"], .75),
                    "p90_mfe": q(z[f"h{h}_mfe"], .90),
                    "median_mae": q(z[f"h{h}_mae"], .5),
                    "p10_mae": q(z[f"h{h}_mae"], .10),
                    "median_net_end_return": q(z[f"h{h}_net_end_return"], .5),
                    "mean_net_end_return": float(z[f"h{h}_net_end_return"].mean()),
                    "positive_net_rate": float((z[f"h{h}_net_end_return"] > 0).mean()),
                })
    return pd.DataFrame(rows)


def target_stop_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in sorted(signals.split.unique()):
        for family in ["ALL"] + sorted(signals.family.unique()):
            g = signals[signals.split == split]
            if family != "ALL":
                g = g[g.family == family]
            for target in TARGETS:
                tc = f"target_{int(round(target*1000))}bp_hour"
                for stop in STOPS:
                    sc = f"stop_{int(round(stop*1000))}bp_hour"
                    t = pd.to_numeric(g[tc], errors="coerce")
                    s = pd.to_numeric(g[sc], errors="coerce")
                    success = t.notna() & (s.isna() | (t < s))
                    failed = s.notna() & (t.isna() | (s <= t))
                    unresolved = ~(success | failed)
                    rows.append({
                        "split": split,
                        "family": family,
                        "target_pct": target,
                        "stop_pct": stop,
                        "signals": len(g),
                        "target_before_stop_rate": float(success.mean()) if len(g) else np.nan,
                        "stop_before_or_same_bar_rate": float(failed.mean()) if len(g) else np.nan,
                        "unresolved_48h_rate": float(unresolved.mean()) if len(g) else np.nan,
                        "resolved_win_rate": float(success.sum() / (success.sum() + failed.sum()))
                            if (success.sum() + failed.sum()) else np.nan,
                    })
    return pd.DataFrame(rows)


def market_regime_summary(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in sorted(signals.split.unique()):
        z0 = signals[signals.split == split]
        for reg in sorted(z0.btc_regime.dropna().unique()):
            for breadth_bucket, g in [
                ("ALL", z0[z0.btc_regime == reg]),
                ("BREADTH_LT_35", z0[(z0.btc_regime == reg) & (z0.breadth_positive_24h < .35)]),
                ("BREADTH_35_65", z0[(z0.btc_regime == reg) & z0.breadth_positive_24h.between(.35, .65)]),
                ("BREADTH_GT_65", z0[(z0.btc_regime == reg) & (z0.breadth_positive_24h > .65)]),
            ]:
                if g.empty:
                    continue
                rows.append({
                    "split": split,
                    "btc_regime": reg,
                    "breadth_bucket": breadth_bucket,
                    "signals": len(g),
                    "median_24h_mfe": q(g.h24_mfe, .5),
                    "median_24h_mae": q(g.h24_mae, .5),
                    "median_24h_net_end_return": q(g.h24_net_end_return, .5),
                    "positive_24h_net_rate": float((g.h24_net_end_return > 0).mean()),
                    "hit_3pct_24h": float((g.h24_mfe >= .03).mean()),
                    "hit_5pct_48h": float((g.h48_mfe >= .05).mean()),
                })
    return pd.DataFrame(rows)


def top_candidate_summary(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = signals.sort_values(["decision_time", "family_score"], ascending=[True, False]).copy()
    base["time_rank"] = base.groupby("decision_time").cumcount() + 1
    rows = []
    for split in sorted(base.split.unique()):
        z0 = base[base.split == split]
        for n in [1, 3, 5]:
            z = z0[z0.time_rank <= n]
            if z.empty:
                continue
            dates = pd.to_datetime(z.decision_time, utc=True).dt.date
            rows.append({
                "split": split,
                "top_n_per_hour": n,
                "signals": len(z),
                "unique_hours": z.decision_time.nunique(),
                "unique_days": dates.nunique(),
                "median_24h_mfe": q(z.h24_mfe, .5),
                "median_24h_mae": q(z.h24_mae, .5),
                "median_24h_net_end_return": q(z.h24_net_end_return, .5),
                "positive_24h_net_rate": float((z.h24_net_end_return > 0).mean()),
                "hit_3pct_24h": float((z.h24_mfe >= .03).mean()),
                "hit_5pct_48h": float((z.h48_mfe >= .05).mean()),
            })
    return pd.DataFrame(rows), base


def write_analysis(signals: pd.DataFrame, fs: pd.DataFrame, ts: pd.DataFrame, tops: pd.DataFrame):
    hold = fs[(fs.split == "HOLDOUT_2026") & (fs.family != "ALL")].copy()
    hold = hold.sort_values(["median_24h_net_end_return", "hit_5pct_48h"], ascending=False)
    best = hold.iloc[0] if len(hold) else None

    pair = ts[
        (ts.split == "HOLDOUT_2026")
        & (ts.family != "ALL")
        & (ts.target_pct.isin([.02, .03, .05]))
        & (ts.stop_pct.isin([.015, .02, .03]))
    ].copy()
    pair = pair.sort_values(["resolved_win_rate", "target_before_stop_rate"], ascending=False)
    best_pair = pair.iloc[0] if len(pair) else None

    top1 = tops[(tops.split == "HOLDOUT_2026") & (tops.top_n_per_hour == 1)]
    top1 = top1.iloc[0] if len(top1) else None

    lines = [
        "# Daily Return Harvest - Round 1",
        "",
        "**Status:** RESEARCH ONLY. No live strategy or automation changes.",
        "",
        "Round 1 is a descriptive short-horizon edge study. It does not optimise exits or simulate a 2,000 USDT portfolio yet.",
        "",
        "## What was tested",
        "",
        f"- Current top {UNIVERSE_SIZE} eligible Binance USDT spot pairs by current 24h quote volume, with historical per-signal liquidity >= {MIN_QUOTE_VOL_24H:,.0f} USDT/day.",
        "- Hourly completed-candle decisions; execution proxy is the next hourly open plus 0.05% entry slippage.",
        "- Three fixed, predeclared signal families: MOMENTUM, PULLBACK_RECLAIM and FRESH_EXPANSION.",
        "- Forward path at 1/3/6/12/24/48 hours, including MFE, MAE and net mark-to-market return after 0.10% fees/side and 0.05% slippage/side.",
        "- Target-before-stop diagnostics use conservative hourly ordering: if target and stop are both touched in the same hourly candle, the stop is counted first.",
        "- Splits: 2021-2023 discovery, 2024-2025 validation, 2026 holdout for this research round.",
        "",
        "## Holdout headline",
        "",
    ]
    if best is not None:
        lines += [
            f"- Best family by median 24h net mark-to-market return: **{best.family}**.",
            f"- Holdout signals: {int(best.signals):,}; median 24h net return {best.median_24h_net_end_return:.2%}; positive 24h rate {best.positive_24h_net_rate:.1%}.",
            f"- Median 24h MFE {best.median_24h_mfe:.2%}; median 24h MAE {best.median_24h_mae:.2%}; +5% reached within 48h on {best.hit_5pct_48h:.1%} of signals.",
        ]
    if best_pair is not None:
        lines += [
            f"- Strongest inspected holdout target/stop ordering among the 2/3/5% targets and 1.5/2/3% stops: {best_pair.family}, target {best_pair.target_pct:.1%}, stop {best_pair.stop_pct:.1%}, resolved win rate {best_pair.resolved_win_rate:.1%}.",
        ]
    if top1 is not None:
        lines += [
            f"- Taking only the highest fixed-score candidate each signal hour produced {int(top1.signals):,} holdout observations across {int(top1.unique_days):,} days; median 24h net mark-to-market return {top1.median_24h_net_end_return:.2%}.",
        ]
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- The 5%/day objective is not used to select or tune these rules. It remains a later portfolio-level acceptance test.",
        "- This round is not a trade backtest: overlapping signals are allowed and capital constraints are ignored.",
        "- Current-liquidity universe selection creates survivorship/current-selection bias in older history. Any promising edge should be re-tested with a point-in-time universe before promotion.",
        "- One-hour bars cannot determine intrabar ordering. Round 2 should use 15m path data for shortlisted cohorts before exit optimisation.",
        "- The 2026 holdout is consumed once these results are inspected. Any subsequent tuning must use a new walk-forward/held-out period or nested validation.",
        "",
        "## Next gate",
        "",
        "Advance a family only if validation and holdout both show positive net expectancy, useful target-before-stop asymmetry, adequate signal frequency, and no obvious dependence on a single BTC/breadth regime. Round 2 should then validate execution paths and exits; portfolio simulation comes after that.",
    ]
    (OUT / "ANALYSIS.md").write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    universe = current_liquid_universe()
    universe.to_csv(OUT / "universe.csv", index=False)
    if universe.empty:
        raise RuntimeError("No eligible universe returned")

    btc = load_symbol("BTCUSDT", INTERVAL, START, END)
    if len(btc) < 240:
        raise RuntimeError("BTC history unavailable")
    bctx = btc_context(btc)

    all_rows = []
    breadth_parts = []
    failures = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_symbol, r, btc): r.symbol for _, r in universe.iterrows()}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                symbol, rows, breadth, err = fut.result()
                if rows:
                    all_rows.extend(rows)
                if breadth:
                    breadth_parts.append(breadth)
                if err:
                    failures.append({"symbol": symbol, "error": err})
                print(symbol, "signals", len(rows), "error", err, flush=True)
            except Exception as e:
                failures.append({"symbol": sym, "error": repr(e)})
                print("ERROR", sym, repr(e), flush=True)

    signals = pd.DataFrame(all_rows)
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)
    if signals.empty:
        raise RuntimeError("No Daily Return Harvest Round 1 signals produced")

    signals["decision_time"] = pd.to_datetime(signals.decision_time, utc=True)
    signals["entry_time"] = pd.to_datetime(signals.entry_time, utc=True)
    signals = add_breadth(signals, breadth_parts)
    bctx["time"] = pd.to_datetime(bctx.time, utc=True)
    signals = signals.merge(bctx, left_on="decision_time", right_on="time", how="left").drop(columns="time")
    fs = family_summary(signals)
    hs = horizon_summary(signals)
    ts = target_stop_summary(signals)
    rs = market_regime_summary(signals)
    tops, ranked = top_candidate_summary(signals)
    ranked = ranked.sort_values(["decision_time", "time_rank", "symbol", "family"]).reset_index(drop=True)

    ranked.to_csv(OUT / "signals.csv.gz", index=False, compression="gzip")
    fs.to_csv(OUT / "family_summary.csv", index=False)
    hs.to_csv(OUT / "horizon_summary.csv", index=False)
    ts.to_csv(OUT / "target_stop_summary.csv", index=False)
    rs.to_csv(OUT / "market_regime_summary.csv", index=False)
    tops.to_csv(OUT / "top_candidate_summary.csv", index=False)

    manifest = {
        "study": "Daily Return Harvest Round 1",
        "status": "RESEARCH ONLY - no live strategy or scheduled task changes",
        "objective": "Discover frequent short-horizon long-only Binance spot edges suitable for later portfolio testing against a 5% net geometric daily stretch objective.",
        "period": {"start": START, "end": END, "interval": INTERVAL},
        "splits": {
            "discovery": "2021-2023",
            "validation": "2024-2025",
            "holdout": "2026",
        },
        "universe": {
            "selection": f"Current top {UNIVERSE_SIZE} eligible Binance USDT spot pairs by current 24h quote volume",
            "historical_liquidity_gate_usdt_24h": MIN_QUOTE_VOL_24H,
            "exclusions": sorted(ALL_EXCLUSIONS),
            "stable_bases": sorted(STABLE_BASES),
            "tokenised_equity_bases": sorted(TOKENISED_EQUITY_BASES),
            "known_limitation": "Current-liquidity selection causes survivorship/current-selection bias in older history.",
        },
        "execution_assumptions": {
            "decision": "completed 1h candle",
            "entry": "next 1h open",
            "fee_rate_each_side": FEE_RATE,
            "slippage_rate_each_side": SLIPPAGE_RATE,
            "target_stop_same_hour": "conservatively count stop first",
        },
        "signal_families": {
            "MOMENTUM": "Short-horizon acceleration + BTC-relative strength + improving volume, capped for extension.",
            "PULLBACK_RECLAIM": "Already-strong asset pulls back 1-6% from 12h high, then reclaims short EMA on a green candle.",
            "FRESH_EXPANSION": "Compressed 24h volatility, near prior 24h high, first price/volume/trade/taker expansion without large extension.",
        },
        "forward_horizons_hours": HORIZONS,
        "targets": TARGETS,
        "stops": STOPS,
        "cooldown_hours_per_symbol_family": COOLDOWN_HOURS,
        "important": [
            "Round 1 is descriptive and permits overlapping signals.",
            "Do not infer a live strategy or 5%/day portfolio result from this round.",
            "Shortlisted signals should receive 15m execution-path validation before exit optimisation.",
            "No Crypto Live Strategy Specification or automation is read or modified.",
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_analysis(ranked, fs, ts, tops)

    print("\nFAMILY SUMMARY\n", fs.to_string(index=False))
    print("\nTOP CANDIDATE SUMMARY\n", tops.to_string(index=False))
    hold_ts = ts[
        (ts.split == "HOLDOUT_2026")
        & (ts.target_pct.isin([.02, .03, .05]))
        & (ts.stop_pct.isin([.015, .02, .03]))
    ]
    print("\nHOLDOUT TARGET/STOP SAMPLE\n", hold_ts.to_string(index=False))
    print("\nTOTAL SIGNALS", len(ranked), "FAILURES", len(failures))


if __name__ == "__main__":
    main()
