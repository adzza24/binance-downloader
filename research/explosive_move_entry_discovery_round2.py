from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from binance_data import load_symbol
from early_breakout_round1 import TASK_EXCLUSIONS

OUT = Path("research/results/explosive_move_entry_discovery_round2")
R1 = Path("research/results/explosive_move_discovery_round1")
START = "2019-01-01"
END = "2026-08-31"
MAX_HOURS = 30 * 24
STOP_PCT = 0.05
TARGET_PCT = 0.20
PROGRESS_LEVELS = [0.00, 0.01, 0.02, 0.03, 0.05]
ARCH_FEATURES = [
    "range_14d", "range_30d", "atr7d_pct", "worst_4h_ret_14d",
    "ret_30d", "dist_30d_high", "rebound_30d_low",
]
MODEL_FEATURES = [
    "ret_1h", "ret_2h", "ret_3h", "ret_6h", "ret_12h", "ret_24h",
    "ret_72h", "ret_7d", "ret_14d", "ret_30d",
    "range_6h", "range_12h", "range_24h", "range_72h", "range_7d",
    "range_14d", "range_30d", "atr24_pct", "atr7d_pct", "atr30d_pct",
    "atr24_vs_7d", "atr7d_vs_30d", "dist_7d_high", "dist_30d_high",
    "rebound_6h_low", "rebound_12h_low", "rebound_24h_low", "rebound_7d_low",
    "rebound_30d_low", "worst_1h_ret_7d", "worst_4h_ret_14d",
    "volume_ratio_current", "volume_ratio_3h", "volume_ratio_6h", "volume_ratio_24h",
    "trade_ratio_current", "trade_ratio_3h", "trade_ratio_6h",
    "taker_buy_current", "taker_buy_3h", "taker_buy_6h", "quote_volume_24h_log",
    "rs_24h", "rs_7d", "rs_30d", "close_vs_ema6", "close_vs_ema12",
    "close_vs_ema24", "close_vs_prev3h_high", "close_vs_prev6h_high",
    "close_vs_prev12h_high", "close_vs_prev24h_high", "low24_vs_prev24",
    "green_body_pct", "close_location", "lower_wick_share",
    "hours_since_anchor",
]

# Research-only local exclusions. This does not alter the Early Breakout task or its list.
ROUND2_EXTRA_EXCLUSIONS = {
    "USUALUSDT", "CFGUSDT", "COWUSDT", "WUSDT", "PYTHUSDT", "STRKUSDT",
    "GPSUSDT", "BBUSDT", "KMNOUSDT", "JTOUSDT", "RESOLVUSDT", "TNSRUSDT",
}
STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR",
    "TRY", "BRL", "GBP", "AUD", "UAH", "RUB", "BIDR", "IDRT", "NGN", "ZAR",
    "VAI", "UST", "USTC", "USD1", "RLUSD", "XUSD", "USDE", "USDS", "PYUSD",
    "EURC", "FDUSD", "BFUSD",
}
TOKENISED_EQUITY_BASES = {"MSTRB", "CRCLB", "SPCXB"}
ALL_EXCLUSIONS = set(TASK_EXCLUSIONS) | ROUND2_EXTRA_EXCLUSIONS


def split_name(year: int) -> str:
    if year <= 2022:
        return "DISCOVERY_2019_2022"
    if year <= 2024:
        return "VALIDATION_2023_2024"
    return "HOLDOUT_2025_2026"


def load_frozen_top50() -> pd.DataFrame:
    u = pd.read_csv(R1 / "universe.csv").sort_values("current_liquidity_rank").copy()
    def eligible(sym: str) -> bool:
        if not sym.endswith("USDT"):
            return False
        base = sym[:-4]
        if base in STABLE_BASES or base in TOKENISED_EQUITY_BASES:
            return False
        if sym in ALL_EXCLUSIONS:
            return False
        if base.endswith(("UP", "DOWN", "BULL", "BEAR")):
            return False
        return True
    u = u[u.symbol.map(eligible)].head(50).copy()
    u["round2_rank"] = np.arange(1, len(u) + 1)
    return u


def build_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("time")
    b = btc.copy().set_index("time").close.reindex(x.index)
    c, h, l, o = x.close, x.high, x.low, x.open

    for n, name in [(1,"1h"),(2,"2h"),(3,"3h"),(6,"6h"),(12,"12h"),(24,"24h"),
                    (72,"72h"),(168,"7d"),(336,"14d"),(720,"30d")]:
        x[f"ret_{name}"] = c.pct_change(n)
    for n, name in [(6,"6h"),(12,"12h"),(24,"24h"),(72,"72h"),(168,"7d"),(336,"14d"),(720,"30d")]:
        hi = h.rolling(n).max(); lo = l.rolling(n).min()
        x[f"range_{name}"] = hi / lo - 1

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr24_pct"] = tr.rolling(24).mean() / c
    x["atr7d_pct"] = tr.rolling(168).mean() / c
    x["atr30d_pct"] = tr.rolling(720).mean() / c
    x["atr24_vs_7d"] = x.atr24_pct / x.atr7d_pct
    x["atr7d_vs_30d"] = x.atr7d_pct / x.atr30d_pct

    x["dist_7d_high"] = c / h.rolling(168).max() - 1
    x["dist_30d_high"] = c / h.rolling(720).max() - 1
    for n, name in [(6,"6h"),(12,"12h"),(24,"24h"),(168,"7d"),(720,"30d")]:
        x[f"rebound_{name}_low"] = c / l.rolling(n).min() - 1
    x["worst_1h_ret_7d"] = c.pct_change().rolling(168).min()
    x["worst_4h_ret_14d"] = c.pct_change(4).rolling(336).min()

    base_vol = x.volume.shift(1).rolling(72).mean()
    base_trades = x.trades.shift(1).rolling(72).mean()
    x["volume_ratio_current"] = x.volume / base_vol
    x["volume_ratio_3h"] = x.volume.rolling(3).mean() / base_vol
    x["volume_ratio_6h"] = x.volume.rolling(6).mean() / base_vol
    x["volume_ratio_24h"] = x.volume.rolling(24).mean() / base_vol
    x["trade_ratio_current"] = x.trades / base_trades
    x["trade_ratio_3h"] = x.trades.rolling(3).mean() / base_trades
    x["trade_ratio_6h"] = x.trades.rolling(6).mean() / base_trades
    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker_buy_current"] = taker
    x["taker_buy_3h"] = taker.rolling(3).mean()
    x["taker_buy_6h"] = taker.rolling(6).mean()
    x["quote_volume_24h_log"] = np.log1p(x.quote_volume.rolling(24).sum())

    x["rs_24h"] = x.ret_24h - b.pct_change(24)
    x["rs_7d"] = x.ret_7d - b.pct_change(168)
    x["rs_30d"] = x.ret_30d - b.pct_change(720)

    for span in [6, 12, 24]:
        ema = c.ewm(span=span, adjust=False).mean()
        x[f"close_vs_ema{span}"] = c / ema - 1
    for n in [3, 6, 12, 24]:
        prev_hi = h.shift(1).rolling(n).max()
        x[f"close_vs_prev{n}h_high"] = c / prev_hi - 1

    low24 = l.rolling(24).min(); prev24 = l.shift(24).rolling(24).min()
    x["low24_vs_prev24"] = low24 / prev24 - 1
    x["green_body_pct"] = c / o - 1
    candle_range = (h-l).replace(0, np.nan)
    x["close_location"] = (c-l) / candle_range
    x["lower_wick_share"] = (np.minimum(o,c)-l) / candle_range
    return x.reset_index()


def conservative_anchor_path(df: pd.DataFrame, start_i: int, max_hours: int = MAX_HOURS) -> dict:
    anchor = float(df.close.iloc[start_i])
    end = min(len(df), start_i + max_hours + 1)
    first5 = None; target20 = None; stop = None
    for j in range(start_i + 1, end):
        lo = float(df.low.iloc[j]); hi = float(df.high.iloc[j])
        if lo <= anchor * (1 - STOP_PCT):
            stop = j
            break
        if first5 is None and hi >= anchor * 1.05:
            first5 = j
        if hi >= anchor * 1.20:
            target20 = j
            break
    if target20 is not None:
        label = "WINNER"
    elif first5 is not None and stop is not None:
        label = "HARD_FAIL"
    else:
        label = "OTHER"
    return {"path_label": label, "first5_index": first5, "target20_index": target20, "stop_index": stop}


def first_progress_index(df: pd.DataFrame, start_i: int, pct: float) -> int | None:
    if pct <= 0:
        return start_i
    anchor = float(df.close.iloc[start_i])
    end = min(len(df), start_i + MAX_HOURS + 1)
    for j in range(start_i + 1, end):
        if float(df.low.iloc[j]) <= anchor * (1 - STOP_PCT):
            return None
        if float(df.high.iloc[j]) >= anchor * (1 + pct):
            return j
    return None


def outcome_from_entry(df: pd.DataFrame, entry_i: int, entry: float) -> dict:
    end = min(len(df), entry_i + MAX_HOURS)
    targets = {10: entry*1.10, 20: entry*1.20, 30: entry*1.30, 50: entry*1.50}
    hit = {10: False, 20: False, 30: False, 50: False}
    hit_idx = {10: None, 20: None, 30: None, 50: None}
    stop_idx = None; min_low = entry; max_high = entry; mae_to_20 = np.nan; hit20_idx = None
    for j in range(entry_i, end):
        lo = float(df.low.iloc[j]); hi = float(df.high.iloc[j])
        min_low = min(min_low, lo); max_high = max(max_high, hi)
        # Conservative treatment of same-hour target/stop ambiguity.
        if lo <= entry * (1 - STOP_PCT):
            stop_idx = j
            break
        for p in [10,20,30,50]:
            if not hit[p] and hi >= targets[p]:
                hit[p] = True; hit_idx[p] = j
                if p == 20:
                    hit20_idx = j
                    mae_to_20 = min_low / entry - 1
    return {
        "hit10_before_stop5": int(hit[10]), "hit20_before_stop5": int(hit[20]),
        "hit30_before_stop5": int(hit[30]), "hit50_before_stop5": int(hit[50]),
        "entry_stop_index": stop_idx, "entry_hit20_index": hit20_idx,
        "post_entry_mae_to_20": mae_to_20,
        "post_entry_mfe_30d": max_high / entry - 1,
        "post_entry_mae_30d": min_low / entry - 1,
        "hours_to_10_from_entry": (hit_idx[10]-entry_i) if hit_idx[10] is not None else np.nan,
        "hours_to_20_from_entry": (hit_idx[20]-entry_i) if hit_idx[20] is not None else np.nan,
        "hours_to_30_from_entry": (hit_idx[30]-entry_i) if hit_idx[30] is not None else np.nan,
        "hours_to_50_from_entry": (hit_idx[50]-entry_i) if hit_idx[50] is not None else np.nan,
    }


def net_binary_returns(cfg: dict) -> tuple[float, float]:
    fee = float(cfg["fee_rate"]); slip = float(cfg["slippage_rate"])
    win = 1.20 * (1-slip) * (1-fee) - 1 - fee
    loss = 0.95 * (1-slip) * (1-fee) - 1 - fee
    return float(win), float(loss)


@dataclass
class ArchModel:
    qlo: pd.Series
    qhi: pd.Series
    imputer: SimpleImputer
    scaler: StandardScaler
    kmeans: KMeans
    cluster_to_name: dict[int, str]
    radius_max: dict[str, float]


def _clip_arch(frame: pd.DataFrame, qlo: pd.Series, qhi: pd.Series) -> pd.DataFrame:
    z = frame[ARCH_FEATURES].copy()
