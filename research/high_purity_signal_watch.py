from __future__ import annotations

import json
import math
import pickle
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "research/high_purity_signal_features.json"
UNIVERSE = ROOT / "research/results/daily_return_harvest/round2_event_discovery/dataset_v1/universe.parquet"
MODEL_DIR = ROOT / "research/results/daily_return_harvest/high_purity_runtime/models"
OUT = ROOT / "research/results/daily_return_harvest/high_purity_runtime/latest_signal_watch.json"
TRIGGER = ROOT / "research/results/daily_return_harvest/high_purity_runtime/watch_trigger.txt"

BINANCE = "https://data-api.binance.vision"
INTERVAL = "1h"
KLINE_LIMIT = 1000
MIN_QUOTE_VOLUME_24H = 1_000_000.0
DELTA_HOURS = [1, 3, 6, 12, 24, 72, 168]

STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR",
    "TRY", "BRL", "GBP", "AUD", "UAH", "RUB", "BIDR", "IDRT", "NGN", "ZAR",
    "VAI", "UST", "USTC", "USD1", "RLUSD", "XUSD", "USDE", "USDS", "PYUSD",
    "EURC", "BFUSD",
}
TOKENISED_EQUITY_BASES = {"MSTRB", "CRCLB", "SPCXB"}

BASE_FEATURES = [
    "ret_1h", "ret_2h", "ret_3h", "ret_6h", "ret_12h", "ret_24h", "ret_48h", "ret_72h", "ret_168h",
    "rs_1h", "rs_3h", "rs_6h", "rs_12h", "rs_24h", "rs_72h", "rs_168h",
    "range_3h", "range_6h", "range_12h", "range_24h", "range_72h", "range_168h",
    "atr6_pct", "atr24_pct", "atr72_pct", "atr168_pct", "atr24_vs_7d", "range24_vs_72h",
    "dist_6h_high", "dist_12h_high", "dist_24h_high", "dist_72h_high", "dist_168h_high",
    "rebound_6h_low", "rebound_12h_low", "rebound_24h_low", "rebound_72h_low", "rebound_168h_low",
    "close_vs_ema6", "close_vs_ema12", "close_vs_ema24", "close_vs_ema72",
    "ema6_slope_3h", "ema24_slope_6h", "ema72_slope_24h",
    "close_vs_vwap24", "close_vs_vwap72",
    "volume_ratio_current", "volume_ratio_3h", "volume_ratio_6h", "volume_ratio_12h", "volume_ratio_24h",
    "trade_ratio_current", "trade_ratio_3h", "trade_ratio_6h", "trade_ratio_12h", "trade_ratio_24h",
    "taker_buy_current", "taker_buy_3h", "taker_buy_6h", "taker_buy_12h", "taker_buy_24h",
    "quote_volume_24h", "quote_volume_7d",
    "green_body_pct", "abs_body_pct", "close_location", "lower_wick_share", "upper_wick_share",
    "btc_ret_1h", "btc_ret_6h", "btc_ret_24h", "btc_ret_72h", "btc_ret_168h",
    "btc_atr24_pct", "btc_vs_ema168",
]
CONTEXT_FEATURES = [
    "xrank_ret_3h", "xrank_ret_6h", "xrank_ret_24h", "xrank_rs_6h", "xrank_rs_24h",
    "xrank_volume_ratio_3h", "xrank_trade_ratio_3h", "xrank_quote_volume_24h",
    "breadth_positive_1h", "breadth_positive_6h", "breadth_positive_24h",
    "median_ret_1h", "median_ret_6h", "median_ret_24h",
]
TRAJECTORY_FEATURES = BASE_FEATURES + CONTEXT_FEATURES

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def eligible_symbol(sym: str) -> bool:
    if not sym.endswith("USDT") or not sym.isascii():
        return False
    base = sym[:-4]
    if base in STABLE_BASES or base in TOKENISED_EQUITY_BASES:
        return False
    return re.search(r"(UP|DOWN|BULL|BEAR)USDT$", sym) is None


def request_json(path: str, params: dict | None = None, attempts: int = 3):
    last = None
    for i in range(attempts):
        try:
            r = requests.get(f"{BINANCE}{path}", params=params, timeout=25)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            if i + 1 < attempts:
                time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Binance request failed {path}: {last}")


def expected_latest_open() -> pd.Timestamp:
    now = pd.Timestamp.now(tz="UTC")
    return now.floor("h") - pd.Timedelta(hours=1)


def fetch_klines(symbol: str, expected: pd.Timestamp) -> pd.DataFrame:
    rows = request_json("/api/v3/klines", {
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": KLINE_LIMIT,
        "endTime": int((expected + pd.Timedelta(hours=1)).timestamp() * 1000) - 1,
    })
    df = pd.DataFrame(rows, columns=KLINE_COLS)
    if df.empty:
        raise RuntimeError(f"No klines for {symbol}")
    df["time"] = pd.to_datetime(pd.to_numeric(df["open_time"]), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df[df["time"] <= expected].drop_duplicates("time").sort_values("time").reset_index(drop=True)
    if df.empty or pd.Timestamp(df.time.iloc[-1]) != expected:
        got = None if df.empty else str(df.time.iloc[-1])
        raise RuntimeError(f"{symbol} latest completed 1h mismatch: expected {expected}, got {got}")
    return df


def build_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("time")
    bsrc = btc.copy().set_index("time")
    b = bsrc.close.reindex(x.index)
    c, h, l, o = x.close, x.high, x.low, x.open

    for n in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        x[f"ret_{n}h"] = c.pct_change(n)
    for n in [1, 3, 6, 12, 24, 72, 168]:
        x[f"rs_{n}h"] = x[f"ret_{n}h"] - b.pct_change(n)
    for n in [3, 6, 12, 24, 72, 168]:
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        x[f"range_{n}h"] = hi / lo - 1
        x[f"dist_{n}h_high"] = c / hi - 1
        x[f"rebound_{n}h_low"] = c / lo - 1

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    x["atr6_pct"] = tr.rolling(6).mean() / c
    x["atr24_pct"] = tr.rolling(24).mean() / c
    x["atr72_pct"] = tr.rolling(72).mean() / c
    x["atr168_pct"] = tr.rolling(168).mean() / c
    x["atr24_vs_7d"] = x.atr24_pct / x.atr168_pct
    x["range24_vs_72h"] = x.range_24h / x.range_72h

    for span in [6, 12, 24, 72]:
        x[f"ema{span}"] = c.ewm(span=span, adjust=False).mean()
        x[f"close_vs_ema{span}"] = c / x[f"ema{span}"] - 1
    x["ema6_slope_3h"] = x.ema6.pct_change(3)
    x["ema24_slope_6h"] = x.ema24.pct_change(6)
    x["ema72_slope_24h"] = x.ema72.pct_change(24)

    x["vwap24"] = x.quote_volume.rolling(24).sum() / x.volume.rolling(24).sum().replace(0, np.nan)
    x["vwap72"] = x.quote_volume.rolling(72).sum() / x.volume.rolling(72).sum().replace(0, np.nan)
    x["close_vs_vwap24"] = c / x.vwap24 - 1
    x["close_vs_vwap72"] = c / x.vwap72 - 1

    vol_base = x.volume.shift(1).rolling(72).mean()
    trade_base = x.trades.shift(1).rolling(72).mean()
    for n, label in [(1, "current"), (3, "3h"), (6, "6h"), (12, "12h"), (24, "24h")]:
        vv = x.volume if n == 1 else x.volume.rolling(n).mean()
        tt = x.trades if n == 1 else x.trades.rolling(n).mean()
        x[f"volume_ratio_{label}"] = vv / vol_base
        x[f"trade_ratio_{label}"] = tt / trade_base

    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker_buy_current"] = taker
    for n in [3, 6, 12, 24]:
        x[f"taker_buy_{n}h"] = taker.rolling(n).mean()

    x["quote_volume_24h"] = x.quote_volume.rolling(24).sum()
    x["quote_volume_7d"] = x.quote_volume.rolling(168).sum()
    x["green_body_pct"] = c / o - 1
    x["abs_body_pct"] = (c-o).abs() / o.replace(0, np.nan)
    cr = (h-l).replace(0, np.nan)
    x["close_location"] = (c-l) / cr
    x["lower_wick_share"] = (np.minimum(o, c)-l) / cr
    x["upper_wick_share"] = (h-np.maximum(o, c)) / cr

    for n in [1, 6, 24, 72, 168]:
        x[f"btc_ret_{n}h"] = b.pct_change(n)
    bt = pd.concat([
        bsrc.high-bsrc.low,
        (bsrc.high-bsrc.close.shift()).abs(),
        (bsrc.low-bsrc.close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["btc_atr24_pct"] = (bt.rolling(24).mean()/bsrc.close).reindex(x.index)
    x["btc_vs_ema168"] = (
        bsrc.close/bsrc.close.ewm(span=168, adjust=False).mean()-1
    ).reindex(x.index)
    return x.reset_index()


def add_cross_section_context(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["time", "symbol"]).reset_index(drop=True)
    for src, dst in [
        ("ret_3h", "xrank_ret_3h"),
        ("ret_6h", "xrank_ret_6h"),
        ("ret_24h", "xrank_ret_24h"),
        ("rs_6h", "xrank_rs_6h"),
        ("rs_24h", "xrank_rs_24h"),
        ("volume_ratio_3h", "xrank_volume_ratio_3h"),
        ("trade_ratio_3h", "xrank_trade_ratio_3h"),
        ("quote_volume_24h", "xrank_quote_volume_24h"),
    ]:
        panel[dst] = panel.groupby("time")[src].rank(pct=True, method="average")
    grp = panel.groupby("time")
    market = pd.DataFrame({
        "breadth_positive_1h": grp.ret_1h.apply(lambda s: float((s > 0).mean())),
        "breadth_positive_6h": grp.ret_6h.apply(lambda s: float((s > 0).mean())),
        "breadth_positive_24h": grp.ret_24h.apply(lambda s: float((s > 0).mean())),
        "median_ret_1h": grp.ret_1h.median(),
        "median_ret_6h": grp.ret_6h.median(),
        "median_ret_24h": grp.ret_24h.median(),
    }).reset_index()
    return panel.merge(market, on="time", how="left")


def build_runtime_row(symbol: str, ff: pd.DataFrame, context: pd.DataFrame, expected: pd.Timestamp) -> dict:
    by_time = ff.set_index("time")
    if expected not in by_time.index:
        raise RuntimeError(f"{symbol} missing expected feature row {expected}")
    ctx = context[context.symbol == symbol].set_index("time")

    row = {}
    for off in [0] + [-h for h in DELTA_HOURS]:
        t = expected + pd.Timedelta(hours=off)
        vals = {}
        if t in by_time.index:
            fr = by_time.loc[t]
            if isinstance(fr, pd.DataFrame):
                fr = fr.iloc[-1]
            for f in BASE_FEATURES:
                vals[f] = float(fr.get(f, np.nan))
        if t in ctx.index:
            cr = ctx.loc[t]
            if isinstance(cr, pd.DataFrame):
                cr = cr.iloc[-1]
            for f in CONTEXT_FEATURES:
                vals[f] = float(cr.get(f, np.nan))
        row[off] = vals

    flat = {}
    now = row[0]
    for f in TRAJECTORY_FEATURES:
        flat[f] = now.get(f, np.nan)
        for dh in DELTA_HOURS:
            cur = now.get(f, np.nan)
            prev = row.get(-dh, {}).get(f, np.nan)
            flat[f"{f}__chg_{dh}h"] = float(cur - prev) if np.isfinite(cur) and np.isfinite(prev) else np.nan
    return flat


def load_models():
    required = [
        MODEL_DIR / "event_hgb_model.pkl",
        MODEL_DIR / "familyB_model.pkl",
        MODEL_DIR / "familyC15_model.pkl",
        MODEL_DIR / "model_manifest.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"Frozen model artifacts missing: {missing}")

    with required[0].open("rb") as f:
        a, a_features = pickle.load(f)
    with required[1].open("rb") as f:
        b, bc_features_b = pickle.load(f)
    with required[2].open("rb") as f:
        c, bc_features_c = pickle.load(f)
    manifest = json.loads(required[3].read_text())
    if bc_features_b != bc_features_c:
        raise RuntimeError("Family B/C feature lists differ")
    return a, b, c, a_features, bc_features_b, manifest


def score_row(row: dict, a, b, c, a_features, bc_features, thresholds):
    xa = pd.DataFrame([[row.get(k, np.nan) for k in a_features]], columns=a_features)
    xb = pd.DataFrame([[row.get(k, np.nan) for k in bc_features]], columns=bc_features)

    score_a = float(a.predict_proba(xa)[0, 1])
    if score_a >= thresholds["A"]:
        return "A", score_a, {"A": score_a}

    score_b = float(b.predict_proba(xb)[0, 1])
    if score_b >= thresholds["B"]:
        return "B", score_b, {"A": score_a, "B": score_b}

    score_c = float(c.predict_proba(xb)[0, 1])
    if score_c >= thresholds["C"]:
        return "C", score_c, {"A": score_a, "B": score_b, "C": score_c}

    return None, None, {"A": score_a, "B": score_b, "C": score_c}


def write_error(exc: Exception):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    trigger_value = TRIGGER.read_text().strip() if TRIGGER.exists() else None
    result = {
        "status": "ERROR",
        "research_only": True,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "trigger_value": trigger_value,
        "error": str(exc),
        "signals": [],
    }
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


def main():
    expected = expected_latest_open()
    trigger_value = TRIGGER.read_text().strip() if TRIGGER.exists() else None
    spec = json.loads(SPEC.read_text())
    a, b, c, a_features, bc_features, manifest = load_models()

    universe = pd.read_parquet(UNIVERSE)
    model_symbols = sorted({str(s) for s in universe["symbol"].tolist() if eligible_symbol(str(s))})
    fetch_symbols = sorted(set(model_symbols) | {"BTCUSDT"})

    exchange = request_json("/api/v3/exchangeInfo")
    trading = {
        s["symbol"] for s in exchange.get("symbols", [])
        if s.get("status") == "TRADING"
    }
    fetch_symbols = sorted(set(fetch_symbols) & trading)
    active_model_symbols = sorted(set(model_symbols) & trading)
    if "BTCUSDT" not in fetch_symbols:
        raise RuntimeError("BTCUSDT unavailable in current exchangeInfo")

    frames = {}
    failures = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(fetch_klines, sym, expected): sym for sym in fetch_symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                frames[sym] = fut.result()
            except Exception as exc:
                failures[sym] = str(exc)

    btc = frames.get("BTCUSDT")
    if btc is None:
        raise RuntimeError(f"BTC data failed: {failures.get('BTCUSDT')}")

    features = {}
    panel_parts = []
    panel_cols = [
        "time", "ret_1h", "ret_3h", "ret_6h", "ret_24h", "rs_6h", "rs_24h",
        "volume_ratio_3h", "trade_ratio_3h", "quote_volume_24h",
    ]
    for sym in active_model_symbols:
        raw = frames.get(sym)
        if raw is None:
            continue
        if len(raw) < 400:
            failures[sym] = f"Insufficient 1h history: {len(raw)} rows"
            continue
        ff = build_features(raw, btc)
        features[sym] = ff
        p = ff[panel_cols].copy()
        p["symbol"] = sym
        panel_parts.append(p)

    minimum_coverage = max(1, math.ceil(len(active_model_symbols) * 0.70))
    if len(features) < minimum_coverage:
        raise RuntimeError(
            f"Insufficient research-universe coverage: {len(features)}/{len(active_model_symbols)} "
            f"symbols with usable data; require at least {minimum_coverage}"
        )

    context = add_cross_section_context(pd.concat(panel_parts, ignore_index=True))
    thresholds = spec["thresholds"]
    signals = []
    evaluated = 0

    diagnostics = [
        "close_vs_ema6", "close_vs_ema12", "ret_12h", "rs_12h",
        "atr24_pct", "atr72_pct", "atr168_pct", "atr24_pct__chg_1h",
        "atr168_pct__chg_1h", "dist_24h_high", "dist_168h_high",
        "range_3h", "range_168h", "btc_ret_24h", "btc_atr24_pct",
        "breadth_positive_24h", "median_ret_6h",
    ]

    for sym, ff in features.items():
        try:
            latest = ff[ff.time == expected]
            if latest.empty:
                failures[sym] = "No expected latest feature row"
                continue
            base = latest.iloc[-1]
            if not np.isfinite(base.get("quote_volume_24h", np.nan)):
                continue
            if float(base.quote_volume_24h) < MIN_QUOTE_VOLUME_24H:
                continue

            row = build_runtime_row(sym, ff, context, expected)
            missing = [k for k in set(a_features + bc_features) if k not in row]
            if missing:
                raise RuntimeError(f"Missing runtime features: {missing[:10]}")
            evaluated += 1
            family, score, family_scores = score_row(
                row, a, b, c, a_features, bc_features, thresholds
            )
            if family is None:
                continue

            signals.append({
                "symbol": sym,
                "family": family,
                "score": score,
                "family_scores": family_scores,
                "decision_time": expected.isoformat(),
                "intended_entry_time": (expected + pd.Timedelta(hours=1)).isoformat(),
                "decision_close": float(base.close),
                "quote_volume_24h": float(base.quote_volume_24h),
                "feature_context": {
                    k: (None if not np.isfinite(row.get(k, np.nan)) else float(row[k]))
                    for k in diagnostics
                },
            })
        except Exception as exc:
            failures[sym] = str(exc)

    signals.sort(key=lambda x: x["score"], reverse=True)
    result = {
        "status": "OK",
        "research_only": True,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "trigger_value": trigger_value,
        "decision_time": expected.isoformat(),
        "model_set": manifest.get("model_set"),
        "model_verification": manifest.get("verification"),
        "thresholds": thresholds,
        "research_universe_symbols": len(model_symbols),
        "currently_trading_research_symbols": len(active_model_symbols),
        "symbols_with_market_data": len(features),
        "mechanically_evaluated": evaluated,
        "failure_count": len(failures),
        "failures": failures,
        "signals": signals,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "trigger_value": trigger_value,
        "decision_time": result["decision_time"],
        "evaluated": evaluated,
        "signals": [(x["symbol"], x["family"], x["score"]) for x in signals],
        "failures": len(failures),
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_error(exc)
