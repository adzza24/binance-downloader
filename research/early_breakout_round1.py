from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from binance_data import load_symbol
from round3a_exit_architecture import capped_stop, net_pnl

OUT = Path("research/results/early_breakout_round1")
REFERENCE_SYMBOLS = ["SYNUSDT", "LSKUSDT", "GUSDT", "CELRUSDT", "ONEUSDT"]
TASK_EXCLUSIONS = {
    "PYPLUSDT", "BEUSDT", "SOMIUSDT", "GMEUSDT", "TUTUSDT",
    "PROVEUSDT", "XPLUSDT", "KAITOUSDT", "METUSDT", "KITEUSDT", "NIGHTUSDT",
    "ENSOUSDT", "WLFIUSDT", "REZUSDT",
}
COOLDOWN_HOURS = 72
MIN_QUOTE_VOL_24H = 500_000.0

DAILY_HORIZONS = list(range(1, 8))
MONTH_HORIZONS = list(range(1, 13))
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]


def q(s: pd.Series, p: float) -> float:
    z = pd.to_numeric(s, errors="coerce").dropna()
    return float(z.quantile(p)) if len(z) else np.nan


def pf(vals) -> float:
    x = np.asarray(vals, float)
    pos = x[x > 0].sum()
    neg = x[x < 0].sum()
    return float(pos / abs(neg)) if neg < 0 else math.inf


def feature_frame(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().set_index("time")
    b = btc.copy().set_index("time")

    for h in (6, 24, 72, 720):
        x[f"ret_{h}h"] = x.close.pct_change(h)
    x["rs_24h"] = x.ret_24h - b.close.pct_change(24).reindex(x.index)
    x["rs_72h"] = x.ret_72h - b.close.pct_change(72).reindex(x.index)

    hi24 = x.high.shift(1).rolling(24).max()
    lo24 = x.low.shift(1).rolling(24).min()
    hi72 = x.high.shift(1).rolling(72).max()
    lo72 = x.low.shift(1).rolling(72).min()
    hi120 = x.high.shift(1).rolling(120).max()
    lo120 = x.low.shift(1).rolling(120).min()
    hi168 = x.high.shift(1).rolling(168).max()

    x["range_24h"] = hi24 / lo24 - 1
    x["range_72h"] = hi72 / lo72 - 1
    x["base_low_120h"] = lo120
    x["base_high_120h"] = hi120
    x["base_range_120h"] = hi120 / lo120 - 1
    x["resistance_168h"] = hi168
    x["dist_resistance_168h"] = x.close / hi168 - 1

    tr = pd.concat([
        x.high - x.low,
        (x.high - x.close.shift()).abs(),
        (x.low - x.close.shift()).abs(),
    ], axis=1).max(axis=1)
    x["atr24_pct"] = tr.rolling(24).mean() / x.close
    x["atr120_pct"] = tr.rolling(120).mean() / x.close

    base_vol = x.volume.shift(1).rolling(72).mean()
    base_trades = x.trades.shift(1).rolling(72).mean()
    x["volume_ratio_current"] = x.volume / base_vol
    x["trade_ratio_current"] = x.trades / base_trades
    x["volume_ratio_6h"] = x.volume.rolling(6).mean() / base_vol
    x["trade_ratio_6h"] = x.trades.rolling(6).mean() / base_trades

    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker_buy_current"] = taker
    x["taker_buy_6h"] = taker.rolling(6).mean()
    x["quote_volume_24h"] = x.quote_volume.rolling(24).sum()

    # Multi-day base context: whether the 120h base followed weakness or an impulse.
    x["prior_30d_return"] = x.close.shift(120) / x.close.shift(720) - 1
    x["distance_from_30d_high"] = x.close / x.high.shift(1).rolling(720).max() - 1

    # Completed 4h bars only: each 4h bar is labelled at the time it becomes fully observable.
    four = x[["open", "high", "low", "close", "volume"]].resample(
        "4h", label="right", closed="left"
    ).agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna()
    four["support_recent"] = four.low.rolling(6).min()
    four["support_prior"] = four.low.shift(6).rolling(6).min()
    four["support_rising_4h"] = four.support_recent >= four.support_prior * 0.98
    four["close_above_4h_mid"] = four.close >= (
        four.low.rolling(12).min() + four.high.rolling(12).max()
    ) / 2
    x["support_rising_4h"] = four.support_rising_4h.reindex(x.index, method="ffill")
    x["close_above_4h_mid"] = four.close_above_4h_mid.reindex(x.index, method="ffill")

    x["green"] = x.close > x.open
    return x.reset_index()


def raw_candidates(ff: pd.DataFrame) -> pd.DataFrame:
    # Research-only v1 operationalisation of the current Early Breakout Market Watch.
    # Deliberately broad: Round 1 characterises forward paths rather than optimising
    # these constants against outcomes.
    compression = (
        (ff.base_range_120h <= 0.18)
        & (
            (ff.range_24h <= ff.range_72h * 0.75)
            | (ff.atr24_pct <= ff.atr120_pct * 0.85)
        )
    )
    support = (
        (ff.base_low_120h > 0)
        & (
            (ff.low.shift(1).rolling(24).min() >= ff.base_low_120h * 1.005)
            | ff.support_rising_4h.fillna(False)
        )
        & ff.close_above_4h_mid.fillna(False)
    )
    relative_strength = (
        (ff.rs_24h >= 0)
        & (ff.rs_72h >= -0.02)
        & ((ff.rs_24h - ff.rs_72h) >= -0.02)
    )
    participation = (
        (ff.volume_ratio_6h >= 1.05)
        & (ff.trade_ratio_6h >= 1.03)
        & (ff.taker_buy_6h >= 0.50)
    )
    liquid = ff.quote_volume_24h >= MIN_QUOTE_VOL_24H
    not_extended = (ff.ret_24h <= 0.12) & (ff.ret_6h <= 0.08)
    history = ff[[
        "base_range_120h","range_24h","range_72h","atr24_pct","atr120_pct",
        "rs_24h","rs_72h","volume_ratio_6h","trade_ratio_6h","taker_buy_6h",
        "resistance_168h","quote_volume_24h"
    ]].notna().all(axis=1)

    common = compression & support & relative_strength & participation & liquid & not_extended & history

    pre = (
        common
        & (ff.dist_resistance_168h >= -0.04)
        & (ff.dist_resistance_168h < 0.005)
        & (ff.volume_ratio_current <= 4.0)
    )
    attempt = (
        common
        & (ff.dist_resistance_168h >= -0.005)
        & (ff.dist_resistance_168h <= 0.05)
        & (ff.volume_ratio_current >= 1.40)
        & (ff.volume_ratio_current <= 5.0)
        & (ff.trade_ratio_current >= 1.20)
        & (ff.taker_buy_current >= 0.53)
        & ff.green
    )

    out = ff[pre | attempt].copy()
    out["stage"] = np.where(attempt.loc[out.index], "BREAKOUT_ATTEMPT", "PRE_BREAKOUT")
    return out


def dedupe_episodes(symbol: str, ff: pd.DataFrame, cand: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if cand.empty:
        return pd.DataFrame()
    cand = cand.sort_values("time")
    rows = []
    last_entry_time = None
    index_by_time = pd.Series(ff.index.to_numpy(), index=ff.time).to_dict()
    for _, r in cand.iterrows():
        t = pd.Timestamp(r.time)
        if last_entry_time is not None and (t - last_entry_time).total_seconds() < COOLDOWN_HOURS * 3600:
            continue
        i = index_by_time.get(t)
        if i is None or i + 1 >= len(ff):
            continue

        entry_bar = ff.iloc[i + 1]
        raw_entry = float(entry_bar.open)
        entry = raw_entry * (1.0 + float(cfg["slippage_rate"]))
        structural = min(float(r.base_low_120h) * 0.995, entry * 0.999)
        base_stop = capped_stop(entry, structural, 0.10)

        within = cand[(cand.time >= t) & (cand.time <= t + pd.Timedelta(hours=72))]
        upgraded = bool((within.stage == "BREAKOUT_ATTEMPT").any()) if r.stage == "PRE_BREAKOUT" else False
        upgrade_hours = np.nan
        if upgraded:
            ut = within.loc[within.stage == "BREAKOUT_ATTEMPT", "time"].iloc[0]
            upgrade_hours = (pd.Timestamp(ut) - t).total_seconds() / 3600

        rows.append({
            "signal_id": f"EB1:{symbol}:{t.isoformat()}",
            "symbol": symbol,
            "signal_time": t,
            "stage": r.stage,
            "entry_time": pd.Timestamp(entry_bar.time),
            "raw_entry_open": raw_entry,
            "entry_price": entry,
            "structural_support": float(r.base_low_120h),
            "round3h_initial_stop": base_stop,
            "round3h_initial_stop_pct": base_stop / entry - 1,
            "upgraded_to_attempt_within_72h": upgraded,
            "hours_to_attempt_upgrade": upgrade_hours,
            "base_range_120h": float(r.base_range_120h),
            "range_24h": float(r.range_24h),
            "range_72h": float(r.range_72h),
            "atr24_pct": float(r.atr24_pct),
            "atr120_pct": float(r.atr120_pct),
            "distance_to_resistance": float(r.dist_resistance_168h),
            "resistance_168h": float(r.resistance_168h),
            "rs_24h": float(r.rs_24h),
            "rs_72h": float(r.rs_72h),
            "volume_ratio_6h": float(r.volume_ratio_6h),
            "trade_ratio_6h": float(r.trade_ratio_6h),
            "volume_ratio_current": float(r.volume_ratio_current),
            "trade_ratio_current": float(r.trade_ratio_current),
            "taker_buy_6h": float(r.taker_buy_6h),
            "taker_buy_current": float(r.taker_buy_current),
            "quote_volume_24h": float(r.quote_volume_24h),
            "ret_24h": float(r.ret_24h),
            "ret_72h": float(r.ret_72h),
            "prior_30d_return": float(r.prior_30d_return) if pd.notna(r.prior_30d_return) else np.nan,
            "distance_from_30d_high": float(r.distance_from_30d_high) if pd.notna(r.distance_from_30d_high) else np.nan,
        })
        last_entry_time = t
    return pd.DataFrame(rows)


def path_metrics(df: pd.DataFrame, sig: pd.Series) -> dict:
    entry_t = pd.Timestamp(sig.entry_time)
    entry = float(sig.entry_price)
    z = df[df.time >= entry_t].copy()
    if z.empty:
        return {}

    result = {}
    for day in DAILY_HORIZONS:
        target = entry_t + pd.Timedelta(days=day)
        w = z[z.time < target]
        complete = bool(len(z) and z.time.iloc[-1] >= target - pd.Timedelta(hours=1))
        result.update(window_metrics(w, entry, f"d{day}", complete))

    for month in MONTH_HORIZONS:
        target = entry_t + pd.DateOffset(months=month)
        w = z[z.time < target]
        complete = bool(len(z) and z.time.iloc[-1] >= target - pd.Timedelta(hours=1))
        result.update(window_metrics(w, entry, f"m{month}", complete))

    year_end = entry_t + pd.DateOffset(years=1)
    wy = z[z.time < year_end]
    year_complete = bool(len(z) and z.time.iloc[-1] >= year_end - pd.Timedelta(hours=1))
    result.update(window_metrics(wy, entry, "y1", year_complete))

    # Exact first-hit times for useful profit milestones.
    for th in THRESHOLDS:
        hits = z[(z.time < year_end) & (z.high >= entry * (1 + th))]
        key = f"hit_{int(th*100)}pct"
        result[key] = bool(len(hits))
        result[f"hours_to_{int(th*100)}pct"] = (
            (pd.Timestamp(hits.iloc[0].time) - entry_t).total_seconds() / 3600
            if len(hits) else np.nan
        )
        if len(hits):
            hit_t = pd.Timestamp(hits.iloc[0].time)
            pre_hit = z[(z.time >= entry_t) & (z.time <= hit_t)]
            result[f"mae_before_{int(th*100)}pct"] = float(pre_hit.low.min() / entry - 1)
        else:
            result[f"mae_before_{int(th*100)}pct"] = np.nan

    # Failure of the original base/support.
    support = float(sig.structural_support)
    lows = z[(z.time < year_end) & (z.low < support)]
    closes = z[(z.time < year_end) & (z.close < support)]
    result["support_low_break_hours"] = (
        (pd.Timestamp(lows.iloc[0].time) - entry_t).total_seconds() / 3600 if len(lows) else np.nan
    )
    result["support_close_break_hours"] = (
        (pd.Timestamp(closes.iloc[0].time) - entry_t).total_seconds() / 3600 if len(closes) else np.nan
    )
    result["full_12m_observed"] = year_complete
    return result


def window_metrics(w: pd.DataFrame, entry: float, prefix: str, complete: bool) -> dict:
    if w.empty:
        return {
            f"{prefix}_observed": False,
            f"{prefix}_mfe_pct": np.nan,
            f"{prefix}_mae_pct": np.nan,
            f"{prefix}_end_return_pct": np.nan,
            f"{prefix}_hours_to_peak": np.nan,
            f"{prefix}_pre_peak_mae_pct": np.nan,
            f"{prefix}_max_peak_drawdown_pct": np.nan,
            f"{prefix}_peak_retained_pct": np.nan,
        }
    highs = w.high.to_numpy(float)
    lows = w.low.to_numpy(float)
    closes = w.close.to_numpy(float)
    peak_i = int(np.argmax(highs))
    peak = highs[peak_i]
    mfe = peak / entry - 1
    mae = lows.min() / entry - 1
    end_ret = closes[-1] / entry - 1
    peak_t = pd.Timestamp(w.iloc[peak_i].time)
    entry_t = pd.Timestamp(w.iloc[0].time)
    pre_peak_mae = lows[:peak_i+1].min() / entry - 1

    running_peak = np.maximum.accumulate(np.maximum(highs, entry))
    dd = lows / running_peak - 1
    max_peak_dd = float(np.nanmin(dd))
    retained = end_ret / mfe if mfe > 0 else np.nan

    return {
        f"{prefix}_observed": bool(complete),
        f"{prefix}_mfe_pct": float(mfe),
        f"{prefix}_mae_pct": float(mae),
        f"{prefix}_end_return_pct": float(end_ret),
        f"{prefix}_hours_to_peak": float((peak_t - entry_t).total_seconds() / 3600),
        f"{prefix}_pre_peak_mae_pct": float(pre_peak_mae),
        f"{prefix}_max_peak_drawdown_pct": max_peak_dd,
        f"{prefix}_peak_retained_pct": float(retained) if np.isfinite(retained) else np.nan,
    }


def simulate_round3h(df: pd.DataFrame, sig: pd.Series, cfg: dict) -> dict | None:
    entry_t = pd.Timestamp(sig.entry_time)
    entry = float(sig.entry_price)
    z = df[df.time >= entry_t].copy()
    if z.empty:
        return None
    # Reference benchmark is only included for signals with a complete 12m path.
    year_end = entry_t + pd.DateOffset(years=1)
    if z.time.iloc[-1] < year_end - pd.Timedelta(hours=1):
        return None

    stop = float(sig.round3h_initial_stop)
    activated = False
    peak = entry
    for _, b in z.iterrows():
        if b.time >= year_end:
            px = float(b.open)
            return finish_round3h(sig, b.time, "ONE_YEAR_MARK", px, net_pnl(entry, [(1.0, px)], cfg))
        low, high, close = float(b.low), float(b.high), float(b.close)
        peak = max(peak, high)

        if not activated:
            if low <= stop:
                return finish_round3h(sig, b.time, "INITIAL_STOP", stop, net_pnl(entry, [(1.0, stop)], cfg))
            if high >= entry * 1.05:
                activated = True
                stop = max(stop, entry)
                continue
        else:
            if low <= stop:
                reason = "GAIN_TRAIL_STOP" if stop > entry * 1.001 else "BREAKEVEN_STOP"
                return finish_round3h(sig, b.time, reason, stop, net_pnl(entry, [(1.0, stop)], cfg))
            if high >= entry * 11.0:
                px = entry * 11.0
                return finish_round3h(sig, b.time, "TAKE_PROFIT_1000", px, net_pnl(entry, [(1.0, px)], cfg))
            pg = peak / entry - 1
            if pg >= 0.30:
                stop = max(stop, entry * 1.10)
                candidate = entry + 0.40 * (peak - entry)
                stop = max(stop, min(candidate, close * 0.999))

    px = float(z.iloc[-1].close)
    return finish_round3h(sig, z.iloc[-1].time, "DATA_END", px, net_pnl(entry, [(1.0, px)], cfg))


def finish_round3h(sig, t, reason, px, pnl):
    return {
        "signal_id": sig.signal_id,
        "symbol": sig.symbol,
        "stage": sig.stage,
        "entry_time": sig.entry_time,
        "exit_time": t,
        "exit_reason": reason,
        "exit_price": px,
        "pnl_usdt": float(pnl),
        "return_on_300_pct": float(pnl / 300.0),
    }


def process_symbol(symbol: str, cfg: dict, btc: pd.DataFrame):
    df = btc.copy() if symbol == "BTCUSDT" else load_symbol(symbol, cfg["interval"], cfg["start"], cfg["end"])
    if len(df) < 900:
        return symbol, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    ff = feature_frame(df, btc)
    cand = raw_candidates(ff)
    sigs = dedupe_episodes(symbol, ff, cand, cfg)
    if sigs.empty:
        return symbol, sigs, pd.DataFrame(), pd.DataFrame()

    paths = []
    exits = []
    for _, sig in sigs.iterrows():
        p = {"signal_id": sig.signal_id, "symbol": symbol, "stage": sig.stage, "entry_time": sig.entry_time}
        p.update(path_metrics(df, sig))
        paths.append(p)
        r = simulate_round3h(df, sig, cfg)
        if r:
            exits.append(r)
    return symbol, sigs, pd.DataFrame(paths), pd.DataFrame(exits)


def horizon_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [(f"d{d}", f"{d}d") for d in DAILY_HORIZONS] + [(f"m{m}", f"{m}m") for m in MONTH_HORIZONS]
    for prefix, label in specs:
        c = f"{prefix}_mfe_pct"
        if c not in all_rows:
            continue
        z = all_rows[all_rows[f"{prefix}_observed"] == True].copy()
        rows.append({
            "horizon": label,
            "observations": int(len(z)),
            "median_mfe_pct": q(z[c], .50),
            "p75_mfe_pct": q(z[c], .75),
            "p90_mfe_pct": q(z[c], .90),
            "median_mae_pct": q(z[f"{prefix}_mae_pct"], .50),
            "p10_mae_pct": q(z[f"{prefix}_mae_pct"], .10),
            "median_end_return_pct": q(z[f"{prefix}_end_return_pct"], .50),
            "median_pre_peak_mae_pct": q(z[f"{prefix}_pre_peak_mae_pct"], .50),
            "p10_pre_peak_mae_pct": q(z[f"{prefix}_pre_peak_mae_pct"], .10),
            "median_peak_retained_pct": q(z[f"{prefix}_peak_retained_pct"], .50),
            "median_max_peak_drawdown_pct": q(z[f"{prefix}_max_peak_drawdown_pct"], .50),
        })
    return pd.DataFrame(rows)


def horizon_hit_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [(f"d{d}", f"{d}d") for d in DAILY_HORIZONS] + [(f"m{m}", f"{m}m") for m in MONTH_HORIZONS]
    for prefix, label in specs:
        obs = all_rows[all_rows[f"{prefix}_observed"] == True]
        for stage_label, z in [("ALL", obs)] + [(st, obs[obs.stage == st]) for st in sorted(obs.stage.dropna().unique())]:
            for th in [0.05, 0.10, 0.20, 0.30, 0.50]:
                rows.append({
                    "horizon": label,
                    "stage": stage_label,
                    "threshold_pct": th,
                    "observations": int(len(z)),
                    "hit_count": int((z[f"{prefix}_mfe_pct"] >= th).sum()),
                    "hit_rate": float((z[f"{prefix}_mfe_pct"] >= th).mean()) if len(z) else np.nan,
                })
    return pd.DataFrame(rows)


def threshold_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    z = all_rows[all_rows.full_12m_observed == True].copy()
    for th in THRESHOLDS:
        pct = int(th * 100)
        hit = z[z[f"hit_{pct}pct"] == True]
        rows.append({
            "threshold_pct": th,
            "complete_12m_signals": int(len(z)),
            "hit_count": int(len(hit)),
            "hit_rate": float(len(hit) / len(z)) if len(z) else np.nan,
            "median_hours_to_hit": q(hit[f"hours_to_{pct}pct"], .50),
            "p75_hours_to_hit": q(hit[f"hours_to_{pct}pct"], .75),
            "median_mae_before_hit_pct": q(hit[f"mae_before_{pct}pct"], .50),
            "p25_mae_before_hit_pct": q(hit[f"mae_before_{pct}pct"], .25),
            "p10_mae_before_hit_pct": q(hit[f"mae_before_{pct}pct"], .10),
        })
    return pd.DataFrame(rows)


def winner_drawdown_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    z = all_rows[all_rows.full_12m_observed == True].copy()
    rows = []
    for th in [0.10, 0.20, 0.30, 0.50, 1.00]:
        w = z[z.y1_mfe_pct >= th]
        rows.append({
            "eventual_mfe_threshold": th,
            "signals": int(len(w)),
            "median_pre_peak_mae_pct": q(w.y1_pre_peak_mae_pct, .50),
            "p25_pre_peak_mae_pct": q(w.y1_pre_peak_mae_pct, .25),
            "p10_pre_peak_mae_pct": q(w.y1_pre_peak_mae_pct, .10),
            "median_hours_to_peak": q(w.y1_hours_to_peak, .50),
            "p75_hours_to_peak": q(w.y1_hours_to_peak, .75),
            "median_max_peak_drawdown_pct": q(w.y1_max_peak_drawdown_pct, .50),
        })
    return pd.DataFrame(rows)


def peak_timing_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    z = all_rows[all_rows.full_12m_observed == True].copy()
    cutoffs = [24, 48, 72, 168, 24*30, 24*90, 24*180, 24*365]
    rows = []
    for h in cutoffs:
        rows.append({
            "within_hours": h,
            "complete_12m_signals": int(len(z)),
            "peak_by_cutoff_count": int((z.y1_hours_to_peak <= h).sum()),
            "peak_by_cutoff_rate": float((z.y1_hours_to_peak <= h).mean()) if len(z) else np.nan,
        })
    return pd.DataFrame(rows)


def baseline_summary(exits: pd.DataFrame) -> pd.DataFrame:
    if exits.empty:
        return pd.DataFrame()
    rows = []
    for label, g in [("ALL", exits)] + [(s, exits[exits.stage == s]) for s in sorted(exits.stage.unique())]:
        vals = g.pnl_usdt.to_numpy(float)
        rows.append({
            "stage": label,
            "trades": int(len(g)),
            "combined_pnl_usdt": float(vals.sum()),
            "profit_factor": pf(vals),
            "win_rate": float((vals > 0).mean()) if len(vals) else np.nan,
            "median_pnl_usdt": float(np.median(vals)) if len(vals) else np.nan,
            "median_return_on_300_pct": float(np.median(vals / 300.0)) if len(vals) else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    cfg = json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    universe = list(dict.fromkeys(cfg["symbols"] + REFERENCE_SYMBOLS))
    universe = [s for s in universe if s not in TASK_EXCLUSIONS]

    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    if len(btc) < 900:
        raise RuntimeError("BTC history unavailable")

    sig_parts, path_parts, exit_parts = [], [], []
    failures = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_symbol, s, cfg, btc): s for s in universe}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                symbol, sigs, paths, exits = fut.result()
                print(symbol, "signals", len(sigs), flush=True)
                if len(sigs): sig_parts.append(sigs)
                if len(paths): path_parts.append(paths)
                if len(exits): exit_parts.append(exits)
            except Exception as e:
                failures.append({"symbol": s, "error": repr(e)})
                print("ERROR", s, repr(e), flush=True)

    signals = pd.concat(sig_parts, ignore_index=True) if sig_parts else pd.DataFrame()
    paths = pd.concat(path_parts, ignore_index=True) if path_parts else pd.DataFrame()
    exits = pd.concat(exit_parts, ignore_index=True) if exit_parts else pd.DataFrame()

    if signals.empty:
        raise RuntimeError("No Early Breakout Round 1 signals produced")

    signals["signal_time"] = pd.to_datetime(signals.signal_time, utc=True)
    signals["entry_time"] = pd.to_datetime(signals.entry_time, utc=True)
    merged = signals.merge(paths, on=["signal_id","symbol","stage","entry_time"], how="left")
    merged = merged.sort_values(["entry_time","symbol"]).reset_index(drop=True)

    merged.to_csv(OUT / "signal_paths.csv", index=False)
    signals.to_csv(OUT / "signals.csv", index=False)
    exits.to_csv(OUT / "round3h_reference_trades.csv", index=False)
    pd.DataFrame(failures).to_csv(OUT / "failures.csv", index=False)

    hs = horizon_summary(merged)
    hh = horizon_hit_summary(merged)
    ts = threshold_summary(merged)
    ws = winner_drawdown_summary(merged)
    ps = peak_timing_summary(merged)
    bs = baseline_summary(exits)

    hs.to_csv(OUT / "horizon_summary.csv", index=False)
    hh.to_csv(OUT / "horizon_hit_rates.csv", index=False)
    ts.to_csv(OUT / "threshold_summary.csv", index=False)
    ws.to_csv(OUT / "winner_pre_peak_drawdown.csv", index=False)
    ps.to_csv(OUT / "peak_timing_summary.csv", index=False)
    bs.to_csv(OUT / "round3h_reference_summary.csv", index=False)

    yearly = signals.assign(year=signals.entry_time.dt.year).groupby(["year","stage"]).size().unstack(fill_value=0)
    yearly.to_csv(OUT / "signals_by_year_stage.csv")
    stage = signals.groupby("stage").agg(
        signals=("signal_id","size"),
        median_base_range=("base_range_120h","median"),
        median_distance_to_resistance=("distance_to_resistance","median"),
        median_rs24=("rs_24h","median"),
        median_volume_ratio_6h=("volume_ratio_6h","median"),
        median_quote_volume_24h=("quote_volume_24h","median"),
    ).reset_index()
    stage.to_csv(OUT / "stage_entry_summary.csv", index=False)

    manifest = {
        "study": "Early Breakout Round 1 - signal cohort and forward-path study",
        "status": "RESEARCH ONLY - separate strategy track; no live task or live strategy edits",
        "task_source": "Early Breakout Market Watch prompt as read 2026-09-25",
        "dataset": {
            "base_universe": cfg["symbols"],
            "added_named_reference_symbols": REFERENCE_SYMBOLS,
            "start": cfg["start"],
            "end": cfg["end"],
            "interval": cfg["interval"],
            "task/account_exclusions": sorted(TASK_EXCLUSIONS),
        },
        "entry_execution": "First qualifying PRE_BREAKOUT or BREAKOUT_ATTEMPT episode; next hourly open plus configured 0.05% slippage; 72h episode cooldown.",
        "entry_operationalisation": {
            "base_compression": "120h base <=18% and either 24h range <=75% of 72h range or 24h ATR <=85% of 120h ATR",
            "support": "24h lows lifted >=0.5% from 120h base low OR completed-4h support not >2% lower than prior 24h support; completed 4h close above 48h midpoint",
            "relative_strength": "24h BTC-relative strength >=0; 72h >=-2%; 24h vs 72h deterioration no worse than -2pp",
            "participation": "6h volume >=1.05x 72h baseline; 6h trade count >=1.03x; 6h taker-buy >=0.50",
            "liquidity": "trailing 24h quote volume >=500k USDT",
            "not_extended": "24h return <=12% and 6h return <=8%",
            "pre_breakout": "within -4% to +0.5% of prior 168h resistance, current volume <=4x baseline",
            "breakout_attempt": "within -0.5% to +5% of resistance plus current volume 1.4-5x, trades >=1.2x, taker-buy >=0.53, green candle",
        },
        "path_outputs": "MFE, MAE, end return, time-to-peak, pre-peak adverse excursion, running-peak drawdown and peak retention at 1-7 days and calendar months 1-12; milestone hit times; support failures.",
        "right_censoring": "Signals without enough future data remain in the cohort but full_12m_observed=false. 12m conditional summaries and Round3H benchmark use only complete 12m paths.",
        "round3h_reference": "Reference only, not breakout-strategy default: structural support stop capped at -10%, +5% breakeven, +30% minimum +10% floor then 60% accumulated-profit giveback (retain 40% of peak gain), +1000% exit. Reference forcibly marks remaining positions at one year for this study.",
        "important": "Round 1 is descriptive. Do not promote any threshold, stop, holding cap, take-profit or trailing rule from this study without a later explicit research round.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("\nCOHORT", len(signals), "complete12m", int(merged.full_12m_observed.sum()), "failures", len(failures))
    print("\nSTAGE ENTRY SUMMARY\n", stage.to_string(index=False))
    print("\nHORIZON SUMMARY\n", hs.to_string(index=False))
    print("\nTHRESHOLD SUMMARY\n", ts.to_string(index=False))
    print("\nWINNER PRE-PEAK DRAWDOWN\n", ws.to_string(index=False))
    print("\nPEAK TIMING\n", ps.to_string(index=False))
    print("\nROUND3H REFERENCE\n", bs.to_string(index=False))


if __name__ == "__main__":
    main()
