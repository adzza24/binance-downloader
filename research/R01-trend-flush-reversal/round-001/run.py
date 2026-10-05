from __future__ import annotations

import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
STATE_PATH = ROOT / "checkpoints/state.json"
SYMBOL_DIR = ROOT / "checkpoints/symbols"
RESULTS = ROOT / "results"
REPO_ROOT = ROOT.parents[2]
sys.path.insert(0, str(REPO_ROOT / "research"))
from binance_data import load_symbol  # noqa: E402


def now_iso() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def load_symbols() -> list[str]:
    cfg = json.loads((REPO_ROOT / "research/config.json").read_text())
    return list(dict.fromkeys(map(str, cfg["symbols"])))


def git_checkpoint(message: str) -> None:
    subprocess.run(["git", "config", "user.name", "github-actions[bot]"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "add", str(ROOT.relative_to(REPO_ROOT))], cwd=REPO_ROOT, check=True)
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT)
    if diff.returncode == 0:
        return
    subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=REPO_ROOT, check=True)


def read_state(chash: str) -> dict:
    default = {
        "study_id": "R01", "round_id": "001", "status": "running",
        "config_hash": chash, "completed_symbols": [], "failed_symbols": [], "updated_at": now_iso(),
    }
    if not STATE_PATH.exists():
        return default
    try:
        state = json.loads(STATE_PATH.read_text())
    except json.JSONDecodeError:
        return default
    if state.get("config_hash") != chash:
        return default
    state["status"] = "running"
    return state


def write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = now_iso()
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def aggregate_4h(df: pd.DataFrame) -> pd.DataFrame:
    x = df.set_index("time")
    return x.resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
        taker_buy_base=("taker_buy_base", "sum"),
    ).dropna(subset=["open", "high", "low", "close"]).reset_index()


def add_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    prev_close = x.close.shift(1)
    tr = pd.concat([x.high - x.low, (x.high - prev_close).abs(), (x.low - prev_close).abs()], axis=1).max(axis=1)
    x["atr20"] = tr.shift(1).rolling(20).mean()
    x["range_atr"] = (x.high - x.low) / x.atr20.replace(0, np.nan)
    rng = (x.high - x.low).replace(0, np.nan)
    body = (x.close - x.open).abs()
    lower_wick = np.minimum(x.open, x.close) - x.low
    x["lower_wick_share"] = lower_wick / rng
    x["wick_body_ratio"] = lower_wick / body.replace(0, np.nan)
    x["close_location"] = (x.close - x.low) / rng
    x["drop_to_low_pct"] = x.low / prev_close - 1.0
    x["rebound_low_close_pct"] = x.close / x.low - 1.0
    x["ema120"] = x.close.ewm(span=120, adjust=False).mean()
    x["ema300"] = x.close.ewm(span=300, adjust=False).mean()
    x["ema300_slope60"] = x.ema300.pct_change(60)
    x["trend_score"] = ((x.close > x.ema300).astype(int) + (x.ema120 > x.ema300).astype(int) + (x.ema300_slope60 > 0).astype(int))
    x["extension_ema300"] = x.close / x.ema300 - 1.0
    x["volume_baseline"] = x.volume.shift(1).rolling(30).median()
    x["volume_ratio"] = x.volume / x.volume_baseline.replace(0, np.nan)
    x["quote_volume_24h"] = x.quote_volume.rolling(6).sum()
    x["prior_low_20d"] = x.low.shift(1).rolling(120).min()
    x["prior_low_60d"] = x.low.shift(1).rolling(360).min()
    x["support_dist_20d"] = x.low / x.prior_low_20d - 1.0
    x["support_dist_60d"] = x.low / x.prior_low_60d - 1.0
    x["support_reclaim_20d"] = ((x.low <= x.prior_low_20d * 1.02) & (x.low >= x.prior_low_20d * 0.94) & (x.close >= x.prior_low_20d * 0.995))
    return x


def add_btc_context(btc4: pd.DataFrame) -> pd.DataFrame:
    b = add_4h_features(btc4)
    b["btc_ret_24h"] = b.close.pct_change(6)
    b["btc_ret_72h"] = b.close.pct_change(18)
    b["btc_bear"] = (b.close < b.ema300) & (b.ema300_slope60 < 0)
    b["btc_regime"] = np.where((b.close > b.ema300) & (b.ema300_slope60 > 0), "BULL", np.where(b.btc_bear, "BEAR", "SIDEWAYS"))
    return b[["time", "btc_ret_24h", "btc_ret_72h", "btc_bear", "btc_regime", "range_atr"]].rename(columns={"range_atr": "btc_range_atr"})


def split_name(ts: pd.Timestamp, cfg: dict) -> str:
    t = pd.Timestamp(ts)
    if t < pd.Timestamp(cfg["development_end"], tz="UTC"):
        return "DEVELOPMENT"
    if t < pd.Timestamp(cfg["validation_end"], tz="UTC"):
        return "VALIDATION"
    return "CONFIRMATION"


def find_entry(df1: pd.DataFrame, flush_row: pd.Series, variant: str, max_hours: int):
    flush_end = flush_row.time + pd.Timedelta(hours=4)
    start = int(df1["time"].searchsorted(flush_end, side="left"))
    if start >= len(df1):
        return None
    if variant == "immediate":
        return start, df1.iloc[start].time, float(df1.iloc[start].open)
    flush_low, flush_high = float(flush_row.low), float(flush_row.high)
    threshold = flush_low + (0.50 if variant == "confirm_mid" else 0.60) * (flush_high - flush_low)
    end = min(len(df1) - 1, start + max_hours)
    for j in range(start, end):
        bar = df1.iloc[j]
        if bar.low > flush_low and bar.close >= threshold and (variant == "confirm_mid" or bar.close > bar.open):
            entry_idx = j + 1
            if entry_idx < len(df1):
                return entry_idx, df1.iloc[entry_idx].time, float(df1.iloc[entry_idx].open)
    return None


def simulate(df1: pd.DataFrame, entry_idx: int, entry: float, target: float, stop: float, horizon: int) -> dict:
    tp, sl = entry * (1.0 + target), entry * (1.0 - stop)
    end = min(len(df1), entry_idx + horizon)
    max_hi, min_lo = entry, entry
    for j in range(entry_idx, end):
        bar = df1.iloc[j]
        hi, lo = float(bar.high), float(bar.low)
        max_hi, min_lo = max(max_hi, hi), min(min_lo, lo)
        hit_tp, hit_sl = hi >= tp, lo <= sl
        if hit_tp and hit_sl:
            return {"result": "AMBIGUOUS_STOP", "gross_return": -stop, "hours": j-entry_idx, "mfe": max_hi/entry-1, "mae": min_lo/entry-1}
        if hit_sl:
            return {"result": "STOP", "gross_return": -stop, "hours": j-entry_idx, "mfe": max_hi/entry-1, "mae": min_lo/entry-1}
        if hit_tp:
            return {"result": "TARGET", "gross_return": target, "hours": j-entry_idx, "mfe": max_hi/entry-1, "mae": min_lo/entry-1}
    if end <= entry_idx:
        return {"result": "NO_DATA", "gross_return": np.nan, "hours": np.nan, "mfe": np.nan, "mae": np.nan}
    close = float(df1.iloc[end-1].close)
    return {"result": "TIMEOUT", "gross_return": close/entry-1, "hours": end-entry_idx, "mfe": max_hi/entry-1, "mae": min_lo/entry-1}


def net_return(gross: float, cfg: dict) -> float:
    if not np.isfinite(gross):
        return np.nan
    return gross - 2.0 * (cfg["fee_rate_each_side"] + cfg["slippage_rate_each_side"])


def process_symbol(symbol: str, btc_context: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df1 = load_symbol(symbol, "1h", cfg["data_start"], "2026-10-01")
    if len(df1) < 24 * 90:
        raise RuntimeError(f"insufficient hourly data: {len(df1)} rows")
    df4 = add_4h_features(aggregate_4h(df1)).merge(btc_context, on="time", how="left")
    safe_end = df1["time"].max() - pd.Timedelta(hours=int(cfg["secondary_horizon_hours"]) + 2)
    f = cfg["candidate_floor"]
    eligible = df4[
        (df4.time >= pd.Timestamp(cfg["test_start"], tz="UTC"))
        & (df4.time < pd.Timestamp(cfg["confirmation_end"], tz="UTC"))
        & (df4.time <= safe_end)
        & (df4.lower_wick_share >= f["lower_wick_share"])
        & (df4.wick_body_ratio >= f["wick_body_ratio"])
        & (df4.range_atr >= f["range_atr"])
        & (df4.drop_to_low_pct <= f["drop_to_low_pct"])
        & (df4.close_location >= f["close_location"])
        & (df4.quote_volume_24h >= f["min_quote_volume_24h"])
        & (df4.extension_ema300 <= 0.30)
    ].copy()
    rows = []
    for _, r in eligible.iterrows():
        base = {
            "symbol": symbol, "flush_time": r.time, "split": split_name(r.time, cfg),
            "lower_wick_share": r.lower_wick_share, "wick_body_ratio": r.wick_body_ratio,
            "range_atr": r.range_atr, "drop_to_low_pct": r.drop_to_low_pct,
            "close_location": r.close_location, "rebound_low_close_pct": r.rebound_low_close_pct,
            "volume_ratio": r.volume_ratio, "trend_score": int(r.trend_score),
            "extension_ema300": r.extension_ema300, "support_reclaim_20d": bool(r.support_reclaim_20d),
            "support_dist_20d": r.support_dist_20d, "support_dist_60d": r.support_dist_60d,
            "quote_volume_24h": r.quote_volume_24h, "btc_ret_24h": r.btc_ret_24h,
            "btc_ret_72h": r.btc_ret_72h, "btc_bear": bool(r.btc_bear) if pd.notna(r.btc_bear) else False,
            "btc_regime": r.btc_regime, "btc_range_atr": r.btc_range_atr,
        }
        for variant in ["immediate", "confirm_mid", "confirm_60"]:
            ent = find_entry(df1, r, variant, cfg["max_confirmation_hours"])
            if ent is None:
                continue
            entry_idx, entry_time, entry_price = ent
            row = {**base, "entry_variant": variant, "entry_time": entry_time, "entry_price": entry_price}
            for horizon in [cfg["primary_horizon_hours"], cfg["secondary_horizon_hours"]]:
                for stop in cfg["stop_pcts"]:
                    out = simulate(df1, entry_idx, entry_price, cfg["primary_target_pct"], stop, horizon)
                    key = f"t3_s{int(stop*1000):02d}_h{horizon}"
                    for k, v in out.items(): row[f"{key}_{k}"] = v
                    row[f"{key}_net_return"] = net_return(out["gross_return"], cfg)
            for target in cfg["secondary_target_pcts"]:
                out = simulate(df1, entry_idx, entry_price, target, 0.02, cfg["secondary_horizon_hours"])
                key = f"t{int(target*100)}_s20_h{cfg['secondary_horizon_hours']}"
                for k, v in out.items(): row[f"{key}_{k}"] = v
                row[f"{key}_net_return"] = net_return(out["gross_return"], cfg)
            rows.append(row)
    return pd.DataFrame(rows)


def wilson(wins: int, n: int, alpha: float = 0.05):
    if n <= 0: return np.nan, np.nan
    z = NormalDist().inv_cdf(1-alpha/2); p = wins/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / d
    return c-h, c+h


def weeks_in_split(split: str, cfg: dict) -> float:
    bounds = {"DEVELOPMENT": (cfg["test_start"], cfg["development_end"]), "VALIDATION": (cfg["development_end"], cfg["validation_end"]), "CONFIRMATION": (cfg["validation_end"], cfg["confirmation_end"])}
    a, b = bounds[split]
    return (pd.Timestamp(b)-pd.Timestamp(a)).days/7.0


def rule_mask(df: pd.DataFrame, rule: dict) -> pd.Series:
    m = ((df.entry_variant == rule["entry_variant"]) & (df.lower_wick_share >= rule["lower_wick_share"]) & (df.wick_body_ratio >= rule["wick_body_ratio"]) & (df.range_atr >= rule["range_atr"]) & (df.close_location >= rule["close_location"]) & (df.volume_ratio >= rule["volume_ratio"]) & (df.trend_score >= rule["trend_score"]))
    if rule["support_filter"] == "near20d": m &= df.support_reclaim_20d
    if rule["btc_filter"] == "not_bear": m &= ~df.btc_bear
    return m


def outcome_metrics(df: pd.DataFrame, stop: float, split: str, cfg: dict) -> dict:
    z = df[df.split == split]
    key = f"t3_s{int(stop*1000):02d}_h{cfg['primary_horizon_hours']}"
    z = z[z[f"{key}_result"].notna() & (z[f"{key}_result"] != "NO_DATA")]
    wins, n = int((z[f"{key}_result"] == "TARGET").sum()), len(z)
    lo, hi = wilson(wins, n)
    return {"signals": n, "wins": wins, "precision": wins/n if n else np.nan, "wilson_low": lo, "wilson_high": hi, "signals_per_week": n/weeks_in_split(split, cfg), "gross_expectancy": z[f"{key}_gross_return"].mean() if n else np.nan, "net_expectancy": z[f"{key}_net_return"].mean() if n else np.nan, "median_mae": z[f"{key}_mae"].median() if n else np.nan, "median_mfe": z[f"{key}_mfe"].median() if n else np.nan, "median_hours": z[f"{key}_hours"].median() if n else np.nan}


def evaluate_grid(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = cfg["grid"]
    dims = [("entry_variant", g["entry_variants"]), ("lower_wick_share", g["lower_wick_share"]), ("wick_body_ratio", g["wick_body_ratio"]), ("range_atr", g["range_atr"]), ("close_location", g["close_location"]), ("volume_ratio", g["volume_ratio"]), ("trend_score", g["trend_score"]), ("support_filter", g["support_filter"]), ("btc_filter", g["btc_filter"])]
    rows = []
    for values in itertools.product(*(v for _, v in dims)):
        rule = dict(zip((k for k, _ in dims), values)); z = events[rule_mask(events, rule)]
        if z.empty: continue
        for stop in cfg["stop_pcts"]:
            dev, val = outcome_metrics(z, stop, "DEVELOPMENT", cfg), outcome_metrics(z, stop, "VALIDATION", cfg)
            rows.append({**rule, "stop_pct": stop, **{f"dev_{k}": v for k, v in dev.items()}, **{f"val_{k}": v for k, v in val.items()}})
    return pd.DataFrame(rows)


def freeze_shortlist(grid: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = cfg["selection"]
    z = grid[(grid.val_signals >= s["min_validation_signals"]) & (grid.val_signals_per_week >= s["min_validation_signals_per_week"])].copy()
    if z.empty: z = grid[grid.val_signals >= max(10, s["min_validation_signals"]//2)].copy()
    return z.sort_values(["val_precision", "val_wilson_low", "val_net_expectancy", "val_signals"], ascending=[False, False, False, False]).head(s["shortlist_size"]).reset_index(drop=True)


def confirmation(shortlist: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    cols = ["entry_variant", "lower_wick_share", "wick_body_ratio", "range_atr", "close_location", "volume_ratio", "trend_score", "support_filter", "btc_filter"]
    rows = []
    for i, r in shortlist.iterrows():
        rule = {c: r[c] for c in cols}; z = events[rule_mask(events, rule)]
        rows.append({"shortlist_rank": i+1, **rule, "stop_pct": float(r.stop_pct), **outcome_metrics(z, float(r.stop_pct), "CONFIRMATION", cfg)})
    return pd.DataFrame(rows)


def summarize_groups(events: pd.DataFrame, best: pd.Series, cfg: dict, group_col: str) -> pd.DataFrame:
    cols = ["entry_variant", "lower_wick_share", "wick_body_ratio", "range_atr", "close_location", "volume_ratio", "trend_score", "support_filter", "btc_filter"]
    rule = {c: best[c] for c in cols}; z = events[rule_mask(events, rule) & (events.split == "CONFIRMATION")].copy()
    stop = float(best.stop_pct); key = f"t3_s{int(stop*1000):02d}_h{cfg['primary_horizon_hours']}"; z["win"] = z[f"{key}_result"] == "TARGET"
    if group_col == "year": z["year"] = pd.to_datetime(z.entry_time, utc=True).dt.year
    rows = []
    for name, g in z.groupby(group_col, dropna=False):
        n, wins = len(g), int(g.win.sum()); lo, hi = wilson(wins, n)
        rows.append({group_col: name, "signals": n, "wins": wins, "precision": wins/n if n else np.nan, "wilson_low": lo, "wilson_high": hi, "net_expectancy": g[f"{key}_net_return"].mean()})
    return pd.DataFrame(rows)


def secondary_exit_summary(events: pd.DataFrame, best: pd.Series, cfg: dict) -> pd.DataFrame:
    cols = ["entry_variant", "lower_wick_share", "wick_body_ratio", "range_atr", "close_location", "volume_ratio", "trend_score", "support_filter", "btc_filter"]
    rule = {c: best[c] for c in cols}; z = events[rule_mask(events, rule) & (events.split == "CONFIRMATION")]; rows = []
    for target in [cfg["primary_target_pct"], *cfg["secondary_target_pcts"]]:
        key = f"t3_s20_h{cfg['primary_horizon_hours']}" if target == cfg["primary_target_pct"] else f"t{int(target*100)}_s20_h{cfg['secondary_horizon_hours']}"
        if f"{key}_result" not in z: continue
        n = len(z); wins = int((z[f"{key}_result"] == "TARGET").sum())
        rows.append({"target_pct": target, "stop_pct": 0.02, "horizon_hours": cfg["primary_horizon_hours"] if target == 0.03 else cfg["secondary_horizon_hours"], "signals": n, "wins": wins, "precision": wins/n if n else np.nan, "gross_expectancy": z[f"{key}_gross_return"].mean(), "net_expectancy": z[f"{key}_net_return"].mean()})
    return pd.DataFrame(rows)


def write_analysis(confirm: pd.DataFrame, secondary: pd.DataFrame, events: pd.DataFrame, cfg: dict) -> None:
    s = cfg["selection"]; lines = ["# R01 Trend Flush Reversal — Round 001 Analysis", "", "Research only. No live strategy changes.", ""]
    if confirm.empty:
        lines += ["## Outcome", "", "No shortlist could be evaluated on locked confirmation data.", ""]
    else:
        best = confirm.sort_values(["precision", "wilson_low", "net_expectancy", "signals"], ascending=[False, False, False, False]).iloc[0]
        accepted = bool(best.precision >= s["acceptance_precision"] and best.signals >= s["acceptance_min_confirmation_signals"] and best.signals_per_week >= s["acceptance_min_confirmation_signals_per_week"])
        lines += ["## Primary result", "", f"- Accepted against the stated primary threshold: **{'YES' if accepted else 'NO'}**", f"- Locked-confirmation signals: **{int(best.signals)}** ({best.signals_per_week:.2f}/week)", f"- +3% before -{best.stop_pct*100:.1f}% precision: **{best.precision*100:.2f}%**", f"- Wilson 95% CI: **{best.wilson_low*100:.2f}% to {best.wilson_high*100:.2f}%**", f"- Mean net return/signal after assumed fees/slippage: **{best.net_expectancy*100:.3f}%**", f"- Median MAE / MFE: **{best.median_mae*100:.2f}% / {best.median_mfe*100:.2f}%**", "", "The confirmation period was not used to choose thresholds. The reported best row is the strongest member of the shortlist frozen from Development/Validation.", "", "## Best locked rule", "", f"`entry={best.entry_variant}, wick_share>={best.lower_wick_share}, wick/body>={best.wick_body_ratio}, range/ATR>={best.range_atr}, close_location>={best.close_location}, volume_ratio>={best.volume_ratio}, trend_score>={int(best.trend_score)}, support={best.support_filter}, BTC={best.btc_filter}, stop={best.stop_pct}`", ""]
        if not accepted: lines += ["The 90% target was not met under the minimum frequency/sample constraints, so Round 001 does **not** establish a 90%-accurate live signal.", ""]
    lines += ["## Dataset", "", f"- Event-entry rows analysed: {len(events):,}", f"- Symbols represented: {events.symbol.nunique() if len(events) else 0}", f"- Development / Validation / Confirmation rows: {(events.split == 'DEVELOPMENT').sum():,} / {(events.split == 'VALIDATION').sum():,} / {(events.split == 'CONFIRMATION').sum():,}", "", "## Secondary exit diagnostic", ""]
    lines.append("No secondary exit diagnostic available." if secondary.empty else secondary.to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## Files", "", "See `shortlist.csv`, `confirmation_results.csv`, `year_summary.csv`, `regime_summary.csv`, `symbol_summary.csv`, `rule_grid.csv` and `events.csv.gz` for the audit trail.", ""]
    (RESULTS / "ANALYSIS.md").write_text("\n".join(lines))


def analyse(events: pd.DataFrame, cfg: dict, chash: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True); events = events.sort_values(["entry_time", "symbol", "entry_variant"]).reset_index(drop=True)
    events.to_csv(RESULTS / "events.csv.gz", index=False, compression="gzip")
    grid = evaluate_grid(events, cfg); grid.to_csv(RESULTS / "rule_grid.csv", index=False)
    shortlist = freeze_shortlist(grid, cfg); shortlist.to_csv(RESULTS / "shortlist.csv", index=False)
    conf = confirmation(shortlist, events, cfg); conf.to_csv(RESULTS / "confirmation_results.csv", index=False)
    if conf.empty:
        for name in ["year_summary.csv", "regime_summary.csv", "symbol_summary.csv"]: pd.DataFrame().to_csv(RESULTS / name, index=False)
        secondary = pd.DataFrame()
    else:
        best = conf.sort_values(["precision", "wilson_low", "net_expectancy", "signals"], ascending=[False, False, False, False]).iloc[0]
        summarize_groups(events, best, cfg, "year").to_csv(RESULTS / "year_summary.csv", index=False)
        summarize_groups(events, best, cfg, "btc_regime").to_csv(RESULTS / "regime_summary.csv", index=False)
        summarize_groups(events, best, cfg, "symbol").to_csv(RESULTS / "symbol_summary.csv", index=False)
        secondary = secondary_exit_summary(events, best, cfg); secondary.to_csv(RESULTS / "secondary_exit_summary.csv", index=False)
    events[events.filter(regex=r"_result$").eq("NO_DATA").any(axis=1)].to_csv(RESULTS / "failures.csv", index=False)
    manifest = {"study_id": "R01", "round_id": "001", "name": "Trend Flush Reversal", "status": "RESEARCH ONLY - no live strategy changes", "config_hash": chash, "generated_at": now_iso(), "symbols": int(events.symbol.nunique()) if len(events) else 0, "event_entry_rows": len(events), "period": [cfg["test_start"], cfg["confirmation_end"]], "costs": {"fee_each_side": cfg["fee_rate_each_side"], "slippage_each_side": cfg["slippage_rate_each_side"]}, "primary_target": cfg["primary_target_pct"], "stops": cfg["stop_pcts"]}
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_analysis(conf, secondary, events, cfg)


def main() -> None:
    cfg = load_config(); chash = config_hash(cfg); symbols = load_symbols()
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True); RESULTS.mkdir(parents=True, exist_ok=True)
    state = read_state(chash); write_state(state)
    btc1 = load_symbol("BTCUSDT", "1h", cfg["data_start"], "2026-10-01")
    if btc1.empty: raise RuntimeError("BTCUSDT data unavailable")
    btc_context = add_btc_context(aggregate_4h(btc1))
    completed = set(state.get("completed_symbols", [])); failures = {x.get("symbol"): x for x in state.get("failed_symbols", []) if isinstance(x, dict)}
    pending = [s for s in symbols if s not in completed]; batch_size = int(cfg["batch_size"])
    for bstart in range(0, len(pending), batch_size):
        for symbol in pending[bstart:bstart+batch_size]:
            try:
                df = process_symbol(symbol, btc_context, cfg); df.to_csv(SYMBOL_DIR / f"{symbol}.csv.gz", index=False, compression="gzip"); completed.add(symbol); failures.pop(symbol, None)
            except Exception as exc:
                failures[symbol] = {"symbol": symbol, "error": repr(exc), "at": now_iso()}
        state.update({"status": "running", "config_hash": chash, "completed_symbols": sorted(completed), "failed_symbols": list(failures.values())}); write_state(state)
        git_checkpoint(f"R01 checkpoint {len(completed)}/{len(symbols)} symbols")
    frames = []
    for symbol in symbols:
        p = SYMBOL_DIR / f"{symbol}.csv.gz"
        if p.exists() and symbol in completed:
            try: frames.append(pd.read_csv(p, parse_dates=["flush_time", "entry_time"]))
            except pd.errors.EmptyDataError: pass
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if events.empty: raise RuntimeError("No candidate events were generated")
    analyse(events, cfg, chash)
    state.update({"status": "complete", "completed_symbols": sorted(completed), "failed_symbols": list(failures.values())}); write_state(state)
    git_checkpoint("Store R01 round 001 research results")
    print((RESULTS / "ANALYSIS.md").read_text())


if __name__ == "__main__":
    main()
