from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from high_purity_signal_watch import (
    BASE_FEATURES,
    CONTEXT_FEATURES,
    DELTA_HOURS,
    add_cross_section_context,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "research/cache/high_purity_replay_source"
MODEL_DIR = ROOT / "research/results/daily_return_harvest/high_purity_runtime/models"
OUT = ROOT / "research/results/daily_return_harvest/high_purity_runtime/replay_2026"

START = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
END_DECISION = pd.Timestamp("2026-09-28 00:00:00", tz="UTC")
END_MARK = pd.Timestamp("2026-09-30 23:00:00", tz="UTC")
PRE_CONTEXT = START - pd.Timedelta(hours=169)

STARTING_CAPITAL = 2000.0
SLEEVES = 2
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
TARGET_NET = 0.03
STOP_NET = -0.02
STALE_HOURS = 24.0
MIN_QUOTE_VOLUME_24H = 1_000_000.0


def shard_dirs() -> list[Path]:
    out = [SRC / f"drh2-shard-{i}" for i in range(4)]
    missing = [str(p) for p in out if not (p / "panel.parquet").exists()]
    if missing:
        raise RuntimeError(f"Missing full shard sources: {missing}")
    return out


def load_models():
    with (MODEL_DIR / "event_hgb_model.pkl").open("rb") as f:
        a, af = pickle.load(f)
    with (MODEL_DIR / "familyB_model.pkl").open("rb") as f:
        b, bf = pickle.load(f)
    with (MODEL_DIR / "familyC15_model.pkl").open("rb") as f:
        c, cf = pickle.load(f)
    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text())
    if bf != cf:
        raise RuntimeError("B/C feature mismatch")
    return a, b, c, af, bf, manifest


def load_context() -> pd.DataFrame:
    frames = []
    for d in shard_dirs():
        p = d / "panel.parquet"
        z = pd.read_parquet(p)
        z["time"] = pd.to_datetime(z["time"], utc=True)
        z = z[(z.time >= PRE_CONTEXT) & (z.time < END_DECISION)].copy()
        frames.append(z)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["symbol", "time"]).sort_values(["time", "symbol"])
    return add_cross_section_context(panel)


def read_symbol_frames() -> list[tuple[str, Path]]:
    out = []
    for d in shard_dirs():
        for p in sorted((d / "features").glob("*.pkl.gz")):
            out.append((p.name.removesuffix(".pkl.gz"), p))
    if not out:
        raise RuntimeError("No full per-symbol feature files found")
    return out


def context_for_symbol(context: pd.DataFrame, symbol: str) -> pd.DataFrame:
    cols = ["time"] + CONTEXT_FEATURES
    z = context.loc[context.symbol.eq(symbol), cols].copy()
    return z.drop_duplicates("time").set_index("time").sort_index()


def trajectory_matrix(ff: pd.DataFrame, ctx: pd.DataFrame, needed: list[str]) -> pd.DataFrame:
    base = ff.set_index("time").sort_index()
    idx = base.index
    merged = pd.DataFrame(index=idx)
    for f in BASE_FEATURES:
        if f in base.columns:
            merged[f] = pd.to_numeric(base[f], errors="coerce")
    for f in CONTEXT_FEATURES:
        if f in ctx.columns:
            merged[f] = pd.to_numeric(ctx[f].reindex(idx), errors="coerce")

    out = pd.DataFrame(index=idx)
    for name in needed:
        if "__chg_" not in name:
            out[name] = merged[name] if name in merged else np.nan
            continue
        src, tail = name.split("__chg_", 1)
        dh = int(tail[:-1])
        cur = merged[src] if src in merged else pd.Series(np.nan, index=idx)
        prev = cur.reindex(idx - pd.Timedelta(hours=dh)).to_numpy()
        out[name] = cur.to_numpy() - prev
    return out


def score_symbol(symbol: str, path: Path, ctx: pd.DataFrame, a, b, c, af, bf, th) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff = pd.read_pickle(path, compression="gzip")
    ff["time"] = pd.to_datetime(ff["time"], utc=True)
    ff = ff.sort_values("time").drop_duplicates("time")

    prices = ff.loc[(ff.time >= START) & (ff.time <= END_MARK), ["time", "open", "high", "low", "close"]].copy()
    if prices.empty:
        return pd.DataFrame(), prices

    needed = list(dict.fromkeys(af + bf))
    xall = trajectory_matrix(ff, ctx, needed)
    xall = xall[(xall.index >= START - pd.Timedelta(hours=1)) & (xall.index < END_DECISION)].copy()
    if xall.empty:
        return pd.DataFrame(), prices

    meta = ff.set_index("time").reindex(xall.index)
    eligible = pd.Series(True, index=xall.index)
    if "eligible" in meta.columns:
        eligible &= meta["eligible"].fillna(False).astype(bool)
    else:
        eligible &= pd.to_numeric(meta.get("quote_volume_24h"), errors="coerce") >= MIN_QUOTE_VOLUME_24H
    if "account_excluded" in meta.columns:
        eligible &= ~meta["account_excluded"].fillna(False).astype(bool)
    eligible &= pd.to_numeric(meta.get("quote_volume_24h"), errors="coerce") >= MIN_QUOTE_VOLUME_24H

    x = xall.loc[eligible].astype(np.float32)
    if x.empty:
        return pd.DataFrame(), prices

    sa = a.predict_proba(x[af])[:, 1]
    family = np.full(len(x), "", dtype=object)
    score = np.full(len(x), np.nan)
    pass_a = sa >= th["A"]
    family[pass_a] = "A"
    score[pass_a] = sa[pass_a]

    rem_b = ~pass_a
    sb = np.full(len(x), np.nan)
    if rem_b.any():
        sb[rem_b] = b.predict_proba(x.loc[rem_b, bf])[:, 1]
    pass_b = rem_b & (sb >= th["B"])
    family[pass_b] = "B"
    score[pass_b] = sb[pass_b]

    rem_c = rem_b & ~pass_b
    sc = np.full(len(x), np.nan)
    if rem_c.any():
        sc[rem_c] = c.predict_proba(x.loc[rem_c, bf])[:, 1]
    pass_c = rem_c & (sc >= th["C"])
    family[pass_c] = "C"
    score[pass_c] = sc[pass_c]

    on = family != ""
    state = pd.DataFrame({
        "symbol": symbol,
        "decision_time": x.index,
        "is_signal": on,
        "family": family,
        "score": score,
    })
    state = state.sort_values("decision_time")
    prev = state.is_signal.shift(1, fill_value=False)
    # If the previous eligible row is not exactly one hour earlier, treat this as a fresh onset.
    gap = state.decision_time.diff().dt.total_seconds().div(3600)
    fresh = state.is_signal & (~prev | gap.ne(1))
    sig = state.loc[fresh].copy()
    if sig.empty:
        return sig, prices

    sig["entry_time"] = sig.decision_time + pd.Timedelta(hours=1)
    w3 = pd.to_numeric(meta.get("w3_2_24h"), errors="coerce") if "w3_2_24h" in meta.columns else pd.Series(np.nan, index=meta.index)
    sig["w3_2_24h"] = w3.reindex(pd.DatetimeIndex(sig.decision_time)).to_numpy()
    return sig.reset_index(drop=True), prices


@dataclass
class Sleeve:
    sleeve_id: int
    cash: float
    symbol: str | None = None
    entry_time: pd.Timestamp | None = None
    raw_entry: float | None = None
    qty: float | None = None
    target_raw: float | None = None
    stop_raw: float | None = None
    family: str | None = None
    signal_score: float | None = None

    @property
    def open(self) -> bool:
        return self.symbol is not None


def price_lookup(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {s: z.set_index("time").sort_index() for s, z in prices.items() if not z.empty}


def raw_open_at(px: dict[str, pd.DataFrame], symbol: str, t: pd.Timestamp) -> float | None:
    z = px.get(symbol)
    if z is None or t not in z.index:
        return None
    v = z.loc[t, "open"]
    if isinstance(v, pd.Series): v = v.iloc[-1]
    return float(v) if np.isfinite(v) else None


def net_liquidation(s: Sleeve, raw_price: float) -> float:
    if not s.open:
        return s.cash
    return float(s.qty * raw_price * (1 - SLIPPAGE_RATE) * (1 - FEE_RATE))


def open_position(s: Sleeve, signal, raw_open: float, log: list[dict], reason: str):
    before = s.cash
    unit_cost = raw_open * (1 + SLIPPAGE_RATE) * (1 + FEE_RATE)
    qty = before / unit_cost
    exit_factor = (1 - SLIPPAGE_RATE) * (1 - FEE_RATE)
    s.symbol = signal.symbol
    s.entry_time = signal.entry_time
    s.raw_entry = raw_open
    s.qty = qty
    s.target_raw = unit_cost * (1 + TARGET_NET) / exit_factor
    s.stop_raw = unit_cost * (1 + STOP_NET) / exit_factor
    s.family = signal.family
    s.signal_score = float(signal.score)
    log.append({"time": signal.entry_time, "event": "ENTRY", "reason": reason, "sleeve": s.sleeve_id, "symbol": s.symbol, "family": s.family, "score": s.signal_score, "balance_before": before, "raw_price": raw_open})


def close_position(s: Sleeve, t: pd.Timestamp, raw_price: float, reason: str, log: list[dict]):
    before = s.cash
    proceeds = net_liquidation(s, raw_price)
    ret = proceeds / before - 1 if before else np.nan
    log.append({"time": t, "event": "EXIT", "reason": reason, "sleeve": s.sleeve_id, "symbol": s.symbol, "family": s.family, "score": s.signal_score, "balance_before": before, "balance_after": proceeds, "net_return": ret, "raw_price": raw_price, "held_hours": (t - s.entry_time).total_seconds()/3600 if s.entry_time is not None else np.nan})
    s.cash = proceeds
    s.symbol = None; s.entry_time = None; s.raw_entry = None; s.qty = None
    s.target_raw = None; s.stop_raw = None; s.family = None; s.signal_score = None


def simulate(signals: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    px = price_lookup(prices)
    sleeves = [Sleeve(i + 1, STARTING_CAPITAL / SLEEVES) for i in range(SLEEVES)]
    signals_by_entry = {t: g.sort_values("symbol") for t, g in signals.groupby("entry_time")}
    all_times = sorted({t for z in px.values() for t in z.index if START <= t <= END_MARK})
    log: list[dict] = []
    equity_rows = []
    skipped_full = skipped_same_symbol = replacements = 0

    for t in all_times:
        batch = signals_by_entry.get(t)
        if batch is not None and len(batch):
            candidates = [r for r in batch.itertuples()]
            candidates = [r for r in candidates if not any(s.open and s.symbol == r.symbol for s in sleeves)]
            skipped_same_symbol += len(batch) - len(candidates)

            for r in candidates:
                free = next((s for s in sleeves if not s.open), None)
                if free is None:
                    stale = [s for s in sleeves if s.open and (t - s.entry_time).total_seconds()/3600 >= STALE_HOURS]
                    if stale:
                        stale.sort(key=lambda s: (s.entry_time, s.sleeve_id))
                        free = stale[0]
                        ro = raw_open_at(px, free.symbol, t)
                        if ro is None:
                            skipped_full += 1
                            continue
                        close_position(free, t, ro, "REPLACED_AFTER_24H", log)
                        replacements += 1
                    else:
                        skipped_full += 1
                        continue
                ro = raw_open_at(px, r.symbol, t)
                if ro is None:
                    skipped_full += 1
                    continue
                open_position(free, r, ro, log, "FREE_SLEEVE" if free.entry_time is None else "REPLACEMENT")

        # Intrabar OCO checks after entries at this hour's open. Ambiguous same-bar touch is stop-first.
        for s in sleeves:
            if not s.open:
                continue
            z = px.get(s.symbol)
            if z is None or t not in z.index:
                continue
            bar = z.loc[t]
            if isinstance(bar, pd.DataFrame): bar = bar.iloc[-1]
            hi, lo = float(bar.high), float(bar.low)
            hit_target = np.isfinite(hi) and hi >= s.target_raw
            hit_stop = np.isfinite(lo) and lo <= s.stop_raw
            if hit_target and hit_stop:
                close_position(s, t + pd.Timedelta(hours=1), s.stop_raw, "STOP_AMBIGUOUS_BAR", log)
            elif hit_stop:
                close_position(s, t + pd.Timedelta(hours=1), s.stop_raw, "STOP", log)
            elif hit_target:
                close_position(s, t + pd.Timedelta(hours=1), s.target_raw, "TARGET", log)

        total = 0.0
        for s in sleeves:
            if not s.open:
                total += s.cash
                continue
            z = px.get(s.symbol)
            if z is not None and t in z.index:
                bar = z.loc[t]
                if isinstance(bar, pd.DataFrame): bar = bar.iloc[-1]
                total += net_liquidation(s, float(bar.close))
            else:
                total += s.cash
        equity_rows.append({"time": t + pd.Timedelta(hours=1), "equity": total})

    # Mark any survivors to market at last available close.
    final_t = max(all_times)
    for s in sleeves:
        if not s.open:
            continue
        z = px[s.symbol]
        zz = z[z.index <= final_t]
        if zz.empty:
            continue
        last_t = zz.index[-1]
        raw = float(zz.loc[last_t, "close"])
        close_position(s, last_t + pd.Timedelta(hours=1), raw, "END_MARK", log)

    trades = pd.DataFrame(log)
    equity = pd.DataFrame(equity_rows)
    if len(equity):
        peak = equity.equity.cummax()
        dd = equity.equity / peak - 1
        max_dd = float(dd.min())
    else:
        max_dd = np.nan
    exits = trades[trades.event.eq("EXIT")].copy() if len(trades) else pd.DataFrame()
    targets = int(exits.reason.eq("TARGET").sum()) if len(exits) else 0
    stops = int(exits.reason.str.startswith("STOP").sum()) if len(exits) else 0
    final_balance = float(sum(s.cash for s in sleeves))
    metrics = {
        "period": "2026-01-01 through available Round 2 data ending 2026-09-30",
        "starting_capital": STARTING_CAPITAL,
        "sleeves": SLEEVES,
        "starting_sleeve_size": STARTING_CAPITAL / SLEEVES,
        "raw_new_signal_onsets": int(len(signals)),
        "raw_w3_rate": float(pd.to_numeric(signals.w3_2_24h, errors="coerce").mean()),
        "entries": int((trades.event == "ENTRY").sum()) if len(trades) else 0,
        "closed_trades": int(len(exits)),
        "targets": targets,
        "stops": stops,
        "replacements_after_24h": int(replacements),
        "end_marks": int(exits.reason.eq("END_MARK").sum()) if len(exits) else 0,
        "skipped_no_capacity_under_24h": int(skipped_full),
        "skipped_same_symbol_already_held": int(skipped_same_symbol),
        "final_balance": final_balance,
        "net_profit": final_balance - STARTING_CAPITAL,
        "total_return_pct": (final_balance / STARTING_CAPITAL - 1) * 100,
        "max_drawdown_pct": max_dd * 100 if np.isfinite(max_dd) else None,
        "mean_closed_trade_return_pct": float(pd.to_numeric(exits.net_return, errors="coerce").mean() * 100) if len(exits) else None,
        "median_closed_trade_return_pct": float(pd.to_numeric(exits.net_return, errors="coerce").median() * 100) if len(exits) else None,
        "median_holding_hours": float(pd.to_numeric(exits.held_hours, errors="coerce").median()) if len(exits) else None,
    }
    return metrics, trades, equity


def main():
    a, b, c, af, bf, manifest = load_models()
    th = manifest["thresholds"]
    context = load_context()
    signals_parts = []
    prices = {}
    for symbol, path in read_symbol_frames():
        ctx = context_for_symbol(context, symbol)
        sig, px = score_symbol(symbol, path, ctx, a, b, c, af, bf, th)
        if len(sig): signals_parts.append(sig)
        if len(px): prices[symbol] = px
        print("SCORE", symbol, "new_signals", len(sig), flush=True)
    signals = pd.concat(signals_parts, ignore_index=True).sort_values(["entry_time", "symbol"]).reset_index(drop=True) if signals_parts else pd.DataFrame()
    metrics, trades, equity = simulate(signals, prices)

    OUT.mkdir(parents=True, exist_ok=True)
    signals.to_csv(OUT / "full_stream_signal_onsets.csv", index=False)
    trades.to_csv(OUT / "trade_log.csv", index=False)
    equity.to_csv(OUT / "equity_curve_hourly.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report = [
        "# High-Purity A/B/C 2026 Full-Stream Portfolio Replay",
        "",
        "Research-only. No live strategy or portfolio was changed.",
        "",
        "## Design",
        "",
        "- Score every eligible historical coin-hour in the frozen Round 2 universe with high-purity-abc-v1.",
        "- New signal = rising edge from non-signal to signal for that symbol.",
        "- Two sleeves starting at 1,000 units each; no GBP/USDT FX modelling.",
        "- Enter at next-hour open with 0.10% fee + 0.05% slippage per side.",
        "- OCO remains +3% net target / -2% net stop with no forced 24h close.",
        "- If both sleeves are occupied and a new signal arrives, oldest position aged >=24h is closed at that hour's open and replaced; otherwise new signal is skipped.",
        "- No predictive ranking. Same-hour overflow is resolved by deterministic symbol order.",
        "- Same-candle target+stop ambiguity is conservatively a stop.",
        "- Survivors at data end are marked to market.",
        "",
        "## Result",
        "",
        "```json",
        json.dumps(metrics, indent=2),
        "```",
    ]
    (OUT / "REPLAY.md").write_text("\n".join(report) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
