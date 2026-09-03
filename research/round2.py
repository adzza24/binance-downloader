from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest
from binance_data import load_symbol


def add_market_state(btc: pd.DataFrame) -> pd.DataFrame:
    x = btc[["time", "close", "high", "low"]].copy()
    x["sma50"] = x.close.rolling(24 * 50).mean()
    x["sma100"] = x.close.rolling(24 * 100).mean()
    x["sma200"] = x.close.rolling(24 * 200).mean()
    x["sma200_prev20d"] = x.sma200.shift(24 * 20)
    hi90 = x.high.rolling(24 * 90).max()
    lo90 = x.low.rolling(24 * 90).min()
    x["mid90"] = (hi90 + lo90) / 2
    x["market_score"] = (
        (x.close > x.sma200).astype(int)
        + (x.sma50 > x.sma100).astype(int)
        + (x.sma200 > x.sma200_prev20d).astype(int)
        + (x.close > x.mid90).astype(int)
    )
    x["market_state"] = np.select(
        [x.market_score <= 1, x.market_score == 2, x.market_score == 3],
        ["BEAR", "RECOVERY", "BULL"],
        default="STRONG_BULL",
    )
    return x


def activity_pass(sig: dict, trigger_return: float, cfg: dict) -> bool:
    f = cfg["round2"]["activity"]
    return (
        f["min_volume_ratio"] <= sig["trigger_volume_ratio"] <= f["max_volume_ratio"]
        and f["min_trade_count_ratio"] <= sig["trade_count_ratio"] <= f["max_trade_count_ratio"]
        and sig["taker_buy_ratio"] >= f["min_taker_buy_ratio"]
        and trigger_return <= f["max_trigger_candle_return"]
        and sig["prior_impulse_pct"] <= f["max_prior_impulse_pct"]
    )


def prior_eligibility(current_time, market_start, prior_a, cfg):
    e = cfg["round2"]["eligibility"]
    if current_time < market_start + pd.Timedelta(days=int(e["warmup_days"])):
        return "WARMUP", 0, np.nan, np.nan
    cutoff = current_time - pd.Timedelta(days=int(e["lookback_days"]))
    resolved = [
        r for r in prior_a
        if pd.Timestamp(r["exit_time"]) < current_time
        and pd.Timestamp(r["entry_time"]) >= cutoff
        and r["status"] == "TRADE"
    ]
    if len(resolved) < int(e["min_resolved_setups"]):
        return "UNPROVEN", len(resolved), np.nan, np.nan
    pnl = np.array([float(r["pnl_usdt"]) for r in resolved])
    win_rate = float((pnl > 0).mean())
    net = float(pnl.sum())
    eligible = net > float(e["min_net_pnl_usdt"]) and win_rate >= float(e["min_win_rate"])
    return ("ELIGIBLE" if eligible else "INELIGIBLE"), len(resolved), win_rate, net


def summarise_variant(trades: pd.DataFrame, variant: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    s = backtest.summarise(trades)
    s.insert(0, "variant", variant)
    return s


def yearly_summary(trades: pd.DataFrame) -> pd.DataFrame:
    x = trades[trades.status == "TRADE"].copy()
    if x.empty:
        return pd.DataFrame()
    x["year"] = pd.to_datetime(x.entry_time, utc=True).dt.year
    return (
        x.groupby(["variant", "method", "year"], as_index=False)
        .agg(
            trades=("pnl_usdt", "size"),
            net_pnl_usdt=("pnl_usdt", "sum"),
            expectancy_usdt=("pnl_usdt", "mean"),
            win_rate=("pnl_usdt", lambda s: float((s > 0).mean())),
        )
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="research/round2_config.json")
    ap.add_argument("--out", default="research/results/round_2")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    if btc.empty:
        raise SystemExit("Unable to load BTCUSDT archive data")
    market = add_market_state(btc)
    market_lookup = market.set_index("time")[["market_score", "market_state"]]

    all_signals = []
    all_trades = []
    eligibility_rows = []
    variants = ["BASELINE", "ACTIVITY", "ACTIVITY_MARKET", "FULL_ELIGIBLE"]

    for symbol in cfg["symbols"]:
        print(f"Round 2 processing {symbol}", flush=True)
        df = btc.copy() if symbol == "BTCUSDT" else load_symbol(
            symbol, cfg["interval"], cfg["start"], cfg["end"]
        )
        if df.empty:
            continue
        signals = backtest.detect_signals(symbol, df, btc, cfg)
        if not signals:
            continue
        by_time = df.set_index("time")[["open", "close"]]
        market_start = pd.Timestamp(df.time.iloc[0])
        prior_a = []

        for sig in sorted(signals, key=lambda r: r["signal_time"]):
            st = pd.Timestamp(sig["signal_time"])
            if st not in by_time.index or st not in market_lookup.index:
                continue
            bar = by_time.loc[st]
            trigger_return = float(bar["close"] / bar["open"] - 1)
            m = market_lookup.loc[st]
            market_score = int(m["market_score"])
            market_state = str(m["market_state"])
            act_ok = activity_pass(sig, trigger_return, cfg)
            status, n_prior, prior_wr, prior_net = prior_eligibility(
                st, market_start, prior_a, cfg
            )
            market_ok = market_score >= int(cfg["round2"]["market"]["min_score"])

            enriched = dict(sig)
            enriched.update({
                "trigger_return": trigger_return,
                "market_score": market_score,
                "market_state": market_state,
                "activity_pass": act_ok,
                "market_pass": market_ok,
                "eligibility_status": status,
                "eligibility_prior_setups": n_prior,
                "eligibility_prior_win_rate": prior_wr,
                "eligibility_prior_net_pnl": prior_net,
            })
            all_signals.append(enriched)
            eligibility_rows.append({
                "signal_id": sig["signal_id"],
                "symbol": symbol,
                "signal_time": st,
                "status": status,
                "prior_setups": n_prior,
                "prior_win_rate": prior_wr,
                "prior_net_pnl_usdt": prior_net,
            })

            accepted = {
                "BASELINE": True,
                "ACTIVITY": act_ok,
                "ACTIVITY_MARKET": act_ok and market_ok,
                "FULL_ELIGIBLE": act_ok and market_ok and status == "ELIGIBLE",
            }
            for variant in variants:
                if not accepted[variant]:
                    continue
                for method in "ABCD":
                    row = backtest.simulate_method(df, sig, method, cfg)
                    row["variant"] = variant
                    all_trades.append(row)

            prior_a.append(backtest.simulate_method(df, sig, "A", cfg))

    s = pd.DataFrame(all_signals)
    t = pd.DataFrame(all_trades)
    e = pd.DataFrame(eligibility_rows)
    s.to_csv(out / "historical_signals_round2.csv", index=False)
    t.to_csv(out / "historical_trades_round2.csv", index=False)
    e.to_csv(out / "eligibility_history.csv", index=False)

    market_daily = market.set_index("time").resample("1D").last().dropna().reset_index()
    market_daily.to_csv(out / "market_state_history.csv", index=False)

    summaries = []
    for variant in variants:
        x = summarise_variant(t[t.variant == variant], variant)
        if not x.empty:
            summaries.append(x)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(out / "backtest_summary_round2.csv", index=False)

    yearly = yearly_summary(t)
    yearly.to_csv(out / "yearly_summary_round2.csv", index=False)

    manifest = {
        "experiment": "round_2_entry_filters",
        "entry_logic": {
            "activity_filter": cfg["round2"]["activity"],
            "market_filter": cfg["round2"]["market"],
            "eligibility_filter": cfg["round2"]["eligibility"],
        },
        "variants": variants,
        "exit_methods": "A/B/C/D unchanged from Round 1",
        "anti_lookahead": (
            "Market state uses only contemporaneous/past BTC candles. Coin eligibility "
            "uses only prior method-A setup outcomes whose exit_time is before the current signal."
        ),
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
