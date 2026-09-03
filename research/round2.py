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


def activity_pass(row: pd.Series, cfg: dict) -> bool:
    f = cfg["round2"]["activity"]
    return (
        f["min_volume_ratio"] <= row.trigger_volume_ratio <= f["max_volume_ratio"]
        and f["min_trade_count_ratio"] <= row.trade_count_ratio <= f["max_trade_count_ratio"]
        and row.taker_buy_ratio >= f["min_taker_buy_ratio"]
        and row.prior_impulse_pct <= f["max_prior_impulse_pct"]
    )


def add_eligibility(signals: pd.DataFrame, trades: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    e = cfg["round2"]["eligibility"]
    a = trades[(trades.method == "A") & (trades.status == "TRADE")].copy()
    a["entry_time"] = pd.to_datetime(a.entry_time, utc=True)
    a["exit_time"] = pd.to_datetime(a.exit_time, utc=True)

    out = []
    for symbol, g in signals.sort_values("signal_time").groupby("symbol"):
        first_signal = g.signal_time.min()
        prior = a[a.symbol == symbol].copy()
        for _, row in g.iterrows():
            st = row.signal_time
            if st < first_signal + pd.Timedelta(days=int(e["warmup_days"])):
                status, n, wr, net = "WARMUP", 0, np.nan, np.nan
            else:
                cutoff = st - pd.Timedelta(days=int(e["lookback_days"]))
                hist = prior[(prior.exit_time < st) & (prior.entry_time >= cutoff)]
                n = len(hist)
                if n < int(e["min_resolved_setups"]):
                    status, wr, net = "UNPROVEN", np.nan, np.nan
                else:
                    pnl = hist.pnl_usdt.astype(float)
                    wr = float((pnl > 0).mean())
                    net = float(pnl.sum())
                    ok = net > float(e["min_net_pnl_usdt"]) and wr >= float(e["min_win_rate"])
                    status = "ELIGIBLE" if ok else "INELIGIBLE"
            out.append({
                "signal_id": row.signal_id,
                "eligibility_status": status,
                "eligibility_prior_setups": n,
                "eligibility_prior_win_rate": wr,
                "eligibility_prior_net_pnl": net,
            })
    return signals.merge(pd.DataFrame(out), on="signal_id", how="left")


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
    ap.add_argument("--round1", default="research/round1_input")
    ap.add_argument("--out", default="research/results/round_2")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    r1 = Path(args.round1)
    signals = pd.read_csv(r1 / "historical_signals.csv")
    trades = pd.read_csv(r1 / "historical_trades.csv")
    signals["signal_time"] = pd.to_datetime(signals.signal_time, utc=True)
    signals["entry_time"] = pd.to_datetime(signals.entry_time, utc=True)
    trades["entry_time"] = pd.to_datetime(trades.entry_time, utc=True)
    trades["exit_time"] = pd.to_datetime(trades.exit_time, utc=True)

    btc = load_symbol("BTCUSDT", cfg["interval"], cfg["start"], cfg["end"])
    if btc.empty:
        raise SystemExit("Unable to load BTCUSDT archive data")
    market = add_market_state(btc)
    lookup = market.set_index("time")[["market_score", "market_state"]]

    signals["market_score"] = signals.signal_time.map(lambda t: lookup["market_score"].get(t, np.nan))
    signals["market_state"] = signals.signal_time.map(lambda t: lookup["market_state"].get(t, np.nan))
    signals["activity_pass"] = signals.apply(lambda r: activity_pass(r, cfg), axis=1)
    signals["market_pass"] = signals.market_score >= int(cfg["round2"]["market"]["min_score"])
    signals = add_eligibility(signals, trades, cfg)

    variants = {
        "BASELINE": pd.Series(True, index=signals.index),
        "ACTIVITY": signals.activity_pass,
        "ACTIVITY_MARKET": signals.activity_pass & signals.market_pass,
        "FULL_ELIGIBLE": (
            signals.activity_pass
            & signals.market_pass
            & (signals.eligibility_status == "ELIGIBLE")
        ),
    }

    round2_trades = []
    for variant, mask in variants.items():
        ids = set(signals.loc[mask, "signal_id"])
        x = trades[trades.signal_id.isin(ids)].copy()
        x["variant"] = variant
        round2_trades.append(x)
    t = pd.concat(round2_trades, ignore_index=True)

    signals.to_csv(out / "historical_signals_round2.csv", index=False)
    t.to_csv(out / "historical_trades_round2.csv", index=False)
    signals[[
        "signal_id", "symbol", "signal_time", "eligibility_status",
        "eligibility_prior_setups", "eligibility_prior_win_rate",
        "eligibility_prior_net_pnl",
    ]].to_csv(out / "eligibility_history.csv", index=False)

    market.set_index("time").resample("1D").last().dropna().reset_index().to_csv(
        out / "market_state_history.csv", index=False
    )

    summaries = []
    for variant in variants:
        x = summarise_variant(t[t.variant == variant], variant)
        if not x.empty:
            summaries.append(x)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary.to_csv(out / "backtest_summary_round2.csv", index=False)
    yearly_summary(t).to_csv(out / "yearly_summary_round2.csv", index=False)

    manifest = {
        "experiment": "round_2_entry_filters",
        "source": "Round 1 historical signals/trades; BTC Binance Vision history for market state",
        "variants": list(variants),
        "exit_methods": "A/B/C/D unchanged from Round 1",
        "activity_filter": cfg["round2"]["activity"],
        "market_filter": cfg["round2"]["market"],
        "eligibility_filter": cfg["round2"]["eligibility"],
        "anti_lookahead": (
            "Market state uses only contemporaneous/past BTC candles. Eligibility uses "
            "only prior method-A outcomes with exit_time strictly before each signal."
        ),
    }
    (out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
