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
    for f in ARCH_FEATURES:
        z[f] = pd.to_numeric(z[f], errors="coerce").clip(qlo[f], qhi[f])
    return z


def fit_archetypes(anchor_features: pd.DataFrame) -> tuple[ArchModel, pd.DataFrame]:
    tr = anchor_features[(anchor_features.anchor_label=="WINNER") & (anchor_features.split=="DISCOVERY_2019_2022")].copy()
    if len(tr) < 100:
        raise RuntimeError(f"Too few discovery winners for archetypes: {len(tr)}")
    qlo = tr[ARCH_FEATURES].quantile(.01); qhi = tr[ARCH_FEATURES].quantile(.99)
    z = _clip_arch(tr, qlo, qhi)
    imp = SimpleImputer(strategy="median"); sc = StandardScaler()
    X = sc.fit_transform(imp.fit_transform(z))
    km = KMeans(n_clusters=3, random_state=42, n_init=30).fit(X)
    tr = tr.copy(); tr["cluster"] = km.labels_
    med = tr.groupby("cluster")["ret_30d"].median().sort_values()
    cap = int(med.index[0]); mom = int(med.index[-1]); base = int([c for c in med.index if c not in {cap,mom}][0])
    mapping = {cap:"CAPITULATION", base:"BASE", mom:"MOMENTUM"}
    dists = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
    tr["arch_name"] = [mapping[int(c)] for c in km.labels_]
    tr["arch_radius"] = dists
    radius_max = {a: float(g.arch_radius.quantile(.80)) for a,g in tr.groupby("arch_name")}
    summary = tr.groupby(["cluster","arch_name"]).agg(
        discovery_winners=("anchor_id","size"), median_ret_30d=("ret_30d","median"),
        median_range_14d=("range_14d","median"), median_range_30d=("range_30d","median"),
        median_atr7d_pct=("atr7d_pct","median"), median_worst_4h_ret_14d=("worst_4h_ret_14d","median"),
        median_dist_30d_high=("dist_30d_high","median"), radius_80pct=("arch_radius",lambda s: s.quantile(.80)),
    ).reset_index()
    return ArchModel(qlo,qhi,imp,sc,km,mapping,radius_max), summary


def assign_archetypes(frame: pd.DataFrame, model: ArchModel) -> pd.DataFrame:
    out = frame.copy()
    z = _clip_arch(out, model.qlo, model.qhi)
    X = model.scaler.transform(model.imputer.transform(z))
    clusters = model.kmeans.predict(X)
    radii = np.linalg.norm(X - model.kmeans.cluster_centers_[clusters], axis=1)
    out["arch_cluster"] = clusters
    out["archetype"] = [model.cluster_to_name[int(c)] for c in clusters]
    out["arch_radius"] = radii
    return out


def build_anchor_cohort(universe: pd.DataFrame, data: dict[str, tuple[pd.DataFrame,pd.DataFrame]]) -> tuple[pd.DataFrame, list[dict]]:
    symbols = set(universe.symbol)
    events = pd.read_csv(R1 / "events.csv")
    events = events[events.symbol.isin(symbols)].copy()
    events["start_time"] = pd.to_datetime(events.start_time, utc=True)
    prec = pd.read_csv(R1 / "precursor_samples.csv.gz", compression="gzip")
    hard = prec[(prec.label=="HARD_FAIL") & (prec.lead_hours==0) & (prec.symbol.isin(symbols))][["symbol","pseudo_start_time","event_id"]].drop_duplicates(["symbol","pseudo_start_time"])
    hard["pseudo_start_time"] = pd.to_datetime(hard.pseudo_start_time, utc=True)

    rows=[]; failures=[]
    for symbol in universe.symbol:
        if symbol not in data:
            continue
        df, ff = data[symbol]
        idx = pd.Series(df.index.to_numpy(), index=df.time).to_dict()
        for _,e in events[events.symbol==symbol].iterrows():
            i = idx.get(pd.Timestamp(e.start_time))
            if i is None: continue
            path = conservative_anchor_path(df, int(i))
            if path["path_label"] != "WINNER":
                failures.append({"symbol":symbol,"kind":"winner_recheck","time":str(e.start_time),"result":path["path_label"]})
                continue
            r = ff.iloc[int(i)]
            row={"anchor_id":str(e.event_id),"source_event_id":str(e.event_id),"symbol":symbol,"anchor_label":"WINNER",
                 "anchor_time":pd.Timestamp(df.time.iloc[int(i)]),"anchor_index":int(i),"anchor_price":float(df.close.iloc[int(i)]),
                 "year":int(pd.Timestamp(df.time.iloc[int(i)]).year),"split":split_name(int(pd.Timestamp(df.time.iloc[int(i)]).year)),
                 "first5_index":path["first5_index"],"target20_index":path["target20_index"],"stop_index":path["stop_index"]}
            for f in ARCH_FEATURES + ["ret_6h","ret_24h","range_24h","range_7d"]:
                row[f]=float(r[f]) if pd.notna(r[f]) else np.nan
            rows.append(row)
        for _,e in hard[hard.symbol==symbol].iterrows():
            i = idx.get(pd.Timestamp(e.pseudo_start_time))
            if i is None: continue
            path = conservative_anchor_path(df, int(i))
            if path["path_label"] != "HARD_FAIL":
                continue
            r=ff.iloc[int(i)]
            aid=f"HARD:{symbol}:{pd.Timestamp(df.time.iloc[int(i)]).isoformat()}"
            row={"anchor_id":aid,"source_event_id":str(e.event_id),"symbol":symbol,"anchor_label":"HARD_FAIL",
                 "anchor_time":pd.Timestamp(df.time.iloc[int(i)]),"anchor_index":int(i),"anchor_price":float(df.close.iloc[int(i)]),
                 "year":int(pd.Timestamp(df.time.iloc[int(i)]).year),"split":split_name(int(pd.Timestamp(df.time.iloc[int(i)]).year)),
                 "first5_index":path["first5_index"],"target20_index":path["target20_index"],"stop_index":path["stop_index"]}
            for f in ARCH_FEATURES + ["ret_6h","ret_24h","range_24h","range_7d"]:
                row[f]=float(r[f]) if pd.notna(r[f]) else np.nan
            rows.append(row)
    a=pd.DataFrame(rows).drop_duplicates("anchor_id")
    return a, failures


def build_checkpoint_samples(anchors: pd.DataFrame, data: dict[str, tuple[pd.DataFrame,pd.DataFrame]], cfg: dict) -> pd.DataFrame:
    rows=[]; slip=float(cfg["slippage_rate"])
    for symbol,g in anchors.groupby("symbol"):
        if symbol not in data: continue
        df,ff=data[symbol]
        for _,a in g.iterrows():
            start_i=int(a.anchor_index); anchor=float(a.anchor_price)
            for p in PROGRESS_LEVELS:
                di=first_progress_index(df,start_i,p)
                if di is None or di+1>=len(df): continue
                ei=di+1; entry=float(df.open.iloc[ei])*(1+slip)
                entry_progress=entry/anchor-1
                r=ff.iloc[di]
                row={
                    "anchor_id":a.anchor_id,"source_event_id":a.source_event_id,"symbol":symbol,
                    "anchor_label":a.anchor_label,"archetype":a.archetype,"arch_radius":a.arch_radius,
                    "split":a.split,"year":int(a.year),"anchor_time":a.anchor_time,
                    "decision_time":pd.Timestamp(df.time.iloc[di]),"entry_time":pd.Timestamp(df.time.iloc[ei]),
                    "progress_level_pct":p,"hours_since_anchor":int(di-start_i),"entry_price":entry,
                    "entry_progress_pct":entry_progress,"within_5pct_entry":bool(entry_progress<=.05),
                }
                for f in MODEL_FEATURES:
                    if f=="hours_since_anchor": continue
                    v=r[f]; row[f]=float(v) if pd.notna(v) else np.nan
                row.update(outcome_from_entry(df,ei,entry))
                rows.append(row)
    return pd.DataFrame(rows)


def auc_direction(z: pd.DataFrame, feature: str) -> tuple[float,float]:
    a=z[[feature,"anchor_label"]].dropna()
    if len(a)<30 or a.anchor_label.nunique()<2: return np.nan,np.nan
    y=(a.anchor_label=="WINNER").astype(int)
    try: auc=roc_auc_score(y,a[feature])
    except ValueError: return np.nan,np.nan
    return max(float(auc),1-float(auc)), (1.0 if auc>=.5 else -1.0)


def feature_contrasts(samples: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    zall=samples[samples.within_5pct_entry]
    for (arch,p,split),z in zall.groupby(["archetype","progress_level_pct","split"]):
        if arch not in {"CAPITULATION","BASE"}: continue
        for f in [x for x in MODEL_FEATURES if x!="hours_since_anchor"]:
            w=pd.to_numeric(z.loc[z.anchor_label=="WINNER",f],errors="coerce").dropna()
            h=pd.to_numeric(z.loc[z.anchor_label=="HARD_FAIL",f],errors="coerce").dropna()
            if min(len(w),len(h))<15: continue
            auc,dirn=auc_direction(z,f)
            rows.append({"archetype":arch,"progress_level_pct":p,"split":split,"feature":f,
                         "winner_n":len(w),"hard_fail_n":len(h),"winner_median":w.median(),"hard_fail_median":h.median(),
                         "separation_auc":auc,"direction":dirn})
    return pd.DataFrame(rows)


def rule_masks(z: pd.DataFrame, arch: str) -> dict[str,pd.Series]:
    if arch=="CAPITULATION":
        return {
            "GREEN_1H": z.ret_1h>0,
            "GREEN_2H": z.ret_2h>0,
            "REBOUND12_1": z.rebound_12h_low>=.01,
            "REBOUND12_2": z.rebound_12h_low>=.02,
            "EMA6_RECLAIM": z.close_vs_ema6>=0,
            "PREV3H_RECLAIM": z.close_vs_prev3h_high>=0,
            "GREEN_VOL125": (z.ret_1h>0)&(z.volume_ratio_current>=1.25),
            "GREEN_TAKER52": (z.ret_1h>0)&(z.taker_buy_current>=.52),
            "REBOUND2_VOL11": (z.rebound_12h_low>=.02)&(z.volume_ratio_3h>=1.10),
            "MICRO_RECLAIM_VOL12": (z.close_vs_prev3h_high>=0)&(z.volume_ratio_current>=1.20),
            "EMA6_TAKER51": (z.close_vs_ema6>=0)&(z.taker_buy_3h>=.51),
            "GREEN_TRADE125": (z.ret_1h>0)&(z.trade_ratio_current>=1.25),
        }
    return {
        "EXPAND3H_1": z.ret_3h>=.01,
        "EXPAND3H_2": z.ret_3h>=.02,
        "PREV6H_BREAK": z.close_vs_prev6h_high>=0,
        "PREV12H_BREAK": z.close_vs_prev12h_high>=0,
        "PREV24H_BREAK": z.close_vs_prev24h_high>=0,
        "VOL_POP15": (z.volume_ratio_current>=1.50)&(z.ret_1h>0),
        "VOL_TRADE13": (z.volume_ratio_current>=1.30)&(z.trade_ratio_current>=1.30)&(z.ret_1h>0),
        "VOL6_EXPAND": (z.volume_ratio_6h>=1.20)&(z.ret_6h>0),
        "BREAK6_VOL13": (z.close_vs_prev6h_high>=0)&(z.volume_ratio_current>=1.30),
        "BREAK12_VOL13": (z.close_vs_prev12h_high>=0)&(z.volume_ratio_current>=1.30),
        "TAKER_BREAK6": (z.close_vs_prev6h_high>=0)&(z.taker_buy_current>=.52),
        "RISING_LOW_EXPAND": (z.low24_vs_prev24>0)&(z.ret_3h>=.01),
    }


def metric_row(sel: pd.DataFrame, all_z: pd.DataFrame, anchors: pd.DataFrame, cfg: dict) -> dict:
    win_ret, loss_ret = net_binary_returns(cfg)
    total_winners = anchors[(anchors.anchor_label=="WINNER")].anchor_id.nunique()
    total_hards = anchors[(anchors.anchor_label=="HARD_FAIL")].anchor_id.nunique()
    sw=sel[sel.anchor_label=="WINNER"].anchor_id.nunique(); sh=sel[sel.anchor_label=="HARD_FAIL"].anchor_id.nunique()
    p20=float(sel.hit20_before_stop5.mean()) if len(sel) else np.nan
    return {
        "selected":len(sel),"selected_winner_anchors":sw,"selected_hard_fail_anchors":sh,
        "anchor_precision":float((sel.anchor_label=="WINNER").mean()) if len(sel) else np.nan,
        "winner_recall":sw/total_winners if total_winners else np.nan,
        "hard_fail_trigger_rate":sh/total_hards if total_hards else np.nan,
        "entry_success20_precision":p20,
        "hit10_rate":float(sel.hit10_before_stop5.mean()) if len(sel) else np.nan,
        "hit30_rate":float(sel.hit30_before_stop5.mean()) if len(sel) else np.nan,
        "hit50_rate":float(sel.hit50_before_stop5.mean()) if len(sel) else np.nan,
        "binary20_stop5_net_expectancy_pct":p20*win_ret+(1-p20)*loss_ret if len(sel) else np.nan,
        "median_entry_progress_pct":float(sel.entry_progress_pct.median()) if len(sel) else np.nan,
        "median_hours_since_anchor":float(sel.hours_since_anchor.median()) if len(sel) else np.nan,
        "median_success_mae_to20":float(sel.loc[sel.hit20_before_stop5==1,"post_entry_mae_to_20"].median()) if (sel.hit20_before_stop5==1).any() else np.nan,
    }


def evaluate_simple_rules(samples: pd.DataFrame, anchors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows=[]; zall=samples[samples.within_5pct_entry].copy()
    for arch in ["CAPITULATION","BASE"]:
        for p in PROGRESS_LEVELS:
            for split in ["DISCOVERY_2019_2022","VALIDATION_2023_2024","HOLDOUT_2025_2026"]:
                z=zall[(zall.archetype==arch)&(zall.progress_level_pct==p)&(zall.split==split)]
                aa=anchors[(anchors.archetype==arch)&(anchors.split==split)]
                if z.empty or aa.empty: continue
                for name,m in rule_masks(z,arch).items():
                    row={"archetype":arch,"progress_level_pct":p,"split":split,"rule":name}
                    row.update(metric_row(z[m.fillna(False)],z,aa,cfg)); rows.append(row)
    return pd.DataFrame(rows)


def choose_simple(simple: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if simple.empty:
        return pd.DataFrame(rows)
    for arch in ["CAPITULATION","BASE"]:
        d=simple[(simple.archetype==arch)&(simple.split=="DISCOVERY_2019_2022")].copy()
        v=simple[(simple.archetype==arch)&(simple.split=="VALIDATION_2023_2024")].copy()
        key=["archetype","progress_level_pct","rule"]
        x=d.merge(v,on=key,suffixes=("_discovery","_validation"))
        x=x[(x.selected_discovery>=40)&(x.selected_validation>=25)&(x.winner_recall_discovery>=.04)&(x.winner_recall_validation>=.03)]
        if x.empty: continue
        x=x.sort_values(["entry_success20_precision_validation","winner_recall_validation","binary20_stop5_net_expectancy_pct_validation"],ascending=False)
        r=x.iloc[0]
        rows.append({"archetype":arch,"progress_level_pct":float(r.progress_level_pct),"rule":r.rule,
                     "validation_selected":int(r.selected_validation),"validation_precision20":float(r.entry_success20_precision_validation),
                     "validation_winner_recall":float(r.winner_recall_validation),
                     "validation_expectancy_pct":float(r.binary20_stop5_net_expectancy_pct_validation)})
    return pd.DataFrame(rows)


@dataclass
class RFBundle:
    archetype: str
    progress: float
    imputer: SimpleImputer
    classifier: RandomForestClassifier
    threshold: float


def fit_rf_models(samples: pd.DataFrame, anchors: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,pd.DataFrame,dict[tuple[str,float],RFBundle]]:
    rows=[]; imps=[]; bundles={}; zall=samples[samples.within_5pct_entry].copy()
    feats=[f for f in MODEL_FEATURES if f in zall.columns]
    for arch in ["CAPITULATION","BASE"]:
        for p in PROGRESS_LEVELS:
            z=zall[(zall.archetype==arch)&(zall.progress_level_pct==p)].copy()
            tr=z[z.split=="DISCOVERY_2019_2022"]; va=z[z.split=="VALIDATION_2023_2024"]; ho=z[z.split=="HOLDOUT_2025_2026"]
            if min(len(tr),len(va))<80 or tr.hit20_before_stop5.nunique()<2 or va.hit20_before_stop5.nunique()<2: continue
            imp=SimpleImputer(strategy="median")
            Xtr=imp.fit_transform(tr[feats]); Xva=imp.transform(va[feats])
            clf=RandomForestClassifier(n_estimators=450,max_depth=5,min_samples_leaf=25,class_weight="balanced",random_state=42,n_jobs=-1)
            clf.fit(Xtr,tr.hit20_before_stop5)
            pva=clf.predict_proba(Xva)[:,1]
            winner_total=anchors[(anchors.archetype==arch)&(anchors.split=="VALIDATION_2023_2024")&(anchors.anchor_label=="WINNER")].anchor_id.nunique()
            candidates=[]
            for th in np.unique(np.quantile(pva,np.linspace(.50,.97,20))):
                m=pva>=th; n=int(m.sum())
                if n<25: continue
                sel=va[m]; sw=sel[sel.anchor_label=="WINNER"].anchor_id.nunique(); recall=sw/winner_total if winner_total else 0
                if recall<.03: continue
                candidates.append((float(sel.hit20_before_stop5.mean()),recall,float(th),n))
            if not candidates: continue
            candidates.sort(key=lambda x:(x[0],x[1],x[3]),reverse=True)
            _,_,th,_=candidates[0]
            bundles[(arch,p)]=RFBundle(arch,p,imp,clf,th)
            for split,frame in [("VALIDATION_2023_2024",va),("HOLDOUT_2025_2026",ho)]:
                if frame.empty: continue
                pp=clf.predict_proba(imp.transform(frame[feats]))[:,1]; sel=frame[pp>=th]
                aa=anchors[(anchors.archetype==arch)&(anchors.split==split)]
                row={"archetype":arch,"progress_level_pct":p,"split":split,"threshold":th,"samples":len(frame)}
                row.update(metric_row(sel,frame,aa,cfg)); rows.append(row)
            for f,v in zip(feats,clf.feature_importances_):
                imps.append({"archetype":arch,"progress_level_pct":p,"feature":f,"importance":float(v)})
    return pd.DataFrame(rows),pd.DataFrame(imps),bundles


def choose_models(modelq: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    if modelq.empty:
        return pd.DataFrame(rows)
    for arch in ["CAPITULATION","BASE"]:
        v=modelq[(modelq.archetype==arch)&(modelq.split=="VALIDATION_2023_2024")].copy()
        v=v[(v.selected>=25)&(v.winner_recall>=.03)]
        if v.empty: continue
        v=v.sort_values(["entry_success20_precision","winner_recall","binary20_stop5_net_expectancy_pct"],ascending=False)
        r=v.iloc[0]
        rows.append({"archetype":arch,"progress_level_pct":float(r.progress_level_pct),"threshold":float(r.threshold),
                     "validation_selected":int(r.selected),"validation_precision20":float(r.entry_success20_precision),
                     "validation_winner_recall":float(r.winner_recall),"validation_expectancy_pct":float(r.binary20_stop5_net_expectancy_pct)})
    return pd.DataFrame(rows)


def discovery_watch_gate_params(anchors: pd.DataFrame, arch_model: ArchModel) -> pd.DataFrame:
    rows=[]
    for arch in ["CAPITULATION","BASE"]:
        g=anchors[(anchors.anchor_label=="WINNER")&(anchors.split=="DISCOVERY_2019_2022")&(anchors.archetype==arch)].copy()
        if g.empty: continue
        row={"archetype":arch,"radius_max":arch_model.radius_max.get(arch,np.nan),"discovery_winners":len(g)}
        if arch=="CAPITULATION":
            row.update({"ret6_max":g.ret_6h.quantile(.75),"ret24_max":g.ret_24h.quantile(.75),"range24_min":g.range_24h.quantile(.25)})
        else:
            row.update({"range7d_max":g.range_7d.quantile(.75),"atr7d_max":g.atr7d_pct.quantile(.75),
                        "abs_ret30_max":g.ret_30d.abs().quantile(.75),"dist30_min":g.dist_30d_high.quantile(.10)})
        rows.append(row)
    return pd.DataFrame(rows)


def apply_arch_to_feature_frame(ff: pd.DataFrame, model: ArchModel) -> pd.DataFrame:
    out=ff.copy(); valid=out[ARCH_FEATURES].notna().sum(axis=1)>=len(ARCH_FEATURES)-1
    out["rt_arch"]="UNKNOWN"; out["rt_arch_radius"]=np.nan
    if valid.any():
        z=_clip_arch(out.loc[valid],model.qlo,model.qhi)
        X=model.scaler.transform(model.imputer.transform(z)); cl=model.kmeans.predict(X)
        rr=np.linalg.norm(X-model.kmeans.cluster_centers_[cl],axis=1)
        out.loc[valid,"rt_arch"]=[model.cluster_to_name[int(c)] for c in cl]
        out.loc[valid,"rt_arch_radius"]=rr
    return out


def watch_gate(row: pd.Series, arch: str, params: pd.Series) -> bool:
    if row.rt_arch!=arch or not np.isfinite(row.rt_arch_radius) or row.rt_arch_radius>params.radius_max:
        return False
    if arch=="CAPITULATION":
        return bool(row.ret_6h<=params.ret6_max and row.ret_24h<=params.ret24_max and row.range_24h>=params.range24_min)
    return bool(row.range_7d<=params.range7d_max and row.atr7d_pct<=params.atr7d_max and abs(row.ret_30d)<=params.abs_ret30_max and row.dist_30d_high>=params.dist30_min)


def rule_pass_row(row: pd.Series, arch: str, rule: str) -> bool:
    z=pd.DataFrame([row])
    m=rule_masks(z,arch).get(rule)
    return bool(m.iloc[0]) if m is not None and len(m) else False


def replay_one(df: pd.DataFrame, ff: pd.DataFrame, arch: str, method: str, progress: float,
               config: dict, params: pd.Series, cfg: dict, rf_bundle: RFBundle | None,
               split: str) -> list[dict]:
    slip=float(cfg["slippage_rate"]); feats=list(MODEL_FEATURES)
    if split=="VALIDATION_2023_2024": y0,y1=2023,2024
    else: y0,y1=2025,2026
    mask=(pd.to_datetime(df.time,utc=True).dt.year>=y0)&(pd.to_datetime(df.time,utc=True).dt.year<=y1)
    ids=np.flatnonzero(mask.to_numpy())
    if len(ids)<2: return []
    start=max(int(ids[0]),2160); end=min(int(ids[-1]),len(df)-2)
    alerts=[]; i=start; cooldown_until=start
    max_watch=72 if arch=="CAPITULATION" else 168
    while i<=end:
        if i<cooldown_until:
            i+=1; continue
        r=ff.iloc[i]
        if not watch_gate(r,arch,params):
            i+=1; continue
        anchor_i=i; anchor=float(df.close.iloc[i]); deadline=min(end,i+max_watch); decision_i=None
        if progress<=0:
            decision_i=i
        else:
            for j in range(i+1,deadline+1):
                if float(df.low.iloc[j])<=anchor*(1-STOP_PCT):
                    break
                if float(df.high.iloc[j])>=anchor*(1+progress):
                    decision_i=j; break
        if decision_i is None:
            i=deadline+1; cooldown_until=i+24; continue
        row=ff.iloc[decision_i]; passed=False; prob=np.nan
        if method=="SIMPLE":
            passed=rule_pass_row(row,arch,config["rule"])
        else:
            if rf_bundle is not None:
                one=pd.DataFrame([{f:(float(row[f]) if f!="hours_since_anchor" and pd.notna(row[f]) else (decision_i-anchor_i if f=="hours_since_anchor" else np.nan)) for f in feats}])
                prob=float(rf_bundle.classifier.predict_proba(rf_bundle.imputer.transform(one[feats]))[0,1])
                passed=prob>=rf_bundle.threshold
        if passed and decision_i+1<len(df):
            ei=decision_i+1; entry=float(df.open.iloc[ei])*(1+slip); adv=entry/anchor-1
            if adv<=.05:
                out=outcome_from_entry(df,ei,entry)
                alerts.append({"method":method,"archetype":arch,"split":split,"symbol":str(config["symbol"]),
                               "watch_start":pd.Timestamp(df.time.iloc[anchor_i]),"decision_time":pd.Timestamp(df.time.iloc[decision_i]),
                               "entry_time":pd.Timestamp(df.time.iloc[ei]),"watch_anchor_price":anchor,"entry_price":entry,
                               "progress_level_pct":progress,"entry_progress_from_watch_pct":adv,
                               "hours_watch_to_entry":decision_i-anchor_i,"model_probability":prob,**out})
                i=decision_i+1; cooldown_until=i+72; continue
        i=decision_i+1; cooldown_until=i+24
    return alerts


def link_alerts_to_events(alerts: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    if alerts.empty: return alerts
    out=alerts.copy(); out["captured_event_id"]=""; out["event_entry_lateness_pct"]=np.nan
    winners=anchors[anchors.anchor_label=="WINNER"].copy()
    for idx,r in out.iterrows():
        g=winners[(winners.symbol==r.symbol)&(winners.archetype==r.archetype)&(winners.split==r.split)]
        if g.empty: continue
        t=pd.Timestamp(r.decision_time)
        cands=[]
        for _,e in g.iterrows():
            if pd.Timestamp(e.anchor_time)<=t:
                first5_time=None
                if pd.notna(e.first5_index):
                    # first5_index is symbol-local and not directly timestamped here; approximate with hours from the stored frame later unavailable.
                    hours5=(int(e.first5_index)-int(e.anchor_index))
                    first5_time=pd.Timestamp(e.anchor_time)+pd.Timedelta(hours=hours5)
                if first5_time is not None and t<=first5_time:
                    lateness=float(r.entry_price)/float(e.anchor_price)-1
                    if lateness<=.05: cands.append((abs(lateness),e.anchor_id,lateness))
        if cands:
            cands.sort(); _,eid,late=cands[0]
            out.at[idx,"captured_event_id"]=eid; out.at[idx,"event_entry_lateness_pct"]=late
    return out


def replay_summary(alerts: pd.DataFrame, anchors: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows=[]; win_ret,loss_ret=net_binary_returns(cfg)
    for (method,arch,split),g in alerts.groupby(["method","archetype","split"]):
        if split=="VALIDATION_2023_2024": start,end=pd.Timestamp("2023-01-01",tz="UTC"),pd.Timestamp("2024-12-31 23:00",tz="UTC")
        else: start,end=pd.Timestamp("2025-01-01",tz="UTC"),pd.Timestamp("2026-08-31 23:00",tz="UTC")
        weeks=(end-start).total_seconds()/(7*24*3600)
        clean=anchors[(anchors.anchor_label=="WINNER")&(anchors.archetype==arch)&(anchors.split==split)].anchor_id.nunique()
        captured=g.loc[g.captured_event_id!="","captured_event_id"].nunique()
        p20=float(g.hit20_before_stop5.mean()) if len(g) else np.nan
        coin_share=float(g.symbol.value_counts(normalize=True).iloc[0]) if len(g) else np.nan
        rows.append({"method":method,"archetype":arch,"split":split,"alerts":len(g),"alerts_per_week":len(g)/weeks,
                     "false_alerts":int((g.hit20_before_stop5==0).sum()),"false_alerts_per_week":float((g.hit20_before_stop5==0).sum()/weeks),
                     "precision20_before_stop5":p20,"hit10_rate":g.hit10_before_stop5.mean(),"hit30_rate":g.hit30_before_stop5.mean(),"hit50_rate":g.hit50_before_stop5.mean(),
                     "binary20_stop5_net_expectancy_pct":p20*win_ret+(1-p20)*loss_ret,
                     "clean_events":clean,"captured_clean_events":captured,"clean_event_recall":captured/clean if clean else np.nan,
                     "median_event_entry_lateness_pct":g.loc[g.captured_event_id!="","event_entry_lateness_pct"].median(),
                     "median_success_mae_to20":g.loc[g.hit20_before_stop5==1,"post_entry_mae_to_20"].median(),
                     "median_hours_to_20":g.loc[g.hit20_before_stop5==1,"hours_to_20_from_entry"].median(),
                     "max_coin_alert_share":coin_share})
    return pd.DataFrame(rows)


def run_causal_replay(universe: pd.DataFrame, data: dict[str, tuple[pd.DataFrame,pd.DataFrame]], arch_model: ArchModel,
                      gate_params: pd.DataFrame, selected_simple: pd.DataFrame, selected_models: pd.DataFrame,
                      bundles: dict[tuple[str,float],RFBundle], anchors: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame,pd.DataFrame]:
    all_alerts=[]; params_map={r.archetype:r for _,r in gate_params.iterrows()}
    configs=[]
    for _,r in selected_simple.iterrows():
        configs.append(("SIMPLE",r.archetype,float(r.progress_level_pct),{"rule":r.rule}))
    for _,r in selected_models.iterrows():
        configs.append(("RF",r.archetype,float(r.progress_level_pct),{"threshold":float(r.threshold)}))
    for symbol in universe.symbol:
        if symbol not in data: continue
        df,ff0=data[symbol]; ff=apply_arch_to_feature_frame(ff0,arch_model)
        for method,arch,p,c in configs:
            if arch not in params_map: continue
            cc=dict(c); cc["symbol"]=symbol
            rb=bundles.get((arch,p)) if method=="RF" else None
            for split in ["VALIDATION_2023_2024","HOLDOUT_2025_2026"]:
                all_alerts.extend(replay_one(df,ff,arch,method,p,cc,params_map[arch],cfg,rb,split))
    alerts=pd.DataFrame(all_alerts)
    if alerts.empty: return alerts,pd.DataFrame()
    alerts=link_alerts_to_events(alerts,anchors)
    return alerts,replay_summary(alerts,anchors,cfg)


def write_analysis(universe: pd.DataFrame, anchors: pd.DataFrame, arch_summary: pd.DataFrame,
                   simple: pd.DataFrame, selected_simple: pd.DataFrame, modelq: pd.DataFrame,
                   selected_models: pd.DataFrame, replay: pd.DataFrame, cfg: dict) -> None:
    lines=["# Explosive Move Entry Discovery Round 2","",
           "**Status:** RESEARCH ONLY. No Early Breakout automation or Crypto Live Strategy Specification changes.","",
           "## Design","",
           "Round 2 keeps the Round 1 clean-event definition (+20% before -5%, maximum 30 days) but tests causal entry decisions separately for CAPITULATION/REVERSAL and BASE/ACCUMULATION. The Round 1 all-period cluster IDs are not reused: archetypes are refit using 2019-2022 discovery winners only, then frozen for 2023-2026 assignment.","",
           "The primary matched comparison is winner vs same-symbol/year HARD FAIL controls from Round 1, with hard controls rechecked to require +5% before a later -5% stop and no +20% first. Candidate decisions are observed at the first 0%, +1%, +2%, +3% and +5% progress checkpoints; decisions use only data known at that hourly close and entries occur at the next hourly open plus configured slippage.","",
           "Simple interpretable reversal/expansion rules are tested first. A shallow Random Forest is then fitted on discovery only; probability thresholds/checkpoints are selected on 2023-2024 validation. 2025-2026 is only reported after selection. A separate whole-history causal replay starts watches from discovery-derived archetype gates and measures real alert frequency and false alerts.","",
           f"Frozen Round 1 ranking was filtered to the first {len(universe)} eligible pairs after removing stablecoins, known unavailable pairs, leveraged-token patterns and identified tokenised-equity symbols.","",
           "## Anchor cohort","",
           "| Archetype | Split | Winners | Hard failures |","|---|---|---:|---:|"]
    ac=anchors[anchors.archetype.isin(["CAPITULATION","BASE"])].groupby(["archetype","split","anchor_label"]).size().unstack(fill_value=0).reset_index()
    for _,r in ac.iterrows():
        lines.append(f"| {r.archetype} | {r.split} | {int(r.get('WINNER',0))} | {int(r.get('HARD_FAIL',0))} |")
    lines += ["","## Discovery-only archetype fit","",
              "| Cluster | Name | Discovery winners | Median 30d return | Median 14d range | Median 7d ATR |","|---:|---|---:|---:|---:|---:|"]
    for _,r in arch_summary.iterrows():
        lines.append(f"| {int(r.cluster)} | {r.arch_name} | {int(r.discovery_winners)} | {100*r.median_ret_30d:.1f}% | {100*r.median_range_14d:.1f}% | {100*r.median_atr7d_pct:.2f}% |")

    lines += ["","## Validation-selected simple triggers","",
              "| Archetype | Progress | Rule | Validation selected | +20/-5 precision | Winner recall | Binary expectancy* |","|---|---:|---|---:|---:|---:|---:|"]
    for _,r in selected_simple.iterrows():
        lines.append(f"| {r.archetype} | {100*r.progress_level_pct:.0f}% | {r.rule} | {int(r.validation_selected)} | {100*r.validation_precision20:.1f}% | {100*r.validation_winner_recall:.1f}% | {100*r.validation_expectancy_pct:.1f}% |")
    lines += ["","## Validation-selected model triggers","",
              "| Archetype | Progress | Validation selected | +20/-5 precision | Winner recall | Binary expectancy* |","|---|---:|---:|---:|---:|---:|"]
    for _,r in selected_models.iterrows():
        lines.append(f"| {r.archetype} | {100*r.progress_level_pct:.0f}% | {int(r.validation_selected)} | {100*r.validation_precision20:.1f}% | {100*r.validation_winner_recall:.1f}% | {100*r.validation_expectancy_pct:.1f}% |")

    lines += ["","## Untouched 2025-2026 matched-cohort results","",
              "| Method | Archetype | Progress | Selected | +20/-5 precision | Winner recall | Median entry advance | Binary expectancy* |","|---|---|---:|---:|---:|---:|---:|---:|"]
    for method,sel,table,keycols in [("SIMPLE",selected_simple,simple,["archetype","progress_level_pct","rule"]),("RF",selected_models,modelq,["archetype","progress_level_pct"])]:
        for _,s in sel.iterrows():
            q=table[(table.split=="HOLDOUT_2025_2026")&(table.archetype==s.archetype)&(table.progress_level_pct==s.progress_level_pct)]
            if method=="SIMPLE": q=q[q.rule==s.rule]
            if q.empty: continue
            r=q.iloc[0]
            lines.append(f"| {method} | {s.archetype} | {100*s.progress_level_pct:.0f}% | {int(r.selected)} | {100*r.entry_success20_precision:.1f}% | {100*r.winner_recall:.1f}% | {100*r.median_entry_progress_pct:.1f}% | {100*r.binary20_stop5_net_expectancy_pct:.1f}% |")

    lines += ["","## Causal replay","",
              "This replay does not start from hindsight event timestamps. It starts a watch only when the frozen discovery-era archetype gate becomes observable, evaluates the selected trigger once at its specified progress checkpoint, enters next-hour open, and applies no exit optimisation. This is the main false-alert stress test.","",
              "| Method | Archetype | Split | Alerts/wk | False/wk | +20/-5 precision | Clean-event recall | Median captured lateness | Binary expectancy* |","|---|---|---|---:|---:|---:|---:|---:|---:|"]
    if replay.empty:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    else:
        for _,r in replay.iterrows():
            lines.append(f"| {r.method} | {r.archetype} | {r.split} | {r.alerts_per_week:.2f} | {r.false_alerts_per_week:.2f} | {100*r.precision20_before_stop5:.1f}% | {100*r.clean_event_recall:.1f}% | {100*r.median_event_entry_lateness_pct:.1f}% | {100*r.binary20_stop5_net_expectancy_pct:.1f}% |")

    win_ret,loss_ret=net_binary_returns(cfg)
    hurdle=(.10-loss_ret)/(win_ret-loss_ret)
    lines += ["","## Interpretation guardrails","",
              f"*Binary expectancy assumes every success exits at +20% and every non-success at -5%, with configured fee/slippage applied. Under that deliberately simple framework, roughly {100*hurdle:.1f}% +20-before--5 precision is required to clear a +10% mean net-return hurdle. It is an entry-quality diagnostic, not an optimised exit test.","",
              "- Primary discrimination remains winner vs HARD FAIL, not winner vs random quiet periods.",
              "- 2025-2026 was not used to choose rule, checkpoint, model threshold or watch-gate parameter.",
              "- Whole-market replay includes ordinary non-event periods, so its false-alert rate is more realistic than matched-control precision.",
              "- The frozen current-liquidity universe still has survivorship bias. Historical point-in-time liquidity reconstruction remains a later robustness task.",
              "- No result is promoted to the live strategy by this workflow.",""]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))


def main():
    cfg=json.loads(Path("research/config.json").read_text())
    OUT.mkdir(parents=True,exist_ok=True)
    universe=load_frozen_top50(); universe.to_csv(OUT/"universe.csv",index=False)
    btc=load_symbol("BTCUSDT","1h",START,END)

    data={}; failures=[]
    def load_one(symbol: str):
        df=load_symbol(symbol,"1h",START,END)
        if len(df)<3000: return symbol,None,None,f"insufficient_history:{len(df)}"
        return symbol,df,build_features(df,btc),None
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(load_one,s):s for s in universe.symbol}
        for fut in as_completed(futs):
            s=futs[fut]
            try:
                symbol,df,ff,err=fut.result()
                if err: failures.append({"symbol":symbol,"kind":"load","error":err})
                else: data[symbol]=(df,ff)
                print(symbol,"rows",0 if df is None else len(df),err or "OK",flush=True)
            except Exception as e:
                failures.append({"symbol":s,"kind":"load","error":repr(e)}); print("ERROR",s,repr(e),flush=True)

    anchors,recheck_fail=build_anchor_cohort(universe,data); failures.extend(recheck_fail)
    arch_model,arch_summary=fit_archetypes(anchors)
    anchors=assign_archetypes(anchors,arch_model)
    anchors.to_csv(OUT/"anchor_cohort.csv",index=False)
    arch_summary.to_csv(OUT/"archetype_summary.csv",index=False)

    samples=build_checkpoint_samples(anchors,data,cfg)
    samples.to_csv(OUT/"checkpoint_samples.csv.gz",index=False,compression="gzip")
    feature_contrasts(samples).to_csv(OUT/"feature_contrasts.csv",index=False)

    simple=evaluate_simple_rules(samples,anchors,cfg); simple.to_csv(OUT/"simple_rule_metrics.csv",index=False)
    selected_simple=choose_simple(simple); selected_simple.to_csv(OUT/"selected_simple.csv",index=False)

    modelq,imp,bundles=fit_rf_models(samples,anchors,cfg)
    modelq.to_csv(OUT/"model_quality.csv",index=False); imp.to_csv(OUT/"model_feature_importance.csv",index=False)
    selected_models=choose_models(modelq); selected_models.to_csv(OUT/"selected_models.csv",index=False)

    gates=discovery_watch_gate_params(anchors,arch_model); gates.to_csv(OUT/"watch_gate_params.csv",index=False)
    alerts,replay=run_causal_replay(universe,data,arch_model,gates,selected_simple,selected_models,bundles,anchors,cfg)
    alerts.to_csv(OUT/"causal_replay_alerts.csv",index=False); replay.to_csv(OUT/"causal_replay_summary.csv",index=False)
    if not alerts.empty:
        alerts.assign(year=pd.to_datetime(alerts.entry_time,utc=True).dt.year).groupby(["method","archetype","split","year"]).agg(
            alerts=("symbol","size"), precision20=("hit20_before_stop5","mean"),
            mean_entry_progress_pct=("entry_progress_from_watch_pct","mean"),
            median_hours_to_20=("hours_to_20_from_entry","median"),
        ).reset_index().to_csv(OUT/"causal_replay_year_summary.csv",index=False)
        alerts.groupby(["method","archetype","split","symbol"]).agg(
            alerts=("symbol","size"), precision20=("hit20_before_stop5","mean"),
            false_alerts=("hit20_before_stop5",lambda x:int((x==0).sum())),
        ).reset_index().to_csv(OUT/"causal_replay_coin_summary.csv",index=False)
    else:
        pd.DataFrame().to_csv(OUT/"causal_replay_year_summary.csv",index=False)
        pd.DataFrame().to_csv(OUT/"causal_replay_coin_summary.csv",index=False)
    pd.DataFrame(failures).to_csv(OUT/"failures.csv",index=False)

    write_analysis(universe,anchors,arch_summary,simple,selected_simple,modelq,selected_models,replay,cfg)
    manifest={
        "study":"Explosive Move Entry Discovery Round 2",
        "status":"RESEARCH ONLY - no live strategy or automation edits",
        "source":"Explosive Move Discovery Round 1 frozen universe/events/hard-failure samples",
        "universe":"First 50 eligible pairs from Round 1 current-liquidity ranking after additional stablecoin/tokenised-equity/known-unavailable filtering; survivorship bias remains.",
        "history":{"start":START,"end":END,"interval":"1h"},
        "event_definition":{"target_pct":TARGET_PCT,"stop_pct":STOP_PCT,"max_hours":MAX_HOURS},
        "hard_failure_recheck":"Round 1 HARD FAIL controls retained only if +5% occurs before a later -5% stop and +20% does not occur first.",
        "archetypes":"KMeans(3) fitted on winsorised/standardised 2019-2022 winner context only; lowest median 30d-return cluster=CAPITULATION, highest=MOMENTUM, remaining=BASE. Momentum is not entry-modelled in this round.",
        "progress_checkpoints_pct":[0,1,2,3,5],
        "execution":"Decision at completed hourly close; entry at next hourly open plus configured slippage; entries above +5% from watch/event anchor excluded.",
        "simple_rules":"Pre-specified interpretable reversal/reclaim rules for capitulation and price-expansion/participation rules for base.",
        "model":{"type":"RandomForestClassifier","n_estimators":450,"max_depth":5,"min_samples_leaf":25,"class_weight":"balanced","fit":"2019-2022","threshold_selection":"2023-2024 validation only; max +20-before--5 precision subject to >=25 selected and >=3% clean-winner recall per checkpoint"},
        "selection":"Best simple rule/checkpoint and RF checkpoint per archetype selected on 2023-2024 validation only. 2025-2026 untouched until reporting.",
        "causal_replay":"Discovery-derived archetype radius/context gates create watches without hindsight event timestamps; validation-selected trigger evaluated once at its progress checkpoint; false alerts measured on all replay alerts.",
        "watch_max_hours":{"CAPITULATION":72,"BASE":168},
        "costs":{"fee_rate":cfg["fee_rate"],"slippage_rate":cfg["slippage_rate"]},
        "economic_hurdle":"Approximately +10% mean net return per accepted trade; binary +20/-5 expectancy is reported as an entry-quality diagnostic only.",
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2))

    print("UNIVERSE",len(universe),"LOADED",len(data),"ANCHORS",len(anchors),"SAMPLES",len(samples),"FAILURES",len(failures))
    print("SELECTED SIMPLE\n",selected_simple.to_string(index=False))
    print("SELECTED MODELS\n",selected_models.to_string(index=False))
    print("REPLAY\n",replay.to_string(index=False) if not replay.empty else "none")

if __name__=="__main__":
    main()
