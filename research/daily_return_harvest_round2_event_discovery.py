from __future__ import annotations

import json
import math
import re
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from binance_data import load_symbol
from early_breakout_round1 import TASK_EXCLUSIONS

OUT = Path("research/results/daily_return_harvest/round2_event_discovery")
TMP = Path("research/cache/daily_return_harvest_round2")
INTERVAL = "1h"
DATA_START = "2021-10-24"
DATA_END = "2026-09-30"
DISCOVERY_START = pd.Timestamp("2021-11-01", tz="UTC")
DISCOVERY_END = pd.Timestamp("2024-05-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-01-01", tz="UTC")
CONFIRMATION_END = pd.Timestamp("2026-09-28", tz="UTC")

UNIVERSE_SIZE = 100
MIN_DISCOVERY_DAYS = 90
MIN_QUOTE_VOL_24H = 1_000_000.0
EVENT_GAP_HOURS = 12
WINNER_BUFFER_HOURS = 72
SAME_COIN_CONTROLS_PER_EVENT = 2
SAME_TIME_CONTROLS_PER_EVENT = 1
HARD_NEGATIVES_PER_EVENT = 1
BASELINE_RANDOM_PER_SYMBOL_MONTH = 8
FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0005
MAX_FORWARD_HOURS = 72
SNAPSHOT_OFFSETS = [0, -1, -2, -3, -6, -12, -24, -48, -72, -168]
DELTA_HOURS = [1, 3, 6, 12, 24, 72, 168]

STABLE_BASES = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI", "BUSD", "EUR", "EURI", "AEUR",
    "TRY", "BRL", "GBP", "AUD", "UAH", "RUB", "BIDR", "IDRT", "NGN", "ZAR",
    "VAI", "UST", "USTC", "USD1", "RLUSD", "XUSD", "USDE", "USDS", "PYUSD",
    "EURC", "BFUSD",
}
TOKENISED_EQUITY_BASES = {"MSTRB", "CRCLB", "SPCXB"}
ACCOUNT_EXCLUSIONS = set(TASK_EXCLUSIONS) | {
    "USUALUSDT", "CFGUSDT", "COWUSDT", "WUSDT", "PYTHUSDT", "STRKUSDT",
    "GPSUSDT", "BBUSDT", "KMNOUSDT", "JTOUSDT", "RESOLVUSDT", "TNSRUSDT",
}

PATH_COLS = [
    "w3_2_24h", "w5_2_48h", "w5_3_48h", "w7_3_72h", "w10_4_72h",
    "target3_hour", "target5_hour", "target7_hour", "target10_hour",
    "stop2_hour", "stop3_hour", "stop4_hour",
    "mfe_24h", "mae_24h", "mfe_48h", "mae_48h", "mfe_72h", "mae_72h",
    "end_net_3h", "end_net_6h", "end_net_12h", "end_net_24h",
    "end_net_48h", "end_net_72h",
]

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


def split_name(ts: pd.Timestamp) -> str:
    t = pd.Timestamp(ts)
    if t < DISCOVERY_END:
        return "DISCOVERY_2021-11_2024-04"
    if t < VALIDATION_END:
        return "VALIDATION_2024-05_2025-12"
    return "CONFIRMATION_2026"


def eligible_symbol(sym: str) -> bool:
    if not sym.endswith("USDT") or not sym.isascii():
        return False
    base = sym[:-4]
    if base in STABLE_BASES or base in TOKENISED_EQUITY_BASES:
        return False
    return re.search(r"(UP|DOWN|BULL|BEAR)USDT$", sym) is None


def load_candidate_pool() -> pd.DataFrame:
    symbols = set()
    for path in [
        Path("research/results/daily_return_harvest/round1/universe.csv"),
        Path("research/results/explosive_move_discovery_round1/universe.csv"),
    ]:
        if path.exists():
            u = pd.read_csv(path)
            if "symbol" in u:
                symbols.update(u.symbol.astype(str))
    cfg = Path("research/config.json")
    if cfg.exists():
        symbols.update(map(str, json.loads(cfg.read_text()).get("symbols", [])))
    cache = Path("research/cache/binance")
    if cache.exists():
        symbols.update(p.name for p in cache.iterdir() if p.is_dir())
    rows = []
    for sym in sorted(s for s in symbols if eligible_symbol(s)):
        rows.append({"symbol": sym, "base_asset": sym[:-4], "account_excluded": sym in ACCOUNT_EXCLUSIONS})
    return pd.DataFrame(rows)


def scan_liquidity(meta: pd.Series) -> dict:
    df = load_symbol(meta.symbol, INTERVAL, DATA_START, "2024-04-30")
    z = df[(df.time >= DISCOVERY_START) & (df.time < DISCOVERY_END)].copy()
    daily = z.assign(day=z.time.dt.floor("D")).groupby("day").quote_volume.sum() if len(z) else pd.Series(dtype=float)
    qv24 = z.quote_volume.rolling(24).sum() if len(z) else pd.Series(dtype=float)
    return {
        "symbol": meta.symbol, "base_asset": meta.base_asset, "account_excluded": bool(meta.account_excluded),
        "hours": len(z), "active_days": int(daily.size),
        "median_daily_quote_volume": float(daily.median()) if len(daily) else np.nan,
        "median_trailing_24h_quote_volume": float(qv24.median()) if len(qv24) else np.nan,
    }


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
    vol_base, trade_base = x.volume.shift(1).rolling(72).mean(), x.trades.shift(1).rolling(72).mean()
    for n, label in [(1, "current"), (3, "3h"), (6, "6h"), (12, "12h"), (24, "24h")]:
        vv = x.volume if n == 1 else x.volume.rolling(n).mean()
        tt = x.trades if n == 1 else x.trades.rolling(n).mean()
        x[f"volume_ratio_{label}"] = vv / vol_base
        x[f"trade_ratio_{label}"] = tt / trade_base
    taker = x.taker_buy_base / x.volume.replace(0, np.nan)
    x["taker_buy_current"] = taker
    for n in [3, 6, 12, 24]: x[f"taker_buy_{n}h"] = taker.rolling(n).mean()
    x["quote_volume_24h"] = x.quote_volume.rolling(24).sum()
    x["quote_volume_7d"] = x.quote_volume.rolling(168).sum()
    x["green_body_pct"] = c / o - 1
    x["abs_body_pct"] = (c-o).abs() / o.replace(0, np.nan)
    cr = (h-l).replace(0, np.nan)
    x["close_location"] = (c-l) / cr
    x["lower_wick_share"] = (np.minimum(o, c)-l) / cr
    x["upper_wick_share"] = (h-np.maximum(o, c)) / cr
    for n in [1, 6, 24, 72, 168]: x[f"btc_ret_{n}h"] = b.pct_change(n)
    bt = pd.concat([bsrc.high-bsrc.low, (bsrc.high-bsrc.close.shift()).abs(), (bsrc.low-bsrc.close.shift()).abs()], axis=1).max(axis=1)
    x["btc_atr24_pct"] = (bt.rolling(24).mean()/bsrc.close).reindex(x.index)
    x["btc_vs_ema168"] = (bsrc.close/bsrc.close.ewm(span=168, adjust=False).mean()-1).reindex(x.index)
    return x.reset_index()


@njit
def _classify_paths(open_, high, low, close):
    n = len(close)
    out = np.full((n, 25), np.nan)
    for i in range(n-1):
        ei = i + 1
        raw = open_[ei]
        if not np.isfinite(raw) or raw <= 0: continue
        entry_cash = raw * (1.0 + SLIPPAGE_RATE) * (1.0 + FEE_RATE)
        exit_factor = (1.0 - SLIPPAGE_RATE) * (1.0 - FEE_RATE)
        t3, t5, t7, t10 = [entry_cash * m / exit_factor for m in (1.03, 1.05, 1.07, 1.10)]
        s2, s3, s4 = [entry_cash * m / exit_factor for m in (0.98, 0.97, 0.96)]
        ht3=ht5=ht7=ht10=hs2=hs3=hs4=-1
        max24=max48=max72=-1e300
        min24=min48=min72=1e300
        end = min(n, ei + MAX_FORWARD_HOURS)
        for j in range(ei, end):
            off, hi, lo = j-ei, high[j], low[j]
            if ht3 < 0 and hi >= t3: ht3 = off
            if ht5 < 0 and hi >= t5: ht5 = off
            if ht7 < 0 and hi >= t7: ht7 = off
            if ht10 < 0 and hi >= t10: ht10 = off
            if hs2 < 0 and lo <= s2: hs2 = off
            if hs3 < 0 and lo <= s3: hs3 = off
            if hs4 < 0 and lo <= s4: hs4 = off
            if off < 24:
                max24=max(max24,hi); min24=min(min24,lo)
            if off < 48:
                max48=max(max48,hi); min48=min(min48,lo)
            if off < 72:
                max72=max(max72,hi); min72=min(min72,lo)
        out[i,0] = 1.0 if ht3>=0 and ht3<24 and (hs2<0 or ht3<hs2) else 0.0
        out[i,1] = 1.0 if ht5>=0 and ht5<48 and (hs2<0 or ht5<hs2) else 0.0
        out[i,2] = 1.0 if ht5>=0 and ht5<48 and (hs3<0 or ht5<hs3) else 0.0
        out[i,3] = 1.0 if ht7>=0 and ht7<72 and (hs3<0 or ht7<hs3) else 0.0
        out[i,4] = 1.0 if ht10>=0 and ht10<72 and (hs4<0 or ht10<hs4) else 0.0
        hits = (ht3,ht5,ht7,ht10,hs2,hs3,hs4)
        for k in range(7): out[i,5+k] = hits[k] if hits[k]>=0 else np.nan
        entry_exec = raw*(1.0+SLIPPAGE_RATE)
        out[i,12]=max24/entry_exec-1.0; out[i,13]=min24/entry_exec-1.0
        out[i,14]=max48/entry_exec-1.0; out[i,15]=min48/entry_exec-1.0
        out[i,16]=max72/entry_exec-1.0; out[i,17]=min72/entry_exec-1.0
        for k, hrs in enumerate((3,6,12,24,48,72)):
            xi = ei + hrs - 1
            if xi < n: out[i,18+k] = close[xi]*exit_factor/entry_cash-1.0
        out[i,24]=1.0
    return out


def classify_paths(ff: pd.DataFrame) -> pd.DataFrame:
    arr = _classify_paths(ff.open.to_numpy(float), ff.high.to_numpy(float), ff.low.to_numpy(float), ff.close.to_numpy(float))
    return pd.DataFrame(arr, columns=PATH_COLS+["path_valid"], index=ff.index)


def eligible_mask(ff: pd.DataFrame) -> pd.Series:
    needed = ["ret_24h", "ret_168h", "rs_24h", "atr24_pct", "atr168_pct", "volume_ratio_3h", "trade_ratio_3h", "taker_buy_3h", "quote_volume_24h"]
    return ((ff.time>=DISCOVERY_START)&(ff.time<CONFIRMATION_END)&(ff.quote_volume_24h>=MIN_QUOTE_VOL_24H)&ff[needed].notna().all(axis=1)&(ff.path_valid==1))


def detect_events(symbol: str, ff: pd.DataFrame):
    ids = np.flatnonzero((ff.w3_2_24h.to_numpy(float)==1.0)&ff.eligible.to_numpy(bool))
    blocked = np.zeros(len(ff), dtype=bool)
    if not len(ids): return pd.DataFrame(), pd.DataFrame(), blocked
    clusters=[]; cluster=[int(ids[0])]
    target_end=int(ids[0]+ff.target3_hour.iloc[ids[0]])
    for idx in ids[1:]:
        idx=int(idx); hit=int(idx+ff.target3_hour.iloc[idx])
        if idx <= target_end + EVENT_GAP_HOURS:
            cluster.append(idx); target_end=max(target_end,hit)
        else:
            clusters.append(cluster); cluster=[idx]; target_end=hit
    clusters.append(cluster)
    events=[]; candidates=[]
    for n, cl in enumerate(clusters,1):
        anchor=cl[0]; t=pd.Timestamp(ff.time.iloc[anchor]); eid=f"DRH2:{symbol}:{t.isoformat()}:{n}"; c=ff.iloc[cl]
        event={
            "event_id":eid,"symbol":symbol,"anchor_time":t,"entry_time":pd.Timestamp(ff.time.iloc[anchor+1]),"split":split_name(t),
            "event_start_time":pd.Timestamp(ff.time.iloc[cl[0]]),"event_end_time":pd.Timestamp(ff.time.iloc[cl[-1]]),
            "entry_window_hours":int(cl[-1]-cl[0]+1),"candidate_hours":len(cl),"account_excluded":bool(ff.account_excluded.iloc[anchor]),
        }
        for lab in ["w3_2_24h","w5_2_48h","w5_3_48h","w7_3_72h","w10_4_72h"]:
            event[f"event_any_{lab}"]=int((c[lab]==1).any()); event[f"anchor_{lab}"]=int(ff[lab].iloc[anchor]==1)
        for col in ["mfe_24h","mae_24h","mfe_48h","mae_48h","mfe_72h","mae_72h","end_net_24h","end_net_48h","end_net_72h"]:
            event[f"anchor_{col}"]=float(ff[col].iloc[anchor])
        events.append(event)
        for i in cl:
            candidates.append({"event_id":eid,"symbol":symbol,"decision_time":pd.Timestamp(ff.time.iloc[i]),"entry_time":pd.Timestamp(ff.time.iloc[i+1]),"hours_from_event_start":int(i-anchor),**{col:ff[col].iloc[i] for col in PATH_COLS}})
        blocked[max(0,cl[0]-WINNER_BUFFER_HOURS):min(len(ff),cl[-1]+WINNER_BUFFER_HOURS+1)] = True
    return pd.DataFrame(events), pd.DataFrame(candidates), blocked


def make_local_samples(symbol: str, ff: pd.DataFrame, events: pd.DataFrame, blocked: np.ndarray) -> pd.DataFrame:
    rows=[]; rng=np.random.default_rng(zlib.crc32(symbol.encode())); eligible=ff.eligible.to_numpy(bool)
    idx_by_time=pd.Series(ff.index.to_numpy(),index=ff.time).to_dict()
    for _,e in events.iterrows():
        rows.append({"sample_id":f"{e.event_id}:WINNER","source_event_id":e.event_id,"symbol":symbol,"decision_time":pd.Timestamp(e.anchor_time),"sample_type":"WINNER_EVENT","split":e.split})
        sm=np.array([split_name(t)==e.split for t in ff.time],dtype=bool); pool=np.flatnonzero(eligible&sm&(~blocked))
        if len(pool):
            for k,i in enumerate(np.atleast_1d(rng.choice(pool,size=min(SAME_COIN_CONTROLS_PER_EVENT,len(pool)),replace=False))):
                rows.append({"sample_id":f"{e.event_id}:SCR:{k}","source_event_id":e.event_id,"symbol":symbol,"decision_time":pd.Timestamp(ff.time.iloc[int(i)]),"sample_type":"SAME_COIN_RANDOM","split":e.split})
        hard=np.flatnonzero(eligible&sm&(~blocked)&(ff.mfe_48h.to_numpy(float)>=0.03)&(ff.w3_2_24h.to_numpy(float)!=1.0))
        if len(hard):
            for k,i in enumerate(np.atleast_1d(rng.choice(hard,size=min(HARD_NEGATIVES_PER_EVENT,len(hard)),replace=False))):
                rows.append({"sample_id":f"{e.event_id}:HARD:{k}","source_event_id":e.event_id,"symbol":symbol,"decision_time":pd.Timestamp(ff.time.iloc[int(i)]),"sample_type":"HARD_NEGATIVE","split":e.split})
    z=ff[eligible].copy()
    if len(z):
        z["month"]=z.time.dt.tz_localize(None).dt.to_period("M")
        for month,g in z.groupby("month"):
            n=min(BASELINE_RANDOM_PER_SYMBOL_MONTH,len(g)); ids=rng.choice(g.index.to_numpy(),size=n,replace=False)
            for i in np.atleast_1d(ids):
                t=pd.Timestamp(ff.time.iloc[int(i)])
                rows.append({"sample_id":f"BASE:{symbol}:{t.isoformat()}","source_event_id":"","symbol":symbol,"decision_time":t,"sample_type":"BASELINE_RANDOM","split":split_name(t)})
    return pd.DataFrame(rows)


def process_symbol(meta: pd.Series, btc: pd.DataFrame):
    symbol=meta.symbol; df=btc.copy() if symbol=="BTCUSDT" else load_symbol(symbol,INTERVAL,DATA_START,DATA_END)
    if len(df)<MIN_DISCOVERY_DAYS*24: return symbol,None,pd.DataFrame(),pd.DataFrame(),pd.DataFrame(),f"insufficient_history:{len(df)}"
    ff=build_features(df,btc); p=classify_paths(ff)
    for col in p.columns: ff[col]=p[col].to_numpy()
    ff["eligible"]=eligible_mask(ff); ff["account_excluded"]=bool(meta.account_excluded); ff["historical_liquidity_rank"]=int(meta.historical_liquidity_rank)
    events,candidates,blocked=detect_events(symbol,ff); samples=make_local_samples(symbol,ff,events,blocked)
    keep=list(dict.fromkeys(["time","open","high","low","close","eligible","account_excluded","historical_liquidity_rank"]+BASE_FEATURES+PATH_COLS))
    (TMP/"features").mkdir(parents=True,exist_ok=True); ff[keep].to_pickle(TMP/"features"/f"{symbol}.pkl.gz",compression="gzip")
    panel_cols=["time","eligible","ret_1h","ret_3h","ret_6h","ret_24h","rs_6h","rs_24h","volume_ratio_3h","trade_ratio_3h","quote_volume_24h","w3_2_24h"]
    panel=ff[panel_cols].copy(); panel["symbol"]=symbol
    return symbol,panel,events,candidates,samples,None


def add_cross_section_context(panel: pd.DataFrame) -> pd.DataFrame:
    panel=panel.sort_values(["time","symbol"]).reset_index(drop=True)
    for src,dst in [("ret_3h","xrank_ret_3h"),("ret_6h","xrank_ret_6h"),("ret_24h","xrank_ret_24h"),("rs_6h","xrank_rs_6h"),("rs_24h","xrank_rs_24h"),("volume_ratio_3h","xrank_volume_ratio_3h"),("trade_ratio_3h","xrank_trade_ratio_3h"),("quote_volume_24h","xrank_quote_volume_24h")]:
        panel[dst]=panel.groupby("time")[src].rank(pct=True,method="average")
    grp=panel.groupby("time")
    market=pd.DataFrame({
        "breadth_positive_1h":grp.ret_1h.apply(lambda s:float((s>0).mean())),"breadth_positive_6h":grp.ret_6h.apply(lambda s:float((s>0).mean())),"breadth_positive_24h":grp.ret_24h.apply(lambda s:float((s>0).mean())),
        "median_ret_1h":grp.ret_1h.median(),"median_ret_6h":grp.ret_6h.median(),"median_ret_24h":grp.ret_24h.median(),
    }).reset_index()
    return panel.merge(market,on="time",how="left")


def choose_same_time_controls(events: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    rng=np.random.default_rng(2442); groups={t:g for t,g in panel.groupby("time",sort=False)}; rows=[]
    for _,e in events.iterrows():
        t=pd.Timestamp(e.anchor_time); g=groups.get(t)
        if g is None: continue
        cand=g[(g.symbol!=e.symbol)&g.eligible.fillna(False)&(g.w3_2_24h!=1.0)]
        if cand.empty: continue
        take=cand.sample(n=min(SAME_TIME_CONTROLS_PER_EVENT,len(cand)),random_state=int(rng.integers(0,2**31-1)))
        for k,r in enumerate(take.itertuples()): rows.append({"sample_id":f"{e.event_id}:STC:{k}","source_event_id":e.event_id,"symbol":r.symbol,"decision_time":t,"sample_type":"SAME_TIME_CONTROL","split":e.split})
    return pd.DataFrame(rows)


def outcome_columns(row: pd.Series) -> dict:
    out={c:int(row[c]==1) for c in ["w3_2_24h","w5_2_48h","w5_3_48h","w7_3_72h","w10_4_72h"]}
    for c in ["mfe_24h","mae_24h","mfe_48h","mae_48h","mfe_72h","mae_72h","end_net_3h","end_net_6h","end_net_12h","end_net_24h","end_net_48h","end_net_72h"]: out[c]=float(row[c])
    return out


def build_sample_outputs(samples: pd.DataFrame, panel: pd.DataFrame):
    ctx=panel[["symbol","time"]+CONTEXT_FEATURES].drop_duplicates(["symbol","time"]).set_index(["symbol","time"])
    history_rows=[]; wide_rows=[]
    for symbol,sg in samples.groupby("symbol"):
        path=TMP/"features"/f"{symbol}.pkl.gz"
        if not path.exists(): continue
        ff=pd.read_pickle(path,compression="gzip"); ff["time"]=pd.to_datetime(ff.time,utc=True); idx_by_time=pd.Series(ff.index.to_numpy(),index=ff.time).to_dict()
        for s in sg.itertuples():
            t=pd.Timestamp(s.decision_time); i=idx_by_time.get(t)
            if i is None: continue
            base=ff.iloc[i]; row={"sample_id":s.sample_id,"source_event_id":s.source_event_id,"symbol":symbol,"decision_time":t,"sample_type":s.sample_type,"split":s.split,"year":int(t.year),"historical_liquidity_rank":int(base.historical_liquidity_rank),"account_excluded":bool(base.account_excluded),**outcome_columns(base)}
            hist={}
            for off in SNAPSHOT_OFFSETS:
                j=i+off
                if j<0 or j>=len(ff): continue
                fr=ff.iloc[j]; ht=pd.Timestamp(fr.time); vals={}
                for f in BASE_FEATURES:
                    v=fr.get(f,np.nan); vals[f]=float(v) if pd.notna(v) else np.nan
                key=(symbol,ht)
                if key in ctx.index:
                    cr=ctx.loc[key]; cr=cr.iloc[0] if isinstance(cr,pd.DataFrame) else cr
                    for f in CONTEXT_FEATURES:
                        v=cr.get(f,np.nan); vals[f]=float(v) if pd.notna(v) else np.nan
                else:
                    for f in CONTEXT_FEATURES: vals[f]=np.nan
                if s.sample_type!="BASELINE_RANDOM": history_rows.append({"sample_id":s.sample_id,"source_event_id":s.source_event_id,"symbol":symbol,"decision_time":t,"sample_type":s.sample_type,"split":s.split,"relative_hour":off,"snapshot_time":ht,**vals})
                hist[off]=vals
            now=hist.get(0,{})
            for f in TRAJECTORY_FEATURES:
                row[f]=now.get(f,np.nan)
                for dh in DELTA_HOURS:
                    cur=now.get(f,np.nan); prev=hist.get(-dh,{}).get(f,np.nan)
                    row[f"{f}__chg_{dh}h"]=float(cur-prev) if np.isfinite(cur) and np.isfinite(prev) else np.nan
            wide_rows.append(row)
    return pd.DataFrame(wide_rows),pd.DataFrame(history_rows)


def qsafe(s: pd.Series,q:float)->float:
    z=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna(); return float(z.quantile(q)) if len(z) else np.nan


def make_condition_library(samples: pd.DataFrame):
    base=samples[(samples.sample_type=="BASELINE_RANDOM")&(samples.split=="DISCOVERY_2021-11_2024-04")]
    exclude={"historical_liquidity_rank","year","w3_2_24h","w5_2_48h","w5_3_48h","w7_3_72h","w10_4_72h","mfe_24h","mae_24h","mfe_48h","mae_48h","mfe_72h","mae_72h","end_net_3h","end_net_6h","end_net_12h","end_net_24h","end_net_48h","end_net_72h"}
    numeric=[c for c in samples.columns if c not in exclude and pd.api.types.is_numeric_dtype(samples[c]) and samples[c].notna().sum()>=200]
    defs=[]; conds={}
    for c in numeric:
        for q,op in [(0.10,"le"),(0.25,"le"),(0.75,"ge"),(0.90,"ge")]:
            th=qsafe(base[c],q)
            if not np.isfinite(th): continue
            name=f"{c}__{op}_q{int(q*100):02d}"; defs.append({"condition":name,"feature":c,"operator":"<=" if op=="le" else ">=","threshold":th,"discovery_quantile":q}); conds[name]=(c,op,th)
    return pd.DataFrame(defs),conds


def apply_condition(df: pd.DataFrame,spec):
    c,op,th=spec; v=pd.to_numeric(df[c],errors="coerce"); return (v<=th) if op=="le" else (v>=th)


def rate_metrics(g:pd.DataFrame,prefix:str,baseline:float)->dict:
    if not len(g): return {f"{prefix}_n":0,f"{prefix}_symbols":0,f"{prefix}_w5_2_rate":np.nan,f"{prefix}_lift":np.nan,f"{prefix}_w3_2_rate":np.nan,f"{prefix}_w5_3_rate":np.nan,f"{prefix}_w7_3_rate":np.nan,f"{prefix}_w10_4_rate":np.nan,f"{prefix}_median_mfe24":np.nan,f"{prefix}_median_mae24":np.nan}
    rate=float(g.w5_2_48h.mean()); return {f"{prefix}_n":int(len(g)),f"{prefix}_symbols":int(g.symbol.nunique()),f"{prefix}_w5_2_rate":rate,f"{prefix}_lift":float(rate/baseline) if baseline>0 else np.nan,f"{prefix}_w3_2_rate":float(g.w3_2_24h.mean()),f"{prefix}_w5_3_rate":float(g.w5_3_48h.mean()),f"{prefix}_w7_3_rate":float(g.w7_3_72h.mean()),f"{prefix}_w10_4_rate":float(g.w10_4_72h.mean()),f"{prefix}_median_mfe24":float(g.mfe_24h.median()),f"{prefix}_median_mae24":float(g.mae_24h.median())}


def mine_intersections(samples:pd.DataFrame,defs:pd.DataFrame,conds:dict):
    base=samples[samples.sample_type=="BASELINE_RANDOM"]; disc=base[base.split=="DISCOVERY_2021-11_2024-04"]; val=base[base.split=="VALIDATION_2024-05_2025-12"]; conf=base[base.split=="CONFIRMATION_2026"]
    baselines={"discovery":float(disc.w5_2_48h.mean()),"validation":float(val.w5_2_48h.mean()) if len(val) else np.nan,"confirmation":float(conf.w5_2_48h.mean()) if len(conf) else np.nan}
    univ=[]; masks={}
    for name,spec in conds.items():
        m=apply_condition(disc,spec); masks[name]=m.to_numpy(bool); g=disc[m]
        if len(g)<100 or g.symbol.nunique()<10: continue
        dm=rate_metrics(g,"discovery",baselines["discovery"]); univ.append({"conditions":name,"condition_count":1,"score":dm["discovery_lift"]*math.sqrt(dm["discovery_n"]),**dm})
    uni=pd.DataFrame(univ)
    if uni.empty: return uni,pd.DataFrame(),baselines,[]
    uni=uni.sort_values(["discovery_lift","discovery_n"],ascending=[False,False])
    feature_for=defs.set_index("condition").feature.to_dict(); top=[]; seen=set()
    for name in uni.conditions:
        f=feature_for[name]
        if f in seen: continue
        top.append(name); seen.add(f)
        if len(top)>=32: break
    sets=[(n,) for n in top]
    for a in range(len(top)):
        for b in range(a+1,len(top)): sets.append((top[a],top[b]))
    top3=top[:16]
    for a in range(len(top3)):
        for b in range(a+1,len(top3)):
            for c in range(b+1,len(top3)): sets.append((top3[a],top3[b],top3[c]))
    rows=[]
    for combo in sets:
        if len({feature_for[n] for n in combo})!=len(combo): continue
        m=np.ones(len(disc),dtype=bool)
        for n in combo: m &= masks[n]
        g=disc[m]
        if len(g)<80 or g.symbol.nunique()<10: continue
        dm=rate_metrics(g,"discovery",baselines["discovery"])
        if not np.isfinite(dm["discovery_lift"]) or dm["discovery_lift"]<1.25: continue
        row={"conditions":" & ".join(combo),"condition_count":len(combo),"score":dm["discovery_lift"]*math.sqrt(dm["discovery_n"]),**dm}
        for df,prefix,key in [(val,"validation","validation"),(conf,"confirmation","confirmation")]:
            mm=np.ones(len(df),dtype=bool)
            for n in combo: mm &= apply_condition(df,conds[n]).to_numpy(bool)
            row.update(rate_metrics(df[mm],prefix,baselines[key]))
        rows.append(row)
    inter=pd.DataFrame(rows)
    if len(inter):
        inter["stability_score"]=inter.discovery_lift.fillna(0)*np.sqrt(inter.discovery_n.clip(lower=1))*np.minimum(inter.validation_lift.fillna(0).clip(lower=0),3.0)
        inter=inter.sort_values(["stability_score","discovery_lift","discovery_n"],ascending=[False,False,False]).reset_index(drop=True)
    return uni,inter,baselines,top


def feature_comparison(samples:pd.DataFrame)->pd.DataFrame:
    d=samples[samples.split=="DISCOVERY_2021-11_2024-04"]; win=d[d.sample_type=="WINNER_EVENT"]
    controls={k:d[d.sample_type==k] for k in ["SAME_COIN_RANDOM","SAME_TIME_CONTROL","HARD_NEGATIVE","BASELINE_RANDOM"]}
    outcomes=set(PATH_COLS+["year","historical_liquidity_rank"])
    numeric=[c for c in samples.columns if pd.api.types.is_numeric_dtype(samples[c]) and c not in outcomes and not c.startswith(("mfe_","mae_","end_net_","w3_","w5_","w7_","w10_"))]
    rows=[]
    for c in numeric:
        wm=pd.to_numeric(win[c],errors="coerce").median()
        if pd.isna(wm): continue
        row={"feature":c,"winner_median":wm,"winner_n":int(win[c].notna().sum())}
        for label,g in controls.items():
            cm=pd.to_numeric(g[c],errors="coerce").median(); row[f"{label.lower()}_median"]=cm; row[f"{label.lower()}_delta"]=wm-cm if pd.notna(cm) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def safe_slug(s:str)->str: return re.sub(r"[^A-Za-z0-9]+","_",s).strip("_").lower()


def write_analysis(events,samples,intersections,baselines,universe,failures):
    disc=events[events.split=="DISCOVERY_2021-11_2024-04"]
    lines=["# Daily Return Harvest Round 2 - Event-first discovery","","**Status:** RESEARCH ONLY. No live strategy, portfolio or automation changes.","","## Dataset","",f"- Universe symbols: {len(universe)}",f"- Discovery winner episodes: {len(disc)}",f"- All event episodes: {len(events)}",f"- Wide samples: {len(samples)}",f"- Failures: {len(failures)}",f"- Discovery baseline W5-before--2 rate: {baselines.get('discovery',np.nan):.3%}",f"- Validation baseline W5-before--2 rate: {baselines.get('validation',np.nan):.3%}","","## Leading intersections",""]
    if intersections.empty: lines.append("No intersections passed the minimum support/lift screen.")
    else:
        cols=["conditions","condition_count","discovery_n","discovery_symbols","discovery_w5_2_rate","discovery_lift","validation_n","validation_symbols","validation_w5_2_rate","validation_lift","confirmation_n","confirmation_w5_2_rate","confirmation_lift","discovery_median_mfe24","discovery_median_mae24"]
        lines.append(intersections.head(25)[[c for c in cols if c in intersections]].to_markdown(index=False))
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))


def main():
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
    pool=load_candidate_pool()
    if pool.empty: raise RuntimeError("Candidate universe empty")
    liq=[]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs={ex.submit(scan_liquidity,r):r.symbol for _,r in pool.iterrows()}
        for fut in as_completed(futs):
            sym=futs[fut]
            try:
                row=fut.result(); liq.append(row); print("LIQ",sym,row["active_days"],row["median_daily_quote_volume"],flush=True)
            except Exception as e:
                liq.append({"symbol":sym,"base_asset":sym[:-4],"account_excluded":sym in ACCOUNT_EXCLUSIONS,"hours":0,"active_days":0,"median_daily_quote_volume":np.nan,"median_trailing_24h_quote_volume":np.nan,"error":repr(e)}); print("LIQ ERROR",sym,repr(e),flush=True)
    liq=pd.DataFrame(liq).sort_values("median_daily_quote_volume",ascending=False,na_position="last"); liq.to_csv(OUT/"candidate_pool_liquidity.csv",index=False)
    universe=liq[(liq.active_days>=MIN_DISCOVERY_DAYS)&liq.median_daily_quote_volume.notna()].head(UNIVERSE_SIZE).copy(); universe["historical_liquidity_rank"]=np.arange(1,len(universe)+1); universe.to_csv(OUT/"universe.csv",index=False)
    if len(universe)<50: raise RuntimeError(f"Only {len(universe)} symbols have sufficient discovery history")
    btc=load_symbol("BTCUSDT",INTERVAL,DATA_START,DATA_END)
    if len(btc)<1000: raise RuntimeError("BTC history unavailable")
    panels=[]; events_parts=[]; cand_parts=[]; sample_parts=[]; failures=[]
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(process_symbol,r,btc):r.symbol for _,r in universe.iterrows()}
        for fut in as_completed(futs):
            sym=futs[fut]
            try:
                symbol,panel,ev,cand,samp,err=fut.result()
                if err: failures.append({"symbol":symbol,"error":err})
                if panel is not None and len(panel): panels.append(panel)
                if len(ev): events_parts.append(ev)
                if len(cand): cand_parts.append(cand)
                if len(samp): sample_parts.append(samp)
                print("PROCESS",symbol,"events",len(ev),"samples",len(samp),flush=True)
            except Exception as e:
                failures.append({"symbol":sym,"error":repr(e)}); print("PROCESS ERROR",sym,repr(e),flush=True)
    if not panels: raise RuntimeError("No panels produced")
    panel=add_cross_section_context(pd.concat(panels,ignore_index=True)); events=pd.concat(events_parts,ignore_index=True) if events_parts else pd.DataFrame(); event_candidates=pd.concat(cand_parts,ignore_index=True) if cand_parts else pd.DataFrame(); local=pd.concat(sample_parts,ignore_index=True) if sample_parts else pd.DataFrame()
    if events.empty: raise RuntimeError("No winning events discovered")
    same_time=choose_same_time_controls(events,panel); sample_index=pd.concat([local,same_time],ignore_index=True).drop_duplicates("sample_id").sort_values(["decision_time","sample_type","symbol"]).reset_index(drop=True)
    samples,history=build_sample_outputs(sample_index,panel)
    if samples.empty: raise RuntimeError("No sample features produced")
    defs,conds=make_condition_library(samples); uni,intersections,baselines,top_conditions=mine_intersections(samples,defs,conds); fcmp=feature_comparison(samples)
    condition_matrix=samples[["sample_id","source_event_id","symbol","decision_time","sample_type","split"]].copy()
    for name in top_conditions[:60]: condition_matrix[name]=apply_condition(samples,conds[name]).fillna(False).astype(np.int8)
    winners=samples[samples.sample_type=="WINNER_EVENT"].copy(); winner_cond=condition_matrix[condition_matrix.sample_type=="WINNER_EVENT"].drop(columns=["source_event_id","symbol","decision_time","sample_type","split"]); enriched=events.merge(winners,left_on="event_id",right_on="source_event_id",how="left",suffixes=("","_sample")).merge(winner_cond,on="sample_id",how="left")
    events.to_csv(OUT/"events.csv",index=False); enriched.to_csv(OUT/"events_enriched.csv.gz",index=False,compression="gzip"); event_candidates.to_csv(OUT/"event_candidates.csv.gz",index=False,compression="gzip"); sample_index.to_csv(OUT/"sample_index.csv.gz",index=False,compression="gzip"); condition_matrix.to_csv(OUT/"condition_matrix.csv.gz",index=False,compression="gzip")
    for split,g in samples.groupby("split"): g.to_csv(OUT/f"samples_wide_{safe_slug(split)}.csv.gz",index=False,compression="gzip")
    for split,g in history.groupby("split"): g.to_csv(OUT/f"feature_history_{safe_slug(split)}.csv.gz",index=False,compression="gzip")
    defs.to_csv(OUT/"condition_dictionary.csv",index=False); uni.to_csv(OUT/"univariate_conditions.csv",index=False); intersections.to_csv(OUT/"intersections.csv",index=False); fcmp.to_csv(OUT/"feature_comparison.csv",index=False); pd.DataFrame(failures).to_csv(OUT/"failures.csv",index=False)
    baseline=[]; base=samples[samples.sample_type=="BASELINE_RANDOM"]
    for split,g in base.groupby("split"): baseline.append({"split":split,"samples":len(g),"symbols":g.symbol.nunique(),"w3_2_24h_rate":g.w3_2_24h.mean(),"w5_2_48h_rate":g.w5_2_48h.mean(),"w5_3_48h_rate":g.w5_3_48h.mean(),"w7_3_72h_rate":g.w7_3_72h.mean(),"w10_4_72h_rate":g.w10_4_72h.mean()})
    pd.DataFrame(baseline).to_csv(OUT/"baseline_rates.csv",index=False)
    manifest={
        "study":"Daily Return Harvest Round 2 - event-first historical winner discovery","status":"RESEARCH ONLY - no live strategy, portfolio or automation changes",
        "discovery_period":["2021-11-01","2024-04-30"],"temporal_validation_period":["2024-05-01","2025-12-31"],"confirmation_period":["2026-01-01","2026-09-27"],
        "universe":{"size_requested":UNIVERSE_SIZE,"size_actual":len(universe),"selection":"Top available symbols by median daily quote volume during discovery, from union of prior research universes/config and restored Binance cache; >=90 active discovery days.","point_in_time_filter":"Trailing 24h quote volume >=1,000,000 USDT.","survivorship_note":"Candidate pool is drawn from symbols known to prior research/cache; delisted historical symbols are not guaranteed. Wider-universe generalisation is reserved for a later holdout.","account_exclusions":"Marked but not removed during pattern discovery."},
        "costs":{"fee_rate_each_side":FEE_RATE,"slippage_rate_each_side":SLIPPAGE_RATE},
        "path_labels_net_of_costs":{"w3_2_24h":"+3% net before -2% net within 24h","w5_2_48h":"+5% net before -2% net within 48h","w5_3_48h":"+5% net before -3% net within 48h","w7_3_72h":"+7% net before -3% net within 72h","w10_4_72h":"+10% net before -4% net within 72h","same_bar_rule":"Target first touched in same hourly candle as stop does not count target-before-stop."},
        "event_definition":"Cluster overlapping +3-before--2 viable decision hours into one episode; anchor is earliest viable causal decision hour.",
        "controls":{"same_coin_random_per_event":SAME_COIN_CONTROLS_PER_EVENT,"same_time_controls_per_event":SAME_TIME_CONTROLS_PER_EVENT,"hard_negative_per_event":HARD_NEGATIVES_PER_EVENT,"baseline_random_per_symbol_month":BASELINE_RANDOM_PER_SYMBOL_MONTH,"winner_buffer_hours":WINNER_BUFFER_HOURS},
        "feature_history_offsets_hours":SNAPSHOT_OFFSETS,"wide_trajectory_deltas_hours":DELTA_HOURS,"intersection_mining":"Quantile-derived causal conditions learned only on discovery BASELINE_RANDOM; distinct-feature singles/pairs/triples; thresholds frozen for validation/confirmation.",
        "full_dataset_note":"Large wide/history tables are retained in the workflow artifact; compact analysis outputs and the enriched event table are intended for repository persistence."
    }
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2,default=str)); write_analysis(events,samples,intersections,baselines,universe,failures)
    print("\nUNIVERSE",len(universe),"EVENTS",len(events),"SAMPLES",len(samples),flush=True); print("\nBASELINES",json.dumps(baselines,indent=2),flush=True)
    if len(intersections): print("\nTOP INTERSECTIONS\n",intersections.head(25).to_string(index=False),flush=True)


if __name__ == "__main__": main()
