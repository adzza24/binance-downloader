from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, early_stopping

from binance_data import load_symbol
from daily_return_harvest_round2_event_discovery import classify_paths, FEE_RATE, SLIPPAGE_RATE

OUT = Path("research/results/daily_return_harvest/btc_forensic_transition_2024")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
LOAD_START = "2023-11-01"
LOAD_END = "2024-12-31"
DECISION_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
DECISION_END = pd.Timestamp("2024-12-28 23:00:00", tz="UTC")  # keeps 72h future inside 2024
WINDOWS = (24, 72, 240)
SEQ_CHANNELS = [
    "log_ret", "range_pct", "body_pct", "upper_wick", "lower_wick", "close_location",
    "volume_rel72", "trades_rel72", "taker_share", "rsi14", "macd_hist_pct",
    "close_vs_ema24", "atr14_pct", "bb_width", "stoch_k",
]
PROFILE_COLS = [
    "log_ret", "body_pct", "range_pct", "lower_wick", "upper_wick", "close_location",
    "volume_rel72", "trades_rel72", "taker_share", "rsi6", "rsi14", "rsi24",
    "macd_pct", "macd_signal_pct", "macd_hist_pct", "macd_fast_hist_pct",
    "atr14_pct", "bb_width", "bb_z", "stoch_k", "stoch_d", "cci20", "mfi14", "adx14",
    "close_vs_ema6", "close_vs_ema12", "close_vs_ema24", "close_vs_ema48", "close_vs_ema72",
    "ema6_slope_3h", "ema24_slope_6h", "ema72_slope_12h", "close_vs_vwap24", "close_vs_vwap72",
]
RNG = np.random.default_rng(20241006)


def safe_div(a, b):
    return a / pd.Series(b, index=a.index).replace(0, np.nan) if not isinstance(b, pd.Series) else a / b.replace(0, np.nan)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    plus = 100 * plus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    minus = 100 * minus_dm.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus-minus).abs() / (plus+minus).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy().sort_values("time").reset_index(drop=True)
    o, h, l, c = x.open, x.high, x.low, x.close
    vol, qv, trades = x.volume, x.quote_volume, x.trades
    cr = (h-l).replace(0, np.nan)

    x["log_ret"] = np.log(c / c.shift())
    x["range_pct"] = h / l.replace(0, np.nan) - 1
    x["body_pct"] = c / o.replace(0, np.nan) - 1
    x["abs_body_pct"] = (c-o).abs() / o.replace(0, np.nan)
    x["upper_wick"] = (h-np.maximum(o, c)) / cr
    x["lower_wick"] = (np.minimum(o, c)-l) / cr
    x["close_location"] = (c-l) / cr
    x["taker_share"] = x.taker_buy_base / vol.replace(0, np.nan)
    x["avg_trade_quote"] = qv / trades.replace(0, np.nan)
    x["obv_delta"] = np.sign(c.diff()).fillna(0) * vol

    for n in [24, 72, 240]:
        for src, name in [(vol, "volume"), (qv, "quote_volume"), (trades, "trades"), (x.avg_trade_quote, "avg_trade_quote")]:
            base = src.shift(1).rolling(n, min_periods=max(6, n//4)).median()
            x[f"{name}_rel{n}"] = src / base.replace(0, np.nan)
        hi, lo = h.rolling(n).max(), l.rolling(n).min()
        x[f"dist_high{n}"] = c / hi.replace(0, np.nan) - 1
        x[f"rebound_low{n}"] = c / lo.replace(0, np.nan) - 1
        vwap = qv.rolling(n).sum() / vol.rolling(n).sum().replace(0, np.nan)
        x[f"close_vs_vwap{n}"] = c / vwap.replace(0, np.nan) - 1

    for n in [3, 6, 12, 24, 48, 72, 168, 240]:
        ema = c.ewm(span=n, adjust=False, min_periods=max(3, n//3)).mean()
        sma = c.rolling(n, min_periods=max(3, n//2)).mean()
        x[f"ema{n}"] = ema
        x[f"close_vs_ema{n}"] = c / ema.replace(0, np.nan) - 1
        x[f"close_vs_sma{n}"] = c / sma.replace(0, np.nan) - 1
    x["ema6_slope_3h"] = x.ema6.pct_change(3)
    x["ema24_slope_6h"] = x.ema24.pct_change(6)
    x["ema72_slope_12h"] = x.ema72.pct_change(12)
    x["ema168_slope_24h"] = x.ema168.pct_change(24)
    x["ema_gap_6_24"] = x.ema6 / x.ema24 - 1
    x["ema_gap_24_72"] = x.ema24 / x.ema72 - 1
    x["ema_gap_72_168"] = x.ema72 / x.ema168 - 1

    for n in [6, 14, 24]:
        x[f"rsi{n}"] = rsi(c, n)

    ema12 = c.ewm(span=12, adjust=False).mean(); ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12-ema26; sig = macd.ewm(span=9, adjust=False).mean()
    x["macd_pct"] = macd / c
    x["macd_signal_pct"] = sig / c
    x["macd_hist_pct"] = (macd-sig) / c
    f6 = c.ewm(span=6, adjust=False).mean(); f18 = c.ewm(span=18, adjust=False).mean()
    fm = f6-f18; fs = fm.ewm(span=6, adjust=False).mean()
    x["macd_fast_hist_pct"] = (fm-fs) / c

    mid = c.rolling(20).mean(); sd = c.rolling(20).std()
    x["bb_width"] = 4 * sd / mid.replace(0, np.nan)
    x["bb_z"] = (c-mid) / sd.replace(0, np.nan)

    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    for n in [6, 14, 24, 72]:
        x[f"atr{n}_pct"] = tr.rolling(n).mean() / c

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    x["stoch_k"] = 100 * (c-lo14) / (hi14-lo14).replace(0, np.nan)
    x["stoch_d"] = x.stoch_k.rolling(3).mean()
    tp = (h+l+c)/3
    tp_mean = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda a: np.mean(np.abs(a-np.mean(a))), raw=True)
    x["cci20"] = (tp-tp_mean) / (0.015*mad.replace(0, np.nan))
    pos_flow = tp*vol
    direction = np.sign(tp.diff())
    pos = pos_flow.where(direction > 0, 0).rolling(14).sum()
    neg = pos_flow.where(direction < 0, 0).rolling(14).sum()
    mfr = pos / neg.replace(0, np.nan)
    x["mfi14"] = 100 - 100/(1+mfr)
    x["adx14"] = adx(h, l, c, 14)

    x["volume_rel24"] = x["volume_rel24"]
    x["volume_rel72"] = x["volume_rel72"]
    x["volume_rel240"] = x["volume_rel240"]
    x["trades_rel24"] = x["trades_rel24"]
    x["trades_rel72"] = x["trades_rel72"]
    x["trades_rel240"] = x["trades_rel240"]
    return x


def temporal_feature_frame(x: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = pd.DataFrame(index=x.index)
    names = []
    for col in cols:
        s = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[f"{col}__now"] = s; names.append(f"{col}__now")
        for lag in [1, 3, 6, 12, 24]:
            out[f"{col}__d{lag}"] = s - s.shift(lag); names.append(f"{col}__d{lag}")
        ds = s.diff()
        for w in WINDOWS:
            r = s.rolling(w, min_periods=max(6, w//2))
            mean, std = r.mean(), r.std()
            mn, mx = r.min(), r.max()
            q10, q25, q75, q90 = r.quantile(.10), r.quantile(.25), r.quantile(.75), r.quantile(.90)
            prefix = f"{col}__w{w}"
            vals = {
                "mean": mean, "std": std, "min": mn, "max": mx,
                "q10": q10, "q25": q25, "q75": q75, "q90": q90,
                "z": (s-mean)/std.replace(0, np.nan),
                "pos": (s-mn)/(mx-mn).replace(0, np.nan),
                "delta": s-s.shift(w-1),
                "diff_mean": ds.rolling(w, min_periods=max(6, w//2)).mean(),
                "diff_std": ds.rolling(w, min_periods=max(6, w//2)).std(),
                "up_share": (ds>0).astype(float).rolling(w, min_periods=max(6, w//2)).mean(),
            }
            q = max(3, w//4)
            late = s.rolling(q, min_periods=q).mean()
            early = late.shift(q)
            vals["late_vs_prev"] = late-early
            short = max(3, min(12, w//4))
            vals["accel"] = (s-s.shift(short)) - (s.shift(short)-s.shift(2*short))
            for k, v in vals.items():
                name = f"{prefix}__{k}"; out[name] = v; names.append(name)
    return out.astype(np.float32), names


def future_extra_labels(x: pd.DataFrame) -> pd.DataFrame:
    o = x.open.to_numpy(float); h = x.high.to_numpy(float); l = x.low.to_numpy(float); c = x.close.to_numpy(float)
    n = len(x)
    cols = [
        "t_up1", "t_up2", "t_up3", "t_up5", "t_dn1", "t_dn2", "t_dn3",
        "mfe_6h", "mae_6h", "mfe_12h", "mae_12h", "w3_2_48h", "w5_2_24h",
    ]
    arr = np.full((n, len(cols)), np.nan, dtype=float)
    ef = (1-SLIPPAGE_RATE)*(1-FEE_RATE)
    for i in range(n-1):
        ei = i+1; raw = o[ei]
        if not np.isfinite(raw) or raw <= 0: continue
        ec = raw*(1+SLIPPAGE_RATE)*(1+FEE_RATE)
        ups = [ec*(1+p)/ef for p in (.01,.02,.03,.05)]
        dns = [ec*(1-p)/ef for p in (.01,.02,.03)]
        hu = [-1]*4; hd = [-1]*3
        max6=max12=-np.inf; min6=min12=np.inf
        for j in range(ei, min(n, ei+72)):
            off=j-ei
            for k,t in enumerate(ups):
                if hu[k]<0 and h[j]>=t: hu[k]=off
            for k,t in enumerate(dns):
                if hd[k]<0 and l[j]<=t: hd[k]=off
            if off<6: max6=max(max6,h[j]); min6=min(min6,l[j])
            if off<12: max12=max(max12,h[j]); min12=min(min12,l[j])
        for k,v in enumerate(hu): arr[i,k] = v if v>=0 else np.nan
        for k,v in enumerate(hd): arr[i,4+k] = v if v>=0 else np.nan
        entry_exec=raw*(1+SLIPPAGE_RATE)
        arr[i,7]=max6/entry_exec-1 if np.isfinite(max6) else np.nan
        arr[i,8]=min6/entry_exec-1 if np.isfinite(min6) else np.nan
        arr[i,9]=max12/entry_exec-1 if np.isfinite(max12) else np.nan
        arr[i,10]=min12/entry_exec-1 if np.isfinite(min12) else np.nan
        u3, d2, u5 = hu[2], hd[1], hu[3]
        arr[i,11]=1.0 if u3>=0 and u3<48 and (d2<0 or u3<d2) else 0.0
        arr[i,12]=1.0 if u5>=0 and u5<24 and (d2<0 or u5<d2) else 0.0
    return pd.DataFrame(arr, columns=cols, index=x.index)


def make_sequences(x: pd.DataFrame, decision_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = x[SEQ_CHANNELS].replace([np.inf,-np.inf], np.nan).to_numpy(np.float32)
    med = np.nanmedian(vals[decision_idx], axis=0)
    mad = np.nanmedian(np.abs(vals[decision_idx]-med), axis=0)
    scale = np.where(mad > 1e-8, 1.4826*mad, np.nanstd(vals[decision_idx],axis=0))
    scale = np.where(scale > 1e-8, scale, 1.0)
    z = np.nan_to_num((vals-med)/scale, nan=0.0, posinf=8.0, neginf=-8.0)
    z = np.clip(z, -8, 8)
    s24=[]; s72=[]; s240=[]
    for i in decision_idx:
        a=z[i-23:i+1]
        b=z[i-71:i+1].reshape(24,3,-1).mean(axis=1)
        d=z[i-239:i+1].reshape(24,10,-1).mean(axis=1)
        s24.append(a); s72.append(b); s240.append(d)
    return np.asarray(s24,np.float32), np.asarray(s72,np.float32), np.asarray(s240,np.float32)


def event_anchors(decisions: pd.DataFrame, gap_hours: int = 12) -> pd.DataFrame:
    pos = decisions.loc[decisions.w3_2_24h.eq(1), ["row_idx","time"]].copy()
    if pos.empty: return pos.assign(event_id=[])
    event=[]; eid=0; prev=None
    for r in pos.itertuples():
        if prev is None or (r.time-prev)/pd.Timedelta(hours=1) > gap_hours:
            eid += 1
        event.append(eid); prev=r.time
    pos["event_id"]=event
    return pos.groupby("event_id",as_index=False).first()


def matched_controls(decisions: pd.DataFrame, anchors: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
    d=decisions.copy(); d["month"]=d.time.dt.month
    atr = x.loc[d.row_idx,"atr14_pct"].to_numpy()
    d["atr_q"] = pd.qcut(pd.Series(atr).rank(method="first"), 5, labels=False, duplicates="drop").to_numpy()
    neg=d[d.w3_2_24h.ne(1)].copy()
    rows=[]
    map_meta=d.set_index("row_idx")[["month","atr_q"]]
    for a in anchors.itertuples():
        m=int(map_meta.loc[a.row_idx,"month"]); q=map_meta.loc[a.row_idx,"atr_q"]
        pool=neg[(neg.month==m)&(neg.atr_q==q)]
        if pool.empty: pool=neg[neg.month==m]
        if pool.empty: continue
        take=pool.sample(n=min(5,len(pool)),random_state=int(RNG.integers(0,2**31-1)))
        for r in take.itertuples(): rows.append({"event_id":a.event_id,"row_idx":r.row_idx,"time":r.time})
    return pd.DataFrame(rows)


def aligned_profiles(x: pd.DataFrame, anchors: pd.DataFrame, controls: pd.DataFrame):
    records=[]
    for kind, frame in [("WINNER",anchors),("CONTROL",controls)]:
        for off in range(-24,7):
            ids=(frame.row_idx.to_numpy(int)+off)
            ids=ids[(ids>=0)&(ids<len(x))]
            if not len(ids): continue
            for col in PROFILE_COLS:
                v=pd.to_numeric(x.loc[ids,col],errors="coerce").to_numpy(float)
                records.append({"kind":kind,"offset_h":off,"feature":col,"n":int(np.isfinite(v).sum()),"median":float(np.nanmedian(v)),"mean":float(np.nanmean(v)),"std":float(np.nanstd(v))})
    prof=pd.DataFrame(records)
    prof.to_csv(OUT/"aligned_profiles.csv",index=False)
    w=prof[prof.kind.eq("WINNER")].set_index(["offset_h","feature"])
    c=prof[prof.kind.eq("CONTROL")].set_index(["offset_h","feature"])
    common=w.index.intersection(c.index); rows=[]
    for idx in common:
        a=w.loc[idx]; b=c.loc[idx]; denom=math.sqrt((a["std"]**2+b["std"]**2)/2) if np.isfinite(a["std"]+b["std"]) else np.nan
        eff=(a["mean"]-b["mean"])/denom if denom and denom>1e-12 else np.nan
        rows.append({"offset_h":idx[0],"feature":idx[1],"effect_size":eff,"winner_median":a["median"],"control_median":b["median"]})
    div=pd.DataFrame(rows)
    div["abs_effect"]=div.effect_size.abs()
    div.sort_values("abs_effect",ascending=False).to_csv(OUT/"aligned_divergence.csv",index=False)
    onset=[]
    for feature,g in div.groupby("feature"):
        pre=g[(g.offset_h<=0)&g.effect_size.notna()].sort_values("offset_h")
        sustained=None
        for off in range(-24,1):
            z=pre[pre.offset_h.between(off,min(off+2,0))]
            if len(z)>=2 and (z.abs_effect>=0.25).all(): sustained=off; break
        best=pre.loc[pre.abs_effect.idxmax()] if len(pre) else None
        onset.append({"feature":feature,"first_sustained_effect_ge_0_25":sustained,"best_pre_offset_h":int(best.offset_h) if best is not None else np.nan,"best_pre_effect":float(best.effect_size) if best is not None else np.nan})
    pd.DataFrame(onset).sort_values("best_pre_effect",key=lambda s:s.abs(),ascending=False).to_csv(OUT/"divergence_onset.csv",index=False)


def univariate_scan(features: pd.DataFrame, y: np.ndarray):
    rows=[]
    for col in features.columns:
        v=pd.to_numeric(features[col],errors="coerce").to_numpy(float)
        ok=np.isfinite(v)
        if ok.sum()<500 or len(np.unique(y[ok]))<2: continue
        try: auc=roc_auc_score(y[ok],v[ok])
        except Exception: continue
        direction=1 if auc>=.5 else -1; score=direction*v
        row={"feature":col,"auc":auc,"separation":max(auc,1-auc),"direction":"high" if direction==1 else "low"}
        for q in [.90,.95,.975,.99]:
            th=np.nanquantile(score[ok],q); sel=ok&(score>=th)
            row[f"precision_top_{int((1-q)*1000)/10:g}pct"]=float(y[sel].mean()) if sel.sum() else np.nan
            row[f"n_top_{int((1-q)*1000)/10:g}pct"]=int(sel.sum())
        rows.append(row)
    pd.DataFrame(rows).sort_values("separation",ascending=False).to_csv(OUT/"univariate_temporal_features.csv",index=False)


def motif_scan(seq24: np.ndarray, seq72: np.ndarray, seq240: np.ndarray, y: np.ndarray, anchor_mask: np.ndarray):
    rows=[]
    for length,arr in [(6,seq24[:,-6:,:]),(12,seq24[:,-12:,:]),(24,seq24),(72,seq72),(240,seq240)]:
        flat=arr.reshape(len(arr),-1)
        scaler=StandardScaler().fit(flat)
        z=scaler.transform(flat)
        pca=PCA(n_components=min(20,z.shape[1],max(2,len(z)-1)),random_state=42).fit(z)
        pc=pca.transform(z)
        win=pc[anchor_mask]
        if len(win)<8: continue
        k=min(8,max(2,len(win)//15))
        km=KMeans(n_clusters=k,random_state=42,n_init=20).fit(win)
        dist=km.transform(pc); nearest=dist.argmin(axis=1); nd=dist[np.arange(len(dist)),nearest]
        wd=km.transform(win); wn=wd.argmin(axis=1); wnd=wd[np.arange(len(wd)),wn]
        for cl in range(k):
            member_w=wnd[wn==cl]
            if len(member_w)<3: continue
            for pct in [50,75,90]:
                th=np.percentile(member_w,pct)
                sel=(nearest==cl)&(nd<=th)
                rows.append({"motif_window":length,"cluster":cl,"winner_distance_pct":pct,"distance_threshold":th,"signals":int(sel.sum()),"wins":int(y[sel].sum()),"precision":float(y[sel].mean()) if sel.sum() else np.nan,"baseline":float(y.mean()),"lift":float(y[sel].mean()/y.mean()) if sel.sum() and y.mean()>0 else np.nan})
    pd.DataFrame(rows).sort_values(["precision","signals"],ascending=[False,False]).to_csv(OUT/"motif_enrichment.csv",index=False)


def onset_metrics(times: pd.Series, y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    on=np.isfinite(scores)&(scores>=threshold)
    prev=np.r_[False,on[:-1]]
    gap=np.r_[True, (times.iloc[1:].reset_index(drop=True)-times.iloc[:-1].reset_index(drop=True)).dt.total_seconds().to_numpy()!=3600]
    onset=on & (~prev | gap)
    days=max(1,(times.iloc[-1]-times.iloc[0]).total_seconds()/86400)
    return {"signals":int(onset.sum()),"wins":int(y[onset].sum()),"precision":float(y[onset].mean()) if onset.sum() else np.nan,"signals_per_day":float(onset.sum()/days),"wins_per_day":float(y[onset].sum()/days)}


def fit_models(meta: pd.DataFrame, static: pd.DataFrame, temporal: pd.DataFrame, seq24,seq72,seq240,y):
    t=meta.time.reset_index(drop=True)
    train=(t<pd.Timestamp("2024-07-01",tz="UTC"))
    dev=(t>=pd.Timestamp("2024-07-08",tz="UTC"))&(t<pd.Timestamp("2024-10-01",tz="UTC"))
    hold=(t>=pd.Timestamp("2024-10-08",tz="UTC"))
    sets={
        "STATIC": static.reset_index(drop=True),
        "TEMPORAL": temporal.reset_index(drop=True),
        "TEMPORAL_SEQ": pd.concat([
            temporal.reset_index(drop=True),
            pd.DataFrame(seq24.reshape(len(seq24),-1),columns=[f"s24_{i}" for i in range(seq24.shape[1]*seq24.shape[2])]),
            pd.DataFrame(seq72.reshape(len(seq72),-1),columns=[f"s72_{i}" for i in range(seq72.shape[1]*seq72.shape[2])]),
            pd.DataFrame(seq240.reshape(len(seq240),-1),columns=[f"s240_{i}" for i in range(seq240.shape[1]*seq240.shape[2])]),
        ],axis=1),
    }
    metrics=[]; frontier=[]; importances=[]
    for name,X in sets.items():
        X=X.replace([np.inf,-np.inf],np.nan).astype(np.float32)
        usable=X.columns[X[train].notna().mean()>.70]
        X=X[usable]
        model=LGBMClassifier(n_estimators=1200,learning_rate=.025,num_leaves=31,max_depth=-1,min_child_samples=40,subsample=.85,colsample_bytree=.70,reg_lambda=5,reg_alpha=1,random_state=42,verbosity=-1,n_jobs=2)
        model.fit(X[train],y[train],eval_set=[(X[dev],y[dev])],eval_metric="binary_logloss",callbacks=[early_stopping(80,verbose=False)])
        scores=np.full(len(X),np.nan)
        scores[dev]=model.predict_proba(X[dev])[:,1]; scores[hold]=model.predict_proba(X[hold])[:,1]
        for split,mask in [("DEV",dev),("HOLDOUT",hold)]:
            auc=roc_auc_score(y[mask],scores[mask]); ap=average_precision_score(y[mask],scores[mask])
            metrics.append({"model":name,"split":split,"rows":int(mask.sum()),"w3_rate":float(y[mask].mean()),"auc":auc,"average_precision":ap})
        dev_scores=scores[dev]; dev_y=y[dev]; dev_t=t[dev].reset_index(drop=True)
        candidates=np.unique(np.quantile(dev_scores[np.isfinite(dev_scores)],np.linspace(.5,.999,250)))
        chosen={}
        for minrate in [.10,.25,.50,1.0]:
            best=None
            for th in candidates:
                m=onset_metrics(dev_t,dev_y,dev_scores,th)
                if m["signals_per_day"]+1e-9<minrate: continue
                if best is None or (m["precision"],m["wins_per_day"])>(best["precision"],best["wins_per_day"]): best={"threshold":float(th),**m}
            if best is None: continue
            chosen[minrate]=best
            hm=onset_metrics(t[hold].reset_index(drop=True),y[hold],scores[hold],best["threshold"])
            frontier.append({"model":name,"min_dev_signals_per_day":minrate,"threshold":best["threshold"],"split":"DEV",**{k:v for k,v in best.items() if k!="threshold"}})
            frontier.append({"model":name,"min_dev_signals_per_day":minrate,"threshold":best["threshold"],"split":"HOLDOUT",**hm})
        for f,imp in zip(X.columns,model.feature_importances_): importances.append({"model":name,"feature":f,"importance":int(imp)})
    pd.DataFrame(metrics).to_csv(OUT/"model_metrics.csv",index=False)
    pd.DataFrame(frontier).to_csv(OUT/"precision_frequency_frontier.csv",index=False)
    pd.DataFrame(importances).sort_values(["model","importance"],ascending=[True,False]).to_csv(OUT/"model_feature_importance.csv",index=False)


def main():
    print("Loading BTC hourly history",flush=True)
    raw=load_symbol(SYMBOL,INTERVAL,LOAD_START,LOAD_END)
    if len(raw)<10000: raise RuntimeError(f"Insufficient BTC history: {len(raw)}")
    x=build_indicators(raw)
    paths=classify_paths(x)
    for col in paths.columns: x[col]=paths[col].to_numpy()
    extra=future_extra_labels(x)
    for col in extra.columns: x[col]=extra[col].to_numpy()

    feature_cols=[c for c in PROFILE_COLS + [
        "abs_body_pct","avg_trade_quote_rel24","avg_trade_quote_rel72","avg_trade_quote_rel240","obv_delta",
        "volume_rel24","volume_rel72","volume_rel240","quote_volume_rel24","quote_volume_rel72","quote_volume_rel240",
        "trades_rel24","trades_rel72","trades_rel240","close_vs_ema168","close_vs_ema240","close_vs_sma24","close_vs_sma72","close_vs_sma168","close_vs_sma240",
        "ema168_slope_24h","ema_gap_6_24","ema_gap_24_72","ema_gap_72_168","atr6_pct","atr24_pct","atr72_pct",
        "close_vs_vwap240","dist_high24","dist_high72","dist_high240","rebound_low24","rebound_low72","rebound_low240",
    ] if c in x.columns]
    feature_cols=list(dict.fromkeys(feature_cols))
    temporal,_=temporal_feature_frame(x,feature_cols)

    decision_mask=(x.time>=DECISION_START)&(x.time<=DECISION_END)&x.path_valid.eq(1)
    decision_idx=np.flatnonzero(decision_mask.to_numpy())
    decision_idx=decision_idx[decision_idx>=239]
    meta=x.loc[decision_idx,["time","w3_2_24h","w5_2_48h","w5_3_48h","w7_3_72h","w10_4_72h","target3_hour","target5_hour","stop2_hour","mfe_24h","mae_24h","mfe_48h","mae_48h","mfe_72h","mae_72h","end_net_24h","end_net_48h","end_net_72h","w3_2_48h","w5_2_24h","t_up1","t_up2","t_up3","t_up5","t_dn1","t_dn2","t_dn3"]].copy()
    meta["row_idx"]=decision_idx
    meta.reset_index(drop=True,inplace=True)
    y=meta.w3_2_24h.to_numpy(int)
    print("Decision hours",len(meta),"W3",int(y.sum()),"rate",float(y.mean()),flush=True)

    static=x.loc[decision_idx,feature_cols].reset_index(drop=True).astype(np.float32)
    temp=temporal.loc[decision_idx].reset_index(drop=True)
    seq24,seq72,seq240=make_sequences(x,decision_idx)

    meta.to_parquet(OUT/"hourly_outcomes_2024.parquet",index=False,compression="zstd")
    static.to_parquet(OUT/"static_indicators_2024.parquet",index=False,compression="zstd")
    temp.to_parquet(OUT/"temporal_features_2024.parquet",index=False,compression="zstd")
    np.savez_compressed(OUT/"raw_sequences_2024.npz",seq24=seq24,seq72=seq72,seq240=seq240,channels=np.asarray(SEQ_CHANNELS))

    decisions=meta[["row_idx","time","w3_2_24h"]].copy()
    anchors=event_anchors(decisions)
    controls=matched_controls(decisions,anchors,x)
    anchors.to_csv(OUT/"w3_event_anchors.csv",index=False)
    controls.to_csv(OUT/"matched_controls.csv",index=False)
    aligned_profiles(x,anchors,controls)

    univariate_scan(temp,y)
    anchor_rows=set(anchors.row_idx.astype(int)); anchor_mask=np.array([int(r) in anchor_rows for r in decision_idx])
    motif_scan(seq24,seq72,seq240,y,anchor_mask)
    fit_models(meta,static,temp,seq24,seq72,seq240,y)

    uni=pd.read_csv(OUT/"univariate_temporal_features.csv")
    div=pd.read_csv(OUT/"divergence_onset.csv")
    motifs=pd.read_csv(OUT/"motif_enrichment.csv")
    mm=pd.read_csv(OUT/"model_metrics.csv")
    fr=pd.read_csv(OUT/"precision_frequency_frontier.csv")
    summary={
        "study":"BTC forensic transition 2024",
        "status":"RESEARCH ONLY - no live strategy or automation changes",
        "decision_start":str(DECISION_START),"decision_end":str(DECISION_END),"hours":len(meta),
        "primary_w3_hours":int(y.sum()),"primary_w3_rate":float(y.mean()),"w3_event_anchors":int(len(anchors)),
        "windows_hours":list(WINDOWS),"sequence_channels":SEQ_CHANNELS,"temporal_input_features":int(temp.shape[1]),
        "2025_2026_touched":False,
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2))
    lines=[
        "# BTC 2024 Forensic Transition Study","","**RESEARCH ONLY. No live strategy or automation changes.**","",
        f"- Decision hours: **{len(meta):,}**",
        f"- W3 (+3% before -2% within 24h, net of modelled costs) hours: **{int(y.sum()):,} ({y.mean():.2%})**",
        f"- W3 event anchors after 12h episode clustering: **{len(anchors):,}**",
        f"- Temporal engineered features: **{temp.shape[1]:,}**",
        "- Rolling contexts: **24h, 72h, 240h**",
        "- Raw sequence channels retained in artifact: **"+", ".join(SEQ_CHANNELS)+"**",
        "- 2025 and 2026 were **not read or evaluated**.","",
        "## Internal 2024 model comparison","",mm.to_markdown(index=False,floatfmt=".4f"),"",
        "## Precision/frequency frontier","",fr.to_markdown(index=False,floatfmt=".4f"),"",
        "## Strongest temporal features","",uni.head(25).to_markdown(index=False,floatfmt=".4f"),"",
        "## Strongest pre-event divergence timing","",div.head(25).to_markdown(index=False,floatfmt=".4f"),"",
        "## Highest-enrichment sequence motifs","",motifs.head(25).to_markdown(index=False,floatfmt=".4f"),"",
    ]
    (OUT/"ANALYSIS.md").write_text("\n".join(lines))
    print("\n".join(lines[:20]),flush=True)


if __name__ == "__main__":
    main()
