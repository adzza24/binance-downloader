// BASE +12h frozen monitor core. Research signal only.
// The calling task supplies parsed family_models.json plus completed 1h candle arrays.

const FEATURE_ORDER = [
  'ret_6h','ret_24h','ret_72h','ret_7d','ret_14d','ret_30d','ret_60d','ret_90d',
  'range_24h','range_72h','range_7d','range_14d','range_30d','range_90d',
  'atr24_pct','atr7d_pct','atr30d_pct','atr24_vs_7d','range24_vs_7d',
  'dist_7d_high','dist_30d_high','dist_90d_high','rebound_7d_low','rebound_30d_low',
  'worst_1h_ret_7d','worst_4h_ret_14d','volume_ratio_6h','volume_ratio_24h',
  'trade_ratio_6h','taker_buy_6h','quote_volume_24h_log','rs_24h','rs_7d','rs_30d'
];
const FAMILY_THRESHOLDS = {CAPITULATION:0.70, BASE:0.65, MOMENTUM:0.35};
const ROUND5 = [
  ['ret_6h_delta_24h','LE',-0.0129633734783224],
  ['ret_24h_delta_12h','LE',0.02910877712005299],
  ['ret_24h_delta_3h','LE',0.03380152204812716]
];
const CP3 = [
  ['range_72h','GE',0.26843286826463253],
  ['atr7d_pct','GE',0.021824046950969615],
  ['rebound_30d_low','GE',0.5852599101174386]
];
const CP6 = [
  ['relative_return_since_open','LE',-0.0398418265942421],
  ['range_7d','GE',0.4186662939780034],
  ['range_90d','GE',1.1617279247896113]
];
const CP12 = [
  ['ret_72h','LE',-0.10039637227522144],
  ['ret_14d','LE',-0.27361717155887444]
];

function stableShard(symbol){let n=0; for(const ch of symbol) n+=ch.codePointAt(0); return n%3;}
function finite(x){return Number.isFinite(x);}
function pass(v,dir,t){return finite(v) && (dir==='GE' ? v>=t : v<=t);}
function pct(a,b){return finite(a)&&finite(b)&&b!==0 ? a/b-1 : NaN;}
function rollingMax(a,i,n){if(i-n+1<0)return NaN; let m=-Infinity; for(let k=i-n+1;k<=i;k++)m=Math.max(m,a[k]); return m;}
function rollingMin(a,i,n){if(i-n+1<0)return NaN; let m=Infinity; for(let k=i-n+1;k<=i;k++)m=Math.min(m,a[k]); return m;}
function rollingMean(a,i,n,shift=0){const end=i-shift, start=end-n+1; if(start<0)return NaN; let s=0,c=0; for(let k=start;k<=end;k++){if(finite(a[k])){s+=a[k];c++;}} return c===n?s/n:NaN;}

function normaliseBars(raw){
  // Accept Binance kline arrays or objects. Return ascending unique completed bars only.
  const out=[];
  for(const r of raw){
    if(Array.isArray(r)) out.push({time:+r[0],open:+r[1],high:+r[2],low:+r[3],close:+r[4],volume:+r[5],quote_volume:+r[7],trades:+r[8],taker_buy_base:+r[9]});
    else out.push({time:+(r.openTime??r.time??r.open_time),open:+r.open,high:+r.high,low:+r.low,close:+r.close,volume:+r.volume,quote_volume:+(r.quoteAssetVolume??r.quote_volume),trades:+(r.numberOfTrades??r.trades),taker_buy_base:+(r.takerBuyBaseAssetVolume??r.taker_buy_base)});
  }
  out.sort((a,b)=>a.time-b.time);
  return out.filter((x,i)=>i===out.length-1||x.time!==out[i+1].time);
}

function buildFeatures(bars, btcBars){
  const bmap=new Map(btcBars.map(x=>[x.time,x.close]));
  const n=bars.length, close=bars.map(x=>x.close), high=bars.map(x=>x.high), low=bars.map(x=>x.low), volume=bars.map(x=>x.volume), trades=bars.map(x=>x.trades), qv=bars.map(x=>x.quote_volume), tb=bars.map(x=>x.taker_buy_base);
  const tr=new Array(n).fill(NaN), r1=new Array(n).fill(NaN), r4=new Array(n).fill(NaN);
  for(let i=0;i<n;i++){
    if(i>0){tr[i]=Math.max(high[i]-low[i],Math.abs(high[i]-close[i-1]),Math.abs(low[i]-close[i-1])); r1[i]=pct(close[i],close[i-1]);}
    if(i>=4)r4[i]=pct(close[i],close[i-4]);
  }
  const out=[];
  const horizons={ret_6h:6,ret_24h:24,ret_72h:72,ret_7d:168,ret_14d:336,ret_30d:720,ret_60d:1440,ret_90d:2160};
  const ranges={range_24h:24,range_72h:72,range_7d:168,range_14d:336,range_30d:720,range_90d:2160};
  for(let i=0;i<n;i++){
    const f={time:bars[i].time};
    for(const [k,h] of Object.entries(horizons)) f[k]=i>=h?pct(close[i],close[i-h]):NaN;
    for(const [k,h] of Object.entries(ranges)){const hi=rollingMax(high,i,h),lo=rollingMin(low,i,h); f[k]=finite(hi)&&finite(lo)&&lo!==0?hi/lo-1:NaN;}
    f.atr24_pct=finite(rollingMean(tr,i,24))?rollingMean(tr,i,24)/close[i]:NaN;
    f.atr7d_pct=finite(rollingMean(tr,i,168))?rollingMean(tr,i,168)/close[i]:NaN;
    f.atr30d_pct=finite(rollingMean(tr,i,720))?rollingMean(tr,i,720)/close[i]:NaN;
    f.atr24_vs_7d=finite(f.atr24_pct)&&finite(f.atr7d_pct)&&f.atr7d_pct!==0?f.atr24_pct/f.atr7d_pct:NaN;
    f.range24_vs_7d=finite(f.range_24h)&&finite(f.range_7d)&&f.range_7d!==0?f.range_24h/f.range_7d:NaN;
    const hi7=rollingMax(high,i,168),hi30=rollingMax(high,i,720),hi90=rollingMax(high,i,2160),lo7=rollingMin(low,i,168),lo30=rollingMin(low,i,720);
    f.dist_7d_high=finite(hi7)?close[i]/hi7-1:NaN; f.dist_30d_high=finite(hi30)?close[i]/hi30-1:NaN; f.dist_90d_high=finite(hi90)?close[i]/hi90-1:NaN;
    f.rebound_7d_low=finite(lo7)?close[i]/lo7-1:NaN; f.rebound_30d_low=finite(lo30)?close[i]/lo30-1:NaN;
    f.worst_1h_ret_7d=rollingMin(r1,i,168); f.worst_4h_ret_14d=rollingMin(r4,i,336);
    const vbase=rollingMean(volume,i,72,1),tbase=rollingMean(trades,i,72,1),v6=rollingMean(volume,i,6),v24=rollingMean(volume,i,24),t6=rollingMean(trades,i,6);
    f.volume_ratio_6h=finite(v6)&&finite(vbase)&&vbase!==0?v6/vbase:NaN; f.volume_ratio_24h=finite(v24)&&finite(vbase)&&vbase!==0?v24/vbase:NaN; f.trade_ratio_6h=finite(t6)&&finite(tbase)&&tbase!==0?t6/tbase:NaN;
    const taker=[]; for(let k=Math.max(0,i-5);k<=i;k++) taker.push(volume[k]!==0?tb[k]/volume[k]:NaN); f.taker_buy_6h=taker.length===6&&taker.every(finite)?taker.reduce((a,b)=>a+b,0)/6:NaN;
    let qsum=0,qok=i>=23; if(qok){for(let k=i-23;k<=i;k++){if(!finite(qv[k])){qok=false;break;}qsum+=qv[k];}} f.quote_volume_24h_log=qok?Math.log1p(qsum):NaN;
    const btcNow=bmap.get(bars[i].time);
    const br=(h)=>{if(i<h)return NaN; const bp=bmap.get(bars[i-h].time); return finite(btcNow)&&finite(bp)?pct(btcNow,bp):NaN;};
    f.rs_24h=finite(f.ret_24h)&&finite(br(24))?f.ret_24h-br(24):NaN; f.rs_7d=finite(f.ret_7d)&&finite(br(168))?f.ret_7d-br(168):NaN; f.rs_30d=finite(f.ret_30d)&&finite(br(720))?f.ret_30d-br(720):NaN;
    out.push(f);
  }
  return out;
}

function rfScore(modelPack,family,feat){
  const m=modelPack.models[family], order=modelPack.feature_order;
  const x=order.map((k,i)=>finite(feat[k])?feat[k]:m.imputer_median[i]);
  let sum=0;
  for(const tr of m.trees){let node=0; while(tr.f[node]>=0){node=x[tr.f[node]]<=tr.x[node]?tr.l[node]:tr.r[node];} sum+=tr.p[node];}
  return sum/m.trees.length;
}

function makeFamilyAlerts(modelPack,features,startIdx){
  const alerts=[];
  for(const family of ['CAPITULATION','BASE','MOMENTUM']){
    const threshold=modelPack.thresholds[family], scores=[];
    for(let i=0;i<features.length;i++) scores.push(rfScore(modelPack,family,features[i]));
    let last=-1e9;
    for(let i=startIdx;i<features.length;i++){
      const valid=FEATURE_ORDER.filter(k=>finite(features[i][k])).length/FEATURE_ORDER.length>=0.80;
      if(valid && scores[i]>=threshold){if(i-last<24)continue; last=i; alerts.push({i,time:features[i].time,family,score:scores[i]});}
    }
  }
  alerts.sort((a,b)=>a.time-b.time || a.family.localeCompare(b.family));
  return alerts;
}

function round5Filter(alerts,features){
  return alerts.filter(a=>{
    const i=a.i; if(i<24)return false;
    const vals={
      ret_6h_delta_24h:features[i].ret_6h-features[i-24].ret_6h,
      ret_24h_delta_12h:features[i].ret_24h-features[i-12].ret_24h,
      ret_24h_delta_3h:features[i].ret_24h-features[i-3].ret_24h
    };
    return ROUND5.every(([k,d,t])=>pass(vals[k],d,t));
  });
}

function buildEpisodes(selected){
  const ep=[]; let cur=null;
  for(const a of selected){
    if(!cur || (a.time-cur.lastTime)>96*3600000){cur={startTime:a.time,startIdx:a.i,lastTime:a.time,family:a.family,openingScore:a.score,alerts:[a]}; ep.push(cur);}
    else {cur.lastTime=a.time;cur.alerts.push(a);}
  }
  return ep;
}

function mfeSinceOpen(bars,i,j){let hi=bars[i].close; for(let k=i+1;k<=j;k++)hi=Math.max(hi,bars[k].high); return hi/bars[i].close-1;}
function checkpointPass(list,vals){return list.every(([k,d,t])=>pass(vals[k],d,t));}

function evaluateBASE12(modelPack,coinRaw,btcRaw,targetTimes){
  const bars=normaliseBars(coinRaw), btc=normaliseBars(btcRaw); const feat=buildFeatures(bars,btc); const byTime=new Map(bars.map((b,i)=>[b.time,i]));
  const targetSet=new Set(targetTimes);
  const earliest=Math.min(...targetTimes)-120*3600000; let startIdx=0; while(startIdx<bars.length && bars[startIdx].time<earliest)startIdx++;
  startIdx=Math.max(2160,startIdx-24);
  const alerts=round5Filter(makeFamilyAlerts(modelPack,feat,startIdx),feat), episodes=buildEpisodes(alerts), out=[];
  for(const e of episodes){
    if(e.family!=='BASE')continue;
    const i=e.startIdx; const j3=i+3,j6=i+6,j12=i+12; if(j12>=bars.length)continue;
    const evalCp=(j,list)=>{
      if(mfeSinceOpen(bars,i,j)>0.05)return false;
      const vals={...feat[j]};
      const btcStart=btc.find(x=>x.time===bars[i].time), btcJ=btc.find(x=>x.time===bars[j].time);
      vals.relative_return_since_open=pct(bars[j].close,bars[i].close)-(btcStart&&btcJ?pct(btcJ.close,btcStart.close):NaN);
      return checkpointPass(list,vals);
    };
    if(evalCp(j3,CP3))continue;
    if(evalCp(j6,CP6))continue;
    if(!evalCp(j12,CP12))continue;
    if(targetSet.has(bars[j12].time)) out.push({symbol:null,signalTime:bars[j12].time,entryPrice:bars[j12].close,candidateStartTime:bars[i].time,baseOpeningScore:e.openingScore,firstAcceptedCheckpoint:12});
  }
  return out;
}

function evaluateBaseEpisodeCheckpoints(coinRaw,btcRaw,candidateStartTime){
  const bars=normaliseBars(coinRaw), btc=normaliseBars(btcRaw), feat=buildFeatures(bars,btc);
  const byTime=new Map(bars.map((b,i)=>[b.time,i]));
  const i=byTime.get(+candidateStartTime);
  if(i===undefined) throw new Error('candidate start candle not found');
  const evalCp=(h,list)=>{
    const j=i+h; if(j>=bars.length) return {due:false,pass:false,mfe:NaN,time:null,close:NaN};
    const mfe=mfeSinceOpen(bars,i,j); if(mfe>0.05) return {due:true,pass:false,mfe,time:bars[j].time,close:bars[j].close,reason:'MFE_GT_5'};
    const vals={...feat[j]};
    const bmap=new Map(btc.map(x=>[x.time,x.close]));
    const bi=bmap.get(bars[i].time), bj=bmap.get(bars[j].time);
    vals.relative_return_since_open=finite(bi)&&finite(bj)?pct(bars[j].close,bars[i].close)-pct(bj,bi):NaN;
    return {due:true,pass:checkpointPass(list,vals),mfe,time:bars[j].time,close:bars[j].close,values:vals};
  };
  const c3=evalCp(3,CP3), c6=evalCp(6,CP6), c12=evalCp(12,CP12);
  let firstAccepted=null;
  if(c3.due&&c3.pass) firstAccepted=3;
  else if(c6.due&&c6.pass) firstAccepted=6;
  else if(c12.due&&c12.pass) firstAccepted=12;
  return {candidateStartTime:+candidateStartTime,candidateOpenPrice:bars[i].close,checkpoint3:c3,checkpoint6:c6,checkpoint12:c12,firstAcceptedCheckpoint:firstAccepted,signalTime:firstAccepted===12?c12.time:null,entryPrice:firstAccepted===12?c12.close:null};
}

return {FEATURE_ORDER,FAMILY_THRESHOLDS,ROUND5,CP3,CP6,CP12,stableShard,normaliseBars,buildFeatures,rfScore,makeFamilyAlerts,round5Filter,buildEpisodes,mfeSinceOpen,checkpointPass,evaluateBASE12,evaluateBaseEpisodeCheckpoints};