#!/usr/bin/env node
"use strict";
// One-time Binance USDT SPOT BASE12 bootstrap. Research only, no orders.
// Does NOT change the frozen model, logic, runbook, or live strategy.
// Top-200 24h quote-volume ranking is BOOTSTRAP PRIORITY, NOT a new signal or liquidity gate.
const fs=require("fs"), path=require("path");
const H=3600000;
const ROOT=__dirname, OUTPUT=path.join(ROOT,"output");
const CORE=new Function(fs.readFileSync(path.join(ROOT,"base12_monitor_core.js"),"utf8"))();
const pack=JSON.parse(fs.readFileSync(path.join(ROOT,"family_models.json"),"utf8"));
if(JSON.stringify(pack.feature_order)!==JSON.stringify(CORE.FEATURE_ORDER)) throw Error("Frozen feature order differs");
for(const [fam,threshold] of Object.entries(CORE.FAMILY_THRESHOLDS)) {
  if(pack.thresholds[fam]!==threshold||pack.models[fam].trees.length!==350) throw Error("Frozen model shape/threshold mismatch: "+fam);
}
const existing=new Set(fs.readFileSync(path.join(ROOT,"existing_symbols.txt"),"utf8").split(/\s+/).filter(Boolean));
const stable=new Set(["USDC","FDUSD","TUSD","USDP","DAI","BUSD","EUR","EURI","AEUR","TRY","BRL","GBP","AUD","UAH","RUB","BIDR","IDRT","NGN","ZAR","VAI","UST","USTC","USD1","RLUSD","XUSD","BFUSD","USDE","USDD","USDY","USDF","USDZ","USDG","USDS","USDJ","USDX","USDO","USDN"]);
const eq=/^(?:AAPL|TSLA|AMZN|NVDA|META|GOOGL|GOOG|MSFT|MSTR|COIN|SPY|QQQ|HOOD|PLTR|NFLX|GME|PYPL|BE|SNDK|SPC|CRCL|SNX|KORU|SOXL|SOXS|INTC|AVGO|MRVL|TQQQ|AAOI|BABA|NBIS|SPCX|SNDKB|SPCXB|CRCLB|MSTRB|NVDAB|QQQB|SNXX|KORUB|SOXLB|GOOGLB|INTCB|AVGOB|SOXSB|TSLAB|MRVLB|TQQQB|AAOIB|BABAB|NBISB)$/;
const cols="symbol,shard,state_revision,initialized_utc,last_scanned_hour_utc,last_raw_capitulation_alert_utc,last_raw_base_alert_utc,last_raw_momentum_alert_utc,episode_start_utc,episode_family,episode_open_score,episode_start_price,episode_last_selected_alert_utc,precursor_alert_count,episode_certainty,checkpoint_3h,checkpoint_6h,checkpoint_12h,signal_time_utc,signal_entry_price,signal_emitted,signal_emitted_at_utc,status,last_updated_utc,last_error".split(",");
const baseUrl=["https://data-api.binance.vision/api/v3","https://api.binance.com/api/v3"];
const cutoff=Math.floor(Date.now()/H)*H-H;
const now=new Date().toISOString();
const csv=v=>{const s=String(v??"");return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
const iso=ms=>ms?new Date(ms).toISOString():"";
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function request(route, maxTries=4){
  let err;
  for(let tryNo=0;tryNo<maxTries;tryNo++){
    for(const base of baseUrl){
      try{
        const response=await fetch(base+route,{headers:{"User-Agent":"base12-frozen-bootstrap/1.0"},signal:AbortSignal.timeout(25000)});
        if(!response.ok)throw Error("HTTP "+response.status+" "+route.slice(0,70));
        return await response.json();
      }catch(e){err=e;}
    }
    await delay((tryNo+1)*1500);
  }
  throw err||Error("Binance request failed");
}
async function candles(symbol){
  const earliest=cutoff-3100*H;
  let start=earliest, all=[];
  while(start<=cutoff){
    const args=new URLSearchParams({symbol,interval:"1h",startTime:String(start),endTime:String(cutoff),limit:"1000"});
    const rows=await request("/klines?"+args.toString());
    if(!Array.isArray(rows)||!rows.length)break;
    all.push(...rows);
    const next=Number(rows[rows.length-1][0])+H;
    if(next<=start)throw Error("Kline pagination did not advance");
    start=next;
    if(rows.length<1000)break;
  }
  all=CORE.normaliseBars(all).filter(r=>r.time<=cutoff);
  if(all.length<2880)throw Error("Insufficient complete 1h bars: "+all.length);
  if(all[all.length-1].time!==cutoff)throw Error("Stale last candle");
  for(let i=1;i<all.length;i++)if(all[i].time-all[i-1].time!==H)throw Error("Missing 1h bar");
  return all;
}
function makeState(symbol,bars,btc){
  const features=CORE.buildFeatures(bars,btc);
  const minIndex=2160;
  if(features.length-minIndex<720)throw Error("Insufficient 30d after 90d warmup");
  const raw=CORE.makeFamilyAlerts(pack,features,minIndex);
  const selected=CORE.round5Filter(raw,features);
  const episodes=CORE.buildEpisodes(selected);
  const lastEp=episodes.length?episodes[episodes.length-1]:null;
  const rawLast=fam=>{const arr=raw.filter(x=>x.family===fam);return arr.length?iso(arr[arr.length-1].time):"";};
  const epFirst=lastEp?lastEp.startTime:0;
  // The 96h preceding the first observable alert are not necessarily known.
  const exact=lastEp && epFirst-bars[minIndex].time>96*H;
  const state={
    symbol,shard:CORE.stableShard(symbol),state_revision:1,initialized_utc:now,
    last_scanned_hour_utc:iso(cutoff),
    last_raw_capitulation_alert_utc:rawLast("CAPITULATION"),
    last_raw_base_alert_utc:rawLast("BASE"),
    last_raw_momentum_alert_utc:rawLast("MOMENTUM"),
    episode_start_utc:iso(epFirst),
    episode_family:lastEp?lastEp.family:"",
    episode_open_score:lastEp?lastEp.openingScore:"",
    episode_start_price:lastEp?bars[lastEp.startIdx].close:"",
    episode_last_selected_alert_utc:lastEp?iso(lastEp.lastTime):"",
    precursor_alert_count:lastEp?lastEp.alerts.length:0,
    episode_certainty:lastEp?(exact?"EXACT":"UNCERTAIN"):"",
    checkpoint_3h:"NOT_DUE",checkpoint_6h:"NOT_DUE",checkpoint_12h:"NOT_DUE",
    signal_time_utc:"",signal_entry_price:"",signal_emitted:"false",
    signal_emitted_at_utc:"",
    status:lastEp&&cutoff-lastEp.lastTime<=96*H?"PENDING":"NONE",
    last_updated_utc:now,last_error:""
  };
  return state;
}
(async()=>{
  fs.mkdirSync(OUTPUT,{recursive:true});
  const outfile=path.join(OUTPUT,"base12_new_candidates.csv");
  const stream=fs.createWriteStream(outfile);
  stream.write(cols.join(",")+"\n");
  const report={
    mode:"TOP_200_BOOTSTRAP_PRIORITY_ONLY",research_only:true,
    started_utc:now,latest_completed_candle_utc:iso(cutoff),
    preexisting_symbol_count:existing.size,volume_rank_limit:200,
    universe_count:0,ranked_count:0,already_initialized_in_top200:0,
    missing_in_top200:0,attempted:0,succeeded:0,failed:[],symbols_new:[]
  };
  const saveReport=()=>fs.writeFileSync(path.join(OUTPUT,"bootstrap_report.json"),JSON.stringify(report,null,2)+"\n");
  try{
    const info=await request("/exchangeInfo");
    const listings=(info.symbols||[]).filter(x=>x.quoteAsset==="USDT"&&x.status==="TRADING"&&x.isSpotTradingAllowed!==false&&
      !stable.has(x.baseAsset)&&!eq.test(x.baseAsset)&&!/(UP|DOWN|BULL|BEAR)USDT$/.test(x.symbol));
    const live=new Set(listings.map(x=>x.symbol));
    report.universe_count=live.size;
    const t=await request("/ticker/24hr");
    if(!Array.isArray(t))throw Error("Expected ticker array");
    const byVol=new Map(t.filter(v=>live.has(v.symbol)).map(v=>[v.symbol,Number(v.quoteVolume)]));
    const ordered=listings.map(x=>x.symbol).filter(s=>Number.isFinite(byVol.get(s))).sort((a,b)=>byVol.get(b)-byVol.get(a)).slice(0,200);
    report.ranked_count=ordered.length;
    const pending=ordered.filter(s=>!existing.has(s));
    report.already_initialized_in_top200=ordered.length-pending.length;
    report.missing_in_top200=pending.length;
    console.log("Eligible",live.size,"Top volume",ordered.length,"Existing",report.already_initialized_in_top200,"Need",pending.length);
    const btc=await candles("BTCUSDT");
    for(let i=0;i<pending.length;i++){
      const symbol=pending[i];
      report.attempted++;
      try{
        const bars=symbol==="BTCUSDT"?btc:await candles(symbol);
        const row=makeState(symbol,bars,btc);
        stream.write(cols.map(c=>csv(row[c])).join(",")+"\n");
        report.succeeded++;
        report.symbols_new.push(symbol);
        console.log("OK",i+1,"/",pending.length,symbol,row.episode_family,row.episode_certainty);
      }catch(e){
        report.failed.push({symbol,error:String(e.message||e)});
        console.log("SKIP",i+1,"/",pending.length,symbol,String(e.message||e));
      }
      saveReport();
    }
    report.completed_utc=new Date().toISOString();
    report.status=report.failed.length?"PARTIAL":"COMPLETE_TOP200";
    saveReport();
    stream.end();
    console.log("RESULT",JSON.stringify({status:report.status,attempted:report.attempted,ok:report.succeeded,failed:report.failed.length}));
  }catch(e){
    report.status="FATAL";report.fatal_error=String(e.stack||e);
    saveReport();stream.end();
    console.error("FATAL",e);
    process.exitCode=1;
  }
})().catch(e=>{console.error(e);process.exitCode=1;});
