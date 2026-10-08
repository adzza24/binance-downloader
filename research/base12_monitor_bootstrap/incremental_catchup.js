#!/usr/bin/env node
"use strict";
// One-off catch-up of EXISTING BASE12 research scanner state.
// Frozen core/model are read-only. No orders, no signal evaluation, no new symbols.
const fs=require("fs"),path=require("path");
const ROOT=__dirname, OUT=path.join(ROOT,"output");
const H=3600000, GAP=96*H, CD=24*H;
const CORE=new Function(fs.readFileSync(path.join(ROOT,"base12_monitor_core.js"),"utf8"))();
const pack=JSON.parse(fs.readFileSync(path.join(ROOT,"family_models.json"),"utf8"));
if(JSON.stringify(CORE.FEATURE_ORDER)!==JSON.stringify(pack.feature_order))throw Error("Frozen feature order mismatch");
for(const [family,threshold] of Object.entries(CORE.FAMILY_THRESHOLDS)){
  if(pack.thresholds[family]!==threshold||pack.models[family].trees.length!==350)
    throw Error("Frozen model family differs: "+family);
}
const columns="symbol,shard,state_revision,initialized_utc,last_scanned_hour_utc,last_raw_capitulation_alert_utc,last_raw_base_alert_utc,last_raw_momentum_alert_utc,episode_start_utc,episode_family,episode_open_score,episode_start_price,episode_last_selected_alert_utc,precursor_alert_count,episode_certainty,checkpoint_3h,checkpoint_6h,checkpoint_12h,signal_time_utc,signal_entry_price,signal_emitted,signal_emitted_at_utc,status,last_updated_utc,last_error".split(",");
function parseCsv(src){
 const lines=[];let row=[],field="",quoted=false;
 for(let i=0;i<src.length;i++){const c=src[i];
   if(quoted){if(c==='"'&&src[i+1]==='"'){field+='"';i++;}else if(c==='"')quoted=false;else field+=c;}
   else if(c==='"')quoted=true;else if(c===','){row.push(field);field="";}
   else if(c==='\n'){row.push(field);lines.push(row);row=[];field="";}
   else if(c!=='\r')field+=c;
 }
 if(quoted)throw Error("Malformed input CSV quotes");
 if(row.length||field){row.push(field);lines.push(row);}
 return lines;
}
const csv=s=>{s=String(s??"");return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
const iso=t=>new Date(t).toISOString();
const input=parseCsv(fs.readFileSync(path.join(ROOT,"state_input.csv"),"utf8"));
if(JSON.stringify(input[0])!==JSON.stringify(columns))throw Error("State input columns differ from 25-column BASE12 schema");
const rows=input.slice(1).map((v,i)=>{
 if(v.length!==columns.length)throw Error("State input row "+i+" has "+v.length+" columns");
 return Object.fromEntries(columns.map((c,j)=>[c,v[j]]));
});
if(new Set(rows.map(r=>r.symbol)).size!==rows.length || rows.length<200)throw Error("Input symbols are missing/duplicate");
for(const r of rows){if(+r.shard!==CORE.stableShard(r.symbol))throw Error("Shard mismatch "+r.symbol);}
const cutoff=Math.floor(Date.now()/H)*H-H;const urlBases=["https://data-api.binance.vision/api/v3","https://api.binance.com/api/v3"];
const pause=ms=>new Promise(r=>setTimeout(r,ms));
async function api(query){let last;
 for(let k=0;k<3;k++){for(const root of urlBases){
  try{const res=await fetch(root+query,{headers:{"User-Agent":"base12-incremental-research/1.0"},signal:AbortSignal.timeout(16000)});
   if(!res.ok)throw Error("HTTP "+res.status+" "+query.slice(0,90));
   return await res.json();
  }catch(e){last=e;}
 }await pause(700*(k+1));}
 throw last||Error("API access unavailable");
}
async function getBars(symbol){
 // 90d features + prior 24h Round-5 differences, with a conservative warmup buffer.
 const minStart=cutoff-2350*H;let next=minStart,data=[];
 while(next<=cutoff){
  const q="/klines?"+new URLSearchParams({symbol,interval:"1h",startTime:String(next),endTime:String(cutoff),limit:"1000"});
  const part=await api(q);if(!Array.isArray(part)||!part.length)break;
  data.push(...part);const after=Number(part[part.length-1][0])+H;if(after<=next)throw Error("Pagination stalled");
  next=after;if(part.length<1000)break;
 }
 const bars=CORE.normaliseBars(data).filter(x=>x.time<=cutoff);
 if(bars.length<2220 || bars[bars.length-1].time!==cutoff)throw Error("Insufficient history or stale latest candle ("+bars.length+")");
 for(let i=1;i<bars.length;i++)if(bars[i].time-bars[i-1].time!==H)throw Error("Non-contiguous 1h candles");
 return bars;
}
const report={type:"BASE12_EXISTING_SYMBOL_CATCHUP",research_only:true,started_at:new Date().toISOString(),
 target_latest_closed_candle_utc:iso(cutoff),input_rows:rows.length,processed:0,already_current:0,failed:[],new_raw_alerts:0,new_selected_alerts:0,new_base_episodes:0,new_momentum_episodes:0,
 updated_symbols:[],result:"IN_PROGRESS"};
function save(){fs.mkdirSync(OUT,{recursive:true});fs.writeFileSync(path.join(OUT,"sync_report.json"),JSON.stringify(report,null,2)+"\n");
 fs.writeFileSync(path.join(OUT,"state_output.csv"),columns.join(",")+"\n"+rows.map(r=>columns.map(c=>csv(r[c])).join(",")).join("\n")+"\n");
}
const parseTime=x=>x?Date.parse(x):NaN;
function step(r,bars,features){
 const since=parseTime(r.last_scanned_hour_utc);
 if(!Number.isFinite(since)||since>cutoff)throw Error("Invalid saved last-scanned candle");
 const byTime=new Map(bars.map((b,i)=>[b.time,i]));
 const first=byTime.get(since+H);
 if(first===undefined)throw Error("First missing completed candle unavailable");
 for(let i=first;i<bars.length;i++){
  const f=features[i],now=bars[i].time;
  if(i<2185)throw Error("Insufficient warmup for 90d + Round5 at "+iso(now));
  if(now<=since)throw Error("Attempted to rescan previous hour");
  const finiteCount=CORE.FEATURE_ORDER.reduce((n,k)=>n+(Number.isFinite(f[k])?1:0),0);
  if(finiteCount/CORE.FEATURE_ORDER.length<0.8)throw Error("Insufficient valid frozen features at "+iso(now));
  for(const fam of ["CAPITULATION","BASE","MOMENTUM"]){
   const score=CORE.rfScore(pack,fam,f);
   if(score<pack.thresholds[fam])continue;
   const key="last_raw_"+fam.toLowerCase()+"_alert_utc";
   const prev=parseTime(r[key]);
   if(Number.isFinite(prev) && now-prev<CD){if(prev>now)throw Error("Future family cooldown timestamp");continue;}
   r[key]=iso(now);report.new_raw_alerts++;
   const alert={i,time:now,family:fam,score};
   if(!CORE.round5Filter([alert],features).length)continue;
   report.new_selected_alerts++;
   const last=parseTime(r.episode_last_selected_alert_utc);
   if(!Number.isFinite(last)||now-last>GAP){
    r.episode_start_utc=iso(now);r.episode_family=fam;r.episode_open_score=String(score);
    r.episode_start_price=String(bars[i].close);r.episode_last_selected_alert_utc=iso(now);
    r.precursor_alert_count="1";r.episode_certainty="EXACT";
    r.checkpoint_3h="NOT_DUE";r.checkpoint_6h="NOT_DUE";r.checkpoint_12h="NOT_DUE";
    r.signal_time_utc="";r.signal_entry_price="";r.signal_emitted="false";r.signal_emitted_at_utc="";
    r.status="PENDING";
    if(fam==="BASE")report.new_base_episodes++;
    if(fam==="MOMENTUM")report.new_momentum_episodes++;
   }else{
    r.episode_last_selected_alert_utc=iso(now);
    r.precursor_alert_count=String((Number(r.precursor_alert_count)||0)+1);
   }
  }
 }
 r.last_scanned_hour_utc=iso(cutoff);
 r.last_updated_utc=new Date().toISOString();
 r.state_revision=String((Number(r.state_revision)||0)+1);
 r.last_error="";
}
(async()=>{
 save();const btc=await getBars("BTCUSDT");
 for(let j=0;j<rows.length;j++){
  const row=rows[j];const since=parseTime(row.last_scanned_hour_utc);
  if(Number.isFinite(since)&&since===cutoff){report.already_current++;continue;}
  const snapshot={...row};
  try{
   const bars=row.symbol==="BTCUSDT"?btc:await getBars(row.symbol);
   const feat=CORE.buildFeatures(bars,btc);
   step(row,bars,feat);
   report.processed++;report.updated_symbols.push(row.symbol);
   console.log("OK",j+1,"/",rows.length,row.symbol);
  }catch(e){
   Object.assign(row,snapshot);
   report.failed.push({symbol:row.symbol,error:String(e.message||e)});
   console.log("FAIL",j+1,"/",rows.length,row.symbol,String(e.message||e));
  }
  if(j%12===0)save();
 }
 report.completed_at=new Date().toISOString();report.result=report.failed.length?"PARTIAL":"COMPLETE";
 save();
 console.log("SUMMARY",JSON.stringify({processed:report.processed,already_current:report.already_current,failed:report.failed.length,cutoff:report.target_latest_closed_candle_utc,new_selected_alerts:report.new_selected_alerts}));
})().catch(e=>{report.result="FATAL";report.fatal=String(e.stack||e);save();console.error(e);process.exitCode=1;});
