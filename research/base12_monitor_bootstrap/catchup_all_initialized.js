#!/usr/bin/env node
"use strict";
// One-off exact incremental catch-up of existing BASE12 research state. No new symbols, orders or strategy edits.
const fs=require("fs"),path=require("path"),H=3600000,ROOT=__dirname,OUT=path.join(ROOT,"output");
const CORE=new Function(fs.readFileSync(path.join(ROOT,"base12_monitor_core.js"),"utf8"))();
const pack=JSON.parse(fs.readFileSync(path.join(ROOT,"family_models.json"),"utf8"));
if(JSON.stringify(pack.feature_order)!==JSON.stringify(CORE.FEATURE_ORDER))throw Error("Frozen feature order mismatch");
for(const [family,threshold] of Object.entries(CORE.FAMILY_THRESHOLDS))
  if(pack.thresholds[family]!==threshold||pack.models[family].trees.length!==350)throw Error("Frozen model mismatch "+family);
const cols="symbol,shard,state_revision,initialized_utc,last_scanned_hour_utc,last_raw_capitulation_alert_utc,last_raw_base_alert_utc,last_raw_momentum_alert_utc,episode_start_utc,episode_family,episode_open_score,episode_start_price,episode_last_selected_alert_utc,precursor_alert_count,episode_certainty,checkpoint_3h,checkpoint_6h,checkpoint_12h,signal_time_utc,signal_entry_price,signal_emitted,signal_emitted_at_utc,status,last_updated_utc,last_error".split(",");
function splitCSV(s){let c=[],v="",quote=false;for(let i=0;i<s.length;i++){let ch=s[i];if(ch==='"'){if(quote&&s[i+1]==='"'){v+='"';i++;}else quote=!quote;}else if(ch===','&&!quote){c.push(v);v="";}else v+=ch;}c.push(v);if(quote)throw Error("unclosed csv quote");return c;}
const read=fs.readFileSync(path.join(ROOT,"catchup_state_input.csv"),"utf8").trim().split(/\r?\n/);
if(JSON.stringify(splitCSV(read[0]))!==JSON.stringify(cols))throw Error("CSV schema/order mismatch");
const rows=read.slice(1).map(line=>{let v=splitCSV(line);if(v.length!==cols.length)throw Error("CSV field count");return Object.fromEntries(cols.map((k,i)=>[k,v[i]]));});
if(rows.length!==231||new Set(rows.map(x=>x.symbol)).size!==231)throw Error("Expected 231 unique rows");
for(const row of rows)if(CORE.stableShard(row.symbol)!==Number(row.shard))throw Error("Shard mismatch "+row.symbol);
const iso=x=>new Date(x).toISOString();
const ms=x=>x?Date.parse(x):NaN;
const now=iso(Date.now()),cutoff=Math.floor(Date.now()/H)*H-H;
const earliest=Math.min(...rows.map(x=>ms(x.last_scanned_hour_utc)))-2205*H;
if(!Number.isFinite(earliest))throw Error("Invalid existing timestamps");
const urls=["https://data-api.binance.vision/api/v3","https://api.binance.com/api/v3"];
const sleep=t=>new Promise(r=>setTimeout(r,t));
async function request(route){
 let last;
 for(let k=0;k<3;k++){
  for(const base of urls)try{
    const r=await fetch(base+route,{headers:{"User-Agent":"base12-oneoff-backfill/1.0"},signal:AbortSignal.timeout(20000)});
    if(!r.ok)throw Error("HTTP "+r.status+" "+route.slice(0,70));
    return await r.json();
  }catch(e){last=e;}
  await sleep(600*(k+1));
 }
 throw last||Error("HTTP failed");
}
async function candles(symbol){
 let start=earliest,all=[];
 while(start<=cutoff){
  const route="/klines?"+new URLSearchParams({symbol,interval:"1h",startTime:String(start),endTime:String(cutoff),limit:"1000"}).toString();
  const part=await request(route);
  if(!Array.isArray(part)||!part.length)break;
  all.push(...part);
  const next=Number(part.at(-1)[0])+H;
  if(next<=start)throw Error("Nonadvancing klines");
  start=next;
  if(part.length<1000)break;
 }
 const bars=CORE.normaliseBars(all).filter(b=>b.time<=cutoff);
 if(!bars.length||bars.at(-1).time!==cutoff)throw Error("Missing latest completed 1h bar");
 for(let i=1;i<bars.length;i++)if(bars[i].time!==bars[i-1].time+H)throw Error("Gap in 1h bars at "+iso(bars[i].time));
 return bars;
}
const fams=[["BASE","last_raw_base_alert_utc"],["CAPITULATION","last_raw_capitulation_alert_utc"],["MOMENTUM","last_raw_momentum_alert_utc"]];
function scanRow(old,bars,btcBars){
 const row={...old},saved=ms(row.last_scanned_hour_utc);if(!Number.isFinite(saved))throw Error("Invalid saved candle");
 if(saved>=cutoff)return {row,processed:false,selected:0};
 const by=new Map(bars.map((b,i)=>[b.time,i]));
 const features=CORE.buildFeatures(bars,btcBars);
 let selected=0;
 for(let t=saved+H;t<=cutoff;t+=H){
  const i=by.get(t);
  if(i===undefined||i<2185)throw Error("Insufficient exactly aligned 90d/24h feature history at "+iso(t));
  const f=features[i];
  const valid=CORE.FEATURE_ORDER.filter(k=>Number.isFinite(f[k])).length/CORE.FEATURE_ORDER.length>=0.80;
  const kept=[];
  for(const [family,lastKey] of fams){
   const rawPrev=ms(row[lastKey]);
   if(Number.isFinite(rawPrev)&&t-rawPrev<24*H)continue;
   if(!valid)continue;
   const score=CORE.rfScore(pack,family,f);
   if(score<CORE.FAMILY_THRESHOLDS[family])continue;
   row[lastKey]=iso(t); // family cooldown is set before Round-5 filtering
   const a={i,time:t,family,score};
   if(CORE.round5Filter([a],features).length)kept.push(a);
  }
  for(const a of kept){
   selected++;
   const last=ms(row.episode_last_selected_alert_utc);
   if(!Number.isFinite(last)||t-last>96*H){
    row.episode_start_utc=iso(t);
    row.episode_family=a.family;
    row.episode_open_score=String(a.score);
    row.episode_start_price=String(bars[i].close);
    row.episode_last_selected_alert_utc=iso(t);
    row.precursor_alert_count="1";
    row.episode_certainty="EXACT";
    row.checkpoint_3h="NOT_DUE";row.checkpoint_6h="NOT_DUE";row.checkpoint_12h="NOT_DUE";
    row.signal_time_utc="";row.signal_entry_price="";row.signal_emitted="false";row.signal_emitted_at_utc="";
    row.status="PENDING";
   }else{
    row.episode_last_selected_alert_utc=iso(t);
    row.precursor_alert_count=String(Number(row.precursor_alert_count||"0")+1);
    row.status="PENDING";
   }
  }
  row.last_scanned_hour_utc=iso(t);
 }
 row.state_revision=String(Number(row.state_revision)+1);
 row.last_updated_utc=now;
 row.last_error="";
 return {row,processed:true,selected};
}
const escape=x=>{const s=String(x??"");return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
async function main(){
 fs.mkdirSync(OUT,{recursive:true});
 const report={kind:"BASE12_ONE_OFF_INCREMENTAL_CATCHUP",research_only:true,created_at_utc:now,cutoff_utc:iso(cutoff),symbols:rows.length,succeeded:0,failed:[],unchanged:0,new_selected_alerts:0};
 const btc=await candles("BTCUSDT");
 let cursor=0;
 async function worker(){
   while(cursor<rows.length){
    const j=cursor++,old=rows[j];
    try{
      const bars=old.symbol==="BTCUSDT"?btc:await candles(old.symbol);
      const ret=scanRow(old,bars,btc);
      rows[j]=ret.row;
      if(ret.processed)report.succeeded++;else report.unchanged++;
      report.new_selected_alerts+=ret.selected;
      if((j+1)%15===0)console.log("PROGRESS",j+1,"of",rows.length,"updated",report.succeeded,"errors",report.failed.length);
    }catch(err){report.failed.push({symbol:old.symbol,error:String(err.message||err)});console.log("ERROR",old.symbol,String(err.message||err));}
   }
 }
 await Promise.all(Array.from({length:5},()=>worker()));
 report.completed_at_utc=iso(Date.now());
 report.status=report.failed.length?"PARTIAL":"SUCCESS";
 const dest=path.join(OUT,"base12_oneoff_catchup_20261008.csv");
 fs.writeFileSync(dest,cols.join(",")+"\n"+rows.map(r=>cols.map(c=>escape(r[c])).join(",")).join("\n")+"\n");
 fs.writeFileSync(path.join(OUT,"base12_oneoff_catchup_report.json"),JSON.stringify(report,null,2)+"\n");
 console.log("FINISHED",JSON.stringify({status:report.status,succeeded:report.succeeded,failed:report.failed.length,unchanged:report.unchanged,cutoff:report.cutoff_utc,alerts:report.new_selected_alerts}));
 if(!report.succeeded && report.failed.length)process.exitCode=2;
}
main().catch(err=>{console.error(err.stack||err);process.exitCode=1});
