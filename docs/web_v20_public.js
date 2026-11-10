/* V20 공개 연구 시연: 저장된 과거 fold CSV만 읽습니다. 신규 예측, 공식 발급, 운영 DB 접근은 없습니다. */
'use strict';
const DATA_URL='./smp_improvement_ml_lgbm/final_fold_demo.csv.gz';
const PREFERRED={1:'V11_ExtraTrees_fold_refit',2:'ml_anchor_blend',3:'ml_anchor_blend'};
const LABELS={1:'V11 ExtraTrees',2:'HGB + seasonal anchor',3:'HGB + seasonal anchor'};
const els={};
const cache=new Map();
let current=null;
const today=()=>new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Seoul',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
const $=id=>document.getElementById(id);
const fmt=(x,d=2)=>Number.isFinite(x)?Number(x).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d}):'—';
const clamp=(x,a,b)=>Math.max(a,Math.min(x,b));
const failure=(text)=>{const el=$('error');el.hidden=false;el.textContent=text;$('dataStatus').textContent='데이터 확인 필요';};
function parseCSV(input){
  const rows=[];let row=[],cell='',inside=false;
  for(let i=0;i<input.length;i++){
    const c=input[i];
    if(c==='"'){if(inside&&input[i+1]==='"'){cell+='"';i++}else inside=!inside}
    else if(c===','&&!inside){row.push(cell);cell=''}
    else if((c==='\n'||c==='\r')&&!inside){if(c==='\r'&&input[i+1]==='\n')i++;row.push(cell);cell='';if(row.length>1||row[0].trim())rows.push(row);row=[]}
    else cell+=c;
  }
  if(cell||row.length){row.push(cell);rows.push(row)}
  const keys=rows.shift().map(s=>s.replace(/^\uFEFF/,'').trim());
  return rows.map(r=>{const o={};keys.forEach((k,i)=>o[k]=r[i]??'');return o});
}
async function loadData(){
  if(!('DecompressionStream'in window))throw Error('이 브라우저는 압축 CSV를 읽을 수 없습니다. 최신 Chrome, Edge 또는 Safari로 접속해 주세요.');
  const res=await fetch(DATA_URL,{cache:'no-cache'});
  if(!res.ok)throw Error('연구 CSV를 다운로드하지 못했습니다 (HTTP '+res.status+'). GitHub Pages의 발행 브랜치와 docs 폴더 설정을 확인해 주세요.');
  const body=res.body.pipeThrough(new DecompressionStream('gzip'));
  const text=await new Response(body).text();
  const rows=parseCSV(text);
  if(!rows.length)throw Error('연구 데이터가 비어 있습니다.');
  for(const row of rows){
    const h=Number(row.horizon_days),day=String(row.target_date||''),series=String(row.series||'');
    const hour=Number(row.hour),price=Number(row.smp);
    if(![1,2,3].includes(h)||!day||!series||!Number.isInteger(hour)||hour<0||hour>23||!Number.isFinite(price))continue;
    const k=h+'|'+day;let g=cache.get(k);if(!g){g={h,day,series:new Map()};cache.set(k,g)}
    if(!g.series.has(series))g.series.set(series,new Map());
    g.series.get(series).set(hour,price);
  }
  if(!cache.size)throw Error('CSV에서 24시간 예측 자료를 찾지 못했습니다. 파일의 horizon_days, target_date, series, hour, smp 열을 확인해 주세요.');
  const available=[1,2,3].filter(h=>[...cache.values()].some(g=>g.h===h&&valid(g)));
  if(!available.length)throw Error('최종 연구 모델과 실제 SMP, 기준선이 함께 있는 24시간 데이터가 없습니다.');
  const hz=$('horizon');for(const op of hz.options)op.disabled=!available.includes(Number(op.value));
  hz.value=String(available[0]);
  hz.addEventListener('change',selectHorizon);
  $('date').addEventListener('change',update);
  $('period').addEventListener('change',update);
  $('latestDate').addEventListener('click',()=>{const dates=[...cache.values()].filter(g=>g.h===Number($('horizon').value)&&valid(g)&&g.day<=today()).map(g=>g.day).sort();$('date').value=dates.at(-1)||today();update();});
  $('date').max=today();$('date').value=today();
  $('calculate').addEventListener('click',calculate);
  $('eff').addEventListener('input',()=>{$('effLabel').textContent=$('eff').value+'%'});
  for(let i=0;i<=24;i++){
    if(i<24)$('start').add(new Option(String(i).padStart(2,'0')+':00',i));
    if(i>0)$('end').add(new Option(i===24?'24:00':String(i).padStart(2,'0')+':00',i));
  }
  $('start').value='0';$('end').value='24';
  $('dataStatus').textContent='GitHub에 저장된 검증 결과 연결 완료';
  selectHorizon();
}
function valid(g){
 const lists=[PREFERRED[g.h],'기준선 SMP','실제 SMP'];
 return lists.every(k=>g.series.has(k)&&g.series.get(k).size===24);
}
function selectHorizon(){
 $('date').max=today();
 const dates=[...cache.values()].filter(g=>g.h===Number($('horizon').value)&&valid(g)&&g.day<=today()).map(g=>g.day).sort();
 if(!cache.get($('horizon').value+'|'+$('date').value)||!valid(cache.get($('horizon').value+'|'+$('date').value)))$('date').value=dates.at(-1)||today();
 update();
}
function selected(){
 const g=cache.get($('horizon').value+'|'+$('date').value);
 if(!g||!valid(g))return null;
 const toArray=(name)=>Array.from({length:24},(_,i)=>g.series.get(name).get(i));
 return {...g,actual:toArray('실제 SMP'),baseline:toArray('기준선 SMP'),predict:toArray(PREFERRED[g.h])};
}
const mae=(a,b)=>a.reduce((s,x,i)=>s+Math.abs(x-b[i]),0)/a.length;
const rmse=(a,b)=>Math.sqrt(a.reduce((s,x,i)=>s+(x-b[i])**2,0)/a.length);
function clearDay(){
 current=null;
 $('dayMetrics').textContent='';$('peakCase').textContent='';$('hourlyError').replaceChildren();
 for(const id of ['cheapHour','cheapPrice','expectedCost','actualCost','regret','saving','energyNote'])$(id).textContent='—';
 $('chargingSlots').replaceChildren();$('weatherHours').replaceChildren();
 $('resultHeading').textContent='선택한 날짜의 예측 자료가 없습니다';
 $('planMessage').textContent='자료가 있는 날짜를 선택하면 충전 계획을 계산할 수 있습니다.';
 $('weatherIcon').textContent='☁';$('weatherTitle').textContent='가격 예보 미발행';
 $('weatherNote').textContent='당일 예측 데이터가 연결되면 가격 날씨가 표시됩니다. 최근 데이터 날짜에서 과거 시연을 확인할 수 있습니다.';
}
function update(){
 $('date').max=today();
 if(!$('date').value||$('date').value>today())$('date').value=today();
 const g=selected();$('error').hidden=true;
 $('dataStatus').textContent='조회 '+$('date').value+' · 저장된 과거 검증 데이터';
 if(!g){clearDay();$('modelName').textContent='선택한 날짜의 예측 없음 · 실시간 예측 미연결';}
 else{
 current=g;
 const min=Math.min(...g.predict),h=g.predict.indexOf(min);
 $('cheapHour').textContent=String(h).padStart(2,'0')+':00';
 $('cheapPrice').textContent='충전 맑음 · 최저 '+fmt(min)+'원/kWh';
 $('modelName').textContent='선정 모델: '+LABELS[g.h]+' · 검증 대상 '+g.day;
  drawWeather(g);drawDiagnostics(g);calculate();
 }
 drawChart(g);
}
function drawDiagnostics(g){
 const lines=(x,y)=>`${fmt(mae(x,y))} / ${fmt(rmse(x,y))}`;
 $('dayMetrics').textContent=`선택일 ${g.day} · 24시간 MAE / RMSE (원/kWh): 선정 모델 ${lines(g.predict,g.actual)} · 단순 기준선 ${lines(g.baseline,g.actual)}`;
 const peakHour=g.actual.indexOf(Math.max(...g.actual)),actual=g.actual[peakHour],pred=g.predict[peakHour],err=pred-actual;
 $('peakCase').textContent=`선택일 최고 SMP ${String(peakHour).padStart(2,'0')}시 · 실제 ${fmt(actual)}원/kWh · AI 오차 ${err<0?'과소예측 ':'과대예측 '}${fmt(Math.abs(err))}원/kWh${actual>=150?' · 150원/kWh 이상 피크':''}`;
 const table=document.createElement('table');table.innerHTML='<thead><tr><th>시각</th><th>실제 SMP</th><th>선정 AI</th><th>AI 오차</th><th>기준선</th><th>기준선 오차</th></tr></thead>';
 const body=document.createElement('tbody');
 for(let h=0;h<24;h++){
  const tr=document.createElement('tr');if(g.actual[h]>=150)tr.className='peak-row';
  const values=[String(h).padStart(2,'0')+':00',fmt(g.actual[h]),fmt(g.predict[h]),fmt(g.predict[h]-g.actual[h]),fmt(g.baseline[h]),fmt(g.baseline[h]-g.actual[h])];
  for(const [i,value]of values.entries()){const td=document.createElement('td');td.textContent=value;if((i===3||i===5)&&Number(value.replaceAll(',',''))<0)td.className='under-prediction';tr.append(td);}
  body.append(tr);
 }
 table.append(body);$('hourlyError').replaceChildren(table);
}
function drawWeather(g){
 const sorted=[...g.predict].sort((a,b)=>a-b),low=sorted[5],high=sorted[18];
 const weather=p=>p<=low?['☀','충전 맑음','sunny']:p>=high?['♨','가격 폭염주의','hot']:['☁','충전 보통','cloudy'];
 $('weatherIcon').textContent='☀';$('weatherTitle').textContent='충전 맑음 시간대를 확인하세요';
 $('weatherNote').textContent='하루 예측 가격 하위 25%는 충전 맑음, 상위 25%는 가격 폭염주의입니다. 동일 가격은 맑음을 우선하며, 기상특보가 아닌 상대적인 SMP 가격 안내입니다.';
 $('weatherHours').replaceChildren(...g.predict.map((p,h)=>{const [icon,label,cls]=weather(p),el=document.createElement('div');el.className='weather-hour '+cls;el.textContent=String(h).padStart(2,'0')+'시 '+icon;el.title=label+' · '+fmt(p)+'원/kWh';el.setAttribute('aria-label',h+'시 '+label+' '+fmt(p)+'원/kWh');return el;}));
}
function svgEl(tag,attrs,text){
 const ns='http://www.w3.org/2000/svg',e=document.createElementNS(ns,tag);
 for(const [k,v]of Object.entries(attrs||{}))e.setAttribute(k,v);
 if(text!==undefined)e.textContent=text;
 return e;
}
function drawChart(g){
 const area=$('chartArea');area.replaceChildren();
 const period=$('period').value,day=$('date').value;
 let labels;
 if(period==='day'){
  labels=Array.from({length:24},(_,i)=>String(i).padStart(2,'0')+':00');
 }else{
  const end=new Date(day+'T00:00:00Z'),start=new Date(end);
  if(period==='month'){start.setUTCDate(1);start.setUTCMonth(start.getUTCMonth()-1);start.setUTCDate(Math.min(end.getUTCDate(),new Date(Date.UTC(start.getUTCFullYear(),start.getUTCMonth()+1,0)).getUTCDate()));}
  else {start.setUTCFullYear(start.getUTCFullYear()-1);}
  start.setUTCDate(start.getUTCDate()+1);
  const groups=[...cache.values()].filter(x=>x.h===Number($('horizon').value)&&valid(x)&&x.day>=start.toISOString().slice(0,10)&&x.day<=day).sort((a,b)=>a.day.localeCompare(b.day));
  labels=groups.map(x=>x.day);
  const avg=(x,key)=>[...x.series.get(key).values()].reduce((a,b)=>a+b,0)/24;
  g=groups.length?{day:labels[0]+' ~ '+labels.at(-1),predict:groups.map(x=>avg(x,PREFERRED[x.h])),baseline:groups.map(x=>avg(x,'기준선 SMP')),actual:groups.map(x=>avg(x,'실제 SMP'))}:null;
 }
 $('chartTitle').textContent=(period==='day'?'하루 · 시간별':period==='month'?'한 달 · 일평균':'1년 · 일평균')+' SMP 가격 곡선';
 if(!g){const p=document.createElement('p');p.className='loading';p.textContent='선택한 기간에 저장된 데이터가 없습니다. 최근 데이터 날짜를 선택해 주세요.';area.append(p);return;}
 if(period!=='day'){const note=document.createElement('p');note.textContent=g.day+' · 자료가 있는 '+labels.length+'일 표시 (누락일은 연결하지 않음)';area.append(note);}
 const W=920,H=325,margin={l:64,r:25,t:22,b:42},w=W-margin.l-margin.r,h=H-margin.t-margin.b;
 const values=[...g.predict,...g.baseline,...g.actual],lo=Math.min(...values),hi=Math.max(...values),diff=Math.max(10,hi-lo);
 const min=lo-diff*.1,max=hi+diff*.1;
 const xp=i=>margin.l+(period==='day'?i/23:(new Date(labels[i])-new Date(labels[0]))/Math.max(86400000,new Date(labels.at(-1))-new Date(labels[0])))*w;
 const yp=v=>margin.t+(max-v)/(max-min)*h;
 const svg=svgEl('svg',{viewBox:'0 0 '+W+' '+H,role:'img','aria-label':g.day+' SMP 예측과 실제값의 가격 비교 그래프',preserveAspectRatio:'none'});
 for(let i=0;i<=5;i++){
  const val=min+(max-min)*(5-i)/5,y=margin.t+h*i/5;
  svg.append(svgEl('line',{x1:margin.l,y1:y,x2:W-margin.r,y2:y,class:'grid'}));
  svg.append(svgEl('text',{x:margin.l-13,y:y+4,'text-anchor':'end'},fmt(val,0)));
 }
 svg.append(svgEl('line',{x1:margin.l,y1:margin.t+h,x2:W-margin.r,y2:margin.t+h,class:'axis'}));
 const ticks=[...new Set(Array.from({length:Math.min(7,labels.length)},(_,i)=>Math.round(i*(labels.length-1)/Math.max(1,Math.min(7,labels.length)-1))))];
 for(const i of ticks)svg.append(svgEl('text',{x:xp(i),y:H-12,'text-anchor':'middle'},period==='day'?labels[i]:labels[i].slice(5)));
 for(const [cls,arr]of [['series-actual',g.actual],['series-base',g.baseline],['series-pred',g.predict]]){
  const pts=arr.map((v,i)=>{const gap=period!=='day'&&i>0&&(new Date(labels[i])-new Date(labels[i-1]))>86400000;return (i&&!gap?'L':'M')+xp(i).toFixed(2)+' '+yp(v).toFixed(2)}).join(' ');
  svg.append(svgEl('path',{d:pts,class:'plot '+cls}));
  arr.forEach((v,i)=>{const name=cls==='series-pred'?'선정 AI':cls==='series-base'?'단순 기준선':'실제 SMP';const fill=cls==='series-pred'?'#0e9188':cls==='series-base'?'#cf995c':'#60758a';const dot=svgEl('circle',{cx:xp(i),cy:yp(v),r:labels.length>40?2:3,fill});dot.append(svgEl('title',{},labels[i]+' · '+name+' '+fmt(v)+'원/kWh'));svg.append(dot);});
 }

 area.append(svg);
}
function readNumber(id){
 const x=Number($(id).value);
 if(!Number.isFinite(x))throw Error('숫자 입력값을 확인해 주세요.');
 return x;
}
function plan(prices,energy,capacity,power,start,end){
 const candidates=[];
 for(let i=start;i<end;i++)candidates.push({i,p:prices[i]});
 candidates.sort((a,b)=>a.p-b.p||a.i-b.i);
 const slots=Array(24).fill(0);let remain=energy;
 for(const item of candidates){
   if(remain<=1e-9)break;
   const used=Math.min(power,remain);slots[item.i]=used;remain-=used;
 }
 if(remain>1e-7)throw Error('연결시간이 부족해 목표 SOC를 달성할 수 없습니다. 충전 종료시각을 늦추거나 출력을 높여 주세요.');
 return slots;
}
const cost=(prices,slots,wear)=>prices.reduce((s,p,i)=>s+p*slots[i],0)+wear;
function calculate(){
 if(!current)return;
 const note=$('planMessage');
 try{
   const cap=readNumber('capacity'),power=readNumber('power'),initial=readNumber('socInitial'),target=readNumber('socTarget');
   const start=readNumber('start'),end=readNumber('end'),eff=readNumber('eff')/100;
   if(!(cap>0&&cap<=300&&power>0&&power<=350&&eff>0&&initial>=0&&initial<=100&&target>initial&&target<=100&&start>=0&&end<=24&&end>start))throw Error('배터리, SOC, 시간 조건을 확인해 주세요. 목표 SOC는 현재 SOC보다 높아야 합니다.');
   const battery=cap*(target-initial)/100,grid=battery/eff,wear=battery*5;
   const pred=plan(current.predict,grid,cap,power,start,end);
   const oracle=plan(current.actual,grid,cap,power,start,end);
   const immediate=Array(24).fill(0);let remaining=grid;
   for(let i=start;i<end&&remaining>1e-9;i++){immediate[i]=Math.min(power,remaining);remaining-=immediate[i]}
   const expCost=cost(current.predict,pred,wear),actCost=cost(current.actual,pred,wear),opt=cost(current.actual,oracle,wear),now=cost(current.actual,immediate,wear);
   const regret=actCost-opt,saving=now-actCost;
   $('resultHeading').textContent=current.day+' 충전 추천';
   $('expectedCost').textContent=fmt(expCost)+'원';
   $('actualCost').textContent=fmt(actCost)+'원';
   $('regret').textContent=fmt(Math.max(0,regret))+'원';
   $('saving').textContent=(saving>=0?'절감 ':'추가 ')+fmt(Math.abs(saving))+'원';
   $('energyNote').textContent=fmt(grid,2)+'kWh (계통 기준)';
   $('chargingSlots').replaceChildren(...pred.map((kw,i)=>{
     const element=document.createElement('div');element.className='slot'+(kw>0?' chosen':'')+(kw>0&&kw<power?' partial':'');
     element.style.setProperty('--fraction',(kw/power*100)+'%');
     element.title=String(i).padStart(2,'0')+':00 · '+fmt(kw,2)+' kWh · 예측 SMP '+fmt(current.predict[i])+'원/kWh';
     return element;
   }));
   const slots=pred.flatMap((kw,h)=>kw>1e-8?[String(h).padStart(2,'0')+':00 ('+fmt(kw,2)+'kWh)']:[]);
   note.className='plan-message';
   note.textContent='추천 시간: '+slots.join(', ')+'. 고정 가정으로 배터리 통과 에너지당 열화비용 5원을 포함했습니다. SMP 기반 가상 정산이며 실제 요금이 아닙니다.';
 }catch(e){
   $('expectedCost').textContent='—';$('actualCost').textContent='—';$('regret').textContent='—';$('saving').textContent='—';$('energyNote').textContent='—';$('chargingSlots').replaceChildren();
   note.textContent=e.message;note.className='plan-message warning';
 }
}
loadData().catch(e=>failure(e.message));
