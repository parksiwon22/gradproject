(() => {
'use strict';
const $ = id => document.getElementById(id);
let data, state = {origin:0, model:'sarimax',day:0,actual:true,baseline:true}, chart;
const fmt = (n,d=2) => Number(n).toLocaleString('ko-KR',{minimumFractionDigits:d,maximumFractionDigits:d});
const date = s => s.slice(0,16).replace('T',' ');
const el = (tag,text,cls) => {const n=document.createElement(tag); if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
function validate(d) {
 if(d.schema_version!==1 || d.mode!=='historical_replay' || d.origins.length!==182) throw Error('지원하지 않는 데이터 버전 또는 발행일 수입니다.');
 for(const o of d.origins){
  const origin=Date.parse(o.forecast_origin);
  if(o.targets.length!==72 || Date.parse(o.observed_cutoff)!==origin-72*3600000)throw Error('72시간 기준 검증에 실패했습니다.');
  for(let i=0;i<72;i++) {
   if(Date.parse(o.targets[i])!==origin+(i+1)*3600000)throw Error('예측 대상시간이 연속적이지 않습니다.');
   for(const arr of [o.actual,o.weekly,...d.models.map(m=>o.predictions[m.id])])if(!arr || arr.length!==72 || !Number.isFinite(arr[i]))throw Error('유효하지 않은 예측값입니다.');
  }
 }
}
function metrics(a,p){let ae=0,se=0;for(let i=0;i<a.length;i++){ae+=Math.abs(a[i]-p[i]);se+=(a[i]-p[i])**2;}return {mae:ae/a.length,rmse:Math.sqrt(se/a.length)};}
function svg(tag,attrs,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;}
function draw(){
 const o=data.origins[state.origin],m=data.models.find(m=>m.id===state.model),p=o.predictions[m.id];
 $('originLabel').textContent=date(o.forecast_origin);$('cutoffLabel').textContent=date(o.observed_cutoff);
 $('targetLabel').textContent=`${date(o.targets[0])} ~ ${date(o.targets[71])}`;
 $('modelTitle').textContent=`${m.label} · 72시간 가격 예측`;
 $('modelPolicy').textContent=`모델 ${Math.round(m.model_weight*100)}% + 주간 기준선 ${Math.round(m.weekly_weight*100)}% · 저장된 과거 결과`;
 const met=metrics(o.actual,p);$('selectedMae').textContent=fmt(met.mae);$('selectedRmse').textContent=fmt(met.rmse);
 document.querySelectorAll('[data-day]').forEach(b=>{const active=Number(b.dataset.day)===state.day;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
 $('toggleActual').setAttribute('aria-pressed',state.actual);$('toggleBaseline').setAttribute('aria-pressed',state.baseline);
 const cards=$('dayCards');cards.replaceChildren();
 for(let day=1;day<=3;day++){
  const start=(day-1)*24,vals=p.slice(start,start+24),b=el('button',undefined,`day-card${state.day===day?' active':''}`);b.type='button';b.setAttribute('aria-label',`D+${day} 구간 보기`);
  const top=el('div',undefined,'day-card-top');top.append(el('strong',`D+${day}`),el('span',o.targets[start].slice(0,10)));b.append(top,el('strong',`${fmt(vals.reduce((a,v)=>a+v,0)/24)} 원/kWh`),el('p',`예측 범위 ${fmt(Math.min(...vals))} ~ ${fmt(Math.max(...vals))}`,'range'),el('small',`이 날 MAE ${fmt(metrics(o.actual.slice(start,start+24),vals).mae)}`));b.onclick=()=>{state.day=day;draw();};cards.append(b);
 }
 const start=state.day?(state.day-1)*24:0,end=state.day?start+24:72,count=end-start;
 const root=$('priceChart');root.replaceChildren();$('chartTooltip').hidden=true;
 const W=1100,H=380,L=65,R=22,T=28,B=50,pw=W-L-R,ph=H-T-B;
 const series=[{id:'prediction',values:p,color:'#138b93',width:3.4},...(state.actual?[{id:'actual',values:o.actual,color:'#243e53',width:2.2}]:[]),...(state.baseline?[{id:'baseline',values:o.weekly,color:'#8b9da9',width:1.8,dash:'6 5'}]:[])];
 const all=series.flatMap(s=>s.values.slice(start,end)),lo=Math.floor((Math.min(...all)-8)/10)*10,hi=Math.ceil((Math.max(...all)+8)/10)*10;
 const x=i=>L+(i-start)/(count-1)*pw,y=v=>T+(hi-v)/(hi-lo)*ph;
 root.append(svg('title',{},`${m.label}의 ${date(o.forecast_origin)} 발행 과거 SMP 예측`));
 for(let j=0;j<=5;j++){const value=lo+(hi-lo)*j/5,yy=y(value);root.append(svg('line',{x1:L,y1:yy,x2:W-R,y2:yy,class:'gridline',stroke:'#e0e8eb'}),svg('text',{x:L-12,y:yy+4,'text-anchor':'end',fill:'#71838f','font-size':12},fmt(value,0)));}
 for(let i=start;i<end;i++)if(i===start||i===end-1||i%12===0){const xx=x(i);root.append(svg('text',{x:xx,y:H-22,'text-anchor':i===start?'start':i===end-1?'end':'middle',fill:'#71838f','font-size':12},`${o.targets[i].slice(5,10)} ${o.targets[i].slice(11,13)}시`));}
 if(!state.day)for(const i of [24,48])root.append(svg('line',{x1:x(i-.5),x2:x(i-.5),y1:T,y2:H-B,class:'day-boundary',stroke:'#b3c8d0','stroke-dasharray':'4 5'}),svg('text',{x:x(i)+7,y:T+12,fill:'#71838f','font-size':11},`D+${i/24+1}`));
 for(const s of series)root.append(svg('polyline',{'data-series':s.id,'data-points':count,points:s.values.slice(start,end).map((v,j)=>`${x(start+j).toFixed(2)},${y(v).toFixed(2)}`).join(' '),fill:'none',stroke:s.color,'stroke-width':s.width,'stroke-dasharray':s.dash||'', 'stroke-linejoin':'round','stroke-linecap':'round'}));
 const cross=svg('line',{y1:T,y2:H-B,stroke:'#138b93','stroke-dasharray':'3 3',visibility:'hidden'}),dot=svg('circle',{r:5,fill:'#138b93',stroke:'white','stroke-width':2,visibility:'hidden'});root.append(cross,dot);
 chart={start,end,x,y,cross,dot,index:start};
 root.onpointermove=e=>{const r=root.getBoundingClientRect(),xx=(e.clientX-r.left)/r.width*W;showPoint(Math.max(start,Math.min(end-1,Math.round(start+(xx-L)/pw*(count-1)))));};
 root.onpointerleave=()=>{$('chartTooltip').hidden=true;cross.setAttribute('visibility','hidden');dot.setAttribute('visibility','hidden');};
 root.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();showPoint(Math.max(start,Math.min(end-1,chart.index+(e.key==='ArrowRight'?1:-1))));}};
}
function showPoint(i){
 chart.index=i;const o=data.origins[state.origin],p=o.predictions[state.model],tip=$('chartTooltip');tip.replaceChildren(el('strong',date(o.targets[i])),el('p',`예측 ${fmt(p[i])} 원/kWh`));
 if(state.actual)tip.append(el('p',`실제 ${fmt(o.actual[i])} · 오차 ${fmt(Math.abs(p[i]-o.actual[i]))}`));if(state.baseline)tip.append(el('p',`주간 기준선 ${fmt(o.weekly[i])}`));
 tip.hidden=false;const width=$('chartWrap').clientWidth;tip.style.left=`${Math.max(4,Math.min(width-240,chart.x(i)/1100*width+12))}px`;tip.style.top='12px';chart.cross.setAttribute('x1',chart.x(i));chart.cross.setAttribute('x2',chart.x(i));chart.cross.setAttribute('visibility','visible');chart.dot.setAttribute('cx',chart.x(i));chart.dot.setAttribute('cy',chart.y(p[i]));chart.dot.setAttribute('visibility','visible');
}
function drawSemester(){
 const latest=new Map();for(const o of data.origins)for(let i=0;i<72;i++){const t=o.targets[i],previous=latest.get(t);if(!previous||o.forecast_origin>previous.origin)latest.set(t,{t,origin:o.forecast_origin,actual:o.actual[i],pred:o.predictions[state.model][i],horizon:i+1});}
 const month=$('semesterMonth').value,rows=[...latest.values()].sort((a,b)=>a.t.localeCompare(b.t)).filter(r=>month==='all'||r.t.slice(5,7)===month);
 const score=metrics(rows.map(r=>r.actual),rows.map(r=>r.pred));$('semesterSummary').textContent=`${month==='all'?'2025년 7~12월':`2025년 ${Number(month)}월`} · ${rows.length.toLocaleString()}시간 · 그래프 구간 MAE ${fmt(score.mae)}원/kWh`;
 const root=$('semesterChart');root.replaceChildren();$('semesterTooltip').hidden=true;const W=1100,H=380,L=65,R=22,T=25,B=50,pw=W-L-R,ph=H-T-B;
 const values=rows.flatMap(r=>[r.actual,r.pred]),lo=Math.floor((Math.min(...values)-8)/10)*10,hi=Math.ceil((Math.max(...values)+8)/10)*10,x=i=>L+i/(rows.length-1)*pw,y=v=>T+(hi-v)/(hi-lo)*ph;
 for(let j=0;j<=5;j++){const value=lo+(hi-lo)*j/5,yy=y(value);root.append(svg('line',{x1:L,x2:W-R,y1:yy,y2:yy,stroke:'#e0e8eb'}),svg('text',{x:L-10,y:yy+5,'text-anchor':'end',fill:'#627d88','font-size':15},fmt(value,0)));}
 for(let i=0;i<rows.length;i++){const t=rows[i].t;if(i===0||i===rows.length-1||(month==='all'?t.slice(8,16)==='01T00:00':t.slice(11,16)==='00:00'&&['08','15','22'].includes(t.slice(8,10))))root.append(svg('text',{x:x(i),y:H-16,'text-anchor':i===0?'start':i===rows.length-1?'end':'middle',fill:'#627d88','font-size':15},t.slice(5,10)));}
 for(const[field,color]of[['actual','#243e53'],['pred','#138b93']])root.append(svg('polyline',{'data-series':field,'data-points':rows.length,points:rows.map((r,i)=>`${x(i).toFixed(2)},${y(r[field]).toFixed(2)}`).join(' '),fill:'none',stroke:color,'stroke-width':month==='all'?1.2:1.8}));
 let current=0;const line=svg('line',{y1:T,y2:H-B,stroke:'#138b93',visibility:'hidden'});root.append(line);
 const show=i=>{current=i;const r=rows[i],tip=$('semesterTooltip');tip.replaceChildren(el('strong',date(r.t)),el('p',`예측 ${fmt(r.pred)} · 실제 ${fmt(r.actual)} 원/kWh`),el('p',`${date(r.origin)} 발행 · D+${Math.ceil(r.horizon/24)}`));tip.hidden=false;tip.style.top='10px';const width=$('semesterWrap').clientWidth;tip.style.left=Math.max(0,Math.min(width-260,x(i)/W*width+10))+'px';line.setAttribute('x1',x(i));line.setAttribute('x2',x(i));line.setAttribute('visibility','visible');};
 root.onpointermove=e=>{const r=root.getBoundingClientRect();show(Math.max(0,Math.min(rows.length-1,Math.round(((e.clientX-r.left)/r.width*W-L)/pw*(rows.length-1)))));};root.onpointerleave=()=>{$('semesterTooltip').hidden=true;line.setAttribute('visibility','hidden');};root.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(rows.length-1,current+(e.key==='ArrowRight'?1:-1))));}};
}

function mode(live){const u=new URL(location.href);if(live)u.hash='live';else if(u.hash==='#live'||!u.hash)u.hash='replay';window.history.replaceState(null,'',u);$('livePanel').hidden=!live;$('replayPanel').hidden=live;for(const [id,active]of [['liveMode',live],['replayMode',!live]]){$(id).classList.toggle('active',active);$(id).setAttribute('aria-pressed',active);}}
function download(){
 const o=data.origins[state.origin],m=data.models.find(m=>m.id===state.model),header=['dataset_version','mode','model','model_weight','forecast_origin','observed_cutoff','target_timestamp','horizon','actual_smp','predicted_smp','weekly_baseline'];
 const rows=o.targets.map((t,i)=>[data.dataset_version,data.mode,m.id,m.model_weight,o.forecast_origin,o.observed_cutoff,t,i+1,o.actual[i],o.predictions[m.id][i],o.weekly[i]].join(','));
 const url=URL.createObjectURL(new Blob(['\uFEFF'+[header.join(','),...rows].join('\r\n')],{type:'text/csv;charset=utf-8'})),a=el('a');a.href=url;a.download=`smp72_${m.id}_${o.forecast_origin.slice(0,10)}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
 try{
  $('loadError').hidden=true;
  const embedded=$('demo-json');if(embedded)data=JSON.parse(embedded.textContent);else{const r=await fetch('./demo/smp72-2025.json',{cache:'no-cache'});if(!r.ok)throw Error(`HTTP ${r.status}`);data=await r.json();}validate(data);
  state.origin=Math.max(0,data.origins.findIndex(o=>o.forecast_origin===data.default_origin));
  $('originSelect').replaceChildren(...data.origins.map((o,i)=>{const n=el('option',date(o.forecast_origin)+' KST');n.value=i;return n;}));$('originSelect').value=state.origin;
  $('modelSelect').replaceChildren(...data.models.map(m=>{const n=el('option',m.label);n.value=m.id;return n;}));$('modelSelect').value=state.model;
  $('originSelect').onchange=e=>{state.origin=Number(e.target.value);draw();};$('modelSelect').onchange=e=>{state.model=e.target.value;draw();drawSemester();};
  for(const id of ['originSelect','modelSelect','downloadCsv'])$(id).disabled=false;
  $('modelRows').replaceChildren(...data.models.map(m=>{const row=el('tr');if(m.id==='sarimax')row.className='best';for(const v of [m.label,`${Math.round(m.model_weight*100)} : ${Math.round(m.weekly_weight*100)}`,fmt(m.metrics.mae),fmt(m.metrics.rmse),fmt(m.metrics.r2,4)])row.append(el('td',v));const td=el('td'),a=el('a','보고서 ↗');a.href=m.report_url;a.target='_blank';a.rel='noopener noreferrer';td.append(a);row.append(td);return row;}));
  draw();drawSemester();document.body.dataset.ready='true';
 }catch(error){$('loadError').hidden=false;$('loadErrorText').textContent=error.message;document.body.dataset.ready='error';}
}
$('semesterMonth').onchange=()=>{if(data)drawSemester();};
$('retry').onclick=init;$('downloadCsv').onclick=download;
$('horizonButtons').onclick=e=>{const b=e.target.closest('[data-day]');if(b&&data){state.day=Number(b.dataset.day);draw();}};
$('toggleActual').onclick=()=>{if(data){state.actual=!state.actual;draw();}};$('toggleBaseline').onclick=()=>{if(data){state.baseline=!state.baseline;draw();}};
$('liveMode').onclick=()=>mode(true);$('replayMode').onclick=()=>mode(false);$('backToReplay').onclick=()=>mode(false);
document.querySelectorAll('.nav nav a').forEach(a=>a.addEventListener('click',()=>mode(false)));
mode(!location.hash||location.hash==='#live');
init();
})();
