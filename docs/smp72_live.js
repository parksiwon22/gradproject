(() => {
'use strict';
const $=id=>document.getElementById(id), fmt=n=>Number(n).toLocaleString('ko-KR',{minimumFractionDigits:2,maximumFractionDigits:2}), date=s=>s.slice(0,16).replace('T',' ');
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const svg=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const labels={smp:'과거 육지 SMP',demand_actual:'실제 전력수요',demand_forecast_history:'과거 수요예측 이력',generation_lng:'LNG 발전량',generation_coal:'석탄 발전량',generation_nuclear:'원자력 발전량',solar_actual:'태양광 발전량',wind_actual:'풍력 발전량',reserve_mw:'공급예비력',reserve_pct:'공급예비율',temperature_c:'서울108 기온',irradiance_mj_m2:'서울108 일사량',wind_ms:'서울108 풍속'};
let data,day=0,choices=[];
function renderWeather(){const chosenDay=day||1,rows=data.rows.slice((chosenDay-1)*24,chosenDay*24),classified=window.SMP72Weather.classify(rows);$('weatherDaySelect').value=chosenDay;$('weatherHourGrid').replaceChildren(...classified.map(r=>{const card=node('article',undefined,'weather-hour '+r.weather);card.append(node('strong',r.target_timestamp.slice(11,13)+'시'),node('span',r.icon,'weather-icon'),node('b',r.weather==='sunny'?'맑음':r.weather==='cloudy'?'흐림':'비슷한 가격'),node('span',r.label),node('span',fmt(r.predicted_smp)+' 원/kWh'));return card;}));}

function validate(d){
 if(!['historical_replay','new_forecast'].includes(d.mode)||d.rows.length!==72)throw Error('예측 형식 검증 실패');
 const origin=Date.parse(d.forecast_origin);if(Date.parse(d.observed_cutoff)!==origin-72*3600000)throw Error('입력 마감시각 오류');
 for(let i=0;i<72;i++){const r=d.rows[i];if(Date.parse(r.target_timestamp)!==origin+(i+1)*3600000||!Number.isFinite(r.predicted_smp)||!Number.isFinite(r.weekly_baseline))throw Error('예측값 또는 대상시간 오류');if(r.actual_smp!==null&&!Number.isFinite(r.actual_smp))throw Error('실제값 형식 오류');}
}
function draw(){
 renderWeather();
 const rows=data.rows,start=day?(day-1)*24:0,end=day?start+24:72,selected=rows.slice(start,end),root=$('livePriceChart');root.replaceChildren();$('liveTooltip').hidden=true;
 document.querySelectorAll('[data-live-day]').forEach(b=>{const active=Number(b.dataset.liveDay)===day;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
 const W=1100,H=380,L=65,R=22,T=25,B=50,pw=W-L-R,ph=H-T-B,values=selected.flatMap(r=>[r.predicted_smp,r.weekly_baseline,r.actual_smp]).filter(Number.isFinite),lo=Math.floor((Math.min(...values)-8)/10)*10,hi=Math.ceil((Math.max(...values)+8)/10)*10;
 const x=i=>L+i/(selected.length-1)*pw,y=v=>T+(hi-v)/(hi-lo)*ph;
 for(let j=0;j<=5;j++){const v=lo+(hi-lo)*j/5,yy=y(v);root.append(svg('line',{x1:L,x2:W-R,y1:yy,y2:yy,stroke:'#e0e8eb'}),svg('text',{x:L-10,y:yy+4,'text-anchor':'end',fill:'#71838f','font-size':16},v.toFixed(0)));}
 for(let i=0;i<selected.length;i++)if(i===0||i===selected.length-1||(i+start)%12===0)root.append(svg('text',{x:x(i),y:H-22,'text-anchor':i===0?'start':i===selected.length-1?'end':'middle',fill:'#71838f','font-size':16},`${selected[i].target_timestamp.slice(5,10)} ${selected[i].target_timestamp.slice(11,13)}시`));
 if(!day)for(const i of[24,48])root.append(svg('line',{x1:x(i-.5),x2:x(i-.5),y1:T,y2:H-B,stroke:'#b3c8d0','stroke-dasharray':'4 5'}));
 for(const[field,color,dash]of[['weekly_baseline','#8b9da9','6 5'],...(selected.every(r=>Number.isFinite(r.actual_smp))?[['actual_smp','#243e53','']]:[]),['predicted_smp','#138b93','']])root.append(svg('polyline',{'data-series':field,'data-points':selected.length,points:selected.map((r,i)=>`${x(i)},${y(r[field])}`).join(' '),fill:'none',stroke:color,'stroke-width':field==='predicted_smp'?3.4:1.8,'stroke-dasharray':dash,'stroke-linejoin':'round'}));
 const line=svg('line',{y1:T,y2:H-B,stroke:'#138b93','stroke-dasharray':'3 3',visibility:'hidden'});root.append(line);let current=0;
 function show(i){current=i;const r=selected[i],tip=$('liveTooltip');tip.replaceChildren(node('strong',date(r.target_timestamp)),node('p',`예측 ${fmt(r.predicted_smp)} 원/kWh`),node('p',`${Number.isFinite(r.actual_smp)?`실제 ${fmt(r.actual_smp)}`:'가격 예측'} · 전주 기준선 ${fmt(r.weekly_baseline)}`));tip.hidden=false;tip.style.left=Math.max(4,Math.min($('liveChartWrap').clientWidth-240,x(i)/W*$('liveChartWrap').clientWidth+12))+'px';tip.style.top='12px';line.setAttribute('x1',x(i));line.setAttribute('x2',x(i));line.setAttribute('visibility','visible');}
 root.onpointermove=e=>{const b=root.getBoundingClientRect(),xx=(e.clientX-b.left)/b.width*W;show(Math.max(0,Math.min(selected.length-1,Math.round((xx-L)/pw*(selected.length-1)))));};root.onpointerleave=()=>{$('liveTooltip').hidden=true;line.setAttribute('visibility','hidden');};root.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(selected.length-1,current+(e.key==='ArrowRight'?1:-1))));}};
}
function selectForecast(value){
 $('evResult').hidden=true;$('evError').hidden=true;$('liveForecastContent').hidden=true;
 $('forecastPreparing').hidden=true;const chosen=choices.find(c=>c.value===value);if(!chosen){data=null;$('liveStatus').textContent=value?'예측 준비 전':'날짜 선택 대기';$('liveNotice').textContent=value?'예측이 준비되면 이곳에서 확인할 수 있어요.':'충전할 날짜를 선택해주세요.';if(value){$('preparingTitle').textContent=value+'의 예측은 아직 준비되지 않았어요';$('forecastPreparing').hidden=false;}return;}
 data=chosen.data;day=0;validate(data);const past=data.mode==='historical_replay',first=data.rows[0].target_timestamp,last=data.rows[71].target_timestamp;
 $('liveStatus').textContent=past?'과거 예측 비교':'SARIMAX 사전 예측';$('liveStatus').classList.toggle('pending',!past);
 $('liveNotice').textContent=`${first.slice(0,10)} ~ ${last.slice(0,10)}의 72시간 예측입니다. ${past?'저장된 실제 SMP와 비교합니다.':`실제 계산시각 ${date(data.calculated_at)} KST. 예정 기준시각과 구분합니다.`}`;
 $('liveOrigin').textContent=date(data.forecast_origin);$('liveCreated').textContent=first.slice(0,10)+' ~ '+last.slice(0,10);$('liveCutoff').textContent=date(data.observed_cutoff);$('liveTarget').textContent=`${date(first)} ~ ${date(last)}`;
 $('liveQualityWarning').textContent=past?'선택한 과거 예측과 당시 시간 조건으로 충전 시뮬레이션을 계산합니다.':'LNG·석탄·원자력·태양광·풍력의 최근 입력은 학습기반 대체값입니다. 입력 보완 조건은 아래 연구 안내에서 확인할 수 있습니다.';
 $('selectedDataPolicy').textContent=past?'저장된 SARIMAX40%+주간60% 혼합 검증 결과입니다. 발전량 결측 대체 및 동월 LNG 제공 조건을 포함합니다.':'11월19일20:05에 계산한 사전 예측입니다. 11월16일23시까지의 실측과 10월 LNG를 사용하고 발전량5종은 대체했습니다. 23시 재발행은 취소된 상태로 새 발행을 주장하지 않습니다.';
 $('livePanel').querySelector('.chart-caption span:first-child').textContent=past?'원/kWh · 실선: 예측 / 진한 선: 실제 / 점선: 전주 기준선':'원/kWh · 실선: 예측 / 점선: 전주 기준선';
 $('livePanel').querySelector('.chart-caption span:last-child').textContent=past?'실제 SMP와 예측값을 비교합니다.':'선택한 기간의 예측 가격 흐름입니다.';
 document.querySelectorAll('[data-live-day]').forEach(b=>{const d=Number(b.dataset.liveDay);if(d)b.textContent=`${data.rows[(d-1)*24].target_timestamp.slice(5,10).replace('-','/')} · D+${d}`;});
 $('weatherDaySelect').replaceChildren(...[1,2,3].map(d=>{const o=node('option',data.rows[(d-1)*24].target_timestamp.slice(0,10));o.value=d;return o;}));$('weatherDaySelect').onchange=()=>{day=Number($('weatherDaySelect').value);draw();};
 $('evStart').value=first.slice(0,10)+'T18:00';$('evEnd').value=data.rows[24].target_timestamp.slice(0,10)+'T07:00';
 $('liveDayCards').replaceChildren(...[1,2,3].map(d=>{const a=data.rows.slice((d-1)*24,d*24).map(r=>r.predicted_smp),b=node('button',undefined,'day-card'),top=node('div',undefined,'day-card-top');top.append(node('strong',`D+${d}`),node('span',data.rows[(d-1)*24].target_timestamp.slice(0,10)));b.append(top,node('strong',`${fmt(a.reduce((s,v)=>s+v,0)/24)} 원/kWh`),node('p',`예측 범위 ${fmt(Math.min(...a))} ~ ${fmt(Math.max(...a))}`,'range'),node('p',data.mode==='historical_replay'?'실제값 비교 가능':'가격 예측'));b.onclick=()=>{day=d;draw();};return b;}));
 $('liveDayButtons').onclick=e=>{const b=e.target.closest('[data-live-day]');if(b){day=Number(b.dataset.liveDay);draw();}};
 $('liveCsv').onclick=()=>{const fields=Object.keys(data.rows[0]),csv='\uFEFF'+[fields.join(','),...data.rows.map(r=>fields.map(k=>r[k]===null?'':r[k]).join(','))].join('\r\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download=`smp72_${data.rows[0].target_timestamp.slice(0,10)}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 function charge(){
  $('evError').hidden=true;$('evResult').hidden=true;
  try{
   if(['evCapacity','evCurrent','evTarget','evPower','evEfficiency','evStart','evEnd'].some(id=>!$(id).value.trim()))throw Error('모든 충전 조건을 입력해주세요.');
   const localKst=id=>$(id).value+':00+09:00';
   const result=window.SMP72Charging.plan(data.rows,{capacity:Number($('evCapacity').value),currentSoc:Number($('evCurrent').value),targetSoc:Number($('evTarget').value),power:Number($('evPower').value),efficiency:Number($('evEfficiency').value),start:localKst('evStart'),end:localKst('evEnd')},data.mode==='historical_replay'?Date.parse(data.forecast_origin):Date.now());
   const time=ms=>{const parts=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(new Date(ms)).map(p=>[p.type,p.value]));return `${parts.month}/${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;};
   $('evEnergy').textContent=`${fmt(result.batteryEnergy)} / ${fmt(result.gridEnergy)} kWh`;$('evCost').textContent=`${fmt(result.cost)} 원`;$('evDifference').textContent=`${fmt(result.difference)} 원`;
   $('evSummary').textContent=`총 ${fmt(result.chargingHours)}시간 충전 · 가능한 즉시 충전의 SMP 가상비용 ${fmt(result.baselineCost)}원.${result.excludedPast?' 현재시각 이전 구간을 제외했습니다.':''} 입력을 바꾸면 다시 계산해주세요.`;
   const weatherMap=new Map(window.SMP72Weather.classify(data.rows).map(r=>[r.target_timestamp,r]));
   $('evSchedule').replaceChildren(...result.allocations.map(r=>{const tr=node('tr'),w=weatherMap.get(r.timestamp),badge=node('span',w.icon+' '+w.label,'schedule-weather '+w.weather),cell=node('td');cell.append(badge);tr.append(cell);for(const value of[`${time(r.start)} ~ ${time(r.start+r.duration*3600000)}`,`${fmt(r.duration*60)}분`,`${fmt(r.energy)} kWh`,`${fmt(r.price)} 원/kWh`])tr.append(node('td',value));return tr;}));
   $('evResult').hidden=false;
  }catch(error){$('evError').textContent=error.message;$('evError').hidden=false;}
 }
 $('chargingForm').onsubmit=e=>{e.preventDefault();charge();};
 $('chargingForm').oninput=()=>{$('evResult').hidden=true;$('evError').hidden=true;};
 charge();
 $('liveForecastContent').hidden=false;draw();document.body.dataset.selectionReady='true';
}
async function init(){try{
 const embedded=$('demo-json');let bundle;if(embedded)bundle=JSON.parse(embedded.textContent);else{const r=await fetch('./demo/smp72-2025.json',{cache:'no-cache'});if(!r.ok)throw Error('검증 자료를 불러오지 못했습니다.');bundle=await r.json();}
 const historical=bundle.origins.map(o=>({value:o.targets[0].slice(0,10),data:{mode:'historical_replay',forecast_origin:o.forecast_origin,observed_cutoff:o.observed_cutoff,rows:o.targets.map((t,i)=>({target_timestamp:t,horizon:i+1,predicted_smp:o.predictions.sarimax[i],actual_smp:o.actual[i],weekly_baseline:o.weekly[i]}))}}));
 let latest=null;try{const e=$('latest-json');if(e)latest=JSON.parse(e.textContent);else{const r=await fetch('./demo/forecast-20261120.json',{cache:'no-cache'});if(r.ok)latest=await r.json();}if(latest)validate(latest);}catch(error){latest=null;}
 choices=[...historical,...(latest?[{value:latest.rows[0].target_timestamp.slice(0,10),data:latest}]:[])];
 const select=$('forecastDateSelect');
 const futureGroup=node('optgroup');futureGroup.label='2026년 11월 · 충전 계획';for(let d=1;d<=30;d++){const value=`2026-11-${String(d).padStart(2,'0')}`,o=node('option',value);o.value=value;futureGroup.append(o);}select.append(futureGroup);
 const historyGroup=node('optgroup');historyGroup.label='2025년 · 과거 예측 비교';for(const c of historical){const o=node('option',`${c.value} ~ ${c.data.rows[71].target_timestamp.slice(0,10)}`);o.value=c.value;historyGroup.append(o);}select.append(historyGroup);
 $('showPreparedForecast').onclick=()=>{select.value=latest?'2026-11-20':'2025-11-20';selectForecast(select.value);};
 select.onchange=()=>selectForecast(select.value);selectForecast('');document.body.dataset.liveReady='true';
 }catch(error){$('liveStatus').textContent='예측 목록 불러오기 실패';$('liveNotice').textContent=error.message;document.body.dataset.liveReady='unavailable';}}
init();
})();
