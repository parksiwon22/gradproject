(() => {
'use strict';
const $=id=>document.getElementById(id), fmt=n=>Number(n).toLocaleString('ko-KR',{minimumFractionDigits:2,maximumFractionDigits:2}), date=s=>s.slice(0,16).replace('T',' ');
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const svg=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const labels={smp:'과거 육지 SMP',demand_actual:'실제 전력수요',demand_forecast_history:'과거 수요예측 이력',generation_lng:'LNG 발전량',generation_coal:'석탄 발전량',generation_nuclear:'원자력 발전량',solar_actual:'태양광 발전량',wind_actual:'풍력 발전량',reserve_mw:'공급예비력',reserve_pct:'공급예비율',temperature_c:'서울108 기온',irradiance_mj_m2:'서울108 일사량',wind_ms:'서울108 풍속'};
let data,day=0;
function validate(d){if(d.mode!=='historical_replay'||d.rows.length!==72||d.forecast_origin!=='2025-11-19T23:00:00+09:00')throw Error('2025년 저장 예측 검증 실패');for(const r of d.rows)if(!Number.isFinite(r.predicted_smp)||!Number.isFinite(r.actual_smp))throw Error('저장된 실제값 또는 예측값 누락');}
function draw(){
 const rows=data.rows,start=day?(day-1)*24:0,end=day?start+24:72,selected=rows.slice(start,end),root=$('livePriceChart');root.replaceChildren();$('liveTooltip').hidden=true;
 document.querySelectorAll('[data-live-day]').forEach(b=>{const active=Number(b.dataset.liveDay)===day;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
 const W=1100,H=380,L=65,R=22,T=25,B=50,pw=W-L-R,ph=H-T-B,values=selected.flatMap(r=>[r.predicted_smp,r.weekly_baseline,r.actual_smp]),lo=Math.floor((Math.min(...values)-8)/10)*10,hi=Math.ceil((Math.max(...values)+8)/10)*10;
 const x=i=>L+i/(selected.length-1)*pw,y=v=>T+(hi-v)/(hi-lo)*ph;
 for(let j=0;j<=5;j++){const v=lo+(hi-lo)*j/5,yy=y(v);root.append(svg('line',{x1:L,x2:W-R,y1:yy,y2:yy,stroke:'#e0e8eb'}),svg('text',{x:L-10,y:yy+4,'text-anchor':'end',fill:'#71838f','font-size':16},v.toFixed(0)));}
 for(let i=0;i<selected.length;i++)if(i===0||i===selected.length-1||(i+start)%12===0)root.append(svg('text',{x:x(i),y:H-22,'text-anchor':i===0?'start':i===selected.length-1?'end':'middle',fill:'#71838f','font-size':16},`${selected[i].target_timestamp.slice(5,10)} ${selected[i].target_timestamp.slice(11,13)}시`));
 if(!day)for(const i of[24,48])root.append(svg('line',{x1:x(i-.5),x2:x(i-.5),y1:T,y2:H-B,stroke:'#b3c8d0','stroke-dasharray':'4 5'}));
 for(const[field,color,dash]of[['weekly_baseline','#8b9da9','6 5'],['actual_smp','#243e53',''],['predicted_smp','#138b93','']])root.append(svg('polyline',{'data-series':field,'data-points':selected.length,points:selected.map((r,i)=>`${x(i)},${y(r[field])}`).join(' '),fill:'none',stroke:color,'stroke-width':field==='predicted_smp'?3.4:1.8,'stroke-dasharray':dash,'stroke-linejoin':'round'}));
 const line=svg('line',{y1:T,y2:H-B,stroke:'#138b93','stroke-dasharray':'3 3',visibility:'hidden'});root.append(line);let current=0;
 function show(i){current=i;const r=selected[i],tip=$('liveTooltip');tip.replaceChildren(node('strong',date(r.target_timestamp)),node('p',`예측 ${fmt(r.predicted_smp)} 원/kWh`),node('p',`실제 ${fmt(r.actual_smp)} · 전주 기준선 ${fmt(r.weekly_baseline)}`));tip.hidden=false;tip.style.left=Math.max(4,Math.min($('liveChartWrap').clientWidth-240,x(i)/W*$('liveChartWrap').clientWidth+12))+'px';tip.style.top='12px';line.setAttribute('x1',x(i));line.setAttribute('x2',x(i));line.setAttribute('visibility','visible');}
 root.onpointermove=e=>{const b=root.getBoundingClientRect(),xx=(e.clientX-b.left)/b.width*W;show(Math.max(0,Math.min(selected.length-1,Math.round((xx-L)/pw*(selected.length-1)))));};root.onpointerleave=()=>{$('liveTooltip').hidden=true;line.setAttribute('visibility','hidden');};root.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(selected.length-1,current+(e.key==='ArrowRight'?1:-1))));}};
}
async function init(){try{
 const embedded=$('demo-json');let bundle;if(embedded)bundle=JSON.parse(embedded.textContent);else{const r=await fetch('./demo/smp72-2025.json',{cache:'no-cache'});if(!r.ok)throw Error('저장된 예측을 불러오지 못했습니다.');bundle=await r.json();}
 const origin=bundle.origins.find(o=>o.forecast_origin==='2025-11-19T23:00:00+09:00');if(!origin)throw Error('2025년 11월19일 발행 예측이 없습니다.');
 data={mode:'historical_replay',forecast_origin:origin.forecast_origin,observed_cutoff:origin.observed_cutoff,rows:origin.targets.map((t,i)=>({target_timestamp:t,horizon:i+1,predicted_smp:origin.predictions.sarimax[i],actual_smp:origin.actual[i],weekly_baseline:origin.weekly[i]}))};validate(data);
 $('liveStatus').textContent='2025년 실제 저장 결과 · 과거 검증 시연';$('liveStatus').classList.remove('pending');
 $('liveNotice').textContent='2025년 11월19일23시 기준으로 예측한 11월20~22일 결과입니다. 실제 SMP와 비교하고, 이 예측으로 충전 시간을 추천합니다.';
 $('liveOrigin').textContent=date(data.forecast_origin);$('liveCreated').textContent='2025.11.20 ~ 11.22';$('liveCutoff').textContent=date(data.observed_cutoff);$('liveTarget').textContent='2025년 11월20일00시 ~ 11월22일23시';
 $('liveQualityWarning').textContent='2025년 학습·평가 결과를 재생하는 화면입니다. 아래 충전 추천도 2025년 당시의 시간 조건으로 계산합니다.';
 $('liveDayCards').replaceChildren(...[1,2,3].map(d=>{const a=data.rows.slice((d-1)*24,d*24).map(r=>r.predicted_smp),b=node('button',undefined,'day-card'),top=node('div',undefined,'day-card-top');top.append(node('strong',`D+${d}`),node('span',data.rows[(d-1)*24].target_timestamp.slice(0,10)));b.append(top,node('strong',`${fmt(a.reduce((s,v)=>s+v,0)/24)} 원/kWh`),node('p',`예측 범위 ${fmt(Math.min(...a))} ~ ${fmt(Math.max(...a))}`,'range'),node('p','저장된 실제값과 비교 가능'));b.onclick=()=>{day=d;draw();};return b;}));
 $('liveDayButtons').onclick=e=>{const b=e.target.closest('[data-live-day]');if(b){day=Number(b.dataset.liveDay);draw();}};
 $('liveCsv').onclick=()=>{const fields=Object.keys(data.rows[0]),csv='\uFEFF'+[fields.join(','),...data.rows.map(r=>fields.map(k=>r[k]===null?'':r[k]).join(','))].join('\r\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='smp72_2025-11-20_22.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 function charge(){
  $('evError').hidden=true;$('evResult').hidden=true;
  try{
   if(['evCapacity','evCurrent','evTarget','evPower','evEfficiency','evStart','evEnd'].some(id=>!$(id).value.trim()))throw Error('모든 충전 조건을 입력해주세요.');
   const localKst=id=>$(id).value+':00+09:00';
   const result=window.SMP72Charging.plan(data.rows,{capacity:Number($('evCapacity').value),currentSoc:Number($('evCurrent').value),targetSoc:Number($('evTarget').value),power:Number($('evPower').value),efficiency:Number($('evEfficiency').value),start:localKst('evStart'),end:localKst('evEnd')},Date.parse(data.forecast_origin));
   const time=ms=>{const parts=Object.fromEntries(new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hourCycle:'h23'}).formatToParts(new Date(ms)).map(p=>[p.type,p.value]));return `${parts.month}/${parts.day} ${parts.hour}:${parts.minute}:${parts.second}`;};
   $('evEnergy').textContent=`${fmt(result.batteryEnergy)} / ${fmt(result.gridEnergy)} kWh`;$('evCost').textContent=`${fmt(result.cost)} 원`;$('evDifference').textContent=`${fmt(result.difference)} 원`;
   $('evSummary').textContent=`총 ${fmt(result.chargingHours)}시간 충전 · 가능한 즉시 충전의 SMP 가상비용 ${fmt(result.baselineCost)}원.${result.excludedPast?' 현재시각 이전 구간을 제외했습니다.':''} 입력을 바꾸면 다시 계산해주세요.`;
   $('evSchedule').replaceChildren(...result.allocations.map(r=>{const tr=node('tr');for(const value of[`${time(r.start)} ~ ${time(r.start+r.duration*3600000)}`,`${fmt(r.duration*60)}분`,`${fmt(r.energy)} kWh`,`${fmt(r.price)} 원/kWh`])tr.append(node('td',value));return tr;}));
   $('evResult').hidden=false;
  }catch(error){$('evError').textContent=error.message;$('evError').hidden=false;}
 }
 $('chargingForm').onsubmit=e=>{e.preventDefault();charge();};
 $('chargingForm').oninput=()=>{$('evResult').hidden=true;$('evError').hidden=true;};
 charge();
 $('liveForecastContent').hidden=false;draw();document.body.dataset.liveReady='true';
 }catch(error){$('liveStatus').textContent='2025년 저장 결과 · 불러오기 실패';$('liveNotice').textContent=error.message;document.body.dataset.liveReady='unavailable';}}
init();
})();
