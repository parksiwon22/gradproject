(() => {
'use strict';
const $=id=>document.getElementById(id), fmt=n=>Number(n).toLocaleString('ko-KR',{minimumFractionDigits:2,maximumFractionDigits:2}), date=s=>s.slice(0,16).replace('T',' ');
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;};
const svg=(tag,attrs,text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,v);if(text!==undefined)n.textContent=text;return n;};
const labels={smp:'과거 육지 SMP',demand_actual:'실제 전력수요',demand_forecast_history:'과거 수요예측 이력',generation_lng:'LNG 발전량',generation_coal:'석탄 발전량',generation_nuclear:'원자력 발전량',solar_actual:'태양광 발전량',wind_actual:'풍력 발전량',reserve_mw:'공급예비력',reserve_pct:'공급예비율',temperature_c:'서울108 기온',irradiance_mj_m2:'서울108 일사량',wind_ms:'서울108 풍속'};
let data,day=0;
function validate(d){
 if(d.mode!=='new_forecast'||!['preliminary','issued'].includes(d.status)||d.rows.length!==72||d.model_weight!==.4||d.weekly_weight!==.6||d.feature_count!==83)throw Error('지원하지 않는 최신 예측 형식');
 const origin=Date.parse(d.forecast_origin);if(Date.parse(d.observed_cutoff)!==origin-72*3600000)throw Error('최신 예측의 실측 마감 오류');
 for(let i=0;i<72;i++){const r=d.rows[i];if(r.horizon!==i+1||Date.parse(r.target_timestamp)!==origin+(i+1)*3600000||!Number.isFinite(r.predicted_smp)||!Number.isFinite(r.weekly_baseline)||r.actual_smp!==null)throw Error('최신 예측값 또는 시간 오류');if(Math.abs(.4*r.sarimax_prediction+.6*r.weekly_baseline-r.predicted_smp)>1e-7)throw Error('혼합 예측 검증 실패');}
 if(d.status==='issued'&&Date.parse(d.calculated_at)<origin)throw Error('기준시각 이전 계산을 정식 발행으로 표시할 수 없음');
}
function draw(){
 const rows=data.rows,start=day?(day-1)*24:0,end=day?start+24:72,selected=rows.slice(start,end),root=$('livePriceChart');root.replaceChildren();$('liveTooltip').hidden=true;
 document.querySelectorAll('[data-live-day]').forEach(b=>{const active=Number(b.dataset.liveDay)===day;b.classList.toggle('active',active);b.setAttribute('aria-pressed',active);});
 const W=1100,H=380,L=65,R=22,T=25,B=50,pw=W-L-R,ph=H-T-B,values=selected.flatMap(r=>[r.predicted_smp,r.weekly_baseline]),lo=Math.floor((Math.min(...values)-8)/10)*10,hi=Math.ceil((Math.max(...values)+8)/10)*10;
 const x=i=>L+i/(selected.length-1)*pw,y=v=>T+(hi-v)/(hi-lo)*ph;
 for(let j=0;j<=5;j++){const v=lo+(hi-lo)*j/5,yy=y(v);root.append(svg('line',{x1:L,x2:W-R,y1:yy,y2:yy,stroke:'#e0e8eb'}),svg('text',{x:L-10,y:yy+4,'text-anchor':'end',fill:'#71838f','font-size':12},v.toFixed(0)));}
 for(let i=0;i<selected.length;i++)if(i===0||i===selected.length-1||(i+start)%12===0)root.append(svg('text',{x:x(i),y:H-22,'text-anchor':i===0?'start':i===selected.length-1?'end':'middle',fill:'#71838f','font-size':12},`${selected[i].target_timestamp.slice(5,10)} ${selected[i].target_timestamp.slice(11,13)}시`));
 if(!day)for(const i of[24,48])root.append(svg('line',{x1:x(i-.5),x2:x(i-.5),y1:T,y2:H-B,stroke:'#b3c8d0','stroke-dasharray':'4 5'}));
 for(const[field,color,dash]of[['weekly_baseline','#8b9da9','6 5'],['predicted_smp','#138b93','']])root.append(svg('polyline',{'data-series':field,'data-points':selected.length,points:selected.map((r,i)=>`${x(i)},${y(r[field])}`).join(' '),fill:'none',stroke:color,'stroke-width':field==='predicted_smp'?3.4:1.8,'stroke-dasharray':dash,'stroke-linejoin':'round'}));
 const line=svg('line',{y1:T,y2:H-B,stroke:'#138b93','stroke-dasharray':'3 3',visibility:'hidden'});root.append(line);let current=0;
 function show(i){current=i;const r=selected[i],tip=$('liveTooltip');tip.replaceChildren(node('strong',date(r.target_timestamp)),node('p',`예측 ${fmt(r.predicted_smp)} 원/kWh`),node('p',`전주 기준선 ${fmt(r.weekly_baseline)}`));tip.hidden=false;tip.style.left=Math.max(4,Math.min($('liveChartWrap').clientWidth-240,x(i)/W*$('liveChartWrap').clientWidth+12))+'px';tip.style.top='12px';line.setAttribute('x1',x(i));line.setAttribute('x2',x(i));line.setAttribute('visibility','visible');}
 root.onpointermove=e=>{const b=root.getBoundingClientRect(),xx=(e.clientX-b.left)/b.width*W;show(Math.max(0,Math.min(selected.length-1,Math.round((xx-L)/pw*(selected.length-1)))));};root.onpointerleave=()=>{$('liveTooltip').hidden=true;line.setAttribute('visibility','hidden');};root.onkeydown=e=>{if(['ArrowLeft','ArrowRight'].includes(e.key)){e.preventDefault();show(Math.max(0,Math.min(selected.length-1,current+(e.key==='ArrowRight'?1:-1))));}};
}
async function init(){try{
 const embedded=$('live-json');if(embedded)data=JSON.parse(embedded.textContent);else{const r=await fetch('./demo/forecast-20261120.json',{cache:'no-cache'});if(!r.ok)throw Error('검증된 최신 모델 결과가 아직 게시되지 않았습니다.');data=await r.json();}validate(data);
 $('liveStatus').textContent=data.status==='preliminary'?'실제 모델 사전 계산본 · 23시 정식 발행 전':'실제 모델 추론 완료 · 계산시각 확인';
 if(data.status==='issued')$('liveStatus').classList.remove('pending');
 $('liveNotice').textContent=data.status==='preliminary'?'23시 예정 기준으로 미리 계산한 새 예측입니다. 아래 실제 계산시각에 확보한 자료를 사용했으며, 23시에 계산된 결과라고 표시하지 않습니다.':'저장된 SARIMAX 모델을 최신 입력 스냅샷에 적용했습니다. 실제 계산시각과 입력 조건을 함께 확인해주세요.';
 $('liveOrigin').textContent=date(data.forecast_origin)+(data.status==='preliminary'?' (예정)':'');$('liveCreated').textContent=date(data.calculated_at);$('liveCutoff').textContent=date(data.observed_cutoff);$('liveTarget').textContent=`${date(data.target_start)} ~ ${date(data.target_end)} · 83개 입력 · 재학습 없음`;
 const entirelyMissing=data.quality.filter(q=>q.missing_hours===168).map(q=>labels[q.variable]||q.variable);$('liveQualityWarning').textContent=entirelyMissing.length?`입력 대체 조건: ${entirelyMissing.join(', ')}의 최근 168시간 원자료가 없어 고정된 학습기반 대체값을 사용했습니다. 최신 실측을 모두 확보한 예측으로 해석하지 않습니다.`:'일부 원자료 결측에는 학습 때 고정한 대체 규칙을 적용했습니다. 아래 입력 상태를 확인해주세요.';
 $('liveQualityRows').replaceChildren(...data.quality.map(q=>{const tr=node('tr');for(const v of[labels[q.variable]||q.variable,`${q.missing_hours}/168시간 (${q.missing_pct}%)`,q.last_source_time?date(q.last_source_time):'없음'])tr.append(node('td',v));return tr;}));
 const issuance=data.forecast_issued_values;$('liveForecastSources').textContent=`대상시간에 실제 연결한 기온 발표본: ${(issuance.temperature_issued_at||[]).map(date).join(', ')||'없음'}. 날씨·강수확률 발표본: ${(issuance.weather_issued_at||[]).map(date).join(', ')||'없음'}.`;
 const si=data.quality.find(q=>q.variable==='irradiance_mj_m2');if(si&&si.day_missing_hours!==undefined)$('liveForecastSources').append(document.createTextNode(` 일사량 원본 결측 ${si.missing_hours}시간 중 태양고도 기준 야간 ${si.night_missing_hours}시간, 낮 ${si.day_missing_hours}시간입니다. 야간 결측을 관측0으로 바꾸지 않았습니다.`));
 $('liveDayCards').replaceChildren(...[1,2,3].map(d=>{const a=data.rows.slice((d-1)*24,d*24).map(r=>r.predicted_smp),b=node('button',undefined,'day-card'),top=node('div',undefined,'day-card-top');top.append(node('strong',`D+${d}`),node('span',data.rows[(d-1)*24].target_timestamp.slice(0,10)));b.append(top,node('strong',`${fmt(a.reduce((s,v)=>s+v,0)/24)} 원/kWh`),node('p',`예측 범위 ${fmt(Math.min(...a))} ~ ${fmt(Math.max(...a))}`,'range'),node('small','실제값·오차는 아직 미연결'));b.onclick=()=>{day=d;draw();};return b;}));
 $('liveDayButtons').onclick=e=>{const b=e.target.closest('[data-live-day]');if(b){day=Number(b.dataset.liveDay);draw();}};
 $('liveCsv').onclick=()=>{const fields=Object.keys(data.rows[0]),csv='\uFEFF'+[fields.join(','),...data.rows.map(r=>fields.map(k=>r[k]===null?'':r[k]).join(','))].join('\r\n'),url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=node('a');a.href=url;a.download='smp72_2026-11-20_22.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
 const snapshot=node('p',`입력 스냅샷: ${date(data.input_snapshot_at||data.calculated_at)} KST. 예정 기준시각 이후 재추론도 이 고정 입력을 사용합니다.`,'sub');$('liveNotice').after(snapshot);
 $('liveForecastContent').hidden=false;draw();document.body.dataset.liveReady='true';
 }catch(error){$('liveStatus').textContent='최신 모델 결과 · 연결 검증 미완료';$('liveNotice').textContent=error.message;document.body.dataset.liveReady='unavailable';}}
init();
})();
