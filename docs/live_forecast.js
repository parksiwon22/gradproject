/* Live research mode is isolated from the preserved historical fold demo. */
(()=>{'use strict';
const $=id=>document.getElementById(id),esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=n=>Number.isFinite(Number(n))?Number(n).toLocaleString('ko-KR',{maximumFractionDigits:1}):'—';
function boot(){
 const main=$('main'); if(!main)return;
 const mode=document.createElement('div');mode.className='live-mode live-mode-bar';mode.innerHTML='<button id="showLive" class="live-active">최신 자동 예측</button><button id="showStudy">과거 연구 시연</button>';
 const live=document.createElement('section');live.id='liveForecast';live.className='live-wrap';
 live.innerHTML=`<div class="live-inner">
 <div class="live-heading"><p class="section-kicker">PUBLIC RESEARCH FORECAST / 자동 갱신</p><h1>육지 SMP 예측과 EV 충전 추천</h1><p>매일 한국시간 23시 기준으로 수집 가능한 자료만 사용합니다. 공식 발급과 분리된 연구용 예측입니다.</p></div>
 <div id="liveStatus" class="live-status" role="status">최신 실행 상태를 확인하는 중입니다…</div>
 <div class="live-controls"><label>기간 <select id="liveHorizon"><option value="1">D+1</option><option value="2">D+2</option><option value="3">D+3</option></select></label><span id="liveTarget">대상일 —</span></div>
 <div class="live-grid"><div class="live-card"><h2>시간대별 SMP 예측</h2><p id="liveModel">모델 정보 —</p><div class="live-legend"><span><i class="forecast-key"></i>AI 예측</span><span id="liveActualLegend" hidden><i class="actual-key"></i><span id="liveActualLabel">공식 SMP</span></span></div><div id="liveChart" class="live-chart"></div><div class="live-hours" id="liveHours"></div></div>
 <div class="live-card"><h2>G2V 충전 시간 추천</h2><div class="live-ev"><label>배터리 kWh<input id="liveCapacity" type="number" min="1" value="60"></label><label>현재 SOC %<input id="liveSoc" type="number" min="0" max="100" value="30"></label><label>목표 SOC %<input id="liveGoal" type="number" min="0" max="100" value="80"></label><label>충전 kW<input id="livePower" type="number" min=".1" value="7" step=".1"></label><label>시작 시각<select id="liveStart"></select></label><label>종료 시각<select id="liveEnd"></select></label></div><button id="liveRecommend" class="live-button">충전 추천 계산</button><p id="liveRecommendation">예측이 준비되면 추천을 표시합니다.</p><p id="liveRegret" class="live-muted">실제 SMP가 확보되기 전에는 사후 비용과 regret을 계산하지 않습니다.</p><p class="live-muted">도매 SMP 기반 가상비용이며 실제 충전요금이 아닙니다. 충전은 선택된 저가 시간대에 나누어 배분합니다.</p></div></div>
 <p class="live-foot">수집·예측 이력: <a href="./live/latest.json" target="_blank" rel="noopener">latest.json</a>. 수신시각이 기준시각보다 늦거나 23시 입력이 불완전하면 발급을 보류합니다. 과거 연구 시연과 실제 자동 실행은 아래 탭에서 구분됩니다.</p></div>`;
 main.prepend(live);main.prepend(mode);
 const legacy=[...main.children].filter(n=>n!==live&&n!==mode); const toggle=which=>{const show=which==='study';live.hidden=show;legacy.forEach(n=>n.hidden=!show);$('showLive').classList.toggle('live-active',!show);$('showStudy').classList.toggle('live-active',show);window.scrollTo({top:0,behavior:'smooth'});};
 $('showLive').onclick=()=>toggle('live');$('showStudy').onclick=()=>toggle('study');
 legacy.forEach(n=>n.hidden=true);
 for(let h=0;h<=24;h++){const o=document.createElement('option');o.value=h;o.textContent=String(h).padStart(2,'0')+':00';$('liveStart').append(o.cloneNode(true));$('liveEnd').append(o);}
 $('liveStart').value='18';$('liveEnd').value='24';
 let payload=null;
 function draw(){
  const sel=payload?.last_successful_forecast?.horizons?.find(x=>x.horizon_days===Number($('liveHorizon').value));
  if(!sel||payload.run?.status!=='issued_research_only'){ $('liveTarget').textContent='대상일 —';$('liveModel').textContent='모델 정보 —';$('liveChart').innerHTML='<p class="live-muted">현재 실행에서 새 예측을 발급하지 않았습니다. 아래 상태와 latest.json의 사유를 확인하세요.</p>';$('liveHours').innerHTML='';$('liveRecommendation').textContent='현재 예측이 없어 추천을 계산할 수 없습니다.';return; }
  $('liveTarget').textContent=`예측 대상일 ${sel.target_date}`;$('liveModel').textContent=`${sel.model_family} · 학습 cutoff ${sel.trained_through} · SHA-256 ${sel.model_sha256.slice(0,12)}…`;
  const vals=sel.predictions.map(Number),actual=Array.isArray(sel.actual_smp)?sel.actual_smp.map(Number):sel.official_smp_at_issue_values,observed=Array.isArray(actual)&&actual.length===24&&actual.every(Number.isFinite),all=observed?vals.concat(actual):vals,lo=Math.min(...all),hi=Math.max(...all),span=Math.max(1,hi-lo),path=a=>a.map((v,i)=>`${(i*400/23).toFixed(1)},${(120-(v-lo)*100/span).toFixed(1)}`).join(' '),pts=path(vals);
  $('liveActualLegend').hidden=!observed;$('liveActualLabel').textContent=Array.isArray(sel.actual_smp)?'사후 공식 실적':'기준시각 전 관측 SMP';
  $('liveChart').innerHTML=`<svg viewBox="0 0 400 140" role="img" aria-label="${esc(sel.target_date)} 시간별 SMP 예측과 공식가격 비교 그래프"><line x1="0" y1="120" x2="400" y2="120" stroke="#dbe5ed"/><polyline points="${pts}" fill="none" stroke="#168c7d" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>${observed?`<polyline points="${path(actual)}" fill="none" stroke="#647b90" stroke-width="2" stroke-dasharray="5 4"/>`:''}</svg><div class="live-axis"><span>00시</span><span>12시</span><span>23시</span></div>`;
  $('liveHours').innerHTML=vals.map((v,i)=>`<span>${String(i).padStart(2,'0')}시<br><b>${money(v)}</b></span>`).join('');recommend();
 }
 function recommend(){
  const sel=payload?.last_successful_forecast?.horizons?.find(x=>x.horizon_days===Number($('liveHorizon').value));if(!sel||payload.run?.status!=='issued_research_only')return;
  const cap=Number($('liveCapacity').value),soc=Number($('liveSoc').value),goal=Number($('liveGoal').value),power=Number($('livePower').value),start=Number($('liveStart').value),end=Number($('liveEnd').value);
  if(!(cap>0&&goal>=soc&&power>0&&end>start)){ $('liveRecommendation').textContent='차량 조건을 확인해 주세요.';return; }
  const need=(goal-soc)*cap/100, efficiency=.9, slots=Array.from({length:24},(_,h)=>h).filter(h=>h>=start&&h<end).sort((a,b)=>sel.predictions[a]-sel.predictions[b]);let remain=need,plan=[];for(const h of slots){if(remain<=0)break;const kwh=Math.min(power,remain/efficiency);plan.push({h,kwh});remain-=kwh*efficiency;}
  if(remain>1e-6){$('liveRecommendation').textContent='선택한 시간 안에 목표 SOC까지 충전할 수 없습니다.';return;}
  const cost=plan.reduce((s,x)=>s+x.kwh*sel.predictions[x.h],0);$('liveRecommendation').innerHTML=`추천 시간 ${plan.map(x=>String(x.h).padStart(2,'0')+'시').join(', ')} · 예상 SMP 가상비용 <b>${money(cost)}원</b> · 충전량 ${money(need)}kWh`;
  const actual=sel.actual_smp;if(Array.isArray(actual)&&actual.length===24&&actual.every(Number.isFinite)){const chosen=plan.reduce((s,x)=>s+x.kwh*actual[x.h],0);let left=need,best=0;for(const h of [...slots].sort((a,b)=>actual[a]-actual[b])){if(left<=0)break;const k=Math.min(power,left/efficiency);best+=k*actual[h];left-=k*efficiency;} $('liveRegret').textContent=`실제 SMP 사후 비용 ${money(chosen)}원 · 최적 계획 대비 regret ${money(chosen-best)}원`;}else $('liveRegret').textContent='실제 SMP가 확보되기 전에는 사후 비용과 regret을 계산하지 않습니다.';
 }
 $('liveHorizon').onchange=draw;$('liveRecommend').onclick=recommend;['liveCapacity','liveSoc','liveGoal','livePower','liveStart','liveEnd'].forEach(id=>$(id).addEventListener('change',recommend));
 fetch('./live/latest.json?ts='+Date.now(),{cache:'no-store'}).then(r=>{if(!r.ok)throw Error('HTTP '+r.status);return r.json();}).then(data=>{payload=data;const run=data.run||{};const last=data.last_successful_forecast;const status=run.status==='issued_research_only'?`이번 실행에서 연구 예측 발급 · ${run.issued_at_kst||run.issued_at_utc}`:`이번 실행은 예측을 보류했습니다 · ${run.reason||'최근 실행 결과 없음'}`;const collected=run.collection_status||'수집 상태 확인 필요';$('liveStatus').innerHTML=`<b>${esc(status)}</b><br><small>${esc(collected)}${last&&run.status!=='issued_research_only'?` · 이전 발급 ${esc(last.issued_at_kst||last.issued_at_utc)} (현재 예측으로 사용하지 않음)`:''}</small>`;draw();}).catch(e=>{$('liveStatus').textContent='최신 실행 상태 파일을 읽지 못했습니다: '+e.message;draw();});
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',boot);else boot();
})();
