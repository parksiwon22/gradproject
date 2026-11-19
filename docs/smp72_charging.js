/* Pure, deterministic fractional-hour G2V allocation. No tariff or V2G assumptions. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.SMP72Charging=api;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
'use strict';
function plan(rows,settings,now=Date.now()){
 const {capacity,currentSoc,targetSoc,power,efficiency}=settings;
 for(const v of[capacity,currentSoc,targetSoc,power,efficiency])if(!Number.isFinite(v))throw Error('모든 충전 조건을 숫자로 입력해주세요.');
 if(capacity<=0||power<=0)throw Error('배터리 용량과 충전 출력은 0보다 커야 합니다.');
 if(currentSoc<0||currentSoc>100||targetSoc<0||targetSoc>100||targetSoc<=currentSoc)throw Error('SOC는 0~100%이며 목표 SOC가 현재 SOC보다 커야 합니다.');
 if(efficiency<=0||efficiency>100)throw Error('충전 효율은 0보다 크고 100% 이하여야 합니다.');
 const start=Date.parse(settings.start),end=Date.parse(settings.end);
 if(!Number.isFinite(start)||!Number.isFinite(end)||start>=end)throw Error('충전 시작과 종료 날짜·시간을 확인해주세요.');
 const ordered=[...rows].sort((a,b)=>Date.parse(a.target_timestamp)-Date.parse(b.target_timestamp));
 if(!ordered.length)throw Error('예측 가격 자료가 없습니다.');
 const first=Date.parse(ordered[0].target_timestamp),last=Date.parse(ordered.at(-1).target_timestamp)+3600000;
 if(start<first||end>last)throw Error('충전 가능 구간을 예측된 72시간 범위 안으로 설정해주세요.');
 const effectiveStart=Math.max(start,now),batteryEnergy=capacity*(targetSoc-currentSoc)/100,gridEnergy=batteryEnergy/(efficiency/100);
 const eligible=ordered.map(r=>{const t=Date.parse(r.target_timestamp),a=Math.max(t,effectiveStart),b=Math.min(t+3600000,end);return {timestamp:r.target_timestamp,start:a,end:b,hours:Math.max(0,(b-a)/3600000),price:r.predicted_smp};}).filter(r=>r.hours>0);
 if(eligible.some(r=>!Number.isFinite(r.price)))throw Error('충전 구간의 예측 가격에 오류가 있습니다.');
 const availableEnergy=eligible.reduce((s,r)=>s+r.hours*power,0);
 if(availableEnergy+1e-8<gridEnergy)throw Error(`선택한 시간에는 목표 SOC에 도달할 수 없습니다. 필요한 계통 전력량 ${gridEnergy.toFixed(2)}kWh, 공급 가능량 ${availableEnergy.toFixed(2)}kWh입니다. 종료시간을 늘리거나 목표 SOC·출력을 조정해주세요.`);
 function allocate(candidates){let remaining=gridEnergy;const allocations=[];for(const r of candidates){if(remaining<1e-9)break;const energy=Math.min(remaining,r.hours*power);allocations.push({...r,energy,duration:energy/power,cost:energy*r.price});remaining-=energy;}return allocations;}
 const cheapest=allocate([...eligible].sort((a,b)=>a.price-b.price||a.start-b.start)).sort((a,b)=>a.start-b.start),earliest=allocate(eligible);
 const cost=cheapest.reduce((s,r)=>s+r.cost,0),baselineCost=earliest.reduce((s,r)=>s+r.cost,0);
 return {batteryEnergy,gridEnergy,chargingHours:gridEnergy/power,availableEnergy,allocations:cheapest,cost,baselineCost,difference:baselineCost-cost,effectiveStart,excludedPast:effectiveStart>start};
}
return {plan};
});
