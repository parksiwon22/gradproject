/* Shared browser/Node EV dispatch functions. V2G uses a 0.1 kWh DP grid. */
'use strict';
(function(root,factory){const api=factory();if(typeof module!=='undefined'&&module.exports)module.exports=api;root.EVOptimization=api;})(typeof globalThis!=='undefined'?globalThis:this,function(){
  const sum=a=>a.reduce((s,x)=>s+Number(x),0);
  function validate(prices,cfg,start=0,end=24){
    if(!Array.isArray(prices)||prices.length!==24||!prices.every(Number.isFinite))throw Error('SMP 예측이 24시간 완전하지 않습니다.');
    if(!(cfg.capacity>0&&cfg.chargeKw>0&&cfg.dischargeKw>=0&&cfg.chargeEff>0&&cfg.chargeEff<=1&&cfg.dischargeEff>0&&cfg.dischargeEff<=1&&cfg.minSoc>=0&&cfg.minSoc<=cfg.initialSoc&&cfg.initialSoc<=cfg.maxSoc&&cfg.maxSoc<=100&&cfg.minSoc<=cfg.targetSoc&&cfg.targetSoc<=cfg.maxSoc&&cfg.degradation>=0&&cfg.sellRatio>=0&&cfg.sellRatio<=1&&Number.isFinite(cfg.purchaseAdder)))throw Error('EV 조건이 올바르지 않습니다.');
    if(!(Number.isInteger(start)&&Number.isInteger(end)&&start>=0&&end<=24&&end>start))throw Error('연결 시간을 확인해 주세요.');
  }
  function score(prices,chargeKw,dischargeKw,cfg,dt=Array(24).fill(1)){
    validate(prices,cfg);if(chargeKw.length!==24||dischargeKw.length!==24||dt.length!==24)throw Error('시간별 dispatch 길이가 24가 아닙니다.');
    let buy=0,sell=0,wear=0,imported=0,exported=0;
    for(let h=0;h<24;h++){const c=Number(chargeKw[h]),d=Number(dischargeKw[h]),t=Number(dt[h]);if(c< -1e-8||d< -1e-8||Math.min(c,d)>1e-8||t<0||c>cfg.chargeKw+1e-7||d>cfg.dischargeKw+1e-7)throw Error('충전·방전 제약을 위반했습니다.');buy+=(Number(prices[h])+cfg.purchaseAdder)*c*t;sell+=Number(prices[h])*d*t*cfg.sellRatio;wear+=cfg.degradation*(cfg.chargeEff*c+d/cfg.dischargeEff)*t;imported+=c*t;exported+=d*t;}
    return {purchaseCost:buy,saleRevenue:sell,degradationCost:wear,netCost:buy-sell+wear,gridImportKwh:imported,gridExportKwh:exported};
  }
  function g2vPlan(prices,cfg,start=0,end=24){
    validate(prices,cfg,start,end);const batteryNeed=Math.max(0,cfg.capacity*(cfg.targetSoc-cfg.initialSoc)/100),gridNeed=batteryNeed/cfg.chargeEff,slots=Array(24).fill(0),order=Array.from({length:end-start},(_,i)=>start+i).sort((a,b)=>(prices[a]+cfg.purchaseAdder)-(prices[b]+cfg.purchaseAdder)||a-b);
    if(gridNeed>cfg.chargeKw*(end-start)+1e-8)throw Error('연결 시간 안에 목표 SOC를 달성할 수 없습니다.');let left=gridNeed;for(const h of order){if(left<=1e-9)break;slots[h]=Math.min(cfg.chargeKw,left);left-=slots[h];}
    const discharge=Array(24).fill(0),cost=score(prices,slots,discharge,cfg);return {chargeKw:slots,dischargeKw:discharge,socPct:socCurve(cfg,slots,discharge),...cost};
  }
  function socCurve(cfg,c,d){const out=[cfg.initialSoc];let e=cfg.capacity*cfg.initialSoc/100;for(let h=0;h<24;h++){e+=cfg.chargeEff*c[h]-d[h]/cfg.dischargeEff;out.push(100*e/cfg.capacity);}return out;}
  function v2gPlan(prices,cfg,start=0,end=24){
    validate(prices,cfg,start,end);const step=.1,e0=cfg.capacity*cfg.initialSoc/100,emin=cfg.capacity*cfg.minSoc/100,emax=cfg.capacity*cfg.maxSoc/100,etarget=cfg.capacity*cfg.targetSoc/100,minK=Math.ceil((emin-e0)/step-1e-9),maxK=Math.floor((emax-e0)/step+1e-9),targetK=Math.ceil((etarget-e0)/step-1e-9),n=maxK-minK+1;
    if(n<=0||n>5001||0<minK||0>maxK)throw Error('SOC DP 상태 범위를 계산할 수 없습니다.');
    let cur=new Float64Array(n);cur.fill(Infinity);cur[-minK]=0;const prev=Array.from({length:24},()=>{const a=new Int32Array(n);a.fill(2147483647);return a;}),maxUp=cfg.chargeKw*cfg.chargeEff,maxDown=cfg.dischargeKw/cfg.dischargeEff;
    for(let h=0;h<24;h++){const next=new Float64Array(n);next.fill(Infinity);const connected=h>=start&&h<end;
      for(let idx=0;idx<n;idx++){const old=cur[idx];if(!Number.isFinite(old))continue;const k=idx+minK,lo=connected?Math.max(minK,Math.ceil(k-maxDown/step-1e-9)):k,hi=connected?Math.min(maxK,Math.floor(k+maxUp/step+1e-9)):k;
        for(let q=lo;q<=hi;q++){const delta=(q-k)*step;let c=0,d=0,wear=0,trade=0;if(delta>1e-9){c=delta/cfg.chargeEff;wear=cfg.degradation*delta;trade=(prices[h]+cfg.purchaseAdder)*c;}else if(delta< -1e-9){d=-delta*cfg.dischargeEff;wear=cfg.degradation*(-delta);trade=-prices[h]*d*cfg.sellRatio;}if(c>cfg.chargeKw+1e-7||d>cfg.dischargeKw+1e-7)continue;const j=q-minK,val=old+trade+wear;if(val<next[j]-1e-9){next[j]=val;prev[h][j]=k;}}
      }cur=next;
    }
    let final=-1,best=Infinity;for(let q=Math.max(minK,targetK);q<=maxK;q++){const v=cur[q-minK];if(v<best){best=v;final=q;}}if(final<0||!Number.isFinite(best))throw Error('출발 SOC 및 최소·최대 SOC를 만족하는 충방전 계획이 없습니다.');
    const c=Array(24).fill(0),d=Array(24).fill(0),soc=Array(25);let q=final;soc[24]=100*(e0+q*step)/cfg.capacity;
    for(let h=23;h>=0;h--){const k=prev[h][q-minK];if(k===2147483647)throw Error('V2G 계획 복원에 실패했습니다.');const delta=(q-k)*step;if(delta>1e-9)c[h]=delta/cfg.chargeEff;else if(delta< -1e-9)d[h]=-delta*cfg.dischargeEff;soc[h]=100*(e0+k*step)/cfg.capacity;q=k;}
    const cost=score(prices,c,d,cfg);return {chargeKw:c,dischargeKw:d,socPct:soc,endingSocPct:soc[24],resolutionKwh:step,...cost};
  }
  function realizedRegret(decisionPlan,actualPrices,cfg,start=0,end=24,mode='G2V'){
    const oracle=mode==='V2G'?v2gPlan(actualPrices,cfg,start,end):g2vPlan(actualPrices,cfg,start,end),realized=score(actualPrices,decisionPlan.chargeKw,decisionPlan.dischargeKw,cfg);return {realizedCost:realized.netCost,oracleCost:oracle.netCost,regret:realized.netCost-oracle.netCost,oracle};
  }
  return {score,g2vPlan,v2gPlan,realizedRegret,socCurve};
});
