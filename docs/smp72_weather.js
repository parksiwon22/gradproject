/* Charging weather is a relative price index, not a meteorological forecast. */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.SMP72Weather=api;})(typeof globalThis!=='undefined'?globalThis:this,()=>{
'use strict';
function classify(rows){
 const groups=new Map();for(const r of rows){if(!Number.isFinite(r.predicted_smp))throw Error('Invalid forecast price');const d=r.target_timestamp.slice(0,10);if(!groups.has(d))groups.set(d,[]);groups.get(d).push(r.predicted_smp);}
 const cuts=new Map();for(const[d,values]of groups){const a=[...values].sort((x,y)=>x-y),n=a.length;cuts.set(d,{median:n%2?a[(n-1)/2]:(a[n/2-1]+a[n/2])/2,flat:a[0]===a.at(-1)});}
 return rows.map(r=>{const c=cuts.get(r.target_timestamp.slice(0,10)),sunny=r.predicted_smp<=c.median;return {...r,weather:c.flat?'flat':sunny?'sunny':'cloudy',icon:c.flat?'⛅':sunny?'☀️':'☁️',label:c.flat?'가격 차이 없음':sunny?'충전하기 좋아요':'조금 기다려요',median:c.median};});
}
return {classify};
});
