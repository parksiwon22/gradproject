const test=require('node:test'),assert=require('node:assert/strict');const {classify}=require('../../docs/smp72_weather.js');
const rows=(date,prices)=>prices.map((p,i)=>({target_timestamp:`${date}T${String(i).padStart(2,'0')}:00:00+09:00`,predicted_smp:p}));
test('sunny for prices at/below daily median, cloudy above',()=>{const r=classify(rows('2026-11-20',[40,10,20,30]));assert.deepEqual(r.map(x=>x.weather),['cloudy','sunny','sunny','cloudy']);assert.ok(r.every(x=>x.median===25));});
test('daily relative index is independent of another day and actual prices',()=>{const input=[...rows('2026-11-20',[1,2]),...rows('2026-11-21',[100,200])];assert.deepEqual(classify(input).map(x=>x.weather),['sunny','cloudy','sunny','cloudy']);assert.deepEqual(classify(input.map(x=>({...x,actual_smp:999}))).map(x=>x.weather),classify(input).map(x=>x.weather));});
test('identical prices have no invented price advantage',()=>{assert.ok(classify(rows('2026-11-20',[5,5,5])).every(x=>x.weather==='flat'&&x.label==='가격 차이 없음'));});
test('invalid prices cannot become a charging weather card',()=>{assert.throws(()=>classify(rows('2026-11-20',[NaN])),/Invalid/);});
