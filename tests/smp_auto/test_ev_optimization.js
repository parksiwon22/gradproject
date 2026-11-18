'use strict';
const assert = require('node:assert/strict');
const fs = require('node:fs');
const zlib = require('node:zlib');
const path = require('node:path');
const ev = require('../../docs/ev_optimization.js');

const cfg = {capacity:60, initialSoc:30, targetSoc:80, minSoc:20, maxSoc:90,
  chargeKw:7, dischargeKw:7, chargeEff:.9, dischargeEff:.9,
  degradation:5, purchaseAdder:0, sellRatio:1};

function testFrozenRegretAndPythonParity() {
  const file = path.resolve(__dirname, '../../docs/live/walkforward_d3_2026.json.gz');
  const payload = JSON.parse(zlib.gunzipSync(fs.readFileSync(file)).toString('utf8'));
  for (const row of [payload.records[0], payload.records[157], payload.records.at(-1)]) {
    const pred = ev.g2vPlan(row.predictions, cfg);
    const realized = ev.score(row.actual_smp, pred.chargeKw, pred.dischargeKw, cfg).netCost;
    const oracle = ev.g2vPlan(row.actual_smp, cfg);
    const regret = realized - oracle.netCost;
    assert.ok(Math.abs(regret - row.metrics.g2v_regret_won) < 1e-5,
      `Python/JS G2V regret differs at ${row.target_date}`);
    assert.ok(Math.abs(realized - row.metrics.g2v_predicted_plan_actual_cost_won) < 1e-5);
    const v2gRealized = ev.score(row.actual_smp, row.metrics.v2g_charge_kw,
      row.metrics.v2g_discharge_kw, cfg).netCost;
    const v2gExpected = ev.score(row.predictions, row.metrics.v2g_charge_kw,
      row.metrics.v2g_discharge_kw, cfg).netCost;
    const v2gOracle = ev.score(row.actual_smp, row.metrics.v2g_oracle_charge_kw,
      row.metrics.v2g_oracle_discharge_kw, cfg).netCost;
    assert.ok(Math.abs(v2gRealized-row.metrics.v2g_actual_cost_won)<1e-3,
      `Python/JS V2G realized cost differs at ${row.target_date}`);
    assert.ok(Math.abs(v2gExpected-row.metrics.v2g_predicted_cost_won)<1e-3);
    assert.ok(Math.abs(v2gOracle-row.metrics.v2g_oracle_cost_won)<1e-3);
    assert.ok(Math.abs(v2gRealized-v2gOracle-row.metrics.v2g_regret_won)<1e-3);
  }
}

function testOracleAndReversedOrdering() {
  const actual = Array.from({length:24}, (_,h)=>40+2*h);
  const matching = ev.g2vPlan(actual, cfg);
  assert.ok(Math.abs(ev.realizedRegret(matching, actual, cfg).regret) < 1e-8);
  const reversed = ev.g2vPlan([...actual].reverse(), cfg);
  assert.ok(ev.realizedRegret(reversed, actual, cfg).regret > 0);
  assert.ok(reversed.chargeKw.some(x=>x>0&&x<7));
}

function testVirtualV2gConstraintsAndConnectionWindow() {
  const prices = Array.from({length:24}, (_,h)=> h>=17&&h<=20?200:50);
  const plan = ev.v2gPlan(prices, cfg, 7, 22);
  assert.ok(plan.chargeKw.every((x,h)=>x===0||h>=7&&h<22));
  assert.ok(plan.dischargeKw.every((x,h)=>x===0||h>=7&&h<22));
  assert.ok(plan.chargeKw.every((x,h)=>!(x>1e-8&&plan.dischargeKw[h]>1e-8)));
  assert.ok(plan.socPct.every(x=>x>=cfg.minSoc-1e-8&&x<=cfg.maxSoc+1e-8));
  assert.ok(plan.socPct.at(-1)>=cfg.targetSoc-1e-8);
  const actualOracle = ev.realizedRegret(ev.v2gPlan(prices,cfg,7,22),prices,cfg,7,22,'V2G');
  assert.ok(Math.abs(actualOracle.regret)<1e-8);
  const noSettlement = ev.v2gPlan(Array(24).fill(100), {...cfg,sellRatio:0,degradation:1000});
  assert.equal(noSettlement.gridExportKwh,0,
    'V2G should leave battery energy unused for export when sale value is zero and wear is high');
}

testFrozenRegretAndPythonParity();
testOracleAndReversedOrdering();
testVirtualV2gConstraintsAndConnectionWindow();
console.log('EV optimization tests passed (Python parity, frozen regret, G2V/V2G constraints).');
