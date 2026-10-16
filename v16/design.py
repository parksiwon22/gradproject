"""Write analysis choices before opening their evaluation scores."""
from v16.common import HOME,write
from v11.store import now

def run():
    path=HOME/'design_before_scores.json'
    if path.exists(): return
    write(path,dict(at=now(),new_fit_requires_eligible_historical_external_data=True,new_price_only_training=0,
        definitions=dict(absolute=150,known_uplift=20,ramp='separate 90th percentile of positive/negative hourly changes in each fold training only',relative_day='highest6 hours per complete day; chronological ties; exclude flat days',daily_peak='maximum level error and earliest-max hour error'),
        inherited_folds='V15 expanding folds, no2026 selection',
        horizon_comparison=dict(train_end='2024-12-31',models='reuse V11 D1 HGB and V15 spring2025 D2 baseline, identical hyperparameters',validation=['2025-04-01','2025-06-30'],stress=['2025-07-01','2025-09-30'],warning='historical latest-vintage event-time comparison, not public-price availability replay; different warmup row availability'),
        sensitivity=dict(model='frozen V14 shape_level_blend outputs',efficiencies=[.8,.9,.95],degradation=[0,5,15],sell_ratios=[.8,.9,1],selection=['2025-10-01','2025-12-31'],audit=['2026-01-01','2026-06-30'],cells=27,qualification='prior selection gain>=10/day, export>=.5kWh/day,active>=10days,loss frequency<=.10; no operational promotion'),
        information_oracles=['point_baseline','perfect_daily_mean','perfect_order_only','perfect_shape','perfect_prices'],
        oracle_warning='Actual targets intentionally used only for value-of-information upper bounds; never fitted model, deployable forecast or new validation success',
        default_policy='G2V unless prospective evidence supports replacement',operating_model_replaced=False))

if __name__=='__main__':run()
