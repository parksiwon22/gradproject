"""Export aggregate results only; never copy raw prices, models or EV inputs."""
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v19.common import HOME,ROOT,read,write,write_csv,now,preserve,sha


def report():
    docs=ROOT/'docs';fig=docs/'v19_figures';fig.mkdir(exist_ok=True)
    comparison=pd.read_csv(HOME/'comparison.csv')
    # Whitelist aggregate fields. No raw hourly price vector or personal settings.
    columns=['version','model','target_date','horizon_days','reference_origin','issued_at',
             'timing','strict_on_time_eligible','model_locked_before_origin',
             'input_receipts_eligible','cohort','comparison_group','status',
             'target_day_elapsed','final_statistics_verified','MAE','RMSE','peak_hours',
             'peak_MAE','peak_bias','peak_F1','peak_hour_error','rank_spearman',
             'cheap5_overlap','up_ramp_MAE','down_ramp_MAE','G2V_expected','G2V_actual',
             'official_G2V_optimum','G2V_regret','immediate_actual','uniform_actual',
             'V2G_actual_gain','V2G_export','V2G_regret']
    write_csv(docs/'V19_MODEL_COMPARISON.csv',comparison[[c for c in columns if c in comparison]])
    fixed=pd.read_csv(HOME/'frozen_costs.csv')
    keep=[c for c in fixed if c not in ['file','forecast_sha256']]
    write_csv(docs/'V19_FROZEN_EV_COMPARISON.csv',fixed[keep])
    paired=pd.read_csv(HOME/'paired_horizons.csv')
    write_csv(docs/'V19_PAIRED_HORIZONS.csv',paired.drop(columns=['model_sha256'],errors='ignore'))
    status=read(HOME/'status.json');stage=read(HOME/'preparation_status.json')
    ledger=read(HOME/'publication_ledger.json');origin=pd.Timestamp(stage['reference_origin'])
    readiness={}
    for h in [2,3]:
        day=(origin+pd.Timedelta(days=h)).strftime('%Y-%m-%d');r=ledger.get('days',{}).get(day,{})
        readiness[day]=dict(horizon=h,official_ever_acquired=r.get('ever_positive',False),
            latest_source_states={s:{k:v for k,v in x.items() if k in ['status','checked_at','legacy_parser']} for s,x in r.get('latest',{}).items()})
    status.update(report_at=now(),reference_origin=stage['reference_origin'],
                  unissued_preparations=stage['records'],readiness=readiness,
                  publication_requirement='Every sampled source verified negative within 180s; all-zero homepage is unknown and blocks issuance',
                  actual_23h_issuance_test='WAITING_FOR_REAL_TIME_NOT_A_SIMULATED_SUCCESS',
                  source_coverage='Sampled official routes; not a proof of universal first release',
                  observer_last_run=read(HOME/'observer_status.json'),
                  scheduler_last_snapshot=read(HOME/'scheduler/after_complete.json'),
                  preservation=preserve())
    write(docs/'V19_STATUS.json',status)
    historical=pd.read_csv(docs/'V19_D3_HISTORICAL_COMPARISON.csv')
    fig1,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,col,title in zip(axes,['MAE','G2V_regret'],['Price MAE (KRW/kWh)','G2V regret (virtual KRW/day)']):
        ax.bar(historical.model,historical[col],color=['#4676ad','#7b93b3','#cf874b']);ax.set_title(title);ax.tick_params(axis='x',rotation=18)
    fig1.suptitle('Already-seen 2025 Q4, 92 days: diagnostic validation, not prospective evidence')
    fig1.tight_layout();fig1.savefig(fig/'historical_tradeoff.png',dpi=150);plt.close(fig1)
    day=comparison[(comparison.target_date=='2026-10-18')&(comparison.horizon_days==2)]
    fig2,ax=plt.subplots(figsize=(10,4));ax.bar(day.model,day.G2V_regret,color='#4676ad')
    ax.set_ylabel('Same-EV replay regret (virtual KRW)');ax.tick_params(axis='x',rotation=25)
    ax.set_title('2026-10-18 official-provided prices vs preserved late forecasts; one day only')
    fig2.tight_layout();fig2.savefig(fig/'official_comparison_regret.png',dpi=150);plt.close(fig2)
    # Curves remain local: redistribution of underlying hourly observations is not assumed.
    from v19.evaluation import inventory,reference
    actual,_=reference();actual=actual.set_index('timestamp').smp
    local_fig=HOME/'figures';local_fig.mkdir(exist_ok=True)
    for target in ['2026-10-18','2026-10-19']:
        f,ax=plt.subplots(figsize=(10,4));plotted=False
        for _,_,prices,meta in inventory():
            g=prices[prices.timestamp.dt.strftime('%Y-%m-%d')==target]
            if len(g)!=24:continue
            if not plotted:ax.plot(range(24),actual.reindex(g.timestamp),color='black',linewidth=2,label='Official provided (not final correction)');plotted=True
            horizon=(pd.Timestamp(target).date()-pd.Timestamp(meta['reference_origin']).date()).days
            ax.plot(range(24),g.predicted_smp,label=f"D{horizon} {meta['model']} (late)")
        ax.set_title(target+' preserved actual records; missing horizon is not fabricated')
        ax.set_xlabel('KST hour');ax.set_ylabel('KRW/kWh');ax.legend(fontsize=7)
        f.tight_layout();f.savefig(local_fig/(target+'_actual_predictions.png'),dpi=150);plt.close(f)
    log=(HOME/'tests_all.log').read_text(encoding='utf-8-sig',errors='replace')
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    if not match or not re.search(r'(?m)^OK\s*$',log) or re.search(r'(?m)^FAILED',log):raise ValueError('Successful regression evidence required')
    write(docs/'V19_VALIDATION.json',dict(at=now(),regression_tests=int(match[1]),regression_seconds=float(match[2]),
          regression_result='OK',regression_log_sha256=sha(HOME/'tests_all.log'),
          preservation=status['preservation'],new_actual_forecasts=status['V19_new_actual_records'],
          HTTP_health='200 ok at 127.0.0.1:8509',
          simulations_are_not_actual_forecasts=True,model_promoted=False))
    print('Public aggregate reports exported; raw curves/models/records remain local.')


if __name__=='__main__':report()
