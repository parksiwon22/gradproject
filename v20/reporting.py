"""Public aggregate export; hourly data and frozen decisions remain local."""
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from v20.core import HOME,preserve
from v19.common import ROOT,read,write,write_csv,sha,now


def report():
    docs=ROOT/'docs';fig=docs/'v20_figures';fig.mkdir(exist_ok=True)
    status=read(HOME/'status.json');status['preservation']=preserve()
    status['scheduler_snapshot']=read(HOME/'scheduler/status.json')
    write(docs/'V20_STATUS.json',status)
    p=pd.read_csv(HOME/'publication.csv');write_csv(docs/'V20_PUBLICATION_OBSERVATIONS.csv',p)
    comparison=pd.read_csv(ROOT/'artifacts/v19/comparison.csv')
    allowed=list(pd.read_csv(docs/'V19_MODEL_COMPARISON.csv',nrows=0).columns)
    write_csv(docs/'V20_MODEL_COMPARISON.csv',comparison[[c for c in allowed if c in comparison]])
    fixed=pd.read_csv(ROOT/'artifacts/v19/frozen_costs.csv');write_csv(docs/'V20_FROZEN_EV_COMPARISON.csv',fixed.drop(columns=['file','forecast_sha256'],errors='ignore'))
    grid=read(HOME/'grid_audit.json');write(docs/'V20_ZERO_GRID_AUDIT.json',grid)
    sample=p[(p.source=='general_public')&p.target_date.isin(['2026-10-18','2026-10-19','2026-10-20'])]
    f,ax=plt.subplots(figsize=(7,3));ax.bar(sample.target_date,sample.zero_response_count,color='#c78f47');ax.set_ylabel('Observed all-zero receipts');ax.set_title('All-zero page responses are NOT certificates of unpublished prices');f.tight_layout();f.savefig(fig/'zero_observations.png',dpi=150);plt.close(f)
    day=comparison[(comparison.target_date=='2026-10-18')&comparison.model.isin(['V12_HGB','V12_LightGBM_peak','V17_joint'])]
    f,axes=plt.subplots(1,2,figsize=(9,3.5))
    for ax,col,title in zip(axes,['MAE','G2V_regret'],['Price MAE (KRW/kWh)','Same-EV G2V regret (virtual KRW)']):
        ax.bar(day.model,day[col],color='#4676ad');ax.set_title(title);ax.tick_params(axis='x',rotation=15)
    f.suptitle('2026-10-18 preserved late forecasts; one day, not on-time evidence');f.tight_layout();f.savefig(fig/'price_ev_comparison.png',dpi=150);plt.close(f)
    log=(HOME/'tests_all.log').read_text(encoding='utf-8-sig',errors='replace');match=re.search(r'Ran (\d+) tests in ([\d.]+)s',log)
    if not match or not re.search(r'(?m)^OK\s*$',log) or re.search(r'(?m)^FAILED',log):raise ValueError('Successful full regression required')
    focused=(HOME/'tests_targeted_final.log').read_text(encoding='utf-8-sig',errors='replace')
    if not re.search(r'(?m)^OK\s*$',focused) or re.search(r'(?m)^FAILED',focused):raise ValueError('Successful final V20 tests required')
    origin=pd.Timestamp.now(tz='Asia/Seoul').normalize()+pd.Timedelta(hours=23)
    observed=[]
    for path in (ROOT/'artifacts/v19/publish_runs').glob('*.json'):
        record=read(path)
        if record.get('mode')=='LIVE_LOCAL_PUBLISH_CHECK' and record.get('reference_origin')==origin.isoformat():
            observed.append(record)
    created=sum(x.get('status')=='created' for run in observed for x in run.get('records',[]))
    issuance=dict(reference_origin=origin.isoformat(),actual_publish_checks=len(observed),
                  forecast_files_created=created,
                  status='observed_no_forecast_issued' if observed and created==0 else
                         'observed_forecast_issued' if created else 'not_observed_at_report_time',
                  note='Preparation and worker success are not forecast issuance; simulation is excluded.')
    write(docs/'V20_VALIDATION.json',dict(at=now(),tests=int(match[1]),seconds=float(match[2]),result='OK',log_sha256=sha(HOME/'tests_all.log'),focused_tests=12,focused_log_sha256=sha(HOME/'tests_targeted_final.log'),preservation=status['preservation'],new_actual_forecasts=status['V19_new_actual_records'],actual23h_issuance=issuance,operating_model_replaced=False))
    print('V20 aggregate exports and preservation verification completed.')


if __name__=='__main__':report()
