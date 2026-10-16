"""Publish aggregates/derived figures only; never raw prices, keys or personal EV."""
import json
from zoneinfo import ZoneInfo
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from v18.common import HOME,ROOT,read,write,csv,sha,now
from v18.evaluation import inventory,reference
from v18.publication import summarize

def run():
    docs=ROOT/'docs';figures=docs/'v18_figures';figures.mkdir(exist_ok=True)
    status=read(HOME/'status.json');sources=summarize();write(docs/'V18_FORWARD_STATUS.json',status);write(docs/'V18_PUBLICATION_WINDOWS.json',sources);write(docs/'V18_MODEL_REGISTRY.json',read(HOME/'model_registry.json'))
    f=csv(HOME/'comparison_index.csv');f.to_csv(docs/'V18_MODEL_COMPARISON.csv',index=False);frozen=csv(HOME/'frozen_decision_index.csv');frozen.to_csv(docs/'V18_FROZEN_DECISION_COMPARISON.csv',index=False)
    prices,_=reference();idx=prices.set_index('timestamp').smp;items=inventory()
    for day in status['primary_dates']:
        fig,ax=plt.subplots(figsize=(12,4))
        times=pd.date_range(pd.Timestamp(day,tz='Asia/Seoul'),periods=24,freq='h');ax.plot(times,idx.reindex(times),label='Official provided SMP',color='black',linewidth=2)
        for _,_,forecast,m in items:
            if m['primary_target_date']!=day or m['model'] not in ['hgb_none','lightgbm_peak','V17_joint_price']:continue
            g=forecast[forecast.timestamp.dt.strftime('%Y-%m-%d').eq(day)];ax.plot(g.timestamp,g.predicted_smp,label='Saved V'+str(m['version'])+' '+m['model'])
        ax.set_title(day+' | saved AI vs later official receipt (not final statistics)');ax.set_ylabel('KRW/kWh');ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=ZoneInfo('Asia/Seoul')));ax.set_xlabel('KST');ax.legend();fig.tight_layout();fig.savefig(figures/(day+'_prior_vs_official.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,6));r=[r for r in sources if r['target_date'] in status['primary_dates']+['2026-10-19'] and r['market']=='land' and r['first_normal_receipt']]
    for i,row in enumerate(r):
        first=pd.Timestamp(row['first_normal_receipt']);ax.scatter(first,i,color='#255e89');last=row['last_strict_normal_nonobservation']
        if last:ax.plot([pd.Timestamp(last),first],[i,i],color='#849aaa');ax.scatter(pd.Timestamp(last),i,marker='x',color='#c06b38')
    ax.set_yticks(range(len(r)),[x['target_date']+' / '+x['source'] for x in r]);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M',tz=ZoneInfo('Asia/Seoul')));ax.set_xlabel('Actual sampled receipt KST; dot=first positive, x=last verified negative');ax.set_title('Source-specific observations; universal release time remains unverified');fig.autofmt_xdate();fig.tight_layout();fig.savefig(figures/'publication_observations.png',dpi=150);plt.close(fig)
    write(docs/'V18_SOURCE_METHODS.json',dict(at=now(),sources=[dict(name='KPX public mainland hourly',url='https://www.kpx.or.kr/smpInland.es?mid=a10404080100',access='unauthenticated'),dict(name='KPX official SMP/demand API',url='https://apis.data.go.kr/B552115/SmpWithForecastDemand/getSmpWithForecastDemand',access='public-service API key, not member trading login'),dict(name='EPSIS hourly SMP',url='https://epsis.kpx.or.kr/epsisnew/selectEkgeGepGscGrid.do?menuId=040202',access='unauthenticated'),dict(name='KPX navigation alias',url='https://www.kpx.or.kr/menu.es?mid=a10404080100',result='Official navigation redirects to same smpInland backend; not an independent publication clock'),dict(name='Official price determination explanation',url='https://www.kpx.or.kr/menu.es?mid=a10401040000',result='Market price determination is distinct from a measured public API release timestamp')],failed_empty_ambiguous_not_treated_as_release_absence=True,original_history_directory='artifacts/v18/publication',raw_data_published=False))
    print('V18 aggregate tables and three figures exported')

if __name__=='__main__':run()
