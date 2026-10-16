"""Diagnostic event catalog, type-specific OOF scores and known-failure analogs."""
import json
import numpy as np,pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from v11.store import ROOT,connect,latest,now
from v11.features import FEATURES
from v14.features import PRICE_EXTRA
from v15.research import HOME,write
from v15.features import build

def catalog(f,mask,definition):
    times=pd.DatetimeIndex(f.timestamp); y=f.actual_smp.to_numpy(); starts=np.r_[True,(times[1:]-times[:-1])!=pd.Timedelta(hours=1)]
    ids=np.cumsum(starts|np.r_[True,mask[1:]!=mask[:-1]])
    records=[]
    for event in np.unique(ids[mask]):
        pos=np.flatnonzero((ids==event)&mask); g=f.iloc[pos]; duration=len(pos); first=pos[0]
        previous=float(f.iloc[first].previous_actual_smp); maximum=float(g.actual_smp.max()); peak=int(g.actual_smp.to_numpy().argmax())
        typ='long_price_regime' if duration>=72 else 'plateau' if duration>=6 else 'short_spike' if duration<=2 else 'multi_hour_jump'
        records.append(dict(definition=definition,start=g.timestamp.iloc[0],end=g.timestamp.iloc[-1],year=int(g.timestamp.iloc[0].year),hours=duration,event_type=typ,maximum=maximum,mean=float(g.actual_smp.mean()),rise_from_previous=maximum-previous,relative_rise=(maximum-previous)/max(abs(previous),10),peak_hour=int(g.timestamp.iloc[peak].hour),max_positive_ramp=float((g.actual_smp-g.previous_actual_smp).max()),known_profile_uplift=float((g.actual_smp-g.daily).max())))
    return records

def run():
    folder=HOME/'events'; folder.mkdir(exist_ok=True)
    if not (folder/'design_before_catalog.json').exists(): write(folder/'design_before_catalog.json',dict(at=now(),absolute_threshold=150,relative_threshold='actual minus known D-2 profile >=20',duration_types=dict(short_max=2,plateau_min=6,regime_min=72),purpose='diagnosis only, not a model feature or tuning target'))
    db=connect(); raw=latest(db,'land'); db.close(); f,cols=build(raw)
    facts=raw[['timestamp','smp']].rename(columns={'smp':'actual_smp'}).copy(); facts['previous_actual_smp']=facts.actual_smp.shift(1)
    facts['daily']=raw.set_index('timestamp').smp.reindex(pd.DatetimeIndex(facts.timestamp)-pd.Timedelta(days=2)).to_numpy()
    events=catalog(facts,(facts.actual_smp>=150).to_numpy(),'absolute150')+catalog(facts,(facts.actual_smp-facts.daily>=20).to_numpy(),'relative20')
    pd.DataFrame(events).to_csv(folder/'event_catalog.csv',index=False)
    pd.DataFrame(events).groupby(['year','definition','event_type']).agg(events=('hours','size'),hours=('hours','sum'),median_duration=('hours','median'),mean_peak=('maximum','mean')).reset_index().to_csv(folder/'type_inventory.csv',index=False)
    incidence=[]
    for year,g in facts.groupby(facts.timestamp.dt.year): incidence.append(dict(year=int(year),hours=len(g),mean=float(g.actual_smp.mean()),high150_hours=int((g.actual_smp>=150).sum()),high150_fraction=float((g.actual_smp>=150).mean()),relative20_hours=int((g.actual_smp-g.daily>=20).sum()),relative20_valid_hours=int(g.daily.notna().sum()),relative20_fraction=float((g.actual_smp-g.daily>=20).mean())))
    pd.DataFrame(incidence).to_csv(folder/'year_incidence.csv',index=False)
    # Only genuinely out-of-training fold outputs enter type performance.
    scores=[]; type_index=pd.Series(index=pd.DatetimeIndex(facts.timestamp),dtype='object')
    for event in events:
        if event['definition']=='absolute150': type_index.loc[event['start']:event['end']]=event['event_type']
    for path in (HOME/'model_study').glob('*_predictions.csv'):
        data=pd.read_csv(path, float_precision='round_trip'); data.timestamp=pd.to_datetime(data.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        for name in ['baseline','event_features_lgb','relative_event_weight','relative_two_stage','event_analog']:
            if name not in data: continue
            subset=data.assign(event_type=type_index.reindex(pd.DatetimeIndex(data.timestamp)).to_numpy())
            for typ,g in subset.dropna(subset=['event_type']).groupby('event_type'):
                if g.empty: continue
                scores.append(dict(fold=path.stem,model=name,event_type=typ,hours=len(g),MAE=float(abs(g[name]-g.actual_smp).mean()),bias=float((g[name]-g.actual_smp).mean()),recall=float((g[name]>=150).mean())))
    pd.DataFrame(scores).to_csv(folder/'type_prediction_scores.csv',index=False)
    train=f[f.target_date<='2025-06-30']; cases=f[(f.target_date>='2026-10-01')&(f.target_date<='2026-10-15')&(f.actual_smp>=150)]; analogs=[]
    for label,fields in [('price21',FEATURES),('price33',FEATURES+PRICE_EXTRA),('event42',cols)]:
        scaler=StandardScaler().fit(train[fields]); neighbor=NearestNeighbors(n_neighbors=20,n_jobs=4).fit(scaler.transform(train[fields])); ds,ids=neighbor.kneighbors(scaler.transform(cases[fields]))
        for (_,case),dist,index in zip(cases.iterrows(),ds,ids):
            nearby=train.iloc[index]; analogs.append(dict(schema=label,timestamp=case.timestamp,actual=case.actual_smp,known_D2_profile=case.daily,neighbor_high_fraction=float((nearby.actual_smp>=150).mean()),neighbor_uplift_fraction=float((nearby.actual_smp-nearby.daily>=20).mean()),neighbor_target_median=float(nearby.actual_smp.median()),neighbor_target_q95=float(nearby.actual_smp.quantile(.95)),mean_feature_distance=float(dist.mean())))
    pd.DataFrame(analogs).to_csv(folder/'known_six_analogs.csv',index=False)
    print('Diagnostic events',len(events),'known high hours',len(cases),flush=True)

if __name__=='__main__': run()
