"""Bounded chronological GRU follow-up; initial summary-sample pilot was not a true time sequence."""
import json,joblib,numpy as np,pandas as pd,torch
from v11.store import connect,latest,now
from v14.features import augment
from v14.sequence import ChronoGRU
from v14.research import HOME,scored,ev_metrics,write
SEQ=['sequence_origin_lag'+str(h) for h in range(144,-1,-24)]
CAL=['hour_sin','hour_cos','dow_sin','dow_cos']

def predict(bundle,f):
    torch.set_num_threads(2); model=ChronoGRU(); model.load_state_dict(bundle['state']); model.eval()
    x=((f[SEQ].to_numpy(dtype='float32')-bundle['center'])/bundle['scale'])[:,:,None]
    with torch.no_grad(): p=model(torch.from_numpy(x),torch.from_numpy(f[CAL].to_numpy(dtype='float32',copy=True))).numpy()
    return f.daily.to_numpy()+p*bundle['scale']

def run():
    folder=HOME/'chronological_followup'; folder.mkdir(exist_ok=True)
    design=dict(at=now(),name='gru_chronological7day_pilot',reason='Initial GRU summary-sample ordering is not chronological; explicitly compare a real lag sequence with target calendar context',train=['2023-01-01','2025-06-30'],calibration=['2025-07-01','2025-09-30'],selection=['2025-10-01','2025-12-31'],max_epochs=6,hidden=16,seed=42,CPU=2,batch_size=256,learning_rate=.002,features=SEQ+CAL,initial_top3_not_reselected=True,no2026_audit_unless_validation_qualifies=True)
    if not (folder/'design_before_scores.json').exists(): write(folder/'design_before_scores.json',design)
    db=connect(); raw=latest(db,'land'); db.close(); data,_,_=augment(raw,external=False)
    def subset(key):
        a,b=design[key]; return data[(data.target_date>=a)&(data.target_date<=b)].reset_index(drop=True)
    train,cal,val=[subset(k) for k in ['train','calibration','selection']]
    path=folder/'gru_chronological7day_pilot.joblib'
    if path.exists(): bundle=joblib.load(path)
    else:
        torch.manual_seed(42); torch.set_num_threads(2); center=float(train[SEQ].to_numpy().mean()); scale=float(train[SEQ].to_numpy().std())
        def tensors(f): return torch.from_numpy(((f[SEQ].to_numpy(dtype='float32')-center)/scale)[:,:,None]),torch.from_numpy(f[CAL].to_numpy(dtype='float32',copy=True)),torch.from_numpy(((f.actual_smp-f.daily).to_numpy()/scale).astype('float32'))
        x,c,y=tensors(train); vx,vc,vy=tensors(cal); model=ChronoGRU(); optimizer=torch.optim.Adam(model.parameters(),lr=.002); best=float('inf'); history=[]; beststate=None
        for epoch in range(6):
            model.train()
            for ids in torch.randperm(len(x)).split(256):
                optimizer.zero_grad(); loss=torch.nn.functional.smooth_l1_loss(model(x[ids],c[ids]),y[ids]); loss.backward(); optimizer.step()
            model.eval()
            with torch.no_grad(): error=float((model(vx,vc)-vy).abs().mean())
            history.append(error)
            if error<best: best=error; beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
            print('Chronological GRU epoch',epoch+1,'calibration_scaled_MAE',round(error,5),flush=True)
        bundle=dict(kind='chrono_gru',state=beststate,center=center,scale=scale,features=SEQ+CAL,history=history,operational_candidate=True,trained_through='2025-06-30'); joblib.dump(bundle,path)
    p=predict(bundle,val); ev,daily=ev_metrics(val,p); result=dict(name=design['name'],split='selection_2025Q4_followup',**scored(val,p,float(train.actual_smp.quantile(.95))),**ev)
    baseline=pd.read_csv(HOME/'study/selection_metrics.csv').query("name=='V12_hgb_baseline'").iloc[0]
    qualifies=all(result[k]<=float(baseline[k]) for k in ['MAE','G2V_regret','V2G_regret'])
    result.update(validation_qualified=qualifies,audit_evaluated=False,not_promoted=True)
    write(folder/'result.json',result); pd.DataFrame([result]).to_csv(folder/'metrics.csv',index=False)
    val[['timestamp','target_date','actual_smp']].assign(predicted_smp=p).to_csv(folder/'predictions.csv',index=False); daily.to_csv(folder/'ev_daily.csv',index=False)
    print(json.dumps(result),flush=True)

if __name__=='__main__': run()
