"""Local scientific figures for seen diagnostics and genuinely pending records."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from v12.research import HOME

def main():
    folder=HOME/'figures'; folder.mkdir(parents=True,exist_ok=True)
    selected=json.loads((HOME/'study/selection_before_audit.json').read_text(encoding='utf-8'))['top3']
    for period in ('selection','retrospective_audit','known_failure_diagnosis'):
        f=pd.read_csv(HOME/'study'/(period+'_predictions.csv')); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul')
        days=f.groupby('target_date').actual_smp.max().nlargest(2).index
        fig,axes=plt.subplots(2,1,figsize=(12,7))
        for ax,day in zip(axes,days):
            g=f[f.target_date==day]
            for name in ['actual_smp','hgb_none']+selected:
                if name in g: ax.plot(g.timestamp,g[name],label=name,linewidth=1.2)
            ax.axhline(150,color='gray',linestyle='--',linewidth=.8); ax.set_title(period+' (already seen): '+day); ax.set_ylabel('KRW/kWh'); ax.legend(fontsize=7,ncol=3); ax.tick_params(axis='x',labelrotation=20)
        fig.tight_layout(); fig.savefig(folder/(period+'_peaks.png'),dpi=150); plt.close(fig)
    records=[]
    for file in (HOME/'forecasts').glob('*.json'):
        r=json.loads(file.read_text(encoding='utf-8'))
        if r.get('status')=='created': records.append(r)
    if records:
        target=max(r['target_date'] for r in records); fig,ax=plt.subplots(figsize=(12,4))
        for r in records:
            if r['target_date']==target:
                f=pd.DataFrame(r['rows']); ax.plot(pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul'),f.predicted_smp,label=r['model'])
        ax.set_title('Saved forward forecasts '+target+'; actual price pending'); ax.set_ylabel('KRW/kWh'); ax.legend(fontsize=8); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(folder/'pending_forward.png',dpi=150); plt.close(fig)
    examples=[]
    for name in ('G2V','V2G'):
        path=HOME/'analysis'/('example_'+name+'.csv')
        if path.exists():
            f=pd.read_csv(path); f.timestamp=pd.to_datetime(f.timestamp,utc=True).dt.tz_convert('Asia/Seoul'); examples.append((name,f))
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    for name,f in examples:
        axes[0].step(f.timestamp,f.charge_kw,label=name+' charge',where='post'); axes[0].step(f.timestamp,-f.discharge_kw,label=name+' discharge',where='post')
        axes[1].step(f.timestamp,f.soc_end,label=name+' SOC',where='post')
    axes[0].set_ylabel('grid kW'); axes[1].set_ylabel('SOC %')
    for ax in axes: ax.legend(fontsize=8); ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('Known official price example; hypothetical settlement'); fig.tight_layout(); fig.savefig(folder/'official_EV_SOC.png',dpi=150); plt.close(fig)

if __name__=='__main__': main()
