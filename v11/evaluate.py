"""Day-aware price, ramp and hypothetical SMP charging metrics."""
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_score,recall_score,f1_score

def metrics(f,p,threshold=150):
    y=f.actual_smp.to_numpy(); p=np.asarray(p); high=y>=threshold; alarms=p>=threshold
    result=dict(MAE=mean_absolute_error(y,p),RMSE=float(np.sqrt(mean_squared_error(y,p))),R2=r2_score(y,p),peak_MAE=float(np.mean(abs(p[high]-y[high]))) if high.any() else None,peak_Bias=float(np.mean(p[high]-y[high])) if high.any() else None,peak_Precision=precision_score(high,alarms,zero_division=0),peak_Recall=recall_score(high,alarms,zero_division=0),peak_F1=f1_score(high,alarms,zero_division=0),false_alarms=int((alarms&~high).sum()),peak_hours=int(high.sum()))
    maximum=[]; timing=[]; up=[]; down=[]; reg=[]; cost=[]; oraclecost=[]; overlap=[]; missed=[]; ramptrue=[]; ramppred=[]
    for ids in f.groupby('target_date',sort=True).indices.values():
        a,b=y[ids],p[ids]; assert len(ids)==24
        maximum.append(abs(a.max()-b.max())); timing.append(abs(int(a.argmax())-int(b.argmax())))
        prev=float(f.iloc[ids[0]].previous_actual_smp)
        da=np.diff(np.r_[prev,a]); dp=np.diff(np.r_[prev,b])
        up.extend(abs(dp[da>=15]-da[da>=15])); down.extend(abs(dp[da<=-15]-da[da<=-15]))
        ramptrue.extend(da>=15); ramppred.extend(dp>=15)
        for length in (2,3,4,5,6,8):
            actual=np.convolve(a,np.ones(length),'valid'); predicted=np.convolve(b,np.ones(length),'valid')
            chosen=int(predicted.argmin()); best=int(actual.argmin())
            cost.append(actual[chosen]); oraclecost.append(actual[best]); reg.append(actual[chosen]-actual[best]); overlap.append(len(set(range(chosen,chosen+length))&set(range(best,best+length)))/length)
            missed.append(int(np.any((a[chosen:chosen+length]>=150)&(b[chosen:chosen+length]<150))))
    result.update(daily_max_MAE=float(np.mean(maximum)),peak_hour_MAE=float(np.mean(timing)),up_ramp_MAE=float(np.mean(up)) if up else None,down_ramp_MAE=float(np.mean(down)) if down else None,ramp_Precision=precision_score(ramptrue,ramppred,zero_division=0),ramp_Recall=recall_score(ramptrue,ramppred,zero_division=0),ramp_F1=f1_score(ramptrue,ramppred,zero_division=0),EV_regret=float(np.mean(reg)),EV_cost=float(np.mean(cost)),EV_oracle_cost=float(np.mean(oraclecost)),cheapest_window_overlap=float(np.mean(overlap)),peak_charging_error_rate=float(np.mean(missed)))
    return result
