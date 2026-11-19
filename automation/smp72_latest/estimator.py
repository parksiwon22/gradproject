"""Explicit two-stage constrained SARIMAX: all 83 exog retained.

beta = Train-only minimum-norm OLS; AR/MA/seasonal/variance conditional MLE.
This is NOT joint MLE of regression and ARIMA coefficients.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.kalman_filter import MEMORY_CONSERVE, MEMORY_NO_FORECAST, MEMORY_NO_PREDICTED


def standardized(x,mean,scale):
    result=(x-mean)/scale
    if not np.isfinite(result).all():raise ValueError('Nonfinite standardized exog')
    return result


class Estimator:
    def __init__(self,order=(1,0,0),seasonal_order=(1,0,0,24),maxiter=100):
        self.order=tuple(order);self.seasonal_order=tuple(seasonal_order);self.maxiter=maxiter
    def specification(self,y,x):
        return SARIMAX(y,exog=x,order=self.order,seasonal_order=self.seasonal_order,trend='n',enforce_stationarity=True,enforce_invertibility=True)
    def fit(self,y,x):
        y=np.asarray(y,dtype=float);x=np.asarray(x,dtype=float)
        if not np.isfinite(y).all() or not np.isfinite(x).all():raise ValueError('Nonfinite training values')
        self.mean=x.mean(0);sd=x.std(0);self.scale=np.where(sd<1e-6,1.,sd)
        self.y_mean=float(y.mean());self.y_scale=max(float(y.std()),1e-6)
        sx=standardized(x,self.mean,self.scale);sy=(y-self.y_mean)/self.y_scale
        self.beta,self.ols_residuals,self.rank,self.singular_values=np.linalg.lstsq(sx,sy,rcond=1e-10)
        self.constant_indices=np.where(sd<1e-6)[0].tolist()
        model=self.specification(sy,sx)
        coefficients={model.param_names[k]:float(self.beta[k]) for k in range(x.shape[1])}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            iteration=[0]
            def progress(params):
                iteration[0]+=1
                if iteration[0]%5==0:print('optimizer iteration',iteration[0],flush=True)
            result=model.fit_constrained(coefficients,disp=False,maxiter=self.maxiter,low_memory=True,cov_type='none',callback=progress)
        self.warnings=[str(w.message) for w in caught]
        self.converged=bool(result.mle_retvals.get('converged',False));self.retvals=result.mle_retvals
        if not self.converged:raise RuntimeError('Conditional SARIMAX optimizer did not converge')
        self.params=np.asarray(result.params)
        if not np.isfinite(self.params).all():raise RuntimeError('Nonfinite SARIMAX coefficients')
        return self
    def reset(self,y,x,last_timestamp):
        # Refilter only the available prefix; never retain end-of-Train future state.
        model=self.specification((np.asarray(y)-self.y_mean)/self.y_scale,standardized(np.asarray(x),self.mean,self.scale))
        self.state=model.filter(self.params,cov_type='none',conserve_memory=MEMORY_CONSERVE & ~(MEMORY_NO_FORECAST | MEMORY_NO_PREDICTED));self.last_timestamp=pd.Timestamp(last_timestamp)
    def update(self,y,x,timestamps,cutoff):
        timestamps=pd.DatetimeIndex(timestamps)
        if not len(timestamps):return
        if timestamps[0]!=self.last_timestamp+pd.Timedelta(hours=1):raise ValueError('State update gap')
        if timestamps[-1]+pd.Timedelta(hours=1)>cutoff:raise ValueError('Unreleased interval in state')
        if not (timestamps.to_series().diff().iloc[1:]==pd.Timedelta(hours=1)).all():raise ValueError('Nonhourly update')
        self.state=self.state.extend((np.asarray(y)-self.y_mean)/self.y_scale,exog=standardized(np.asarray(x),self.mean,self.scale))
        self.last_timestamp=timestamps[-1]
    def predict(self,x,origin,cutoff):
        origin=pd.Timestamp(origin);cutoff=pd.Timestamp(cutoff)
        if cutoff!=origin-pd.Timedelta(hours=72):raise ValueError('Incorrect fixed origin cutoff')
        if self.last_timestamp+pd.Timedelta(hours=1)!=cutoff:raise ValueError('State includes wrong interval')
        bridge=int((origin-self.last_timestamp).total_seconds()/3600)
        if len(x)!=bridge+72:raise ValueError('Missing bridge exog')
        forecast=self.state.get_forecast(steps=len(x),exog=standardized(np.asarray(x),self.mean,self.scale))
        means=np.asarray(forecast.predicted_mean)[bridge:]*self.y_scale+self.y_mean
        ci=np.asarray(forecast.conf_int())[bridge:]*self.y_scale+self.y_mean
        if not np.isfinite(means).all() or not np.isfinite(ci).all():raise RuntimeError('Nonfinite SARIMAX prediction')
        return means,ci,bridge
    def save(self,path):
        np.savez(path,params=self.params,beta=self.beta,mean=self.mean,scale=self.scale,y_mean=self.y_mean,y_scale=self.y_scale,order=self.order,seasonal_order=self.seasonal_order)
    @classmethod
    def load(cls,path):
        data=np.load(path);obj=cls(tuple(data['order']),tuple(data['seasonal_order']))
        for name in ['params','beta','mean','scale','y_mean','y_scale']:setattr(obj,name,data[name])
        return obj
