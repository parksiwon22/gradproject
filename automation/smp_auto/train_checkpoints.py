"""Build separate research refit checkpoints from the saved V20 source DB."""
from __future__ import annotations
import hashlib, json, os, sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(os.environ.get("SMP_SOURCE", ROOT.parent/"gradproject"))
sys.path.insert(0, str(SOURCE))
from v11.features import FEATURES as D1_FEATURES, build as build_d1
from v11.store import connect, latest
from v17.features import FEATURES as DX_FEATURES, build as build_dx

CUTOFF="2026-10-26"
OUT=ROOT/"automation"/"models"

def main():
    dbpath=SOURCE/"artifacts"/"v11"/"prices.sqlite"
    if not dbpath.exists(): raise RuntimeError(f"Missing saved source DB: {dbpath}")
    db=connect(SOURCE/"artifacts"/"v11")
    try: prices=latest(db,"land")
    finally: db.close()
    cutoff=pd.Timestamp(CUTOFF,tz="Asia/Seoul")+pd.Timedelta(hours=23)
    prices=prices[prices.timestamp<=cutoff].drop_duplicates("timestamp").sort_values("timestamp")
    if prices.empty or prices.timestamp.min()>pd.Timestamp("2023-04-01",tz="Asia/Seoul"):
        raise RuntimeError("Insufficient saved land SMP history (need data from 2023-04)")
    OUT.mkdir(parents=True,exist_ok=True)
    records={}
    for h in (1,2,3):
        frame=(build_d1(prices,days_ahead=1) if h==1 else build_dx(prices,days_ahead=h))
        features=D1_FEATURES if h==1 else DX_FEATURES
        frame=frame[frame.target_date<=CUTOFF]
        if frame.target_date.nunique()<400: raise RuntimeError(f"Insufficient D+{h} training days")
        y=(frame.actual_smp-frame.daily).to_numpy(float)
        if h==1:
            model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=2,random_state=42)
            anchor=0.0; family="V11-style ExtraTrees residual refit"
        else:
            model=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,early_stopping=False,l2_regularization=2.,random_state=42)
            anchor=.05; family="HGB residual + seasonal anchor refit"
        model.fit(frame[features].to_numpy(float),y)
        path=OUT/f"d{h}_selected_refit.joblib"
        joblib.dump(dict(model=model,horizon_days=h,features=features,anchor_weight=anchor,
                         trained_through=CUTOFF,family=family,model_origin="AUTOMATION_REFIT_20261109",
                         anchor_method="median_same_hour_days_7_14_21_28" if h>1 else "none",
                         anchor_days=[7,14,21,28] if h>1 else [],vintage_verified=False),path,compress=3)
        records[str(h)]={"file":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
                         "family":family,"trained_through":CUTOFF,
                         "train_start":str(frame.target_date.min()),"training_days":int(frame.target_date.nunique()),
                         "training_hours":int(len(frame)),"anchor_weight":anchor,
                         "anchor_method":"median_same_hour_days_7_14_21_28" if h>1 else "none",
                         "anchor_days":[7,14,21,28] if h>1 else [],
                         "vintage_verified":False,"research_metrics_applied":False}
    manifest={"created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"source_db_sha256":hashlib.sha256(dbpath.read_bytes()).hexdigest(),
              "source_commit":"fb6fc173d682b5cce2657b65c0b97c11e26ebd32","data_cutoff":"2026-10-26T23:00:00+09:00",
              "model_origin":"new refit; not the serialized fold model files","models":records,
              "warning":"The published 275-day CV metrics do not apply to these refit weights. Historical first_seen/vintage is unverified."}
    (OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(records,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
