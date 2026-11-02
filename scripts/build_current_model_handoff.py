"""Build current-model standalone ZIP after collecting public mainland SMP."""
from __future__ import annotations
import hashlib
import json
import shutil
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from v11.features import FEATURES as D1_FEATURES, build as build_d1
from v17.features import FEATURES as DX_FEATURES, build as build_dx
from v11.store import ROOT, connect, latest

OUT=ROOT/"dist"/"SMP_Current_Model_Complete"
CUTOFF="2026-10-26"

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    dbpath=ROOT/"artifacts"/"v11"/"prices.sqlite"
    if not dbpath.exists():
        raise RuntimeError("Missing price database; collect official SMP first")
    db=connect()
    try: prices=latest(db,"land")
    finally: db.close()
    if prices.empty or prices.timestamp.min()>pd.Timestamp("2023-04-01",tz="Asia/Seoul"):
        raise RuntimeError("Official history insufficient")
    limit=pd.Timestamp(CUTOFF,tz="Asia/Seoul")+pd.Timedelta(hours=23)
    prices=prices[prices.timestamp<=limit].drop_duplicates("timestamp").sort_values("timestamp")
    (OUT/"models").mkdir(parents=True,exist_ok=True)
    (OUT/"artifacts"/"v11").mkdir(parents=True,exist_ok=True)
    details={}
    for h in (1,2,3):
        frame=(build_d1(prices,days_ahead=1) if h==1 else build_dx(prices,days_ahead=h))
        features=D1_FEATURES if h==1 else DX_FEATURES
        if frame.target_date.nunique()<400:raise RuntimeError(f"D+{h} insufficient training days")
        X=frame[features].to_numpy(float)
        y=(frame.actual_smp-frame.daily).to_numpy(float)
        if h==1:
            model=ExtraTreesRegressor(n_estimators=150,min_samples_leaf=10,max_features=.8,n_jobs=2,random_state=42)
            weight=0.0; family="V11-style ExtraTrees residual"
        else:
            model=HistGradientBoostingRegressor(max_iter=250,max_leaf_nodes=31,learning_rate=.05,
                 early_stopping=False,l2_regularization=2.,random_state=42)
            weight=.05;family="HGB residual and weekly anchor"
        model.fit(X,y)
        path=OUT/"models"/f"d{h}_refit.joblib"
        joblib.dump(dict(model=model,horizon_days=h,features=features,anchor_weight=weight,
            trained_through=CUTOFF,family=family,model_origin="NEW_20261102_HANDOFF_REFIT",
            vintage_verified=False),path,compress=3)
        details[str(h)]=dict(checkpoint=path.name,sha256=digest(path),
            training_days=int(frame.target_date.nunique()),training_start=frame.target_date.min(),
            training_cutoff=CUTOFF,family=family,anchor_weight=weight)
    with sqlite3.connect(dbpath) as src,sqlite3.connect(OUT/"artifacts"/"v11"/"prices.sqlite") as dst:src.backup(dst)
    snapshot=OUT/"artifacts"/"v11"/"prices.sqlite"
    with sqlite3.connect(snapshot) as con: count=con.execute("SELECT COUNT(*) FROM prices WHERE market='land'").fetchone()[0]
    if count<25000:raise RuntimeError(f"Price DB incomplete: {count}")
    for src in ["v11/__init__.py","v11/features.py","v11/store.py","v11/collect.py",
                "v17/__init__.py","v17/features.py"]:
        dst=OUT/src;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/src,dst)
    for src in ["docs/smp_improvement_ml_lgbm/final_recommended_models.csv",
                "docs/smp_improvement_ml_lgbm/all_fold_paired_summary.csv",
                "docs/smp_improvement_ml_lgbm/final_hourly_mae.csv",
                "docs/smp_improvement_ml_lgbm/FINAL_RESEARCH_REPORT.md"]:
        p=ROOT/src
        if p.exists():
            dst=OUT/src;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst)
    shutil.copy2(ROOT/"scripts"/"current_model_handoff_app.py",OUT/"app.py")
    shutil.copy2(ROOT/"scripts"/"current_model_handoff_predict.py",OUT/"forecast.py")
    (OUT/"requirements.txt").write_text("numpy==2.2.6\npandas==2.3.3\nscikit-learn==1.7.2\njoblib==1.5.3\nscipy>=1.14,<1.18\nstreamlit==1.66.1\naltair==6.3.0\n",encoding="utf-8")
    (OUT/"RUN_WINDOWS.bat").write_text("@echo off\ncd /d \"%~dp0\"\npy -3.12 -m venv .venv\nif errorlevel 1 py -3 -m venv .venv\nif errorlevel 1 exit /b 1\n.venv\\Scripts\\python.exe -m pip install -r requirements.txt\nif errorlevel 1 exit /b 1\n.venv\\Scripts\\python.exe -m streamlit run app.py\npause\n",encoding="utf-8")
    (OUT/"README_FIRST.md").write_text(
        "# 현재 SMP 연구모델 전체 실행본\n\n"
        "## 포함 자료\n"
        "D+1 ExtraTrees와 D+2/D+3 HGB-anchor로 새로 재학습한 joblib 3개,\n"
        "KPX EPSIS 육지 시간별 가격 SQLite, 독립적인 Streamlit 예측 및 EV 충전 웹, 학습/데이터 소스와 연구 검증표.\n\n"
        "## Windows 실행\n"
        "Python 3.12를 설치하고 RUN_WINDOWS.bat을 더블클릭합니다. http://localhost:8501 에 접속하세요.\n"
        "SMP 데이터 갱신: .venv\\Scripts\\python.exe -m v11.collect --market land\n\n"
        "## 중요한 구분\n"
        "ZIP의 모델 3개는 2026-10-26까지 데이터로 새로 학습한 공유용 체크포인트입니다.\n"
        "기존 275일 시간순 연구에 사용한 과거 fold 모델 파일과 다릅니다.\n"
        "따라서 기존 D+1 9.248 / D+2 10.175 / D+3 10.429 MAE는 이 모델 파일의 검증 결과가 아닙니다.\n"
        "과거 공개시점 first_seen이 검증되지 않으면 사전예측 성능으로 주장하면 안 됩니다.\n"
        "모델 학습 기준일 이전의 날짜에 이 체크포인트를 사용하지 마세요.\n"
        "예측 시각 23:00에 입력 SMP가 없으면 추론을 중단합니다.\n"
        "SMP 기반 충전 비용은 연구용 가상 비용이며 실제 전기차 충전 요금이 아닙니다.\n\n"
        "## 원본 연구\nhttps://github.com/parksiwon22/gradproject/tree/research/smp-performance-20261027\n",
        encoding="utf-8")
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),
      price_source="KPX EPSIS mainland historical hourly SMP",price_rows=count,
      price_db_sha256=digest(snapshot),models=details,
      note="NEW REFIT CHECKPOINTS - historical 275 day fold metrics DO NOT apply to these weights",
      source_commit="0fa98350509195e61a5286760cdc50b79831c928")
    (OUT/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    archive=ROOT/"dist"/"SMP_Current_Model_Complete.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():z.write(p,str(Path("SMP_Current_Model_Complete")/p.relative_to(OUT)))
    print("BUILD_OK",archive,archive.stat().st_size,count,flush=True)

if __name__=="__main__":main()
