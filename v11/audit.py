"""Record source hashes, original feature portability, and optional demand data."""
import hashlib
import json
import numpy as np
import pandas as pd
from v11.store import ROOT,HOME,connect,latest,now,export

def main():
    db=connect(); official=latest(db,'land'); export(db); db.close()
    original=pd.read_csv(ROOT/'data/aligned_hourly_all_features_2023.csv',parse_dates=['timestamp'])
    original.timestamp=original.timestamp.dt.tz_localize('Asia/Seoul')
    compare=original[['timestamp','smp']].merge(official[['timestamp','smp']],on='timestamp',suffixes=('_original','_official'))
    delta=compare.smp_original-compare.smp_official
    result=dict(at=now(),compared_hours=len(compare),exact_matches=int(delta.abs().lt(1e-8).sum()),MAE=float(delta.abs().mean()),max_abs=float(delta.abs().max()),note='Comparison does not establish historical publication timing or redistribution rights')
    path=HOME/'demand_official.csv'
    if path.exists():
        for encoding in ['utf-8-sig','cp949']:
            try: demand=pd.read_csv(path,encoding=encoding); break
            except UnicodeDecodeError: continue
        demand=demand[demand.iloc[:,0].notna()]
        demand=demand.rename(columns={demand.columns[0]:'date'})
        long=demand.melt(id_vars='date',var_name='ending_hour',value_name='national_demand_mwh')
        long['hour']=long.ending_hour.str.extract(r'(\d+)').astype(int)-1
        long['timestamp']=pd.to_datetime(long.date).dt.tz_localize('Asia/Seoul')+pd.to_timedelta(long.hour,unit='h')
        long['available_at']=None; long['source']='data.go.kr/15065266'; long['hour_mapping_status']='ending-hour assumption; not independently certified'; long['usable_in_live_model']=False
        assert len(long)==8760 and not long.timestamp.duplicated().any() and np.isfinite(long.national_demand_mwh).all()
        long.to_csv(HOME/'national_demand_2025.csv',index=False)
        joined=official.merge(long[['timestamp','national_demand_mwh']],on='timestamp',how='left')
        joined['complete_original_63_features']=False
        joined['live_features']='SMP+calendar; optional annual demand excluded because publication vintage missing'
        joined.to_csv(HOME/'integrated_dataset.csv',index=False)
        result['optional_demand']=dict(hours=len(long),unit='MWh per one-hour interval',geography='national mainland+Jeju, not mainland-only',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source='https://www.data.go.kr/data/15065266/fileData.do',available_at='unknown annual final data',used_for_training=False)
    features=json.loads((ROOT/'artifacts/models/day_ahead_feature_metadata.json').read_text(encoding='utf-8'))['exact_feature_order']
    lines=['# V5 63피처의 V11 이식 점검','', '원 V5 소스와 모델은 수정하지 않았다. 생성식은 model/v5.py의 preprocess_hourly/build_one_day_rows를 기준으로 분류했다. 공식 자료의 event time과 실제 발표/수신시각은 별개다.','', '| V5 피처 | 의미/생성 | 2024~2026 이식 상태 |','|---|---|---|']
    for f in features:
        if any(x in f for x in ['holiday']): meaning='공휴일/대체/임시휴일 달력 플래그'; status='정확한 당시 선공개 휴일 달력 미확보; V11 제외'
        elif any(x in f for x in ['gas_','fuel_trade']): meaning='월별/일별 가스·연료 단가의 origin 값'; status='원자료 정의/발표 vintage 미확보; V11 제외'
        elif any(x in f for x in ['weather_','solar_radiation','solar_generation','wind_','renewable','net_load','reserve','demand']): meaning='origin 실제 관측/파생값; demand lag1/24, renewable 합, net_load=수요-신재생'; status='같은 지역·정의의 다년 관측 및 available_at 미확보; V11 제외. 전국2025 수요는 별도 보존'
        elif any(x in f for x in ['smp','profile','cutoff','horizon']): meaning='horizon, 과거 동일시간24/168h SMP 또는 origin lag/rolling/change'; status='공식 육지 SMP로 재구성 가능; V11은 독립21피처; 역사 발표 vintage 미확인'
        else: meaning='origin/target 달력의 시간·요일·월 sin/cos와 주말'; status='재구성 가능; 목표 달력 외 미래 관측은 사용 안 함'
        lines.append('| '+f+' | '+meaning+' | '+status+' |')
    lines+=['','전국 수요 MWh는 1시간 평균 MW와 수치가 같아도 원 V5 national_demand_mw와 동일 원천/정의라고 인증하지 않는다. 현재 모든 연도의 원63피처 완비 날짜는 0일로 표시한다. 기상청 최신예보 키와 과거 forecast-vintage archive가 없어 실제기상을 미래예보로 대체하지 않는다.']
    (ROOT/'docs/V11_FEATURE_AUDIT.md').write_text('\n'.join(lines),encoding='utf-8')
    (HOME/'source_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
