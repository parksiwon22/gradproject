# 5번 발전량 — 2023~2025 팀 공유 v2

**팀원은 `generation_2023_2025_hourly.csv`를 받으세요.** 26,304개 시간 행이며 2023·2025년은 연간 공식 CSV, 2024년은 API 원본입니다. 원본 결측을 남긴 자료이며 학습용 대체값이나 72시간 지연을 아직 적용하지 않았습니다.

|연도|시간 행|LNG 결측|유연탄 결측|무연탄 결측|원자력 결측|출처|
|---|---:|---:|---:|---:|---:|---|
|2023|8,760|0|0|0|0|annual_2023|
|2024|8,784|24|24|768|768|api_2024|
|2025|8,760|0|0|0|0|annual_2025|

## 파일·열

- 통합: generation_2023_2025_hourly.csv. 연도별: generation_2023_hourly.csv / generation_2024_hourly.csv / generation_2025_hourly.csv. 두 종류를 중복 합산하지 마세요.
- timestamp: Asia/Seoul 시간 구간 시작. 00:00은00~01시 구간.
- lng_generation_mwh / bituminous_coal_generation_mwh / anthracite_coal_generation_mwh / nuclear_generation_mwh: LNG·유연탄·무연탄·원자력 MWh.
- coal_generation_mwh: 유연탄+무연탄. 한 성분이라도 빠지면 합계도 결측.
- 각 *_missing: 1은 결측, 0은 값 존재. 실제 발전량0과 구분합니다.
- source_kind: annual_2023 / api_2024 / annual_2025. 개별 출처 파일·SHA256은 manifest.json.
- missing_intervals.csv: 남은 핵심 발전원 누락 구간.

## 채택 근거·주의

2025년 필수4종 모두8,760시간입니다. API와 겹치는1월1일96개 값은 최대0.0005MWh 차이로 반올림 범위에서 일치합니다. 2023년도 필수4종 모두8,760시간이지만 API대조에서는 반올림 이상 차이가 있으므로 연간 파일을 한 해 단일 출처로 사용했습니다. 서로 동일한 정산 발표본이라고 주장하지 않습니다. 차이 통계는 manifest.json에 있습니다.

2024년 연간 후보에는 신재생 연료원만 있어 LNG·석탄·원자력 보충에 사용할 수 없습니다. API의2024-02-29 전체24시간과8월 원자력·무연탄744시간 누락을 그대로 유지했습니다. 석탄 합계는 총768시간 결측입니다. **2023~2025년 완전 무결측 자료가 아닙니다.**

정의: 전력시장 참여 발전기의 송전단 전력거래량입니다. 설비용량 MW·입찰량·전국 총발전량과 다릅니다. 지역 필드가 없어 육지 단독이라고 표시하지 않습니다. [연간 공식 자료](https://www.data.go.kr/data/15127395/fileData.do) / [API](https://www.data.go.kr/data/15106744/openapi.do).

## 학습 연결

이 CSV는 동일 시각 원자료입니다. 모든72시간 목표에 공통 마감 `forecast_origin−72h`를 적용하고, 해당 마감에 종료한 구간까지만 사용해야 합니다. target_timestamp별72시간 차감 방식은 금지합니다. 결측 대체는Train으로만 적합하고 결측 표시를 보존하세요. 연간 자료의 과거 공개시각·수정 이력을 복원하지 않았으므로72시간 지연은 여전히 연구 가정입니다.

이 버전은 기존 공통 v3 및 LSTM/SARIMAX 학습 결과를 덮어쓰지 않습니다. 재학습 시 네 모델 모두 새 공통 버전과 같은 평가 행을 사용하세요. 현재는 원자료 재추출·검사·공유까지만 수행했습니다.

재현: `1120gradpj/.venv/bin/python 1120gradpj/scripts/export_generation_share_v2.py`.
