# V5 재현 기록

2026-10-13, Windows CPU에서 ZIP의 원본 데이터·원본 알고리즘으로 전체 학습을 실행했습니다.
환경 버전, 입력 SHA-256, ZIP 목록, 원본과의 수치 차이는 reproduction.json에 있습니다.

- 학습 대상일: 2023년 1~8월, 이력/lag 생성 후 235일·5,640행
- 검증 대상일: 2023년 9월, 30일·720행
- 최종 테스트 대상일: 2023년 10~12월, 92일·2,208행
- 입력 이력 168시간, 특징 63개, seed 42
- 전일 23시 cutoff에서 다음 날 00~23시 24개 horizon 직접 예측
- 검증 MAE로 모델과 혼합 비율 선택, 테스트로 재조정하지 않음
- 일별 프로파일 비율 0.69, 최종 프로파일 비율 0.31, 급등 확률 임계값 0.45
- 선택 모델: validation_selected_profile_hgb_blend
- 최종 MAE: **11.093786148024114원/kWh**
- 최종 R²: **0.6429194848696833**

ZIP 결과와 MAE·R² 및 2,208개 최종 선택 예측이 일치합니다.
경로 설정과 CLI 출력 문구 외 모델 학습 알고리즘은 변경하지 않았습니다.
원본은 33.84초, 이번 학습은 13.91초로 실행 시간은 환경에 따라 다릅니다.
일별 기준선 MAE는 13.5758, 충전 regret는 2·3·4·5·6·8시간 구간의 평균을 사용합니다.

검증 명령:

```bash
python -m pip install -r requirements-model.txt
python -m pip check
python -m model.v5 --full
python -m scripts.predict --date 2023-10-01
python -m services.v5_web_adapter
python -m unittest discover -s tests -v
python -m scripts.smoke_server
python -m scripts.reproduction_report --zip ../smp_project_v5_package.zip
```

저장 모델을 새 프로세스에서 불러와 24행 예측을 생성했습니다.
7개 단위·통합 테스트가 통과했습니다. 시간 분할 경계, 충전 구간의 끊김과 동률,
24시간 완전성, 전체 재현 지표, 저장 모델과 학습 출력 일치, 웹 변환,
Streamlit 화면 실행과 SOC/시간 입력을 검증합니다.
서버 HTTP health 확인은 별도 smoke_server로 수행합니다.
첫 샌드박스 서버 확인은 30초 시간 초과로 실패했으며 정상 권한 재실행에서
HTTP health 응답이 통과했습니다. 서버는 확인 후 종료했습니다.

첫 샌드박스 실행에서는 7개 테스트 통과 뒤 Streamlit 임시 폴더 정리 시
PermissionError가 발생했습니다. 정상 권한에서 동일 테스트를 재실행하여 통과했습니다.
pip 설치는 모든 패키지가 이미 충족됐고 종료 코드는 0이었지만 샌드박스 임시 폴더
정리 경고가 있었습니다. pip check는 충돌이 없었습니다.
joblib의 NumPy 2.5 shape 설정 deprecation, Windows 물리 코어 조회 fallback,
AppTest의 bare ScriptRunContext 경고가 관찰됐으며 숨기지 않았습니다.

웹은 역사적 하루 데모입니다. 새 데이터 확보·미래 기간 일반화·API 실서비스·V6 성능은
검증하지 않았습니다. Linux/macOS 및 독립 가상환경 신규 설치도 아직 검증하지 않았습니다.
