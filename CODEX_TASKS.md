# Codex: C 최소형 개발 순서 및 완료조건

수정 2026-10-14. RESEARCH_DIRECTION.md를 먼저 읽고 아래5개명세 및 CODEX_PROMPT.md를 따른다. 이 커밋은 연구 추천 및 개발 지시이며 V6구현완료가 아니다. 원 V5와main,기존outputs는 보존한다.

## 1. 시작가능범위

브랜치 work/v5-audit-v6-prep-20261013에는원본 V5/portable코드/설정/테스트/문서가있다. 기존7개unittest와full재실행검증은PROJECT_AUDIT.md를따른다. 이번수정은문서뿐이다. private학습자료와원joblib은공개 Git에없으며별도Codex환경이ChatGPT첨부에접근할수있다고가정하지않는다. 현재접근가능ZIP가있으면import도구로직접연결한다. 인증키없이합성fixture와snapshot replay모듈부터개발할수있다.

## 2. 순서

| 순서 | 개발내용 | 완료조건 |
| --- | --- | --- |
| 0 | 현재branch/baseSHA 확인,원본 hash 및 회귀 테스트 | V5/main/outputs보존;원 성능11.09와재실행11.42구분 |
| 1 | price schema와snapshot replay | market/product/interval/version/first_seen/assumption분리 |
| 2 | 단방향 charging LP | SOC/효율/부분시간/출발제약,불가능요청명시 |
| 3 | v6 09시feature builder와HGB | lag24 미래정보차단,lead15~38,동일시점profilebaseline |
| 4 | mixed-price planning와event reoptimization | 오늘known/내일unknown구분,실행한행동고정,SOC상태연결 |
| 5 | 가상V2G MILP | 동일slot충방전금지,수출gate,terminalSOC/throughput열화 |
| 6 | 정책평가/민감도 | P0/P1/P2/P3/oracle,공표전후접속,같은조건/가정표시 |
| 7 | app_v6 UI | 추천/출처/SOC/가상경제성,AppTest와렌더 |
| 8 | 승인검토자료 | 실제변경/실패/성능/한계/데이터권한/다음선택보고 |

새파일은v6/, app_v6.py, tests/test_v6_*에한정하는구성을권장한다. 모델family확장/실시간API/기상/실제알림/사용자DB/차량군/실차제어는기본 순서 뒤에둔다. 코드복잡도를낮추기위해 scipy.optimize.linprog/milp의지원버전을확인후환경을고정하고해결기status를반드시검사한다.

## 3. 고정해야할정의

origin 제안D-1 09시,next-day24h와optimization horizon의차이를config에서관리한다. 최적화는현재연결~출발,최대48시간이다. 오전예측을내일24h에대해서만평가하더라도오늘충전과내일미루기의선택을replay해야조기정보가치를평가할수있다. 발표후연결군은예측증분가치0이가능한대조군이다.

가격처리는정각17/18/23시만으로officialavailable=true가아니다. 실측수신이없으면rule_assumed/schedule_assumed로표시한다. 타깃이origin전에알려진경우예측을안쓰고known으로처리한다. 알수없는가격으로추천이불가능하면권한있는snapshot또는명시적가정으로fallback한다.

V2G 실제모드export=false. simulated_mode에서만방전변수를허용한다. 구매요금과가상판매요금을따로입력하고SMP를실소매요금/실판매단가로치환하지않는다. 모든정책에같은끝SOC,이동에너지,효율/열화와connected조건을적용한다. 원 자료가 없어도합성 최적화 검증은진행할수있지만실시장 수익성은주장하지않는다.

## 4. 필수테스트

09시origin뒤전일profile특성차단,target_price불법사용차단,knownfuture가격자격분리,미래revision불변,정렬/자정/구간길이/coverage검사,연결없음/출력0/비정상SOC/eff/부족시간,부분시간에너지보존,충방전모드배타,exportgate,높은열화무방전,판매0에서도음수가격구매대응과가짜동시cycle방지,기존행동고정과가격 갱신,SOC갱신/조기 출발/연결 중단,모든공식정보가초기부터있는상황에서forecast의존0,동일조건oracle비용하한,solverinfeasible/timeout와secret마스킹을검증한다.

열화비0과동일구매/판매가격에서양의손실사이클로공짜수익을만들면실패다. 다른말단SOC로정책수익을비교하면실패다. 고정소매요금에서시간이동만으로절감액이생기면실패다. 실제자료없는원화결과에가상 표시가없으면실패다. 미래확정값으로과거계획을바꾸면실패다.

## 5. 완료보고

코드/테스트/자료접근/구현기능을실제로수행한범위만보고한다. modelselection은validation,최종test는동결후한번.23시V5 archive에09시후보가못미쳐도서로다른정보집합이므로연구실패로단정하지않는다. P3가P2만개선하고P1을못개선하면AI증분가치없음/제한됨으로기록한다.

main직접수정/병합,공개배포,실서비스V2G제어/실제전송은사용자승인전금지. 최종C선정/수익 가정/모델승격/공개데이터/운영키등록은검토자료로남기되사소한구현판단은합리적인기본값으로진행한다.
