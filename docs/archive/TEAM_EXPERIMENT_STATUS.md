# LG Aimers 제구 예측 — 팀 실험 현황판

최종 갱신: **2026-08-31 KST**

이 문서는 팀원이 지금까지 완료한 실험, 실패한 exact variant, 아직 하지 않은 실험을 한 번에
확인하기 위한 공유용 정본이다. 새 실험을 맡기 전에 이 파일에서 중복 여부와 현재 owner를 확인한다.

> 상태 갱신: E115는 2026-08-30 16:53 PDT에 완료됐다. 이전 상태 문서의 RUNNING/미측정 표기는
> 이 문서의 완료 결과로 대체한다.

## 1. 현재 최고

| 항목 | 값 |
|---|---|
| 제출물 | `candidate_v18g030.zip` |
| Public LB | **1092.808353586** |
| SHA256 | `ED1B729B9B83EB1F0D344B475939353149D5ADC83009A40FF116DF6E5A168557` |
| 실행시간 | 4초 |
| 구성 | target5 + team categorical + CS79 + affine + frozen hand/pressure residual |
| 행 독립 QA | batch/reverse/permutation/duplicate/split/singleton 모두 max diff `0` |

현재 파일은 수정하거나 이름을 바꾸지 않는다. 새 후보는 별도 이름으로 만들고 이 SHA를 parent로 기록한다.

## 2. 증거등급

| 등급 | 뜻 |
|---|---|
| E0 | 아이디어·설계만 있음 |
| E1 | 데이터 감사 또는 복원 가능성 확인 |
| E2 | 단일시드·오라클·진단 |
| E3 | paired 다중시드 local 검증 |
| E4 | 과거 연도에서 선택을 잠근 뒤 다음 연도 확인 |
| E5 | 우리 팀 direct Public LB |

우선순위는 `E5 > E4 > E3 > E2 > E1 > E0`이다. direct LB 음수는 같은 축의 양수 local을 덮는다.

## 3. 실제 LB 계보

| 단계 | 핵심 변경 | Public LB | 직전 기준 대비 | 판정 |
|---|---|---:|---:|---|
| 운영진 RF | baseline | 549.51193 | — | 기준 |
| HGB65 | 상황·누적 피처 | 830.322760105 | +280.81 | 채택 |
| HGB/CB w=.3 | 혼합 | 864.3312059823 | +34.01 | 역사적 채택 |
| HGB/CB w=.5 | 혼합 | 880.5655581163 | +16.23 | 역사적 채택 |
| CatBoost 8seed | CB 단독 | 898.61863054 | +18.05 | 채택 |
| CatBoost 16seed | 시드 2배 | 897.6644306949 | −0.95 | 기각 |
| CS76 | 현 시즌 진행분 복원 | 990.957167531 | +92.34 | 채택 |
| CS79 | 구종 관련 3피처 | 993.6345481775 | +2.68 | 채택 |
| target5 | 5-class → P(success) | 1033.9866361712 | +40.35 | 채택 |
| lr.04+season weight | 두 변경 묶음 | 1006.3794404841 | −27.61 | 기각 |
| team categorical | 팀 ID categorical 처리 | 1018.1353 | 약 +24.5 vs CS79 | 채택축 |
| target5+team_cat | 검증축 결합 | 1059.0501189623 | +25.06 vs target5 | 채택 |
| IT300 | 추론 tree 500→300 | 1040.0366 | −19.01 | 기각 |
| shift −.012 | 중심 probe | 약 1065.26 | +6.21 | 진단 |
| shift −.0066 | 중심 정점 | 약 1076.81 | +17.76 | 채택 |
| scale 1.15 | scale probe | 약 1073.66 | −3.15 | 기각 |
| affine s=1.06 | 중심+scale | 약 1081.67 | +22.62 vs 1059 | 채택 |
| frozen hand/pressure residual γ=.30 | 보호형 residual | **1092.808353586** | 약 +11.14 | 현재 최고 |

## 4. 채택된 핵심 축

### A. 현 시즌 진행분 복원

행의 `asof_*` 누적값에서 이전 시즌 종료 시점의 train 고정 상수를 빼서 현재 시즌의 투수·타자
성공률과 표본량을 복원한다. direct LB **+92.3**으로 가장 큰 단일 개선이다.

### B. target5

성공/미들/리버스/미들∩리버스/빅미스의 5개 상호배타 class를 학습하고 제출에는 class 0의 확률만
쓴다. direct LB **+40.35**.

### C. team categorical

13개 팀 ID를 연속 숫자가 아니라 CatBoost categorical로 처리한다. target5와 결합해 direct
**+25.06**.

### D. affine 출력 보정

전역 중심 `−0.0066`, 중심점 `.49`, scale `1.06`을 각 행에 독립 적용한다. direct 누적
**+22.6**. 현재 정점 근처이므로 추가 global bracket은 종료한다.

### E. 보호형 hand/pressure residual

`(game_type,pitcher) → batter_hand → pressure×hand`의 강수축 과거 통계 correction을 γ=.30으로
적용한다. 2022/2023/2024 시간 검증이 모두 양수였고 direct **+11.14**.

## 5. 완료·기각한 측정 범위

### 모델·앙상블

- CatBoost 16seed: direct −0.95.
- HGB를 현 CatBoost와 혼합: 2025 endpoint 열세가 커 최적 이득 약 +0.21.
- XGBoost/LightGBM 단순 partner: 독립 이득 약 +0~1.
- OVA: 혼합 기대 +4 이하.
- depthwise/lossguide: local −82.9/−88.1.
- EBM: `d=-412.7, K=315.2`, 최적 가중 0.
- ExtraTrees 완성모델: `d≈-332, K≈446`, 최적 가중 0.
- ResMLP: `d=-389, K=335`, 최적 가중 0; 시드 분산이 CatBoost의 6.6배.
- `has_time=True`: paired local −16.5/−23.0, 평균 −19.7. LB에는 제출하지 않음.
- IT300: local 양수였지만 direct LB −19.01. 현재 챔피언 기반 배포·혼합·교차 후보로 재사용하지 않음.

### 타깃·head

- M4: paired −11.8.
- C6: seed42 −16.7.
- MC7: 두 시드 평균 −22.8. 9-class 후속도 중단.
- reverse/breaking current-season CS feature 확장: cs_rev −19.4, cs_rev_brk −12.6.
- conditional binary success head: raw −19.79, equal-mean shape −10.16.
- exact reverse-only binary head는 미실행.

### categorical·상호작용

- 팀×팀/팀×역할 categorical: paired −20.2.
- count×hand + base/out: paired −10.1.
- forced one-hot: paired −6.0.
- `max_ctr_complexity=1`: paired −4.3.
- 선수 ID high-card categorical: 약 −160~−174.
- 단순 pitcher×count/inning/baseout/stint child: 시간 검증 gate 실패.
- batter×pitcher_hand child: 시간 검증 gate 실패.

### 신인·콜드스타트·매치업

- rookie expert: 신인행 `d=-51.75, K=42.32`, 최적 가중 0.
- `f_rk_ratedev` 팀 harness: seed42 −24.2, seed7 +16.6, 평균 −3.8로 불안정.
- HGB65 no-pitcher rookie router, 2024 신인행: d=-39.84, K=43.63, 최적이득 +0.08.
- 별도 whole-fold HGB ID ablation, 2024: no-pitcher −12.4, no-both −6.7.
- batter ID 제거: 선택시드 양수 뒤 독립시드 전부 반전.
- 저차원 pitcher-batter FM: 전체 +1.65지만 bootstrap CI가 0 포함, 미등장 pair 음수.

### residual·최근 상태

- 성공한 hand/pressure residual을 성분별로 분리: 평균 −2.43.
- 추가 hierarchy child 7종: leave-one-year-out 생존자 0.
- team match/rookie/middle/batter early 등 5개 신호의 보호형 residual: 전부 weight 0.
- 시간순 ExtraTrees residual adapter: discovery에서 gamma 0.
- prev1/3/5 분모는 81.9~84.2% 정확도로 복원했지만 workload residual은 gamma 0.

### TrackMan·물리

- train 행 키 존재 79.7%, 유일 조인 54.9%, 투수 ID 일치 100%까지 데이터 감사 완료.
- 직접 물리 피처: LB 약 −7.81.
- CS 위 물리: local 순증 +0.2.
- release consistency, distillation 재검증, arrival-region 모델: 모두 gate 실패.

### 보정

- 기존 Platt/isotonic: 연도 전이 약 −150.9.
- 보호형 residual γ 재탐색: 현 .30 위 이론 상방 최대 약 +0.32라 종료.
- 2024 정답을 본 segment 상한도 game_type×p50 4그룹에서 global 초과 +7.61,
  12개 count 전부에서 +10.59. +100 카드가 아니므로 제출 실험을 하지 않음.

## 6. 방금 완료한 구조 후보

| ID | 질문 | owner | 상태 | 제출 생성 |
|---|---|---|---|---|
| E115 | 독립 LGB/HGB residual + source-season Team EB + pitcher×24문맥 rank-6 SVD와 현 챔피언 구조의 2-seed historical OOF proxy 간 K | Codex | **OFFLINE SCREEN PASS — endpoint build/QA 1회 자격** | 없음 |

E115는 더 낮은 단독점수를 채택하려는 실험이 아니라 현 모델과 다른 함수공간의 오차가 충분히
직교하는지를 측정한 것이다. 후보 자체 fold metric과 target row order를 exact 재현했다. 비교 기준은
배포된 8시드 ZIP이 아니라 동일 구조의 2-seed historical OOF proxy다.

| 검증연도 | 단독 차이 `d` | 다양성 `K` | 최적 가중 `w*` | 이론 혼합이득 |
|---:|---:|---:|---:|---:|
| 2022 | −628.683 | 663.393 | 0.0262 | +0.454 |
| 2023 | +1805.519 | 1395.174 | 1.0000 | +1805.519 |
| 2024 | −56.920 | 175.413 | 0.3378 | **+20.011** |

사전 gate인 exact metric 재현, 세 폴드 중 2개 이상 계산상 이득 +20, 최소 K≥150을 통과했다.
하지만 2023의 +1805.519는 w*=1인 endpoint 우세이며 interior blend 증거가 아니다. interior blend는
2024 +20.011 한 번이고 2022는 +0.454에 그쳤다. 따라서 **채택이나 혼합 승인이 아니라 2025 endpoint를
빌드하고 runtime·행 독립성 QA를 할 1회 자격만 얻은 상태**다. 검증 환경도 배포 환경과 달라 compatibility는
아직 확인하지 않았다. ZIP이나 제출물은 생성하지 않았다.

## 7. 아직 하지 않았거나 미완료인 exact 질문

| 우선 | 질문 | 현 상태 | 예상 역할 |
|---:|---|---|---|
| 1a | E115의 2025 endpoint build + runtime/행독립 QA + endpoint 1회 | offline screen만 통과, bundle 미생성 | endpoint가 2025에서도 생존하는지 확인 |
| 1b | E115 endpoint와 챔피언의 50:50 다양성 probe | 1a 생존 시에만 | 실제 2025 K 확인 |
| 2 | rank-6 pitcher×24문맥 correction을 현 챔피언 residual 위에 직접 적용 | 미실행 | +0~10급 가능성 |
| 3 | 불확실 확률 구간만의 강수축 residual | 미실행 | 소액·직교 후보 |
| 4 | 투수 동적 AR(1) 잠재상태 | 미실행 | CS와 중복 여부 먼저 확인 |
| 5 | count×runner×prediction-bin 안정셀 correction | 미실행 | 마지막 소액 결합 |
| 6 | exact reverse-only binary head | 미실행 | 인접 반증 강해 낮은 우선순위 |
| 7 | full CAT5 no-pitcher | exp/86 완료 결과 없음 | 인접 no-ID 반증 강함 |
| 8 | `exp/99_microfeat.py` 자체 실행 | 결과 없음 | 같은 RK harness가 불안정해 낮은 우선순위 |
| 차단 | row_id의 날짜·경기·순번 구조 | 사용 승인 없음 | 서면 승인 전 read-only 감사만 허용; 학습·라우팅·보정·후처리·배포 사용 금지 |

## 8. 남은 제출 9회 운영 규칙

1. 오프라인 global-equivalent +20 미만 후보는 제출하지 않는다.
2. 단독점수 `d`와 다양성 `K`를 함께 계산하고 `d<-K`면 즉시 폐기한다.
3. endpoint가 살아야만 50:50 한 번으로 2025 `K`를 잰다.
4. 50:50 결과로 계산한 최적 혼합 기대가 +20 이상일 때만 최적비 제출을 쓴다.
5. 서로 다른 구조 후보가 둘 이상 살아야 3-way 결합을 한다.
6. 마지막 1회는 실행 오류·SHA 검증·최종 후보 확인용으로 남긴다.
7. IT300, HT, season weighting, hand delta, rookie, FM, NN, TrackMan, conditional head는
   여러 개를 합쳐 재등판시키지 않는다.

## 9. 팀원 작업 등록 양식

새 실험을 시작할 때 아래를 이 파일 끝에 먼저 추가한다.

```text
ID:
owner:
시작 시각/KST:
parent ZIP + SHA256:
정확한 질문:
변경점 하나:
discovery / confirmation years:
seeds / threads:
자동 중단 gate:
산출물 경로:
상태: IDEA / RUNNING / PASS / FAIL / DEPLOYED / LB-DEAD
```

완료 시에는 raw gain, equal-mean shape, F/R 기여, bootstrap, `d/K/w*`, 실제 LB 여부와
“정확히 무엇을 죽였는지”를 함께 기록한다. 비슷한 실험의 실패를 exact 미실행 변형의 실패로
확대해서 쓰지 않는다.
