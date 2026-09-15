# 1위권 공개 저장소(x2_qp_cheese, LB 1081.68) 해부 종합 — 5인 독자 결과 통합

**저장소 사실관계** (독자 전원 일치): 공개 커밋 3개, 문서 최종 갱신 2026-08-09(DL:L3). 최고점 V18(1081.68246, 27위)의 학습·추론 코드는 저장소에 **없음** — 보존된 것은 기각된 V25.5 노트북 1개(DL:L164 "전달된 산출물은 노트북 하나", 출력 셀 0개)와 초기 HGB/CatBoost 블렌드 스크립트 3종(LB 562/720/608)뿐. V18은 문서 서술 수준까지만 복원 가능.

약어: DL = docs/development_log_v8_v25_5.md, EX = docs/experiments.md, PG = docs/progress_2026-08-06.md, FE = docs/feature_engineering_experiments.md, EDA = docs/eda_findings.md, NB = notebooks/kaggle_kbo_v25_5_*.ipynb(셀), H = 우리 HANDOFF.md(줄).

---

## 1. LB 궤적 대조

| 그 팀 버전 | 변경점 | 그 팀 LB | 출처 | | 우리 제출 | 변경점 | 우리 LB | 출처 |
|---|---|---:|---|---|---|---|---:|---|
| E003 | HGB 단독 | 562.24 | EX:L8 | | 운영진 RF | 베이스라인 | 549.51 | H:44 |
| E012 | HGB 52.5% + 시즌가중 CatBoost 47.5% | 720.10 (151위) | EX:L15, PG:L15 | | hand delta 67피처 | 투수×타자손 편차 피처 | 808.09 | H:45 |
| E016 | +파생 85개(투수·상황·타자) blend — 로컬 660.14 | **608.36 (−112)** | FE:L25, PG:L16 | | submit4 | HGB 65피처 | 830.32 | H:46 |
| V2 | 2024 stratified 50:50 기준 초기 anchor | 1021.62 | DL:L28 | | submit8 | CatBoost 8시드 단독 | 898.62 | H:52 |
| V10.5 | champion 보호 + residual gate + fallback | 1023.91 | DL:L29,46 | | submit10 | CS 시즌진행분(76피처) | 990.96 (+92.3) | H:54 |
| V14 | shared backbone + 3-domain(R_CORE/R_ANCHOR/F) ExtraTrees residual | ≈1044.26 (+20.4) | DL:L30,57 | | submit12 | +구종 3피처(CS79) | 993.63 (+2.7) | H:55 |
| V15 | PITCHER_CONDITIONAL + EXACT_ASOF | 1053.67 (+9.41) | DL:L31,63 | | submit14 | 5클래스 MultiClass | 1033.99 (+40.4) | H:59 |
| V17 | R_CORE evidence/disagreement residual (내부 +8.86, 전 시드 양수) | 1049.998 (**−3.67**) | DL:L32,65 | | target5_teamcat | +팀 categorical(cat5) | 1059.05 (+25.1) | H:809 |
| **V18** | temporal-stable conditional residual (2022→23 +21.21, 2023→24 +24.74) | **1081.68246 (+28.01)** | DL:L33,66 | | submit20 | 사후 중심 −0.0066 | 1076.81 (+17.8) | H:1144 |
| V22.2 | risk-aware Meta-MoE γ=0.35 (내부 950.04, +4.55) | 미제출 | DL:L78 | | **submit22** | 스케일 1.06 | **1081.67 (+22.6)** | H:1146 |
| V19~V25.5 (12버전) | 선수상태·MoE·gating·비선형 잔차·계층 EB 대체·F 분리 | LB 갱신 0 | DL:L74-90 | | | | | |

**판정**: 두 팀은 서로 다른 경로로 **같은 고원(1081.67 vs 1081.68)** 에 도달했다. 그 팀은 V2→V18 사이 잔차 3단계(+20, +9, +28)로 올랐고, 08-09 이후 3주간 12버전이 전부 실패. 그들의 상승분 중 우리가 이미 흡수한 것: EXACT_ASOF(=우리 CS, +92.3로 훨씬 크게 실현), 팀 신호 global 모델(team cat +25). 우리에게만 있는 것: 사후 중심·스케일 보정(+22.6, 그들 LB 이력에 보정 단계 없음, DL:L26-34). 그들에게만 있는 것: **V18 조건부 잔차(+28)** 하나.

---

## 2. 구성요소 대조표

| 축 | 그 팀 (V18 기준, 코드 없는 부분은 V25.5·문서로 추정) | 우리 (submit22) | 차이 |
|---|---|---|---|
| 백본 | CatBoost 계열 shared backbone(알고리즘·피처 미공개, H:1187). V25.5 재현판은 R: Cat d7/lr.035/l2 28/it620 Logloss + LGB 수치전용, F: 별도 Cat + row-local state(NB 셀6) | CatBoost MultiClass 5클래스, P(성공)=class0, it500/d6/lr.08/l2=10, 8시드 | 우리는 타깃 분해(5클래스 LB +40)가 핵심; 그들은 binary Logloss. 그들 V25.5 R-only 792.65(2024) < 우리 cat5 2시드 873.9(전체 행, H:1189) |
| 타깃 | binary control_success | 5클래스{성공/미들/리버스/미들∩리버스/빅미스} | 그들에게 타깃 분해 시도 기록 없음 |
| 선수 ID | 초기: ID categorical CTR(constants.py:8-20); V25.5: raw ID 제거, 계층 EB로만 표현(NB 셀6 L10) — 792 < 945로 열세 | pitcher_id/batter_id 수치(데뷔순 경력신호), 선수 CTR −160.9 기각(exp/23) | 양쪽 모두 "ID CTR ✗, 계층 EB 대체 ✗" 결론 일치 |
| 피처군 | 47 원본 + 17단 계층 EB(rate/delta/rel/logn/trusted_delta/trend/backoff) + 현시즌 분해(k80/160, 중립 k20/50/100) + 문맥 범주(count_state, pressure_state, inning_phase, baseout_state, month_phase, team_match) + 최근폼 차분·구종 entropy(NB 셀3-5) | 79 = 원본 47 + 파생 18 + CS 11 + 구종 3 | 그들 문맥 범주 = 우리 조합 categorical 기각 축(H:807-878); 계층 EB delta는 우리 hand delta 계열; 구종 3(복원 라벨 기반)은 우리만 |
| as-of 처리 | cn=max(asof_n−Σ과거시즌 n,0), cs=clip(asof_n·rate−Σ과거 y,0,cn), rate_k=(cs+k·prior)/(cn+k), prior=계층 EB 투수율 k80/160(NB 셀4 L91-118). success 1개만 | attach_cs(): cs_n=n_now−N_end, cs_S=car·n_now−S_end, K=50, MIN_CS_N=5. 7 rate(succ/ball/strike/middle/fb + 타자 2) — 라벨 불필요 | 대수적 동일. 우리가 더 넓게(7 rate) 적용, LB +92.3. 그들의 중립 prior·k20 변형은 우리 K스캔(최적 50~100, H:326) 상 열등 |
| 잔차·계층 구조 | **챔피언 보호 + 조건부 잔차**: (game_type,pitcher) k220 → ×batter_hand k110 → ×pressure_state×batter_hand k220, trusted_delta=delta×n/(n+k), 다음시즌 전이 2구간 양수만 채택(DL:L66-68, NB 셀1 HIER) | 단일 모델, 잔차 없음. FM 잔차 +1.6(기각), 조건부 head −20(기각) | **우리 미측정 축 — exp/103 프로브 PRE-GATE PASS, exp/104 재학습 진행 중** |
| 도메인 분리 | V14 R/F 도메인 잔차 어댑터(+20); V13 완전 분리 MoE(오류상관 0.998, 실패); V25.5 R/F 분리 모델(F −8484 붕괴) | game_type을 cat 피처로 단일 모델. F/R Brier 비 0.998(exp/92) | 우리 설계가 그들 최종 결론(F 분리 금지)과 일치 |
| 블렌딩 | 초기 HGB+Cat 볼록 블렌드(720); V25.5 Cat+LGB 50%(−25.18 역효과) | 8시드 평균 단일 모델. HGB 혼합 +0.21·2025 전이 −68, LGB "9번째 시드" 기각 | 일치 |
| 보정 | 없음(클리핑만). V8 weight opt+calibration A→B −7.53 실패(DL:L42). 딥러닝 노트북에 logit-Platt 시도, LB 미확인 | 확률공간 아핀: 0.49+1.06·(p−0.49)−0.0066, LB 3점 곡선으로 결정, +22.6 | **우리 고유 이득** |
| 검증 | 초기 2024 홀드아웃 + early-stopping best_iter 재사용(낙관 편향, train_gbdt.py:118-128) → V17 사건 후 **strict next-season 2전이(2022→23, 2023→24) discovery 잠금 + 2024 confirmation-only**, paired gain, bootstrap p05, 최악 월/도메인, 블렌드는 minimax(DL:L153-159, NB 셀6 choose_*_blend) | 2019~23→2024 홀드아웃 + 4년 폴드 paired σ + LB 오라클. exp/103/104에서 discovery 잠금 채택(H:1182) | 프로토콜 동형. 우리에 없던 것: 투수 클러스터 bootstrap p05 표준화(exp/103에서 1회 사용), 블렌드 가중 minimax |

---

## 3. ★ 이식 후보 순위 (그 팀이 LB로 실증했고 우리에게 없는 것만)

### 1순위 — V18형 챔피언 보호 조건부 잔차 (진행 중인 exp/103→104 완주)
- **메커니즘**: ≤target 시즌 라벨만으로 (game_type,pitcher)→×batter_hand(k110)→×pressure_state×batter_hand(k220) 계층 EB 표를 만들고, 챔피언 예측에 γ·Σ(delta×rel)을 더함. pressure_state=full(3-2)/high(3볼 or 2스트)/normal. γ·k는 discovery 연도에서만 잠금.
- **그 팀 LB 이득**: V15→V18 **+28.01**(DL:L33), 유일하게 3주간 살아남은 축. 내부 2022→23 +21.21, 2023→24 +24.74(DL:L66).
- **우리 기대치**: exp/103 무학습 프로브(lab/103_v18_pregate.txt): γ=0.30을 2021~2023 최악연도 raw gain 최대화로 잠근 뒤 2024 HGB OOF +24.13, **현 it500 CAT5 2시드 위 raw +25.40 / shape +23.78 / F +2.67 / R +22.74, 투수 bootstrap 95% [+11.41, +40.17], P(>0)=1.0** → PRE-GATE PASS. 단 우리 백본이 그들보다 강하고(cat5 873.9 vs R 792.65) ID 수치·CS를 이미 가지므로 일부 흡수됐을 수 있음. 현실적 기대 **0~15, 낙관 +28**(H:1183). 그 팀 기준으로도 "옮겨지는 크기"(≥+20 반복 양수)에 해당하며 우리 노이즈 바닥 σ=15.3 위.
- **비용**: exp/104(정확한 79피처 CAT5, target 2022·2023, seeds 42/7, γ 탐색 없음)가 현재 2022 폴드 학습 중(lab/104_cat5_yearfold_result.txt, Pool train 728,387). 폴드당 ~1.5h → 잔여 ~4~6h. 통과 시 **배포는 재학습 불필요**: ≤2024 train 집계표(수천 행) + script.py 행 단위 pressure_state 계산·조인 → 기존 8시드 예측 위 보정. LB 슬롯 1.
- **규정 위험**: 없음. train 전용 상수표를 각 test 행에 독립 조인(CS 11과 동일 원리). test 내부 집계 없음.
- **죽은 축과의 관계**: 우리 hand delta(로컬 4년 +16.8 → LB −22.2, H:426)와 같은 [F-타겟] 정보. 차이 3가지: (a) 피처 투입이 아니라 챔피언 보호 잔차, (b) EB 수축 delta×reliability(얇은 셀 자동 억제), (c) γ를 2024 안 보고 잠금. 그 팀 V25.5가 같은 정보를 **피처**로 넣었을 때 792 < 945로 오히려 무너진 것(DL:L143)이 "잔차 형태만 살아남는다"는 방증. 그래도 실전 실패 전례가 있으므로 exp/104 게이트(두 시드 raw/shape 양수, ensemble raw≥+8, shape≥+5, R≥0, F≥−5, bootstrap p2.5>0; H:1205)를 두 연도 모두 통과할 때만 배포.
- **주의(독자 1·5 교차)**: 2022 폴드는 F 구레짐(.709) 포함이라 BSS가 부풀려짐(그 팀 V18 2022 내부 2500.55 vs 2023 729.56, NB 셀1 BSS_REFERENCE; DL:L142). 2022 폴드에서 F 기여가 과대·과소 어느 쪽으로든 튀어도 R 기여 중심으로 판정할 것. 또한 "2022·2023 게이트 통과 → 2024는 이미 봤음(exp/103)"이므로 2024는 확인 1회로만 취급.

### 2순위 — 검증·선택 프로토콜 보강 (비용 0, 슬롯 절약)
- **메커니즘**: (a) 구조·가중·γ 선택은 discovery 연도에서만, 확인 연도는 1회 읽기; (b) 절대 BSS 대신 동일 행 paired gain; (c) 투수 클러스터 bootstrap p05를 6줄 표준출력에 추가; (d) 블렌드/γ 격자 선택은 평균 최대가 아니라 **최악 연도 gain 최대화(minimax)**, 최선 단독 대비 min_gain ≥ −2 필터.
- **그 팀 실증**: V17 같은 시즌 pitcher split 내부 +8.86 전 시드 양수 → LB −3.67(DL:L65); V25.5 평균 최대화로 LGB 50% 선택 → 2024 R −25.18, minimax였으면 25% → 792.35(DL:L139-140).
- **우리**: exp/103이 이미 (a)(b)(d) 적용. 남은 것은 (c) 표준화와, 남은 슬롯 결정 전부를 이 틀로 강제하는 것. 우리 "로컬 +10 미만 축은 슬롯에 쓰지 말 것" 규칙과 정합(그 팀 로컬 +84 → LB −112 사례, FE:L25, PG:L16).

### 3순위 — train/serve skew 검사 (≤2023 번들을 script.py 경로로 2024 채점)
- **메커니즘**: 그 팀 scripts/evaluate_submission_zip.py 개념 — ≤2023 학습 번들을 실제 data/·output/ 구조로 2024 행에 실행해 BSS 계산, 학습 피처와 추론 피처(상수표·조인·NaN·pressure_state)가 같은지 점수로 확인.
- **우리**: 가짜서버 245,789행·행독립 프로브·5행 diff(exp/102)로 결정론은 확인했으나, script.py 경로 점수가 로컬 실험 점수(lab/89_cat5 873.9)와 일치하는지는 미확인. **1순위 잔차 표를 script.py에 넣을 때 skew 위험이 가장 큼**. 비용: 기존 ≤2023 2시드 모델(exp/104 산출물) 재사용 시 ~10분. LB 슬롯 1개를 코드 오류로 날리지 않기 위한 보험.

### 4순위 (조건부) — 계층 자식 확장: batter×pitcher_hand k200, pitcher×count k190, pitcher×inning k230, pitcher×baseout k280, pitcher_stint k180
- **메커니즘**: 1순위 구조에 자식 2~4개 추가, backoff=Σrel·delta/Σrel. 그 팀 공개 HIER 17단 중 exp/103 미사용분(NB 셀1 L63-81).
- **그 팀 LB 이득**: **단독 실증 없음** — V18 공개 설명은 hand·pressure 두 축만 확정(DL:L68); 17단 전체를 피처로 쓴 V25.5는 792.
- **우리**: 1순위가 2연도 게이트 통과한 뒤에만, 무학습 ~20분으로 γ 재잠금. 우리 보류표 "투수×카운트 조건부 delta = hand delta 패턴, 셀 더 얇음"(H:646), 타자 냉동표 전 구간 실패(H:946)와 충돌 → 타자 측은 k≥300, discovery 3년 모두 양수일 때만. 기대 소액.

### 후순위 (그 팀도 LB 미실증, 기대 ≤+2, 44h 예산 밖)
- logit 공간 Platt(a·logit p + b, Brier 최소화): 우리 확률 아핀과 거의 동치. 수 초 피팅 가능하나 슬롯 가치 낮음.
- 투수편 승리기대(pitcher_we), 구종 entropy/dominant_share, success−reverse 대비, 다중 k reliability, 불확실도 √(p(1−p)/(n+1)): 전부 트리 근사 가능한 단조/저차 변환, 그 팀에서도 묶음 +3.88(EX:L114)이 전부.
- F/R 별도 δ 사후 중심(우리 H:1157 +1.5~5): 그 팀 F 레짐 분석이 방향을 지지하나 그들 실증은 아님.

---

## 4. 그 팀이 실패한 것 중 우리가 하려던/보류한 것 (중복 방지)

| 우리 계획·보류 항목 | 그 팀 실측 | 출처 | 판정 |
|---|---|---|---|
| F specialist(보류 중) | F 전용 Cat 2023/24 −8775.62/−8484.02, 예측평균 .62/.61 vs 실제 .473/.459; 고정 블렌드 25%로 전체 137.18 | DL:L122-134, L141 | **보류 유지 확정**. 우리 F/R Brier 비 0.998 |
| 투수×카운트 조건부 delta(§4-3 보류) | pitcher_count k190은 V25.5 피처 묶음 안에서만 존재, R 792 < 945 | NB 셀1, DL:L143 | 피처로는 금지; 1순위 잔차의 자식으로만(4순위) |
| 조건부 head / 비선형 잔차 | V24 HGB·얕은 LGB 잔차 943 < 945.49 폐기; V23 row gating 동일값 REJECT | DL:L79-80 | 우리 −20과 일치, 재시도 금지 |
| 신인 전용/라우팅 expert | V13 3-domain MoE expert 오류상관 ≥0.998; V25.4 팀별 specialist 388.84 vs global 768.98 | DL:L56, L89 | 우리 w*=0과 일치 |
| 혼합 파트너(LGB/HGB) | V25.5 Cat+LGB 50% −25.18 | DL:L139 | 우리 "9번째 시드"·2025 전이 −68과 일치 |
| 시즌 가중 | Cat +21.66 로컬(2024 단일), LGB −12.10, XGB −40.30, **LB 단독 미확인** | EX:L107-112 | 우리 LB −28로 이미 사망. 그들 수치는 반증 아님 |
| Trackman 잔차/증류 | V10.4 champion 위 최적 alpha=0 | DL:L45 | 우리 ≈0과 일치 |
| NN(TabM 등) | V11.2 −64.37, V11.3 weight 0, V25.2 MLP 2023 −5105 | DL:L53-54, L87 | 일치 |
| 상황 파생 묶음(카운트·이닝·LI 22개) | 일괄 추가 −53.12 | FE:L23 | 우리 볼카운트 파생 사망과 일치 |
| 직접 pitcher×batter 표 | 평균 reliability 1.5~1.7%, k=700 강수축; E016 중요도 1위였으나 LB −112 | DL:L18, FE:L43-46, PG:L16 | 우리 FM +1.6·페어 표 기각과 일치 |
| 타자 결측 플래그(우리 미보유) | 결측행 성공률 +7.87%p(830건) 발견만, 실증 없음 | EDA:L63-67 | CatBoost NaN 자체 분기, 우리 결측플래그 축 사망 → 불필요 |
| **2022 폴드 해석** | 2022 내부 BSS 2500(F 구레짐 고예측성), discovery에 2022 섞으면 F 가중 오염 | NB 셀1, DL:L142 | **exp/104 2022 폴드 판정 시 F 기여 할인** |
| 로컬 점수 해석 일반 | 홀드아웃 early-stopping best_iter를 재학습에 재사용 → 로컬 낙관 편향 | train_gbdt.py:118-128, L264 | 우리 it500 고정·8시드는 해당 없음. 그 팀 초기 로컬 수치는 우리 로컬과 직접 비교 금지 |

---

## 5. 추론 코드의 규정 관점 특이점

- 저장소 내 제출 스크립트 3종(submission/script.py:17-22, blend_script.py:43-58, feature_blend_script.py:29-39): test.csv 로드 → row_id 제거 → 저장된 파이프라인/모델로 **행 단위 predict_proba** → clip[0,1] → CSV. 대치(median/most_frequent), Ordinal/OneHot(min_frequency=5), 범주 레벨(gbdt.py:30-36), rate prior(학습 열 평균, features.py:82-92), pressure_threshold(season<2024 li 75분위), 딥러닝 표준화 통계 모두 train에서 fit 후 model/에 저장. **test 내부 groupby·분포·순서·타깃인코딩·사후 분포 보정 없음 → 위반 소지 없음.**
- V25.5 노트북: test 추론·zip 생성 셀 자체가 없음(검증 전용). 현시즌 분해는 각 행 자신의 asof와 train 상수표만 사용(NB 셀4 L193-195 hist=season<target) — 우리 CS와 동일 원리. preflight가 game_type∈{R,F} 외 값에 RuntimeError(NB 셀2 L150-151) — 우리와 무관.
- V18 추론 코드는 부재하므로 규정 검토 불가. 문서상 잔차 표는 past-only 라벨 집계라 우리 1순위 배포 방식(≤2024 train 집계표 조인)과 동일하게 합법.
- 사후 보정: 그 팀은 클리핑만. 우리 중심·스케일 상수는 LB 3점 곡선(2025 test 전체 분포)으로 정했다는 점에서 "평가 데이터 전체를 보고 만든 사후 보정값" 해석 여지가 있으나, 상수 2개가 script.py에 고정돼 test 행 간 계산은 없음 — 이미 제출·확정된 사항이므로 여기서는 지적만 기록.

---

## 결론 (44h · 기계 1대 · 슬롯 5)

이 저장소에서 나오는 **새 카드는 1순위 하나**뿐이며 이미 exp/103→104로 진행 중이다. 나머지는 우리 죽은 축과 거의 1:1로 겹쳐 "재도전 금지 목록"을 독립 실측으로 확증해 줄 뿐이다. 두 팀이 같은 1081 고원에서 정체했다는 사실은, 1170 컷 격차를 이 저장소로 메울 수 없음을 뜻한다 — 1순위(기대 0~15)와 우리 자체 P1(f_rk_ratedev)을 합쳐도 +10~30 범위. 실행 순서 권고: exp/104 완주(4~6h, 다른 CatBoost 병행 금지) → 게이트 통과 시 3순위 skew 검사(10분) → 슬롯 1 배포. 실패 시 잔여 슬롯은 자체 P1 계열로.

참고 파일: c:/Users/gwonn/Desktop/open/lab/103_v18_pregate.txt, c:/Users/gwonn/Desktop/open/lab/104_cat5_yearfold_result.txt(UTF-16, 2022 폴드 진행 중), c:/Users/gwonn/Desktop/open/reference/x2_qp_cheese_LGAimers/docs/development_log_v8_v25_5.md
