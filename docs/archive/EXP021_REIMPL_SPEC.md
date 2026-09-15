# EXP-021 엔드포인트 재구현 구현 사양서 (통합본)

> **목적**: 참조 저장소의 코드를 한 줄도 옮기지 않고, 수식·표·의사코드만으로 EXP-021 추론 엔드포인트를 제3자가 처음부터 재작성할 수 있게 한다.
> **검증 기준**: 실제 2025 `data/test.csv` 5행에 대해 최종 예측
> `[0.3810933855090245, 0.3729165651008432, 0.4441587362008253, 0.5022321253158382, 0.5162418838966417]` (평균 `0.44332853920463455`) 를 재현하면 완료.
> **출처 인용(내용 비인용)**: `reference/mk-isos_lg-aimers-9-pitch-control/experiments/` 의 `exp021_submission_inference.py`(873줄), `build_exp021_final_candidates.py`, `train_exp018_constrained_multiscale.py`, `train_exp019_r_full_residual.py`, `train_exp019_histgb_residual.py`, `train_exp019_team_eb_ensemble.py`, `train_exp020_pitcher_count_eb_atop_team.py`, `train_exp020_low_rank_pitcher_context_eb.py`, `temporal_residual_features.py`, `temporal_multirate_features.py`.
> **산출물(숫자) 재사용**: `candidate_exp021_w0356557_v002_src/model/` 의 JSON/텍스트 11종. 본 문서에 적힌 모든 수치는 이 파일들을 직접 열어 실측한 값이거나(§2, §8), 실측 산출물로 계산한 골든 트레이스(§9)다.

---

## 0. 개요와 최종 수식

### 0.1 무엇을 만드는가

EXP-021 엔드포인트는 **확률 하나를 뱉는 완결된 추론기**다. 구조는 "시간 베이스 위에 4개의 가산 보정을 순차로 얹는 5단 잔차 스택"이며, 각 단계마다 `[0,1]` 클립이 들어간다. 학습된 모든 숫자는 이미 파일로 동결되어 있으므로 **재구현 대상은 오직 추론 코드**다(재학습은 §3.3/§4.6/§5.3/§6.6에 기록만 남긴다).

```
data/test.csv
  │  (row_id 제거)
  ├─ [F1] 정적 파생 21열
  ├─ [F2] temporal 45열      ← history_state.json
  ├─ [F3] multirate 69열     ← multirate_state.json
  │
  ├─ b_tmp = temporal_base_global_30            ← 확률 베이스
  ├─ (+) e_group   ← group_effects.json         → clip → b_grp
  ├─ (+) 0.75·r_lgb / 1.00·r_hgb (R행만, 각각 clip) → 50:50 평균 → b_bone
  ├─ (+) e_team    ← team_effects.json          → clip → b_team
  └─ (+) e_lowrank ← lowrank_effects.json       → clip → p
                                                       ↓
                                       output/submission.csv
```

**주의**: 우리 상위 `script.py`(champion 0.6434428305247574 / exp021 0.35655716947524263 가중합 래퍼)는 이 문서의 대상이 아니다. 본 문서는 exp021 엔드포인트 단독을 기술한다.

### 0.2 최종 수식 (단일 식)

행 $i$, 클립 $C(x)=\min(\max(x,0),1)$, 정규시즌 지시함수 $\mathbb 1_R = [\,\texttt{game\_type}_i = \texttt{"R"}\,]$.

$$
\begin{aligned}
b^{\text{tmp}}_i &= \mathrm{f32}\!\big(0.7\,\hat s^{P,30}_i + 0.3\,\hat s^{B,30}_i\big) \qquad(\texttt{temporal\_base\_global\_30})\\[3pt]
e^{\text{group}}_i &= 0.7\,g^{\text{base}}[c_i,h^P_i,h^B_i] \;+\; 0.3\,g^{\text{rev}}[c_i,h^P_i,h^B_i,\beta_i]\\[3pt]
b^{\text{grp}}_i &= C\!\big(b^{\text{tmp}}_i + e^{\text{group}}_i\big)\\[6pt]
b^{\text{bone}}_i &= 0.5\,C\!\big(b^{\text{grp}}_i + 0.75\,r^{\text{lgb}}_i\,\mathbb 1_R\big)\;+\;0.5\,C\!\big(b^{\text{grp}}_i + 1.00\,r^{\text{hgb}}_i\,\mathbb 1_R\big)\\[6pt]
e^{\text{team}}_i &= \tfrac12\cdot\tfrac14\!\!\sum_{y\in\{21,22,23,24\}}\!\! t^{P}_y[\text{ptm}_i,h^P_i,h^B_i]\;+\;\tfrac12\cdot\tfrac14\!\!\sum_{y}\!\! t^{B}_y[\text{btm}_i,h^P_i,h^B_i]\\[3pt]
b^{\text{team}}_i &= C\!\big(b^{\text{bone}}_i + e^{\text{team}}_i\big)\\[6pt]
e^{\text{low}}_i &= \tfrac14\sum_{y\in\{21,22,23,24\}} V_y\big[\text{pit}_i,\;\kappa(c_i,h^B_i)\big]\\[3pt]
\boxed{\,p_i} &= C\!\big(b^{\text{team}}_i + e^{\text{low}}_i\big)
\end{aligned}
$$

기호: $c_i$=`count_index`, $h^P$=`pitcher_hand`, $h^B$=`batter_hand`, $\beta$=`reverse_rate_bin`, ptm/btm=팀 id, pit=`pitcher_id`, $\kappa$=컨텍스트 위치(0..23).

**클립은 정확히 5지점**: (a) 그룹 가산 후, (b) LGB 가지 내부, (c) HGB 가지 내부, (d) 팀 가산 후, (e) 저랭크 가산 후. **50:50 평균 직후에는 클립하지 않는다**(두 피가산항이 이미 [0,1]).

**미등록 키는 전부 0 기여**(그룹/팀/저랭크 공통). 팀·저랭크의 시즌 평균 분모는 **항상 4**이며, 매칭된 시즌 수로 바꾸면 안 된다.

**비-R 행**은 두 GBDT 잔차가 모두 무시되어 $b^{\text{bone}}=b^{\text{grp}}$ 가 된다.

### 0.3 행 독립성 (실격 방지)

모든 계산은 (a) 그 행 자신의 컬럼값, (b) 학습기가 동결해 파일로 저장한 사전 통계 — 둘만 쓴다. test 내부의 집계·분포·순서·빈도는 어디에도 없다. 원 저장소에서 배치를 훑는 곳이 두 군데 있으나 둘 다 **행별로 재작성해도 결과가 동일**하며, 재작성에서는 반드시 행별 방식을 쓴다:

1. **시간 가드**(§3.5): `(season <= 2024).any()` 전역 검사 → 예측값에 관여하지 않는 입력 검증. 행별 검사로 대체.
2. **원-핫 인코딩**: `get_dummies(dummy_na=True)` 후 스키마 순서로 재색인(누락 열 0 채움) → **고정 어휘 12열 방식**(§3.6)으로 대체하면 결과 동일.

### 0.4 실행 환경

추론: python 3.11 / numpy 1.26.4 / pandas 2.0.3 / lightgbm 4.6.0. **scikit-learn 은 추론에 불필요**(HGB 는 JSON→순수 numpy 순회). 필요 라이브러리: `json`, `os`/`pathlib`, `numpy`, `pandas`, `lightgbm`.

| 경로 상수 | 값 |
|---|---|
| 모델 디렉터리 | `./model` |
| 테스트 입력 | `./data/test.csv` |
| 제출 양식 | `./data/sample_submission.csv` |
| 출력 | `./output/submission.csv` (디렉터리 자동 생성) |
| 행 ID / 타깃 컬럼명 | `row_id` / `control_success` |

---

## 1. 입력·컬럼 규약

### 1.1 CSV 읽기와 검증

두 CSV 모두 **`encoding="utf-8-sig"`** 로 읽는다(BOM 제거). 이 엔드포인트는 그 외 컬럼명 정규화(strip/replace)를 하지 않는다. 방어적으로 `.str.strip()` 을 넣더라도 **컬럼 이름·순서는 절대 바꾸지 말 것**(검증 #1이 순서까지 본다).

읽은 직후 아래 5개를 순서대로 검사하고, 하나라도 위배되면 `ValueError` 로 중단한다.

| # | 위배 조건 |
|---|---|
| 1 | `sample.columns != ["row_id", "control_success"]` (순서 포함) |
| 2 | `len(test) != len(sample)` |
| 3 | `test.row_id` 또는 `sample.row_id` 에 결측 존재 |
| 4 | `test.row_id` 또는 `sample.row_id` 에 중복 존재 |
| 5 | 두 `row_id` 집합이 다름 |

출력 직전 검사(§7.3)도 별도로 있다.

### 1.2 test.csv 48열의 처리 구분

| 구분 | 컬럼 | 처리 |
|---|---|---|
| 식별자 | `row_id` | 피처 프레임 생성 전 제거. 최종 매핑에만 사용 |
| 가드 전용 | `season` | §3.5 시간 가드에만 사용. **모델 입력 아님** |
| 게이트 전용 | `game_type` | 문자열 `"R"` 비교(§6.5). **모델 입력 아님** |
| 조회 키 전용 | `pitcher_id`, `batter_id`, `pitcher_team_id`, `batter_team_id` | 상태/효과표 조인 키. **모델 입력 아님** |
| 원-핫 대상 | `top_bottom`(문자열 `T`/`B`), `base_state`(3글자 문자열) | §3.6 |
| 그대로 통과 (20) | `game_month, game_dayofweek, inning, balls_before, strikes_before, outs_before, run_top_before, run_bot_before, run_total_before, score_diff_home, score_diff_pitcher_team, runner_on_1b, runner_on_2b, runner_on_3b, num_runners_on, home_win_expectancy, away_win_expectancy, li, pitcher_hand, batter_hand` | 결측 대치 없음, float32 캐스팅만 |
| 그대로 통과 (asof 19) | `asof_pitcher_n, asof_pitcher_success_rate, asof_pitcher_reverse_rate, asof_pitcher_middle_rate, asof_pitcher_ball_rate, asof_pitcher_strike_rate, asof_pitcher_prev{1,3,5}_game_success_rate, asof_pitcher_prev{1,3,5}_game_middle_rate, asof_batter_n, asof_batter_success_rate, asof_batter_middle_rate, asof_pitcher_pitchmix_n, asof_pitcher_fastball_rate, asof_pitcher_breaking_rate, asof_pitcher_offspeed_rate` | **결측 대치 없음. NaN 그대로 GBDT 행렬에 들어간다** |

### 1.3 결측 처리 대원칙 (가장 틀리기 쉬운 규칙)

> **원본 `asof_*` 열은 절대 채우지 않는다.** NaN 을 그대로 LightGBM/HGB 에 넘겨 네이티브 결측 경로를 타게 한다.
> 대치는 오직 §3.3(temporal) / §3.4(multirate) **파생 피처 계산 내부**에서만 일어나고, 그 결과는 새 열에만 들어가며 원본 열을 덮어쓰지 않는다.
> 그래서 파생 피처(temporal 45 + multirate 69)에는 NaN 이 하나도 없고, NaN 은 원본 asof 통과 열과 그로부터 나온 4개 차분 피처에만 존재한다(§9.4 NaN 개수 표가 이 사실의 회귀 테스트다).

### 1.4 관측 사실

- `pitcher_hand`, `batter_hand` 는 **정수 코드**(값 1, 2). 문자열이 아니다. `same_hand` 는 정수 비교.
- `home_win_expectancy` / `away_win_expectancy` 는 **퍼센트 단위(0~100)** (예 29.7 / 70.3, 합 100).
- `count_index = balls×4 + strikes` 의 유효값은 12종: `{0,1,2,4,5,6,8,9,10,12,13,14}` (3,7,11,15 발생 불가).
- `base_state` 는 3글자(1/2/3루 점유, 빈 자리 `_`).

---

## 2. 산출물 파일 스키마 표

`model/` 의 13개 파일 중 이 엔드포인트가 읽는 것은 **7개**다.

| 파일 | 사용 여부 | 역할 |
|---|---|---|
| `metadata.json` | ○ | 후보 문자열 확인(`candidate = "strict_lowrank_s300_r6"`) |
| `feature_schemas.json` | ○ | GBDT 입력 열 이름·순서 |
| `history_state.json` | ○ | temporal 45열 복원 |
| `multirate_state.json` | ○ | multirate 69열 복원 |
| `group_effects.json` | ○ | $e^{\text{group}}$ |
| `rfull_lightgbm.txt` | ○ | $r^{\text{lgb}}$ |
| `histgradientboosting.json` | ○ | $r^{\text{hgb}}$ |
| `team_effects.json` | ○ | $e^{\text{team}}$ |
| `lowrank_effects.json` | ○ | $e^{\text{low}}$ |
| `pitcher_count_effects.json` | **×** | aggressive 후보 전용. strict 경로에서 미사용 |
| `blend_metadata.json` | × | 상위 블렌드 래퍼용 |
| `model.pkl`, `v18_tables.npz`, `champion_inference.py` | × | champion 엔드포인트용 |

### 2.1 `metadata.json` (주요 키)

| 키 | 값 |
|---|---|
| `experiment` | `"EXP-021"` |
| `candidate` | **`"strict_lowrank_s300_r6"`** |
| `training_rows` | 1,475,092 |
| `training_seasons` | [2019, 2020, 2021, 2022, 2023, 2024] |
| `history_through_season` | 2024 |
| `source_effect_seasons` | [2021, 2022, 2023, 2024] |
| `source_combination` | `"equal average; missing mapping contributes zero"` |
| `probability_calibration` | `"identity"` (사후 보정 없음) |
| `full_fit_backbone.base` | `"EXP018 temporal base + last3 group"` |
| `full_fit_backbone.R_lightgbm` | `"rfull_l63_m1000_i300 weight 0.75"` |
| `full_fit_backbone.R_histgradientboosting` | `"hist_l15_d4_m3000_i160 weight 1.0"` |
| `full_fit_backbone.ensemble` | `"fixed 50:50"` |
| `full_fit_backbone.F` | `"EXP018 temporal base + last3 group"` |
| 검증 집계 (2022–24) | Brier 0.24470458400867237 / 0.2477311440988816 / 0.24763380341629648, skill 1789.5967932082258 / 907.5416355312283 / 869.9211702032806, mean_skill 1189.0198663142448 |
| 학습 환경 기록 | python 3.12.6, numpy 2.5.1, pandas 3.0.5, lightgbm 4.6.0, scikit-learn 1.9.0 |

### 2.2 `feature_schemas.json`

| 키 | 타입 | 길이 |
|---|---|---|
| `lightgbm` | 순서 있는 문자열 리스트 | **186** |
| `histgradientboosting` | 순서 있는 문자열 리스트 | **84** |

리스트 인덱스 = 모델 입력 행렬의 열 인덱스. 전체 목록은 §6.2 / §6.3.

### 2.3 `history_state.json`

| 키 | 타입 | 실측 |
|---|---|---|
| `through_season` | int | **2024** |
| `league_rate` | float | **0.4861049201797189** (float32: 0.48610490560531616) |
| `pitcher` | list[dict] | **792** 원소, 필드 `pitcher_id`, `prior_n`, `prior_successes` |
| `batter` | list[dict] | **830** 원소, 필드 `batter_id`, `prior_n`, `prior_successes` |

- `prior_n` = 2024 종료 시점 그 선수의 **career 누적 투구 수**(정수값 float). pitcher 범위 2.0…15450.0, batter 범위 1.0…13928.0. **양쪽 모두 합계 = 1,475,092.0** (= 학습 행 수).
- `prior_successes` 합계 = **772,603.0** (양쪽 동일). 772603/1475092 = 0.5237659752747625.
- id 범위: pitcher 20700…24633, batter 20889…24632. 학습에 없던 id 는 테이블에 **존재하지 않는다**.

샘플: pitcher 22548→(248, 133), 22703→(2933, 1655), 23415→(2982, 1871), 21813→(3085, 1560), 24198→(87, 53) / batter 23649→(184, 97), 23635→(810, 413), 23143→(1565, 812), 22026→(11099, 5782), 23646→(12645, 6575).

### 2.4 `multirate_state.json`

| 키 | 내용 |
|---|---|
| `through_season` | **2024** |
| `global_rates` | 10개 스칼라 (§8.3 표) |
| `tables` | 3개 그룹 |

| 그룹 | id 열 | 행 수 | 값 필드 |
|---|---|---|---|
| `pitcher_control` | `pitcher_id` | 792 | `prior_n`, `prior_success_count`, `prior_reverse_count`, `prior_middle_count`, `prior_ball_count`, `prior_strike_count` |
| `batter_control` | `batter_id` | 830 | `prior_n`, `prior_success_count`, `prior_middle_count` |
| `pitcher_pitchmix` | `pitcher_id` | 792 | `prior_n`, `prior_fastball_count`, `prior_breaking_count`, `prior_offspeed_count` |

세 그룹 모두 `prior_n` 합 = 1,475,092.0. 실측 항등식:
- `pitcher_control.prior_n` ≡ `history_state.pitcher.prior_n` (792행 전부 일치)
- `pitcher_control.prior_success_count` ≡ `history_state.pitcher.prior_successes` (792행 전부 일치)
- `pitcher_pitchmix`: `fastball + breaking + offspeed = prior_n − 1` 이 792행 전부에서 성립(이유는 §3.4.5).

샘플: pitcher_control 22548→n248/succ133/rev52/mid27/ball104/str111, 22703→n2933/1655/504/362/1042/1303 / batter_control 23649→n184/succ97/mid24, 23635→n810/413/113 / pitchmix 22548→n248/fb174/br31/os42, 22703→n2933/1672/690/570.

### 2.5 `group_effects.json`

| 키 | 레코드 필드 | 행수 | effect 범위 |
|---|---|---|---|
| `base` | `count_index`, `pitcher_hand`, `batter_hand`, `effect` | **48** (=12×2×2, 완전) | −0.07769743358179122 ~ +0.02689694839120108 |
| `reverse` | 위 3개 + `reverse_rate_bin`, `effect` | **688** (희소) | −0.06990556727719334 ~ +0.04723161639589252 |

`reverse_rate_bin` 실측 존재값: `{-1, 0,1,…,17, 20}`. 수축 상수(100/300)는 이 파일에 저장되어 있지 않다(§4.5 참조).

### 2.6 `rfull_lightgbm.txt`

표준 LightGBM 텍스트 모델(`lgb.Booster(model_file=…)` 로 그대로 로드).

| 항목 | 실측 |
|---|---|
| 헤더 | `tree` / `version=v4` / `num_class=1` / `num_tree_per_iteration=1` |
| `max_feature_idx` | 185 → `num_feature() = 186` |
| objective | `regression` (L2) |
| 트리 수 | **300** |
| feature_names | `Column_0`…`Column_185` (일반명) → **이름 결속 정보 없음. `feature_schemas.json["lightgbm"]` 순서가 유일한 결속** |
| 파라미터 블록 | 파일 끝 `parameters:` … `end of parameters` 에 원 하이퍼파라미터 전부 보존 (§8.5에 전재) |
| `boost_from_average` | 1 — 초기 평균이 첫 트리 bias 로 흡수(tree 0 루트 `internal_value` ≈ 5.06e-07). **모델이 자기완결적이므로 별도 초기값을 더하면 안 된다** |
| tree 0 루트 `internal_count` | 1,117,198 (bagging 0.85 적용 후) |

### 2.7 `histgradientboosting.json`

| 키 | 타입 | 값 |
|---|---|---|
| `format` | str | **`"numeric_hgb_v1"`** — 다르면 즉시 `ValueError` |
| `n_features` | int | **84** |
| `baseline` | float | **4.8898374092703725e-09** |
| `trees` | list | 길이 **160** |

`trees[k]` 는 노드 수 $N_k$ 길이의 7개 배열:

| 필드 | dtype | 의미 |
|---|---|---|
| `value` | float | 노드값. **리프에서만 사용** |
| `feature_idx` | int | 분기 피처 열 인덱스(0..83). 리프에서는 0, 무의미 |
| `num_threshold` | float | 수치 임계값. 리프에서는 0.0 |
| `missing_go_to_left` | 0/1 | 결측 시 왼쪽 여부 |
| `left`, `right` | int | 자식 노드 인덱스. 리프에서는 0 |
| `is_leaf` | 0/1 | 리프 여부 |

루트는 항상 인덱스 0. 실측 지문: 총 노드 4,416개, 트리별 노드 수 ∈ {19,21,23,25,27,29}, 리프 10~15개, 최대 깊이 **4**, 분기 피처 인덱스 0~82(서로 다른 70종), 리프값 범위 [−0.0015397169887587743, +0.001466065840579519], **비리프 노드 중 `value ≠ 0` 인 것이 1,968개**, 범주형 분기 없음.

### 2.8 `team_effects.json`

```
{ "source_seasons": [2021,2022,2023,2024],
  "pitcher_team": [ {season, records:[{pitcher_team_id, pitcher_hand, batter_hand, effect}]} ×4 ],
  "batter_team":  [ {season, records:[{batter_team_id,  pitcher_hand, batter_hand, effect}]} ×4 ] }
```

레코드 수 **실측(재확인)**: `pitcher_team` = 2021:44, 2022:44, 2023:44, **2024:48** / `batter_team` = 2021:44, 2022:44, 2023:44, **2024:48**.
team_id 도메인 12~21, 23 (2024에만 **25** 추가). hand 도메인 {1,2}, 4조합 모두 존재.
effect 범위: pitcher_team 2021 [−0.021537, +0.040656], 2022 [−0.022367, +0.029454], 2023 [−0.021414, +0.042449], 2024 [−0.016635, +0.032567] / batter_team 2021 [−0.021783, +0.038402], 2022 [−0.028387, +0.027560], 2023 [−0.019095, +0.029860], 2024 [−0.018912, +0.025572].
**수축 상수 1000.0 은 이 파일에 저장되어 있지 않다**(코드 상수).

### 2.9 `lowrank_effects.json`

```
{ "smoothing": 300.0, "rank": 6,
  "contexts": [ {position, count_index, batter_hand} ×24 ],
  "sources":  [ {season, pitcher_ids:[int]×P, values:[[float]×24]×P} ×4 ] }
```

`source_weights` 키는 **없다** → 균등 평균 경로.

| season | P | values | 값 범위 | mean(abs) | σ₁..σ₆ |
|---|---|---|---|---|---|
| 2021 | 386 | (386,24) | −0.06023632788577382 ~ +0.05147612447246095 | 0.0031946 | 0.30482789, 0.26619443, 0.19833226, 0.18963916, 0.18678364, 0.17325649 |
| 2022 | 390 | (390,24) | −0.0607095034037196 ~ +0.05817342467305488 | 0.0031088 | 0.28919349, 0.27070458, 0.20923849, 0.19904887, 0.18040001, 0.17201940 |
| 2023 | 382 | (382,24) | −0.0461553128742133 ~ +0.0603917702456991 | 0.0031217 | 0.27894465, 0.24584236, 0.21157353, 0.18891093, 0.17557634, 0.17310804 |
| 2024 | 391 | (391,24) | −0.04878602051837508 ~ +0.05115330511210226 | 0.0029846 | 0.29965717, 0.23236197, 0.20480507, 0.18215033, 0.17953970, 0.17142046 |

σ₇ 이후는 전부 0 → 저장 행렬 랭크가 정확히 6임이 확인된다. `pitcher_ids` 는 네 시즌 모두 오름차순 정렬(2021 처음 5개: 21170, 21372, 21402, 21414, 21432).

### 2.10 `pitcher_count_effects.json` (미사용, 기록용)

```
{ "smoothing": 600.0, "sources": [ {season, records:[{pitcher_id, count_index, batter_hand, effect}]} ×4 ] }
```
레코드 수 실측: 2021 **8540**, 2022 **8450**, 2023 **8323**, 2024 **8492**. 2021 기준 고유 투수 386명, effect 범위 [−0.035521, +0.030258], count_index 도메인 {0,1,2,4,5,6,8,9,10,12,13,14}, batter_hand {1,2} (386×24=9264 중 8540 관측, 포화도 0.922).

---

## 3. 피처·상태 복원

### 3.1 전체 순서

```
F ← test 에서 row_id 제거 (47열)
[F1] 정적 파생 21열 추가                  (§3.2)
[가드 G1] season 검사 (history)           (§3.5)
[F2] temporal 45열 추가                   (§3.3)
[가드 G2] season 검사 (multirate)         (§3.5)
[F3] multirate 69열 추가                  (§3.4)
b_tmp ← F["temporal_base_global_30"] (float32 → float64 승격)
```

### 3.2 정적 파생 21열 (F1)

| # | 열 | 정의 | dtype |
|---|---|---|---|
| 1 | `count_index` | `balls_before×4 + strikes_before` | int8 |
| 2 | `count_out_index` | `count_index×3 + outs_before` | int8 |
| 3 | `is_full_count` | `[balls=3 ∧ strikes=2]` | int8 |
| 4 | `has_two_strikes` | `[strikes=2]` | int8 |
| 5 | `has_three_balls` | `[balls=3]` | int8 |
| 6 | `count_advantage` | `strikes_before − balls_before` | int8 |
| 7 | `runner_in_scoring_position` | `[runner_on_2b=1 ∨ runner_on_3b=1]` | int8 |
| 8 | `bases_loaded` | `[1b=1 ∧ 2b=1 ∧ 3b=1]` | int8 |
| 9 | `same_hand` | `[pitcher_hand = batter_hand]` | int8 |
| 10 | `late_inning` | `[inning ≥ 7]` | int8 |
| 11 | `close_game` | `[ |score_diff_pitcher_team| ≤ 1 ]` | int8 |
| 12 | `log_li` | `ln(1 + max(li, 0))` | float32 |
| 13 | `score_pressure` | `|score_diff_pitcher_team| × log_li` (**float32로 반올림된 log_li 사용**) | float32 |
| 14 | `win_expectancy_gap` | `home_win_expectancy − away_win_expectancy` | float32 |
| 15 | `pitcher_batter_success_gap` | `asof_pitcher_success_rate − asof_batter_success_rate` | float32, **NaN 전파** |
| 16 | `pitcher_recent_success_delta_1_5` | `prev1_success − prev5_success` | float32, **NaN 전파** |
| 17 | `pitcher_recent_success_delta_3_5` | `prev3_success − prev5_success` | float32, **NaN 전파** |
| 18 | `pitcher_recent_middle_delta_1_5` | `prev1_middle − prev5_middle` | float32, **NaN 전파** |
| 19 | `log_pitcher_n` | `ln(1 + max(asof_pitcher_n, 0))` | float32 |
| 20 | `log_batter_n` | `ln(1 + max(asof_batter_n, 0))` | float32 |
| 21 | `log_pitchmix_n` | `ln(1 + max(asof_pitcher_pitchmix_n, 0))` | float32 |

- 15~18은 **대치 금지**. 한쪽이라도 NaN 이면 NaN 이 모델 행렬까지 간다.
- 13번은 계수가 정수라 float64 계산 후 float32 캐스팅과 float32 산술이 동일하다. 단 **입력 `log_li` 는 이미 float32 로 반올림된 값**이어야 한다.
- int8 오버플로 위험 없음(최대 44).

### 3.3 temporal 45열 (F2) — 시즌 진행분 복원

**핵심 메커니즘**: `career 누적 asof 값 − 2024 종료 스냅샷 = 2025 시즌 진행분`. 행별 뺄셈이므로 행 독립.

기호: $L$ = `league_rate` = 0.4861049201797189. 엔티티 $e \in \{$pitcher, batter$\}$, 행의 id 로 조회.

| 양 | 정의 | 미등록 id |
|---|---|---|
| $P_n$ | 테이블 `prior_n` | 0.0 |
| $P_s$ | 테이블 `prior_successes` | 0.0 |
| `prior_exists` | id 존재 여부 1/0 | 0 |
| $C_n$ | `asof_{e}_n` (**fillna 하지 않음**, float 캐스팅만) | — |
| $C_s$ | $\mathrm{rint}\big(C_n \times \text{fillna}(\texttt{asof\_\{e\}\_success\_rate},\,0.0)\big)$ | — |

복원:
$$n^{se}_{\text{raw}} = C_n - P_n,\qquad S^{se}_{\text{raw}} = C_s - P_s$$
가드 G3, G4(§3.5) 통과 후:
$$n^{se} = \max(n^{se}_{\text{raw}}, 0),\qquad S^{se} = \mathrm{clip}(S^{se}_{\text{raw}},\,0,\,n^{se})$$

비율 — **분모 > 0 일 때만 계산하고, 아니면 fallback 값을 그대로 둔다**(`0/0` 을 만들고 errstate 로 덮지 말 것):
$$\rho^{pr} = \begin{cases}P_s/P_n & P_n>0\\ L & \text{else}\end{cases}\qquad
\rho^{se} = \begin{cases}S^{se}/n^{se} & n^{se}>0\\ L & \text{else}\end{cases}$$
$$\pi = \frac{P_s + 200.0\,L}{P_n + 200.0}\quad(\text{선수 사전확률; }P_n=0\text{ 이면 자동으로 }L)$$

생성 열 (접두사 `temporal_{e}_`, 엔티티당 21열, 전부 float32):

| 접미 | 값 |
|---|---|
| `prior_exists` | 0/1 (int8) |
| `prior_n` | $P_n$ |
| `log_prior_n` | $\ln(1+P_n)$ |
| `prior_rate` | $\rho^{pr}$ |
| `prior_rate_shrunk_200` | $\pi$ |
| `season_n` | $n^{se}$ |
| `log_season_n` | $\ln(1+n^{se})$ |
| `season_rate` | $\rho^{se}$ |
| `season_minus_prior_rate` | $\rho^{se}-\rho^{pr}$ |
| `season_global_{k}` | $(S^{se}+kL)/(n^{se}+k)$ |
| `season_player_{k}` | $(S^{se}+k\pi)/(n^{se}+k)$ |
| `reliability_{k}` | $n^{se}/(n^{se}+k)$ |

수축 강도 $k \in \{10.0, 30.0, 100.0, 300.0\}$, 접미사는 `int(k)` = `10`/`30`/`100`/`300`.

추가 3열:

| 열 | 값 |
|---|---|
| `temporal_prior_league_rate` | 상수 $L$ (float32 → 0.48610490560531616) |
| `temporal_base_global_30` | $0.7\cdot$`temporal_pitcher_season_global_30` $+\;0.3\cdot$`temporal_batter_season_global_30` |
| `temporal_base_player_30` | $0.7\cdot$`temporal_pitcher_season_player_30` $+\;0.3\cdot$`temporal_batter_season_player_30` |

합 = 1 + 21×2 + 2 = **45**.

#### 3.3.1 ★ 정밀도 규칙 (재현에 결정적)

1. 21열은 **float64 계산 후 float32 저장**.
2. `temporal_base_global_30` / `_player_30` 두 열만은 **이미 float32 로 저장된 두 열에 대한 float32 산술**로 계산한다. 실측 차이:

| 계산 경로 | row 1 값 |
|---|---|
| float32 산술 (**정답**) | `0.3964788615703583` |
| float64 산술 후 float32 캐스팅 (오답) | `0.3964788317680359` |

차이 3.0e−8 이 그대로 최종 확률에 전파되어 16자리 재현이 깨진다. 구현: `np.float32(0.7)*A32 + np.float32(0.3)*B32`.
3. `rint` 는 **round-half-to-even(은행가 반올림)**. `floor(x+0.5)` 로 대체하면 경계에서 틀린다.
4. 다운스트림에서 float32 → float64 확대는 무손실.

#### 3.3.2 (재학습 시에만) 상태 생성 규칙

시즌 오름차순(2019→2024)으로 진행하며 시즌 $s$ 처리 후:
1. 그 시즌 행을 id 로 그룹핑, 각 그룹에서 **`asof_{e}_n` 최대인 행**(동점 시 첫 행)을 선택.
2. `n_last` = 그 행의 `asof_{e}_n`, `rate_last` = 그 행의 success_rate(결측 0.0).
3. `prior_n ← n_last + 1.0` (마지막 투구 자신을 포함시키는 보정)
4. `prior_successes ← rint(n_last × rate_last) + control_success(그 행)`
5. 같은 id 는 **나중 시즌 값으로 덮어씀(keep last)**.

`league_rate` = 마지막 처리 시즌 행들의 `control_success` 평균(= 2024 한 시즌 값). 복원 오차는 선수당 최대 ±0.5.

### 3.4 multirate 69열 (F3)

#### 3.4.1 그룹 정의 (처리 순서 고정)

| # | group | id 열 | 표본수 열 | metric → rate 열 (순서 고정) |
|---|---|---|---|---|
| 1 | `pitcher_control` | `pitcher_id` | `asof_pitcher_n` | success→`asof_pitcher_success_rate`, reverse→`asof_pitcher_reverse_rate`, middle→`asof_pitcher_middle_rate`, ball→`asof_pitcher_ball_rate`, strike→`asof_pitcher_strike_rate` |
| 2 | `batter_control` | `batter_id` | `asof_batter_n` | success→`asof_batter_success_rate`, middle→`asof_batter_middle_rate` |
| 3 | `pitcher_pitchmix` | `pitcher_id` | `asof_pitcher_pitchmix_n` | fastball→`asof_pitcher_fastball_rate`, breaking→`asof_pitcher_breaking_rate`, offspeed→`asof_pitcher_offspeed_rate` |

#### 3.4.2 그룹 공통 (그룹당 3열, 접두사 `multirate_{group}_`)

$$C_n = \text{fillna}(\text{표본수 열},\,0.0)\quad\textbf{(★ temporal 과 달리 fillna 한다)}$$
$$n^{se}_{\text{raw}} = C_n - P_n,\qquad n^{se} = \max(n^{se}_{\text{raw}}, 0)$$
가드 G5(§3.5) 통과 필요.

| 열 | 값 |
|---|---|
| `season_n` | $n^{se}$ |
| `log_season_n` | $\ln(1+n^{se})$ |
| `reliability_30` | $n^{se}/(n^{se}+30.0)$ |

#### 3.4.3 metric 단위 (metric당 6열, 접두사 `multirate_{group}_{metric}_`)

$G$ = `global_rates[f"{group}_{metric}"]`, $P_m$ = `prior_{metric}_count`(미등록 0.0).

$$\text{rate} = \text{fillna}(\text{rate 열},\;\mathbf{G})\quad\textbf{(★ 여기서만 결측을 전역률로 대치)}$$
$$c^{car} = \mathrm{rint}(C_n \times \text{rate}),\qquad c^{se} = \mathrm{clip}(c^{car}-P_m,\;0,\;n^{se})\ \ (\text{가드 없이 조용히 클램프})$$
$$\rho^{pr} = \begin{cases}P_m/P_n & P_n>0\\ G & \text{else}\end{cases}\quad
\pi = \frac{P_m + 200.0\,G}{P_n + 200.0}\quad
\rho^{se} = \begin{cases}c^{se}/n^{se} & n^{se}>0\\ G & \text{else}\end{cases}$$

| 열 | 값 |
|---|---|
| `prior_rate` | $\rho^{pr}$ |
| `prior_shrunk_200` | $\pi$ |
| `season_rate` | $\rho^{se}$ |
| `season_global_30` | $(c^{se}+30.0\,G)/(n^{se}+30.0)$ |
| `season_player_30` | $(c^{se}+30.0\,\pi)/(n^{se}+30.0)$ |
| `season_minus_prior` | $\rho^{se}-\rho^{pr}$ |

총 3×3 + 10×6 = 9 + 60 = **69열**. 전부 float64 계산 → float32 저장.

> **이름 함정**: temporal 은 `_prior_rate_shrunk_200` / `_season_minus_prior_rate`, multirate 는 `_prior_shrunk_200` / `_season_minus_prior` 로 **미묘하게 다르다.** §6.2 스키마 목록을 정본으로 삼을 것.

#### 3.4.4 temporal vs multirate 차이 요약 (혼동 주의)

| 항목 | temporal | multirate |
|---|---|---|
| 리그 fallback | `league_rate` = **0.4861049201797189** (2024 한 시즌) | `global_rates[*]`, success = **0.5237659752747625** (career 가중) |
| 표본수 결측 | fillna **안 함** | fillna(0.0) **함** |
| rate 결측 대치 | 0.0 (성공수 재구성용) | **전역률 $G$** |
| successes 범위 가드 | 있음 | **없음** (clip 만) |
| 수축 강도 | 10/30/100/300 | 30 단일 |
| 선수 prior 강도 | 200 | 200 |
| HGB 스키마 포함 | 일부 17열 | **전혀 없음** |

두 경로는 같은 $n^{se}$ 를 계산하지만 fallback 이 다르므로 콜드스타트 행에서 값이 갈린다. 예: 5행 샘플 row 5 에서 `temporal_pitcher_season_global_30 = 0.47649097` vs `multirate_pitcher_control_success_season_global_30 = 0.50546098`.

#### 3.4.5 (재학습 시에만) 상태 생성 규칙

시즌 $s$ 처리 후: 그룹 id 로 묶어 표본수 열 최대 행 선택 → `n_last`(결측 0.0) → `prior_n ← n_last + 1.0` → 각 metric `count ← rint(n_last × rate)` (rate 결측 시 아래 DEFAULT) → 그룹의 "정답 metric"(pitcher_control/batter_control 은 `success`, pitchmix 는 **없음**)이면 `count ← count + control_success(그 행)` → keep-last 병합.

**pitchmix 에 정답 metric 이 없어서** 카운트 3개 합이 `n_last` 에 머물고 `prior_n` 만 +1 된다 → §2.4의 `합 = prior_n − 1` 항등식. 선수당 최대 1투구의 보수적 오차이며 복원 clip 이 흡수한다.

시즌마다 `global_rates[group_metric] = Σ prior_{metric}_count / Σ prior_n` 재계산(합 0이면 갱신 생략).

**DEFAULT_GLOBAL_RATES**(첫 시즌·결측 대치 초기값, 추론에는 미사용): pitcher_control success 0.5 / reverse 0.2 / middle 0.15 / ball 0.35 / strike 0.45, batter_control success 0.5 / middle 0.15, pitchmix fastball 0.5 / breaking 0.35 / offspeed 0.15.

### 3.5 가드 (assertion) 5종 — 전부 그대로 구현할 것

| # | 위치 | 조건 (참이면 `ValueError`) | 취지 |
|---|---|---|---|
| G1 | temporal 부착 직전 | `season ≤ history_state.through_season(2024)` | 추론 시즌은 이력 이후여야 함 |
| G2 | multirate 부착 직전 | `season ≤ multirate_state.through_season(2024)` | 동일 |
| G3 | temporal, $e\in$\{pitcher,batter\} | $n^{se}_{\text{raw}} < -1\text{e-}6$ | career 누적이 저장 스냅샷보다 작음 |
| G4 | temporal, 동일 | $S^{se}_{\text{raw}} < -0.01$ **또는** $S^{se}_{\text{raw}} - n^{se}_{\text{raw}} > +0.01$ | 재구성 성공수가 [0, 시즌표본수] 밖 |
| G5 | multirate, 3그룹 | $n^{se}_{\text{raw}} < -1\text{e-}6$ | 동일(표본수만) |

- **G4 의 우변은 클램프 전 $n^{se}_{\text{raw}}$** 이다(클램프 후 $n^{se}$ 가 아니다). 순서를 지킬 것.
- 허용오차 상수 `−1e-6`, `−0.01`, `+0.01` 을 그대로 쓸 것.
- multirate 에는 성공수 가드(G4 상당)가 **없다.** 이 비대칭은 원본 그대로다.
- G3 좌변에 fillna 를 하지 않으므로 `asof_{e}_n` 이 NaN 이면 비교가 False 가 되어 통과하고 `max(NaN,0)=NaN` 으로 전파된다. 원문 동작 보존을 위해 **여기서만 fillna 하지 말 것**.
- G1/G2 는 원문이 프레임 전체 `.any()` 로 검사하지만, **행별 검사로 재작성**한다(§0.3).

### 3.6 원-핫 12열 — 고정 어휘 방식

| 열 이름 | 조건 |
|---|---|
| `top_bottom_B` | `top_bottom == "B"` |
| `top_bottom_T` | `top_bottom == "T"` |
| `top_bottom_nan` | `top_bottom` 결측 |
| `base_state_123` | `base_state == "123"` |
| `base_state_12_` | `== "12_"` |
| `base_state_1_3` | `== "1_3"` |
| `base_state_1__` | `== "1__"` |
| `base_state__23` | `== "_23"` |
| `base_state__2_` | `== "_2_"` |
| `base_state___3` | `== "__3"` |
| `base_state____` | `== "___"` (밑줄 3개 값 → 열 이름은 밑줄 4개) |
| `base_state_nan` | `base_state` 결측 |

- dtype int8. `_nan` 열은 결측이 없어도 항상 존재하며 전부 0.
- **어휘 밖 값은 12열이 전부 0** 이 된다 — 예외를 던지지 않는다(원문 동작과 동일).
- 이 방식은 `get_dummies(dummy_na=True)` + 스키마 재색인(누락 0 채움)과 결과가 동일하면서 행 독립성이 명시적이다.

---

## 4. EB 계층

세 EB 성분(그룹 / 팀 / pitcher×count)과 저랭크(§5)는 **동일한 골격**을 공유한다.

### 4.1 공통 EB 골격

소스 시즌 $s$ 의 행 집합 $R_s$, 정답 $y_i$, 그 시즌의 고정 기준 예측 $b_i$:

$$\text{raw}_i = y_i - b_i,\qquad m_s = \frac{1}{|R_s|}\sum_{i\in R_s}\text{raw}_i,\qquad \text{resid}_i = \text{raw}_i - m_s$$

($\Rightarrow \sum \text{resid} = 0$. 저랭크 학습 코드는 $|\text{mean}| \le 10^{-12}$ 를 단언한다.)

키 조합 $g$ 에 대해 **prior = 0 을 향한 수축**:

$$n_g = |\{i: \text{key}(i)=g\}|,\qquad S_g = \sum_{\text{key}(i)=g}\text{resid}_i,\qquad
\text{effect}_g(s) = \frac{S_g}{n_g + k} = \frac{n_g}{n_g+k}\cdot\overline{\text{resid}}_g$$

- 별도 부모 평균을 더하지 않는다. $n_g=0$ 인 키는 테이블에 없고 조회 시 **0**.
- $k$: 그룹 100/300, 팀 **1000.0**, pitcher×count **600.0**, 저랭크 **300.0**.

**"계층"의 의미**: 부모 효과값을 prior 로 넣는 게 아니라, **부모의 보정된 예측을 다음 단계의 기준 예측 $b_i$ 로 갈아끼우는 순차 잔차 부스팅**이다.

```
level 0 : backbone (fixed 50:50)
level 1 : team_base = backbone + Δ_team            ← 잔차 y − backbone 에서 학습
level 2 : (a) + Δ_pc  또는  (b) + Δ_lowrank        ← 잔차 y − team_base 에서 학습
```
따라서 pitcher×count 와 저랭크는 **형제 관계**이며 동시에 더해지는 경로는 없다.

| 성분 | 기준 예측 $b_i$ |
|---|---|
| Team EB | fixed 50:50 백본 OOF |
| pitcher×count EB | team all_prior_s1000 OOF |
| low-rank | team all_prior_s1000 OOF (동일) |

### 4.2 소스 시즌 결합

$$\Delta(\text{row}) = \frac{1}{4}\sum_{s\in\{2021,2022,2023,2024\}}\text{effect}_{\text{key(row)}}(s)$$

- 가중치 **완전 균등 1/4**. 최신성·표본수 가중 없음.
- **분모는 항상 4**(매칭 시즌 수가 아님). 1개 시즌에만 등장한 투수는 효과가 자동으로 1/4 로 더 축소된다 — 저표본 안전장치.
- 학습·검증 스크립트에서는 검증 시즌 $v$ 에 대해 소스가 $\{s: s<v\}$ 로 제한되지만(2021 검증 → 소스 없음 → 보정 0), **배포는 검증 시즌이 2025이므로 자연히 4시즌 전부**다. 재현 대상은 배포 규칙이다.

### 4.3 그룹 효과 $e^{\text{group}}$ (추론)

조회 키:

| 테이블 | 키 |
|---|---|
| `base` | (`count_index`, `pitcher_hand`, `batter_hand`) |
| `reverse` | (`count_index`, `pitcher_hand`, `batter_hand`, `reverse_rate_bin`) |

$$\beta_i = \begin{cases}\big\lfloor \texttt{asof\_pitcher\_reverse\_rate}_i \,/\, 0.05 \big\rfloor & \text{유한값(isfinite)일 때}\\ -1 & \text{NaN/Inf 일 때}\end{cases}$$

- 키는 각 컬럼을 **int 캐스팅**한 튜플. 미등록이면 **0.0**.
- 합성: $e^{\text{group}} = 0.7\,g^{\text{base}} + 0.3\,g^{\text{rev}}$, 이어서 $b^{\text{grp}} = C(b^{\text{tmp}} + e^{\text{group}})$.

### 4.4 팀 효과 $e^{\text{team}}$ (추론)

| 패밀리 | 키 순서 |
|---|---|
| `pitcher_team` | `pitcher_team_id`, `pitcher_hand`, `batter_hand` |
| `batter_team` | `batter_team_id`, `pitcher_hand`, `batter_hand` |

```
Δ_pitcher_team = (1/4) Σ_{s∈{2021..2024}} lookup(pitcher_team[s], key)   # 미스 0.0
Δ_batter_team  = (1/4) Σ_{s}              lookup(batter_team[s],  key)   # 미스 0.0
Δ_team         = 0.50·Δ_pitcher_team + 0.50·Δ_batter_team
b_team         = clip(b_bone + Δ_team, 0, 1)
```

패밀리 가중 **0.50/0.50 고정**. 등가중 평균이므로 **4시즌 평균 테이블을 미리 한 번 만들어 조회 1회로 축약해도 결과가 동일**하다(권장 벡터화).

### 4.5 그룹 효과표 생성 규칙 (재학습 시)

`group_effects.json` 은 EXP018 규칙으로 만들어졌다.

1. 초기 잔차 시즌별 **전체 행**(R/F 구분 없음) 중심화: $e_i = (y_i - b_i) - \text{mean}\{y_j-b_j : s_j=s_i\}$, float32.
2. 그룹 키 4종: `count_index`(int8), `pitcher_hand`(int8), `batter_hand`(int8), `reverse_rate_bin`(int16).
3. 학습 윈도: 직전 최대 **3시즌** (`EXP018_GROUP_WINDOW = 3`).
4. coarse 효과: 키 3개, 수축 $\lambda = 100.0$ / reverse 효과: 키 4개, $\lambda = 300.0$. 미등장 키 0.0.
5. 혼합 0.7 / 0.3.
6. **배포용 최종 효과표의 적합 구간은 2022~2024** (= $\max(s)-3+1$ 이상).

> **비대칭 주의(재현 필수)**: GBDT 학습 타깃 계산에 쓰이는 그룹 예측은 **폴드별 leave-future-out OOF** 인 반면, 추론에서 쓰는 `group_base` 는 **2022~2024 로 한 번 적합한 최종 표**다. 두 값은 다르다.

### 4.6 pitcher×count EB (배포 미사용, 기록용)

| 항목 | 값 |
|---|---|
| 키 | `pitcher_id`, `count_index`, `batter_hand` |
| 기준 예측 | team all_prior_s1000 OOF |
| 수축 $k$ | **600.0** (파일에 저장됨) |
| 소스 시즌 | 2021~2024, 균등 1/4 |

적용식(후보 `r_gated_team_pc_all` 전용):
```
gated = team_base  if game_type == "R"  else  backbone     # F행은 team 보정을 되돌림
pred  = clip(gated + Δ_pc, 0, 1)
```
`strict_lowrank_s300_r6` 에서는 이 파일이 **로드조차 되지 않는다.**

### 4.7 팀 효과표 생성 규칙 (재학습 시)

미채택 후보 포함 기록: EXP-019 에는 `all_prior_s1000`(prior_window=None, smoothing 1000.0)과 `prior1_s500`(prior_window=1, smoothing 500.0) 두 후보가 있었고 **배포는 `all_prior_s1000`** 이다.

---

## 5. 저랭크 성분

### 5.1 컨텍스트 격자 (24개, 정적 선언 — 데이터에서 유도하지 말 것)

```
COUNT_INDICES = [ b*4 + s  for b in 0..3  for s in 0..2 ] = [0,1,2, 4,5,6, 8,9,10, 12,13,14]
BATTER_HANDS  = [1, 2]
CONTEXTS      = [ (c,h) for c in COUNT_INDICES for h in BATTER_HANDS ]      # 24개
```
**정렬 키: count_index 가 바깥 루프, batter_hand 가 안쪽 루프.**

| pos | count | hand | | pos | count | hand | | pos | count | hand |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 1 | | 8 | 5 | 1 | | 16 | 10 | 1 |
| 1 | 0 | 2 | | 9 | 5 | 2 | | 17 | 10 | 2 |
| 2 | 1 | 1 | | 10 | 6 | 1 | | 18 | 12 | 1 |
| 3 | 1 | 2 | | 11 | 6 | 2 | | 19 | 12 | 2 |
| 4 | 2 | 1 | | 12 | 8 | 1 | | 20 | 13 | 1 |
| 5 | 2 | 2 | | 13 | 8 | 2 | | 21 | 13 | 2 |
| 6 | 4 | 1 | | 14 | 9 | 1 | | 22 | 14 | 1 |
| 7 | 4 | 2 | | 15 | 9 | 2 | | 23 | 14 | 2 |

이 표는 `lowrank_effects.json` 의 `contexts` 배열과 일치함을 실측 확인했다. **구현 시에는 파일의 `position` 필드를 그대로 신뢰해 딕셔너리를 만들 것**(순서 재계산 금지).

### 5.2 추론 적용

```
ctx = CONTEXT_POSITION[(count_index_i, batter_hand_i)]      # 24격자 밖이면 KeyError (원본 동작)
for s in [2021, 2022, 2023, 2024]:
    j ← pitcher_ids[s] 에서 pitcher_id_i 의 위치, 없으면 −1
    v_s ← values[s][j, ctx]  if j ≥ 0  else  0.0
Δ_lowrank = mean over 4 sources of v_s                       # 분모 항상 4
p = clip(b_team + Δ_lowrank, 0, 1)
```

- **미등록 투수는 정확히 0 기여**. "매칭된 소스만 평균" 으로 바꾸면 값이 달라진다.
- `source_weights` 키가 있으면 가중평균 경로(유효성: 길이 일치, 전부 유한, 음수 없음, 합 > 0; 위배 시 `ValueError`)지만 **우리 산출물에는 없으므로 균등 평균만 구현하면 된다**.
- 벡터화 시 4개 소스를 투수 id 합집합 기준 **(n_pitchers × 24) 합/4 배열 하나**로 축약 가능(단 "합/4" 여야 하고 "등장 소스만 평균" 이면 안 됨).

### 5.3 (재학습 시) 행렬 적합 절차

```
입력: 소스 시즌 s 전체 행 R_s (R행+F행 모두, game_type 필터 없음)
      y_s, b_s = team all_prior_s1000 OOF

1. raw ← y_s − b_s ; m_s ← mean(raw) ; resid ← raw − m_s     # assert |mean(resid)| ≤ 1e-12
2. pitcher_ids ← 그 시즌 pitcher_id 의 오름차순 정렬 유일값 (factorize sort=True)
   P ← len(pitcher_ids) ;  row(i) ← 위치 ;  col(i) ← position(count_index_i, batter_hand_i)
3. Sum[P×24] ← 0 ; Cnt[P×24] ← 0
   for i in R_s: Sum[row,col] += resid_i ; Cnt[row,col] += 1     # assert Cnt.sum() == |R_s|
4. A ← Sum / (Cnt + 300.0)          # 미관측 셀: 0/(0+300) = 0.0, 마스킹하지 않음
5. U, σ, Vᵀ ← SVD(A, full_matrices=False)
   Â ← (U[:, :6] * σ[:6]) @ Vᵀ[:6, :]                          # rank 정확히 6
6. 내보내기: {season, pitcher_ids, values: Â}
```

**명시적 부정(자주 잘못 구현되는 지점)**
- 행렬의 **행/열 중심화 없음**. 중심화는 §4.1 의 시즌 전체 잔차 평균 제거 한 번뿐.
- SVD 에 **관측수 가중 없음**(표준 비가중 SVD). 관측수는 오직 4단계 분모에만 들어간다.
- **마스크드 SVD / ALS 대치가 아니다.** 미관측 셀을 0으로 채워 분해한다. 랭크 절단이 0 위로 정보를 번지게 하는 것이 이 성분의 존재 이유다.
- 재구성값에 별도 스케일·수축 없음(`correction_weight = 1.0`). U/σ/Vᵀ 를 따로 저장하지 않고 $\hat A$ 만 저장.

실험 격자(참고): smoothing (300.0, 600.0) × rank (2, 4, 6, 8, 12) → 배포는 **300.0 / 6**. R-specific 변종(smoothing 300.0, rank 4, 소스·적용 모두 R행만)은 미배포.

---

## 6. 잔차 학습기 2종과 JSON 추론 규칙

### 6.1 학습 타깃·행 필터·가중치 (재학습 시에만 필요, 두 학습기 완전 공유)

**타깃**: $z_i^{(0)} = \mathrm{f32}(y_i - g_i)$ (여기서 $g_i$ = §4.5 규칙의 leave-future-out 그룹 OOF 예측, `clip(b + c, 0, 1)`), 이어서 **시즌 × R행 그룹별로만** 평균 제거:

$$z_i = z_i^{(0)} - \text{mean}\{z_j^{(0)} : s_j = s_i,\ R_j\}\quad(R_i \text{ 인 행})$$

클립 없음. 범위 대략 [−1, 1]. 한 줄 요약: **타깃 = 실제 라벨 − 시간안전 그룹 OOF 예측, 시즌별 R행 평균을 0으로 맞춘 것.**

**행 필터**: 시즌 필터 없음(2019~2024 전부), `game_type == "R"` 행만. 원본 총 1,475,092행 중 R행 수는 산출물에 기록되어 있지 않다(§10-1).

**시즌 균등 표본가중치** (R행만 뽑은 시즌 벡터, 길이 $m$):
```
w_i = 1 / count(season == s_i)            (float32)
w  *= m / sum(w)                          # 총합 = m, 시즌별 총가중치 동일
```
두 학습기가 이 **동일한 $z$ 와 동일한 $w$** 를 쓴다.

### 6.2 LightGBM 입력 186열 (`feature_schemas.json["lightgbm"]`, 이 순서 그대로)

블록 구성: 0–19 원본 통과 20 / 20–38 asof 19 / 39–59 정적 파생 21 / 60–104 temporal 45 / 105–173 multirate 69 / 174–185 더미 12.

```
  0 game_month                     63 temporal_pitcher_log_prior_n
  1 game_dayofweek                 64 temporal_pitcher_prior_rate
  2 inning                         65 temporal_pitcher_prior_rate_shrunk_200
  3 balls_before                   66 temporal_pitcher_season_n
  4 strikes_before                 67 temporal_pitcher_log_season_n
  5 outs_before                    68 temporal_pitcher_season_rate
  6 run_top_before                 69 temporal_pitcher_season_minus_prior_rate
  7 run_bot_before                 70 temporal_pitcher_season_global_10
  8 run_total_before               71 temporal_pitcher_season_player_10
  9 score_diff_home                72 temporal_pitcher_reliability_10
 10 score_diff_pitcher_team        73 temporal_pitcher_season_global_30
 11 runner_on_1b                   74 temporal_pitcher_season_player_30
 12 runner_on_2b                   75 temporal_pitcher_reliability_30
 13 runner_on_3b                   76 temporal_pitcher_season_global_100
 14 num_runners_on                 77 temporal_pitcher_season_player_100
 15 home_win_expectancy            78 temporal_pitcher_reliability_100
 16 away_win_expectancy            79 temporal_pitcher_season_global_300
 17 li                             80 temporal_pitcher_season_player_300
 18 pitcher_hand                   81 temporal_pitcher_reliability_300
 19 batter_hand                    82 temporal_batter_prior_exists
 20 asof_pitcher_n                 83 temporal_batter_prior_n
 21 asof_pitcher_success_rate      84 temporal_batter_log_prior_n
 22 asof_pitcher_reverse_rate      85 temporal_batter_prior_rate
 23 asof_pitcher_middle_rate       86 temporal_batter_prior_rate_shrunk_200
 24 asof_pitcher_ball_rate         87 temporal_batter_season_n
 25 asof_pitcher_strike_rate       88 temporal_batter_log_season_n
 26 asof_pitcher_prev1_game_success_rate   89 temporal_batter_season_rate
 27 asof_pitcher_prev3_game_success_rate   90 temporal_batter_season_minus_prior_rate
 28 asof_pitcher_prev5_game_success_rate   91 temporal_batter_season_global_10
 29 asof_pitcher_prev1_game_middle_rate    92 temporal_batter_season_player_10
 30 asof_pitcher_prev3_game_middle_rate    93 temporal_batter_reliability_10
 31 asof_pitcher_prev5_game_middle_rate    94 temporal_batter_season_global_30
 32 asof_batter_n                  95 temporal_batter_season_player_30
 33 asof_batter_success_rate       96 temporal_batter_reliability_30
 34 asof_batter_middle_rate        97 temporal_batter_season_global_100
 35 asof_pitcher_pitchmix_n        98 temporal_batter_season_player_100
 36 asof_pitcher_fastball_rate     99 temporal_batter_reliability_100
 37 asof_pitcher_breaking_rate    100 temporal_batter_season_global_300
 38 asof_pitcher_offspeed_rate    101 temporal_batter_season_player_300
 39 count_index                   102 temporal_batter_reliability_300
 40 count_out_index               103 temporal_base_global_30
 41 is_full_count                 104 temporal_base_player_30
 42 has_two_strikes               105 multirate_pitcher_control_season_n
 43 has_three_balls               106 multirate_pitcher_control_log_season_n
 44 count_advantage               107 multirate_pitcher_control_reliability_30
 45 runner_in_scoring_position    108 multirate_pitcher_control_success_prior_rate
 46 bases_loaded                  109 multirate_pitcher_control_success_prior_shrunk_200
 47 same_hand                     110 multirate_pitcher_control_success_season_rate
 48 late_inning                   111 multirate_pitcher_control_success_season_global_30
 49 close_game                    112 multirate_pitcher_control_success_season_player_30
 50 log_li                        113 multirate_pitcher_control_success_season_minus_prior
 51 score_pressure                114..119 multirate_pitcher_control_reverse_{prior_rate, prior_shrunk_200,
 52 win_expectancy_gap                      season_rate, season_global_30, season_player_30, season_minus_prior}
 53 pitcher_batter_success_gap    120..125 multirate_pitcher_control_middle_{동일 6}
 54 pitcher_recent_success_delta_1_5  126..131 multirate_pitcher_control_ball_{동일 6}
 55 pitcher_recent_success_delta_3_5  132..137 multirate_pitcher_control_strike_{동일 6}
 56 pitcher_recent_middle_delta_1_5   138 multirate_batter_control_season_n
 57 log_pitcher_n                     139 multirate_batter_control_log_season_n
 58 log_batter_n                      140 multirate_batter_control_reliability_30
 59 log_pitchmix_n                    141..146 multirate_batter_control_success_{동일 6}
 60 temporal_prior_league_rate        147..152 multirate_batter_control_middle_{동일 6}
 61 temporal_pitcher_prior_exists     153 multirate_pitcher_pitchmix_season_n
 62 temporal_pitcher_prior_n          154 multirate_pitcher_pitchmix_log_season_n
                                      155 multirate_pitcher_pitchmix_reliability_30
                                      156..161 multirate_pitcher_pitchmix_fastball_{동일 6}
                                      162..167 multirate_pitcher_pitchmix_breaking_{동일 6}
                                      168..173 multirate_pitcher_pitchmix_offspeed_{동일 6}
                                      174 top_bottom_B        180 base_state_1__
                                      175 top_bottom_T        181 base_state__23
                                      176 top_bottom_nan      182 base_state__2_
                                      177 base_state_123      183 base_state___3
                                      178 base_state_12_      184 base_state____
                                      179 base_state_1_3      185 base_state_nan
```
("동일 6" = `_prior_rate, _prior_shrunk_200, _season_rate, _season_global_30, _season_player_30, _season_minus_prior` 순서)

### 6.3 HGB 입력 84열 (`feature_schemas.json["histgradientboosting"]`, 이 순서 그대로)

```
 0 game_month                          42 same_hand
 1 game_dayofweek                      43 late_inning
 2 inning                              44 close_game
 3 balls_before                        45 log_li
 4 strikes_before                      46 score_pressure
 5 outs_before                         47 win_expectancy_gap
 6 run_top_before                      48 pitcher_batter_success_gap
 7 run_bot_before                      49 pitcher_recent_success_delta_1_5
 8 run_total_before                    50 pitcher_recent_success_delta_3_5
 9 score_diff_home                     51 pitcher_recent_middle_delta_1_5
10 score_diff_pitcher_team             52 log_pitcher_n
11 runner_on_1b                        53 log_batter_n
12 runner_on_2b                        54 log_pitchmix_n
13 runner_on_3b                        55 temporal_prior_league_rate
14 num_runners_on                      56 temporal_pitcher_prior_exists
15 home_win_expectancy                 57 temporal_pitcher_log_prior_n
16 away_win_expectancy                 58 temporal_pitcher_prior_rate_shrunk_200
17 li                                  59 temporal_pitcher_log_season_n
18 pitcher_hand                        60 temporal_pitcher_season_global_30
19 batter_hand                         61 temporal_pitcher_season_player_30
20 asof_pitcher_reverse_rate           62 temporal_pitcher_reliability_30
21 asof_pitcher_middle_rate            63 temporal_batter_prior_exists
22 asof_pitcher_ball_rate              64 temporal_batter_log_prior_n
23 asof_pitcher_strike_rate            65 temporal_batter_prior_rate_shrunk_200
24 asof_pitcher_prev1_game_success_rate 66 temporal_batter_log_season_n
25 asof_pitcher_prev3_game_success_rate 67 temporal_batter_season_global_30
26 asof_pitcher_prev5_game_success_rate 68 temporal_batter_season_player_30
27 asof_pitcher_prev1_game_middle_rate  69 temporal_batter_reliability_30
28 asof_pitcher_prev3_game_middle_rate  70 temporal_base_global_30
29 asof_pitcher_prev5_game_middle_rate  71 temporal_base_player_30
30 asof_batter_middle_rate              72 top_bottom_B
31 asof_pitcher_fastball_rate           73 top_bottom_T
32 asof_pitcher_breaking_rate           74 top_bottom_nan
33 asof_pitcher_offspeed_rate           75 base_state_123
34 count_index                          76 base_state_12_
35 count_out_index                      77 base_state_1_3
36 is_full_count                        78 base_state_1__
37 has_two_strikes                      79 base_state__23
38 has_three_balls                      80 base_state__2_
39 count_advantage                      81 base_state___3
40 runner_in_scoring_position           82 base_state____
41 bases_loaded                         83 base_state_nan
```

블록: 0–19 원본 20 / 20–33 asof 14 / 34–54 정적 파생 21 / 55–71 temporal 17 / 72–83 더미 12.

**LGB 대비 빠진 것**: asof 5열(`asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_batter_n`, `asof_batter_success_rate`, `asof_pitcher_pitchmix_n`), temporal 의 `_prior_n/_prior_rate/_season_n/_season_rate/_season_minus_prior_rate` 및 강도 10/100/300 계열 전부, **multirate 69열 전부**. 이 축소가 HGB 를 LGB 와 상관 낮은 학습기로 만드는 핵심이다.

> **multirate 는 HGB 에 안 들어가지만 LGB 를 위해 반드시 계산해야 한다.** 마찬가지로 강도 10/100/300 temporal 도 LGB 전용.

### 6.4 행렬 조립 규칙 (두 모델 공통)

1. 스키마 리스트에 중복 이름이 있으면 `ValueError`.
2. 프레임에 `top_bottom`, `base_state` **두 컬럼만** 원-핫(§3.6). `game_type` 은 문자열로 남는다.
3. 스키마 이름 중 프레임에 없는 것이 있으면 `ValueError` — 단 접두사 `top_bottom_` / `base_state_` 인 것은 **면제**(그 범주가 배치에 없었을 뿐).
4. 스키마 순서대로 재색인, 없는 열은 **0** 으로 채움. 스키마 밖 열(`season`, `game_type`, id 4종, 원본 `top_bottom`/`base_state` 등)은 버림.
5. 전체를 **float32** 로 캐스팅. **NaN 은 NaN 으로 보존**.
6. 폭 검증: `booster.num_feature() != 186` 이면 `ValueError`, `hgb_state["n_features"] != 84` 이면 `ValueError`, 실제 행렬 열 수 ≠ 84 이면 `ValueError`.

실측(5행): LGB (5,186) NaN 42개, HGB (5,84) NaN 39개.

### 6.5 LightGBM 추론 $r^{\text{lgb}}$

1. 표준 로더로 `rfull_lightgbm.txt` → Booster.
2. 폭 검증(§6.4-6).
3. 186열 **float32** 행렬로 예측. `num_iteration` 지정 없음 = 300트리 전부.
4. 결과를 **float64 로 캐스팅**. 이것이 $r^{\text{lgb}}$ (확률 스케일 잔차).

> `Column_k` 일반명이라 **이름 기준 재정렬을 하면 오류 없이 조용히 틀린다.** `feature_schemas.json` 순서를 그대로 쓸 것.
> `boost_from_average` 초기값이 첫 트리에 흡수되어 있으므로 **별도 초기값을 더하지 말 것.**

학습 하이퍼파라미터는 §8.5 (모델 파일에서 실측 전재).

### 6.6 HGB JSON 추론 $r^{\text{hgb}}$ — 순수 numpy 순회 ★가장 틀리기 쉬운 부분★

sklearn 피클을 쓰지 않는 이유: 평가 서버의 sklearn 버전이 학습 환경과 달라도 동작해야 하기 때문. 트리 노드 배열만 숫자로 내보내고 numpy 로 순회한다.

$$\hat r^{\text{hgb}}(x) = \text{baseline} + \sum_{k=0}^{159} v_k\big(\ell_k(x)\big)$$

노드 $n$ 에서 ($f$=`feature_idx[n]`, $\theta$=`num_threshold[n]`):

$$\text{다음 노드} = \begin{cases}
\text{left}[n] & x_f \text{ NaN 이고 missing\_go\_to\_left}[n]=1\\
\text{right}[n] & x_f \text{ NaN 이고 missing\_go\_to\_left}[n]=0\\
\text{left}[n] & x_f \text{ 유효하고 } x_f \le \theta\\
\text{right}[n] & x_f \text{ 유효하고 } x_f > \theta
\end{cases}$$

**반드시 지킬 4가지**

1. **학습률(0.025)을 다시 곱하지 말 것.** sklearn 은 리프값에 학습률을 이미 반영해 예측기를 만든다. 원 저장소는 export 직후 sklearn 네이티브 예측 대비 4,096행에서 `max|Δ| ≤ 1e-12` 를 단언한다.
2. **리프에서만 value 누적.** 비리프 노드 중 1,968개가 `value ≠ 0` 이므로, 내부 노드에서 더하면 결과가 완전히 망가진다.
3. **경계는 `<=` 가 왼쪽** (sklearn HistGradientBoosting 규약).
4. **NaN 을 채우지 말 것.** 결측 여부가 분기의 1급 정보다.

의사코드 (벡터화, 행 독립):
```
if state.format ≠ "numeric_hgb_v1": ValueError
if X.shape[1] ≠ state.n_features:   ValueError

pred ← 길이 n, 값 = state.baseline, dtype float64
for tree in state.trees:
    value/feat/thr/miss_left/left/right/leaf ← float64/intp/float64/bool/intp/intp/bool 로 변환
    node ← zeros(n, int) ; active ← ones(n, bool)
    while any(active):
        rows ← active 인덱스 ; cur ← node[rows]
        done ← rows 중 leaf[cur] 참
            pred[done] += value[node[done]] ; active[done] ← False
        go   ← rows 중 leaf[cur] 거짓
            m ← node[go] ; v ← X[go, feat[m]]
            go_left ← where(isnan(v), miss_left[m], v <= thr[m])
            node[go] ← where(go_left, left[m], right[m])
return pred
```
최대 깊이 4 이므로 while 은 트리당 최대 5회. 스칼라 루프로 구현해도 결과 동일(벡터화는 속도 문제).

**(재학습 시) 직렬화 절차**: (1) 반복당 트리 수가 1이 아니면 예외 (2) 각 부스팅 반복 예측기의 노드 구조체에서 7필드 추출 (3) 범주형 분기가 하나라도 있으면 예외 (4) `baseline` = 초기 예측 배열 첫 원소 (5) `n_features` = 학습 피처 수 (6) **필수 사후 검증**: 학습 행렬에서 균등 간격 최대 4,096행을 뽑아 sklearn 네이티브 예측과 numpy 순회 예측 비교, `max|Δ| > 1e-12` 이면 실패.

### 6.7 게이트와 backbone 합성

```
is_R = (str(game_type) == "R")
# GBDT 예측은 마스크와 무관하게 전 행 계산 후 마스크로 선택 (값은 동일, 비용만의 문제)
B_lgb = is_R ? clip(b_grp + 0.75 · r_lgb, 0, 1) : b_grp
B_hgb = is_R ? clip(b_grp + 1.00 · r_hgb, 0, 1) : b_grp
b_bone = 0.5 · B_lgb + 0.5 · B_hgb                    # 평균 후 클립 없음
```

**가중치 비대칭 주의**: LGB **0.75**, HGB **1.00**. 잔차 블렌드 후보 격자 {0.25, 0.50, 0.75, 1.00} 에서 각각 고른 값이며, 50:50 결합은 폴드 평가 전에 고정된 값이다.

(참고, 배포 미사용) HGB 검증 스크립트가 계산한 오라클 가중치 진단식:
$$w^{\ast} = \mathrm{clip}\!\left(\frac{-\overline{(r-y)\delta}}{\overline{\delta^2}},\,0,\,1\right),\qquad \delta = \text{candidate} - \text{reference}$$

---

## 7. 최종 합성·클립·출력

### 7.1 최상위 흐름 의사코드

```
STEP 1  test  ← read_csv("./data/test.csv", encoding="utf-8-sig")
        sample← read_csv("./data/sample_submission.csv", encoding="utf-8-sig")
STEP 2  입력 검증 5종 (§1.1)
STEP 3  metadata / history_state / multirate_state / feature_schemas 로드
STEP 4  F ← test.drop(row_id)
STEP 5  F += 정적 파생 21열            (§3.2)
STEP 6  가드 G1 → F += temporal 45열   (§3.3, §3.5)
STEP 7  가드 G2 → F += multirate 69열  (§3.4, §3.5)
STEP 8  b_tmp ← F["temporal_base_global_30"]  (float32 → float64)
STEP 9  group_effects 로드 → e_group ; b_grp ← clip(b_tmp + e_group, 0, 1)
STEP 10 Booster 로드 + HGB JSON 로드 ; 폭 검증
        X_lgb (n,186) float32 ; X_hgb (n,84) float32
        r_lgb ← Booster.predict(X_lgb) → float64
        r_hgb ← numpy 순회(X_hgb)      → float64
STEP 11 is_R 게이트 → B_lgb, B_hgb → b_bone = 0.5/0.5     (§6.7)
STEP 12 team_effects 로드 → e_team ; b_team ← clip(b_bone + e_team, 0, 1)
STEP 13 lowrank_effects 로드 → e_low ; p ← clip(b_team + e_low, 0, 1)
STEP 14 row_id → p 매핑 → sample 의 row_id 순서로 재배열 → 검증 → CSV 저장
```

### 7.2 클립 5지점 재확인

| 순번 | 위치 | 식 |
|---|---|---|
| 1 | 그룹 가산 후 | `clip(b_tmp + e_group, 0, 1)` |
| 2 | LGB 가지 내부 (R행만) | `clip(b_grp + 0.75·r_lgb, 0, 1)` |
| 3 | HGB 가지 내부 (R행만) | `clip(b_grp + 1.00·r_hgb, 0, 1)` |
| 4 | 팀 가산 후 | `clip(b_bone + e_team, 0, 1)` |
| 5 | 저랭크 가산 후 | `clip(b_team + e_low, 0, 1)` |

**50:50 평균 직후에는 클립하지 않는다.** 중간 클립을 생략하거나 순서를 바꾸면 극단 행에서 값이 달라진다.

### 7.3 출력

1. `dict(zip(test.row_id, predictions))` 를 만들고 **sample 의 row_id 순서**로 꺼낸다(순서 차이에 안전).
2. NaN 이 하나라도 있으면 `ValueError`.
3. 전부 `[0.0, 1.0]` 범위가 아니면 `ValueError`.
4. `./output` 디렉터리 생성 후 `./output/submission.csv` 저장. 컬럼 `[row_id, control_success]`, `index=False`, `encoding="utf-8"`.

### 7.4 실행되지 않는 분기 (재작성 시 전부 삭제)

`candidate = "strict_lowrank_s300_r6"` 이므로 원 스크립트의 9가지 후보 분기 중 **strict 경로 하나만** 실행된다. 나머지 8개 후보와 그에 딸린 5개 매퍼(exact pitchtype, physical ridge, pitcher_count, rspecific, recency)는 전부 제거한다 — 코드가 대폭 짧아지고 원문과의 구조적 유사성도 사라진다.

기록용(구현 불필요) 다른 후보의 합성 규칙: aggressive = `clip(where(R, team_base, backbone) + Δ_pc, 0, 1)`; dualrank_consensus_50 = 0.5·strict + 0.5·rspecific(비-R 행은 보정 0); strict_aggressive_consensus_50 = 0.5/0.5; recency_aggressive_consensus_50/_70 = 0.5/0.5 및 0.7/0.3; trackman_recent_consensus_50/_25 = trackman 가중 0.5/0.25 (exact 보정 계수 0.25); public_simplex_act_25_60_15 = 0.25·aggressive + 0.60·recency + 0.15·exact; trackman_direct_recent_w010/w0125 = recent + 0.10/0.125×exact; trackman_physical_recent_w015 = recent + 0.10·exact + 0.15·physical. **exact/physical 보정은 모두 ±0.03 클립.**

### 7.5 성능 메모 (245,789행 기준)

- 효과 조회를 파이썬 dict + 행별 튜플 순회로 하면 O(N) 파이썬 루프가 10회(base, reverse, pitcher_team×4, batter_team×4) 돈다. **pandas merge 또는 다차원 numpy 룩업 배열로 벡터화할 것** — 결과 동일, 훨씬 빠르고 코드 형태도 원문과 완전히 달라진다.
  - base: (count_index 15 × pitcher_hand 3 × batter_hand 3) 0-초기화 배열 + fancy indexing.
  - reverse: bin 범위 −1..20 → (15 × 3 × 3 × 22) 배열, 오프셋 +1.
  - team: 등가중 평균이므로 **4시즌 평균 테이블을 미리 만들어** 패밀리당 조회 1회.
  - lowrank: 투수 id 합집합 기준 **합/4 배열 하나**로 축약(§5.2 주의).
- HGB 순회: 160트리 × 최대 깊이 4 → while 5회 이내. 충분히 빠르다.
- 메모리: (245789, 186) float32 ≈ 183MB, (245789, 84) ≈ 83MB. **두 행렬을 동시에 들지 말고 순차 생성·해제** 권장.
- 산출물 총 용량 수 MB(`lowrank_effects.json` 877,781 B 가 최대). 10GB 제한과 무관.

---

## 8. 상수 총표

### 8.1 결합 가중치·클립

| 이름 | 값 | 위치 |
|---|---|---|
| temporal base 투수/타자 가중 | **0.7 / 0.3** | `temporal_base_global_30`, `_player_30` |
| group base/reverse 혼합 | **0.7 / 0.3** | $e^{\text{group}}$ |
| LightGBM 잔차 가중 | **0.75** | R행에만 |
| HistGB 잔차 가중 | **1.00** | R행에만 |
| backbone 앙상블 | **0.5 / 0.5** | LGB 가지 : HGB 가지 |
| team 패밀리 가중 | **0.5 / 0.5** | pitcher_team : batter_team |
| team 시즌 결합 | 등가중 1/4 (2021·2022·2023·2024) | 미매칭 0 기여, 분모 항상 4 |
| lowrank 소스 결합 | 등가중 1/4 (동일 4시즌) | `source_weights` 키 부재 |
| 확률 클립 | **[0.0, 1.0]**, 5지점 | §7.2 |

### 8.2 수축(shrinkage) 상수

| 이름 | 값 | 저장 위치 |
|---|---|---|
| 선수 사전확률 수축 | **200.0** | temporal `_prior_rate_shrunk_200`, multirate `_prior_shrunk_200` (코드 상수) |
| temporal 시즌 수축 강도 | **10.0, 30.0, 100.0, 300.0** | 코드 상수 |
| multirate 시즌 수축 강도 | **30.0** (단일) | 코드 상수 |
| 그룹 coarse 수축 $\lambda$ | **100.0** | 코드 상수(JSON 미저장) |
| 그룹 reverse 수축 $\lambda$ | **300.0** | 코드 상수(JSON 미저장) |
| 그룹 학습 윈도 | **3시즌** (배포 표는 2022–2024) | 코드 상수 |
| Team EB 수축 $k$ | **1000.0** | 코드 상수(JSON 미저장) |
| Team EB 미채택 후보 | prior_window=1, smoothing **500.0** | 참고 |
| pitcher×count EB 수축 $k$ | **600.0** | `pitcher_count_effects.json.smoothing` |
| low-rank 수축 $k$ | **300.0** | `lowrank_effects.json.smoothing` |
| low-rank 랭크 | **6** | `lowrank_effects.json.rank` |
| low-rank 실험 격자 | smoothing (300.0, 600.0) × rank (2,4,6,8,12) | 참고 |
| low-rank R-specific 변종 | smoothing 300.0, rank 4 | 미배포 |
| low-rank 보정 스케일 | **1.0** (재스케일 없음) | |

### 8.3 전역 확률 상수

| 이름 | 값 | float32 |
|---|---|---|
| `league_rate` ($L$) | **0.4861049201797189** | 0.48610490560531616 |
| `through_season` (양쪽) | **2024** | |
| `pitcher_control_success` | 0.5237659752747625 | 0.5237659811973572 |
| `pitcher_control_reverse` | 0.22891250172870573 | 0.22891250252723694 |
| `pitcher_control_middle` | 0.14952491098860274 | 0.14952491223812103 |
| `pitcher_control_ball` | 0.36937763881846014 | 0.36937764286994934 |
| `pitcher_control_strike` | 0.44309710851933304 | 0.4430971145629883 |
| `batter_control_success` | 0.5237659752747625 | 0.5237659811973572 |
| `batter_control_middle` | 0.14951677590279114 | 0.1495167762041092 |
| `pitcher_pitchmix_fastball` | 0.5411296380157984 | 0.5411296486854553 |
| `pitcher_pitchmix_breaking` | 0.29567647306066336 | 0.29567646980285645 |
| `pitcher_pitchmix_offspeed` | 0.16265697325997294 | 0.1626569777727127 |
| HGB `baseline` | **4.8898374092703725e-09** | |

> `league_rate`(0.4861) ≠ `global_rates[*_success]`(0.5238). 절대 혼용 금지.
> DEFAULT_GLOBAL_RATES(재학습 전용): 0.5 / 0.2 / 0.15 / 0.35 / 0.45 / 0.5 / 0.15 / 0.5 / 0.35 / 0.15.

### 8.4 이산화·가드·정의 상수

| 이름 | 값 |
|---|---|
| `reverse_rate` bin 폭 | **0.05** (floor 분할) |
| `reverse_rate` 결측 bin | **−1** (판정은 `isfinite`) |
| reverse bin 실측 존재값 | −1, 0…17, 20 |
| $n^{se}$ 음수 허용오차 | **−1e-6** (temporal, multirate 공통) |
| $S^{se}$ 하한/상한 허용오차 | **−0.01 / +0.01** (temporal 만) |
| 잔차 중심화 검증 허용오차 | **1e-12** |
| HGB export 패리티 허용오차 | **1e-12** (샘플 4,096행) |
| `count_index` | `balls×4 + strikes`, 유효 12값 {0,1,2,4,5,6,8,9,10,12,13,14} |
| `count_out_index` | `count_index×3 + outs`, 값역 0..44 |
| `late_inning` | `inning ≥ 7` |
| `close_game` | `|score_diff_pitcher_team| ≤ 1` |
| `li` / `asof_*_n` 클립 하한 | **0** |
| GBDT 게이트 | `game_type` 문자열 == `"R"` |
| 반올림 | `np.rint` = round-half-to-even |
| hand 도메인 | {1, 2} |
| 컨텍스트 개수 | **24** = 12 × 2, 정렬 키 count 바깥 / hand 안쪽 |
| exact/physical 보정 클립 (미사용 후보) | **±0.03** |

### 8.5 LightGBM 모델 제원 (모델 파일 파라미터 블록 실측 전재)

| 파라미터 | 값 |
|---|---|
| boosting / objective / metric | gbdt / regression(L2) / l2 |
| num_iterations | **300** |
| learning_rate | **0.015** |
| num_leaves | **63** |
| min_data_in_leaf | **1000** |
| max_depth | −1 (무제한) |
| max_bin | 255 (min_data_in_bin 3, bin_construct_sample_cnt 200000) |
| bagging_fraction / bagging_freq | **0.85 / 1** |
| feature_fraction / bynode | **0.85 / 1** |
| lambda_l1 / lambda_l2 | **0.5 / 8** |
| min_sum_hessian_in_leaf | 0.001 |
| min_gain_to_split | 0 |
| seed | **42** |
| 파생 시드 | bagging 400, feature_fraction 30056, data_random 175, extra 12879, drop 17869, objective **16083** |
| deterministic / force_col_wise | 1 / 1 |
| num_threads (저장값) | 22 |
| early_stopping_round | **0** (조기중단 없음) |
| boost_from_average | 1 |
| use_missing / zero_as_missing | 1 / 0 |
| verbosity | −1 |
| linear_tree / extra_trees | 0 / 0 |
| 파일 제원 | v4, num_class 1, num_tree_per_iteration 1, max_feature_idx 185, 트리 300 |
| metadata 라벨 | `rfull_l63_m1000_i300`, 학습 소요 200.66391348838806초 |

미채택 LGB 후보(검증 실험, 동일 하이퍼 공유): `rfull_l15_m3000_i200`(200트리·15리프·min3000), `rfull_l31_m2000_i300`(300·31·2000).

### 8.6 HistGradientBoosting 제원

| 파라미터 | 값 |
|---|---|
| loss | squared_error |
| learning_rate | **0.025** |
| max_iter | **160** |
| max_leaf_nodes | **15** |
| max_depth | **4** |
| min_samples_leaf | **3000** |
| l2_regularization | **30.0** |
| max_features | **0.70** |
| max_bins | **127** |
| early_stopping | **False** |
| random_state | **42** |
| 피처 폭 / 트리 수 | 84 / 160 |
| format / baseline | `numeric_hgb_v1` / 4.8898374092703725e-09 |
| 총 노드 / 트리별 노드 | 4,416 / {19,21,23,25,27,29} |
| 리프 수 / 최대 깊이 | 10~15 / 4 |
| 분기 피처 인덱스 | 0~82 (70종) |
| 리프값 범위 | [−0.0015397169887587743, +0.001466065840579519] |
| 비리프 `value ≠ 0` | 1,968개 |
| metadata 라벨 | `hist_l15_d4_m3000_i160`, 학습 소요 63.08717656135559초 |

미채택 HGB 후보: max_iter 120 / lr 0.035 / leaf 7 / depth 3 / min_samples_leaf 5000 / l2 20.0 / max_bins 63 / max_features 0.70.

### 8.7 피처 개수 요약

정적 파생 21 / temporal 45 (=1 + 21×2 + 2) / multirate 69 (=33+15+21) / 원본 통과 39 (=20+19) / 원-핫 12.
LGB 186 = 20+19+21+45+69+12. HGB 84 = 20+14+21+17+12.

### 8.8 상위 블렌드 래퍼 상수 (영역 밖, 참고)

champion 0.6434428305247574 / exp021 0.35655716947524263 / measured_K 171.5001496146865 / probe_weight 0.45 / probe_score 1113.1142204091 / champion_score 1092.808353586 / exp021_endpoint_score 1043.6074197937 / projected_score 1114.611684697336.

---

## 9. 검증 절차 (5행 정답값 대조)

### 9.1 입력 요약 (로컬 `data/test.csv`, 5행)

| # | row_id | season | pit | bat | p_team | b_team | p_hand | b_hand | B/S/O | li | HWE/AWE | base_state | top_bottom | game_type | count_index | ctx_pos |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TEST_000001 | 2025 | 21813 | 23143 | 16 | 21 | 2 | 2 | 1/0/2 | 0.34 | 29.7/70.3 | `___` | T | R | 4 | 7 |
| 2 | TEST_000017 | 2025 | 24745 | 22026 | 14 | 16 | 1 | 2 | 2/0/1 | 1.29 | 21.4/78.6 | `1__` | B | R | 8 | 13 |
| 3 | TEST_000213 | 2025 | 24198 | 24790 | 19 | 14 | 2 | 2 | 0/0/2 | 0.83 | 41.8/58.2 | `1__` | B | R | 0 | 1 |
| 4 | TEST_005332 | 2025 | 24713 | 23646 | 19 | 16 | 1 | 2 | 0/0/0 | 0.87 | 54.8/45.2 | `___` | B | R | 0 | 1 |
| 5 | TEST_035185 | 2025 | 24713 | 24742 | 19 | 16 | 1 | 2 | 0/0/0 | 0.92 | 55.1/44.9 | `___` | B | R | 0 | 1 |

asof 요약: pitcher_n = 3465 / 86 / 486 / 0 / 9, pitcher_success_rate = 0.489466 / 0.290698 / 0.475309 / 결측 / 0.444444, batter_n = 1999 / 11153 / 0 / 12645 / 0, batter_success_rate = 0.506753 / 0.520308 / 결측 / 0.519968 / 결측.
상태 테이블 조회: 투수 21813→(3085,1560), 24198→(87,53), **24745·24713 미등록**. 타자 23143→(1565,812), 22026→(11099,5782), 23646→(12645,6575), **24790·24742 미등록**.

### 9.2 단계별 골든 트레이스 (가장 강력한 회귀 테스트)

| 단계 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| `temporal_base_global_30` | 0.3964788615703583 | 0.3659464716911316 | 0.4600697159767151 | **0.48610490560531616** | 0.4793751835823059 |
| `temporal_base_player_30` | 0.39798325300216675 | 0.3696135878562927 | 0.46189624071121216 | 0.49610579013824463 | 0.4793751835823059 |
| $g^{\text{base}}$ | −0.01395816 | +0.02434253 | −0.00728898 | +0.02564826 | +0.02564826 |
| $g^{\text{rev}}$ | −0.01338544 | +0.00691927 | −0.01607533 | −0.01434262 | +0.03361667 |
| $e^{\text{group}}$ | −0.01378634315692847 | +0.019115549421999518 | −0.009924881436229549 | +0.013650997279894604 | +0.028038784515518195 |
| $b^{\text{grp}}$ | 0.3826925184134298 | 0.3850620211131311 | 0.45014483454048554 | 0.49975590288521077 | 0.5074139680978241 |
| $r^{\text{lgb}}$ | +0.0072841573108296765 | −0.013183374123506144 | +0.0005115927451147388 | −0.0014069268187024686 | +0.011785380634986445 |
| $r^{\text{hgb}}$ | −0.006911080337004893 | −0.008684118732940005 | +0.002675139383738191 | +0.003681785353099663 | +0.0064909414992134935 |
| `is_R` | True | True | True | True | True |
| $B^{\text{lgb}}$ (×0.75) | 0.3881556363965521 | 0.3751744905205015 | 0.4505285290993216 | 0.4987007077711839 | 0.516253003574064 |
| $B^{\text{hgb}}$ (×1.00) | 0.37578143807642495 | 0.3763779023801911 | 0.45281997392422374 | 0.5034376882383105 | 0.5139049095970376 |
| $b^{\text{bone}}$ | 0.3819685372364885 | 0.3757761964503463 | 0.45167425151177265 | 0.5010691980047473 | 0.5150789565855508 |
| $e^{\text{team}}$ | −0.0041630263 | −0.0028596313 | −0.0057933344 | +0.0011629273 | +0.0011629273 |
| $b^{\text{team}}$ | 0.37780551094337195 | 0.3729165651008432 | 0.4458809170646161 | 0.5022321253158382 | 0.5162418838966417 |
| $e^{\text{low}}$ | +0.0032878746 | 0.0 | −0.0017221809 | 0.0 | 0.0 |
| **최종 $p$** | **0.38109338550902455** | **0.3729165651008432** | **0.44415873620082535** | **0.5022321253158382** | **0.5162418838966417** |

평균 **0.4433285392046346**. 과제 제시 정답과 최대 절대오차 **5.551e-17**(부동소수점 최종자리) — 사양의 데이터 흐름이 정확함이 실증되었다.

$e^{\text{low}}$ 시즌별 기여(분모는 항상 4):

| source | 매칭 (r1..r5) | $v_s$ |
|---|---|---|
| 2021 | T,F,F,F,F | [+0.0030952956, 0, 0, 0, 0] |
| 2022 | F,F,T,F,F | [0, 0, −0.0068887235, 0, 0] |
| 2023 | T,F,F,F,F | [−0.0005732847, 0, 0, 0, 0] |
| 2024 | T,F,F,F,F | [+0.0106294874, 0, 0, 0, 0] |

(참고, 배포 미사용) $\Delta_{pc}$ = [+0.0016426684, 0.0, −0.0011230621, 0.0, 0.0].

### 9.3 이 5행이 잡아내는 함정

- **r4 의 $b^{\text{tmp}}$ 가 리그 상수 0.4861049201797189 과 정확히 일치** → 투수·타자 모두 $n^{se}=0$ 인 콜드스타트. temporal fallback 을 잘못 구현하면 여기서 즉시 틀린다.
- **r4 의 타자 23646 은 $C_n = P_n = 12645$** → $n^{se}=0$ 이므로 `season_global_30 = L = 0.48610…` 이지만 `season_player_30 = π = 0.5194411…`. 두 값이 갈리는 좋은 검증 포인트.
- **r2/r4/r5 의 $e^{\text{low}} = 0$** → 투수 24745·24713 이 4개 소스 전부에 미등록. NaN 이나 평균 폴백이 아니라 **정확히 0 기여**여야 한다.
- **r3 은 투수 24198 이 2022 소스에만 존재** → 원 효과 −0.00688…이 1/4 로 축소되어 −0.00172…. 분모를 매칭 시즌 수(1)로 바꾸면 4배 틀린다.
- **r4/r5 의 $g^{\text{base}}$ 동일** → 같은 (count_index, pitcher_hand, batter_hand) 셀.
- **r4 의 multirate 는 $G$ 로, temporal 은 $L$ 로 떨어진다** → 두 fallback 혼용 여부를 잡아낸다.
- 5행 전부 `game_type == "R"` 이므로 **R/F 게이팅 분기는 이 샘플로 검증되지 않는다**(로직은 반드시 유지).

### 9.4 보조 체크포인트

**행렬 NaN 개수** (강력한 피처 레이어 회귀 테스트):

| row | LGB 186열 | HGB 84열 |
|---|---|---|
| 1 | 0 | 0 |
| 2 | 9 | 9 |
| 3 | 3 | 2 |
| 4 | 18 | 17 |
| 5 | 12 | 11 |

내역: r2 = prev-game 6 + delta 3. r3 = `asof_batter_success_rate` + `asof_batter_middle_rate` + gap (HGB 는 success_rate 제외 → 2). r4 = 투수 asof 비율 5 + prev 6 + pitchmix 3 + gap 1 + delta 3 (HGB 는 `asof_pitcher_success_rate` 제외 → 17). r5 = prev 6 + delta 3 + batter 2 + gap 1.

**정적 파생 골든 값**:

| 열 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| `count_index` | 4 | 8 | 0 | 0 | 0 |
| `count_out_index` | 14 | 25 | 2 | 0 | 0 |
| `count_advantage` | −1 | −2 | 0 | 0 | 0 |
| `same_hand` | 1 | 0 | 1 | 0 | 0 |
| `close_game` | 0 | 0 | 1 | 1 | 1 |
| `log_li` | 0.2926696240901947 | 0.8285518288612366 | 0.604315996170044 | 0.6259384155273438 | 0.6523252129554749 |
| `score_pressure` | 0.5853392481803894 | 2.4856555461883545 | 0.604315996170044 | 0.0 | 0.0 |
| `win_expectancy_gap` | −40.599998474121094 | −57.20000076293945 | −16.399999618530273 | 9.600000381469727 | 10.199999809265137 |
| `pitcher_batter_success_gap` | −0.01728699915111065 | −0.22960999608039856 | NaN | NaN | NaN |
| `log_pitcher_n` | 8.1507568359375 | 4.465908050537109 | 6.188263893127441 | 0.0 | 2.3025851249694824 |
| `log_batter_n` | 7.600902557373047 | 9.31955337524414 | 0.0 | 9.445096015930176 | 0.0 |

**temporal 주요 열** (r1 pitcher 수동 검산이 가능한 형태):

| 열 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| `temporal_prior_league_rate` | 0.48610490560531616 (5행 동일) | | | | |
| `temporal_pitcher_prior_exists` | 1 | 0 | 1 | 0 | 0 |
| `temporal_pitcher_prior_n` | 3085.0 | 0.0 | 87.0 | 0.0 | 0.0 |
| `temporal_pitcher_prior_rate` | 0.5056726336479187 | 0.48610490560531616 | 0.6091954112052917 | 0.48610490560531616 | 0.48610490560531616 |
| `temporal_pitcher_prior_rate_shrunk_200` | 0.5044812560081482 | 0.48610490560531616 | 0.5234180688858032 | 0.48610490560531616 | 0.48610490560531616 |
| `temporal_pitcher_season_n` | 380.0 | 86.0 | 399.0 | 0.0 | 9.0 |
| `temporal_pitcher_season_rate` | 0.35789474844932556 | 0.2906976640224457 | 0.44611528515815735 | 0.48610490560531616 | 0.4444444477558136 |
| `temporal_pitcher_season_global_30` | 0.36727598309516907 | 0.341234028339386 | 0.44891175627708435 | 0.48610490560531616 | 0.47649097442626953 |
| `temporal_pitcher_season_player_30` | 0.3686205744743347 | 0.341234028339386 | 0.4515210688114166 | 0.48610490560531616 | 0.47649097442626953 |
| `temporal_pitcher_reliability_30` | 0.9268292784690857 | 0.7413793206214905 | 0.9300699234008789 | 0.0 | 0.23076923191547394 |
| `temporal_batter_prior_n` | 1565.0 | 11099.0 | 0.0 | 12645.0 | 0.0 |
| `temporal_batter_season_n` | 434.0 | 54.0 | 0.0 | 0.0 | 0.0 |
| `temporal_batter_season_rate` | 0.46313363313674927 | 0.3888888955116272 | 0.48610490560531616 | 0.48610490560531616 | 0.48610490560531616 |
| `temporal_batter_season_global_30` | 0.46461886167526245 | 0.4236088991165161 | 0.48610490560531616 | 0.48610490560531616 | 0.48610490560531616 |
| `temporal_batter_season_player_30` | 0.466496080160141 | 0.43583253026008606 | 0.48610490560531616 | 0.5194411277770996 | 0.48610490560531616 |

r1 pitcher 수동 검산: $P_n{=}3085, P_s{=}1560 \Rightarrow \rho^{pr}=0.50567260$ ✓ / $C_n{=}3465$, $C_s = \mathrm{rint}(3465\times0.489466)=\mathrm{rint}(1696.00\ldots)=1696$ / $n^{se}=380$, $S^{se}=136$ / $\rho^{se}=136/380=0.35789474$ ✓ / $\pi=(1560+200\times0.4861049201797189)/3285=0.50448126$ ✓ / `season_global_30` $=(136+30\times L)/410=0.36727598$ ✓ / `reliability_30` $=380/410=0.92682927$ ✓.

**multirate 주요 열**:

| 열 | r1 | r2 | r3 | r4 | r5 |
|---|---|---|---|---|---|
| `multirate_pitcher_control_season_n` | 380.0 | 86.0 | 399.0 | 0.0 | 9.0 |
| `..._success_prior_rate` | 0.5056726336479187 | 0.5237659811973572 | 0.6091954112052917 | 0.5237659811973572 | 0.5237659811973572 |
| `..._success_prior_shrunk_200` | 0.5067741870880127 | 0.5237659811973572 | 0.549662709236145 | 0.5237659811973572 | 0.5237659811973572 |
| `..._success_season_rate` | 0.35789474844932556 | 0.2906976640224457 | 0.44611528515815735 | 0.5237659811973572 | 0.4444444477558136 |
| `..._success_season_global_30` | 0.3700316548347473 | 0.3509739637374878 | 0.4515454173088074 | 0.5237659811973572 | 0.5054609775543213 |
| `..._success_season_minus_prior` | −0.14777787029743195 | −0.2330683022737503 | −0.1630801111459732 | 0.0 | −0.07932153344154358 |
| `..._ball_season_rate` | 0.371052622795105 | 0.5 | 0.378446102142334 | 0.36937764286994934 | 0.3333333432674408 |
| `multirate_batter_control_season_n` | 434.0 | 54.0 | 0.0 | 0.0 | 0.0 |
| `..._middle_prior_rate` | 0.13354632258415222 | 0.15271645784378052 | 0.1495167762041092 | 0.15800711512565613 | 0.1495167762041092 |
| `multirate_pitcher_pitchmix_fastball_season_rate` | 0.7184210419654846 | 0.5348837375640869 | 0.38596490025520325 | 0.5411296486854553 | 0.6666666865348816 |

검산: r2 `ball_season_rate` — $C_n{=}86$, rate 0.5 → $c^{car}=\mathrm{rint}(43.0)=43$, $P_m{=}0$, $n^{se}{=}86$ → 43/86 = 0.5 ✓.

**원-핫 (5행)**: r1 = `top_bottom_T`, `base_state____` / r2·r3 = `top_bottom_B`, `base_state_1__` / r4·r5 = `top_bottom_B`, `base_state____`. 나머지 10열은 5행 모두 0.

### 9.5 검증 순서 권장

1. §9.4 NaN 개수 표 + `temporal_base_global_30` 5값 → 피처 레이어 통과.
2. §9.2 의 $b^{\text{grp}}$ → 그룹 효과 통과.
3. $r^{\text{lgb}}$, $r^{\text{hgb}}$ → GBDT 로딩·순회 통과 (HGB 가 정확히 0.025배로 작으면 §6.6-1 함정).
4. $b^{\text{bone}}$ → 게이트·가중치·클립 순서 통과.
5. $e^{\text{team}}$, $e^{\text{low}}$ → 조회·시즌 평균 통과.
6. 최종 $p$ 5값 일치.
7. **행 독립 회귀 테스트**: 5행짜리 배치와 대량 배치에서 같은 행이 정확히 같은 피처·같은 예측을 내는지 확인.

### 9.6 재현 실패를 부르는 함정 체크리스트

1. `temporal_base_*` 를 float64 산술로 계산 (→ 3.0e−8 오차, 16자리 재현 실패).
2. `rint` 를 `floor(x+0.5)` 로 대체.
3. HGB 리프값에 학습률 0.025 재적용 (→ 예측이 0.025배).
4. HGB 내부 노드 value 누적 (→ 완전히 다른 값).
5. NaN 을 0이나 평균으로 대치 (→ 두 GBDT 모두 크게 틀림).
6. LGB 가중치를 1.00, HGB 를 0.75 로 뒤바꿈.
7. 50:50 평균 전 클립 생략, 또는 평균 후에만 클립.
8. 피처를 이름 기준으로 재정렬 (LGB 는 `Column_k` 일반명이라 **오류 없이 조용히 틀린다**).
9. 더미 열 순서를 ASCII 오름차순 + `_nan` 마지막이 아닌 순서로 생성.
10. `season` 이나 `game_type_*` 더미를 모델 입력에 포함.
11. 팀/저랭크 시즌 평균 분모를 매칭 시즌 수로 변경.
12. 미등록 키를 0 이 아닌 평균 등으로 폴백.
13. 추론 `group_base` 를 학습타깃용 OOF 그룹 예측으로 계산 (→ 백본 전체가 어긋남).
14. CSV 를 `utf-8-sig` 가 아닌 인코딩으로 읽어 첫 컬럼명에 BOM 잔류.
15. temporal 과 multirate 의 fallback($L$ vs $G$)을 혼용.

---

## 10. 불확실 항목과 확인 방법 / 영역 간 모순 정리

### 10.1 영역 간 모순 (본 통합 과정에서 실측으로 판정)

| # | 모순 | 판정 (실측 근거) |
|---|---|---|
| M1 | `team_effects.pitcher_team` 2024 레코드 수: 영역A "44" vs 영역C "48" | **48 이 맞다.** 파일 직접 카운트: pitcher_team [44,44,44,**48**], batter_team [44,44,44,**48**]. 영역A 표가 오기. |
| M2 | HGB 스키마 인덱스: 영역D 2열 표가 off-by-one (`log_pitchmix_n` 을 55로 표기) | **영역A/실측 리스트가 맞다.** 실측: 54=`log_pitchmix_n`, 55=`temporal_prior_league_rate`, 70/71=`temporal_base_global_30`/`_player_30`. §6.3 이 정본. |
| M3 | LGB 학습 하이퍼파라미터: 영역A는 "확인 못함(불확실)", 영역D는 전체 제시 | **해소.** 모델 파일 `parameters:` 블록 실측 전재(§8.5). learning_rate 0.015 등 확정. |
| M4 | `pitcher_count_effects.smoothing`: 영역A "미확인" | **해소: 600.0** (파일 실측). 단 strict 후보에서 미사용. |
| M5 | `temporal_base_global_30` 정밀도: 영역A "float32 반올림 후 승격"만 언급 vs 영역B "float32 산술 필수" | **영역B 채택.** B가 두 경로의 실측 차이(3.0e−8)를 제시했고 A의 골든값과도 일치. §3.3.1. |
| M6 | 클립 횟수 표기: 영역A 본문 "총 4회… 정확히는 5개 위치" | **5지점으로 통일**(§7.2). |
| M7 | 영역C §7 표의 r4 `ctx_pos` 를 3으로 적었다가 본문에서 1로 정정 | **1 이 맞다**(count_index 0 · batter_hand 2 → position 1). 제시된 $\Delta$ 값들은 매핑을 그대로 쓴 계산이라 영향 없음. |
| M8 | reverse bin 결측 판정: 학습 스크립트는 `notna()`, 추론은 `isfinite()` | NaN 에 대해 동치. `±inf` 가 있으면 갈린다. **재구현은 추론 기준 `isfinite` 를 따른다.** |

### 10.2 불확실 항목 (추측하지 않고 표시)

| # | 불확실 항목 | 재현 영향 | 확인 방법 |
|---|---|---|---|
| U1 | **R행 정확한 개수.** 산출물·스크립트에 기록 없음. LGB tree 0 루트 `internal_count = 1,117,198` 에서 bagging 0.85 역산 → **약 1,314,350 (추정)** | 없음(추론은 동결 모델 사용) | train.csv 에서 `game_type == "R"` 카운트 |
| U2 | **`asof_*` 열의 공식 정의.** 코드가 전제하는 것은 "career 누적, 시즌 리셋 없음, 단조증가, 현재 투구 미포함". 상태 테이블 `prior_n` 합 = 학습 행 수(1,475,092)로 간접 검증됨 | 전제가 깨지면 가드 G3 가 예외를 던져 추론 실패 | 대회 데이터 스펙 문서 / train.csv 에서 선수별 `asof_n` 단조성 확인 |
| U3 | **`pitcher_control` 의 ball/strike 카운트 분할 의미.** 샘플(id 22548: ball 104 + strike 111 = 215 ≠ 248)로 보아 상호배타 완전분할이 아님 | 없음(저장된 숫자를 그대로 사용) | 대회 타깃 정의 문서 |
| U4 | **재학습 비트 단위 재현성.** 산출물은 py3.12.6/np2.5.1/pd3.0.5/sk1.9.0 에서 생성. sklearn 버전이 다르면 HGB 비닝·`max_features` RNG 소비가 달라질 수 있음. LGB `deterministic=True` 의 스레드 수 독립성도 실측 미검증(저장 `num_threads=22`) | 없음(산출물 재사용) | 동일 버전 환경에서 재학습 후 MD5 비교 |
| U5 | **`max_features` 가 어느 sklearn 버전부터 있는지** (1.4 전후로 추정, **불확실**) | 재학습 시에만 문제. 추론은 sklearn 불필요 | sklearn 릴리스 노트 |
| U6 | **HGB 리프값에 학습률이 이미 반영되어 있다는 판단**은 간접 근거 3가지에 기반: (a) 원 빌드의 1e-12 패리티 단언, (b) 리프값 최대 0.00154 가 lr 0.025 반영 스케일과 부합, (c) 5행 재현 일치 | 결론(다시 곱하지 말 것)은 확실 | sklearn 내부 구현 확인 또는 재학습 후 패리티 테스트 |
| U7 | **`base_state` 어휘 8종의 완전성.** 스키마에 8종이 있으나 train.csv 전수 확인 안 함 | 없음(어휘 밖 값은 12열 0 → 원문 동작 동일) | train.csv `base_state.unique()` |
| U8 | **multirate 에 성공수 가드(G4 상당)가 없는 것**이 의도인지 누락인지 코드만으로 판단 불가 | 없음(원본 동작 그대로 기술) | 원 저자 확인 |
| U9 | **시간 가드 `.any()` 전역 검사가 규칙상 문제인지** — 예측에 관여하지 않는 순수 입력 검증이라 실질 위험 없다고 판단했으나 이는 규칙 해석 영역 | 없음(행별 검사 권고로 회피) | 대회 규칙 문의 |
| U10 | **`pitcher_hand`/`batter_hand` 코드 의미**(1=우완/2=좌완 등) 미확인 | 없음(정수 코드를 키로 사용) | 데이터 딕셔너리 |
| U11 | **저랭크 $\hat A$ 가 정말 k=300 포화 행렬에서 나왔는지** 직접 재현 못 함. 저장 메타데이터(`smoothing: 300.0`, `rank: 6`)와 σ₇ 이후가 0(랭크 6)이라는 실측으로만 뒷받침 | 없음(추론은 $\hat A$ 를 그대로 읽음) | train.csv + OOF 로 재학습 |
| U12 | **`source_weights` 키의 실제 값.** `lowrank_recency_effects.json`(우리 model/ 에 없음)에서 쓰였을 것으로 보이나 미확인 | 없음(우리 산출물에 키 없음) | 참조 저장소의 recency 산출물 확인 |
| U13 | **실전 245,789행에서만 나타날 케이스** — 미등록 team_id, contexts 룩업 KeyError, 가드 발동, 비-R 행 존재 여부. 로컬 5행으로는 검증 불가. 특히 **저랭크 contexts 룩업은 24격자 밖이면 KeyError 로 죽는다** — 방어적 기본값(0 기여)을 넣을지 원본처럼 예외로 둘지는 **판단 보류** | 실행 시 예외 위험 | train.csv 로 대량 스모크 테스트(§9.5-7)를 돌려 가드/룩업 예외가 없음을 확인 |
| U14 | **상위 블렌드 래퍼가 champion/exp021 두 컴포넌트의 row_id 배열이 완전히 동일 순서일 것을 요구**(불일치 시 ValueError). 이 제약이 EXP-021 재작성에 추가 요구를 거는지 미검토 | 래퍼 통합 시 확인 필요 | `script.py` 검토 |
| U15 | **pitcher×count 적용식(§4.6)은 "코드에서 읽은 정의"이지 "정답으로 확인된 경로"가 아니다** (strict 후보에서 미실행이라 5행 정답으로 검증되지 않음) | 없음(구현 대상 아님) | aggressive 후보 산출물로 별도 검증 |
| U16 | `train_exp019_histgb_residual` 의 피처 행렬 출처 함수(`train_exp017_rolling_residual` 의 데이터 준비부)는 **읽지 않았다**. 배포 빌드 규칙은 완전히 확인했고 결과 폭이 84로 일치 | 없음 | 해당 스크립트 정독 |

### 10.3 재작성 최종 체크리스트

- [ ] `utf-8-sig` 로 두 CSV 읽기 + 입력 검증 5종.
- [ ] 시간 가드 2곳(행별로 재작성) + 재구성 가드 3종, 허용오차 −1e-6 / −0.01 / +0.01.
- [ ] 정적 21 + temporal 45 + multirate 69 + 더미 12를 §6.2/§6.3 이름표 그대로 생성(오타 하나면 `ValueError`).
- [ ] 수축 상수: 선수 prior **200.0** 공통, temporal 강도 **10/30/100/300**, multirate 강도 **30** 만.
- [ ] `temporal_base_*` 는 **float32 산술**, `rint` 는 **half-to-even**, GBDT 입력은 **float32 + NaN 보존**.
- [ ] 결합 가중치: temporal base 0.7/0.3, group 0.7/0.3, LGB 0.75 / HGB 1.0, backbone 0.5/0.5, team 패밀리 0.5/0.5 × 시즌 등가중 4, lowrank 등가중 4.
- [ ] reverse bin 폭 **0.05**, 결측 bin **−1**, 판정은 `isfinite`.
- [ ] `game_type == "R"` 게이트가 GBDT 잔차 **두 개 모두**에 걸린다.
- [ ] 클립 [0,1] 이 정확히 5지점(그룹 후 / 각 가지 안 / 팀 후 / 저랭크 후), 평균 직후엔 없음.
- [ ] strict 경로 외 8개 후보 분기와 5개 매퍼를 전부 삭제, `pitcher_count_effects.json` 로드도 제거.
- [ ] 출력: sample_submission 의 row_id 순서, 컬럼 `[row_id, control_success]`, `index=False`, `encoding="utf-8"`, `./output/submission.csv`, 디렉터리 자동 생성, NaN·범위 검증.
- [ ] §9.2 골든 트레이스 전 단계 재현 → 최종 5값 일치.
- [ ] 대량 배치 행 독립 회귀 테스트 + 10분 추론 시간 여유 확인.
