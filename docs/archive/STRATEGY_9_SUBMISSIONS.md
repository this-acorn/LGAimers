# 남은 9회로 +100을 노리는 전략

최종 갱신: 2026-08-30 PDT / 2026-08-31 KST

## 결론부터

현재 챔피언 1092.808에서 +100의 목표점은 **1192.808**이다. 지금 장부의 +1~10 후보를
모두 성공시켜도 도달하지 못한다. +100은 대략 Brier MSE `0.00025`를 더 줄여야 하므로,
다음 셋 중 하나가 한 번 터져야 한다.

1. 현 챔피언과 오류가 매우 다른 고품질 독립모델.
2. 2025의 구조변화를 정확히 맞힌 행 단위 expert.
3. 운영진이 허용한 새 공식 정보원의 복원.

따라서 제출 9회는 작은 후보 9개를 시험하는 예산이 아니라, **독립 endpoint의 오류 직교성을
측정하고 최적 조합을 확정하는 예산**으로 쓴다.

## 왜 평범한 앙상블로는 안 되는가

후보 B의 단독점수 차이를 `d=S_B-S_A`, 챔피언 A와의 다양성을 `K`라 하면 B 비중 `w`의 점수는

```text
S(w) = S_A + d·w + K·w(1-w)
w*   = clip((d+K)/(2K), 0, 1)
gain = (d+K)^2/(4K),  K>-d일 때만
```

이다. 같은 계열에서 관측된 `K≈26`이면 동급 모델 두 개의 최대 이득도 `K/4≈6.5`다.
지금까지 큰 쪽인 `K≈117.5`조차 동급일 때 최대 +29.4다. +100은 “조금 다른 CatBoost”가
아니라 `K≈500`급의 완전히 다른 오류가 필요하다.

## 문샷 1 — mk-isos EXP-021 전체 백본

공개 EXP-021 strict는 동일 대회에서 **1043.6074197937**을 받았다. 우리 챔피언 대비
`d=-49.200934`다. 단독으로는 낮지만 함수계열이 다르다.

- R행 LightGBM residual.
- HistGradientBoosting residual과 50:50.
- source-season Team EB.
- 투수별 24개 `count×batter_hand` residual을 rank-6 SVD로 공유.
- affine 없이 행 독립 추론.

외부팀의 935.81→1043.61(+107.80)은 이 전체 composite의 효과이며, rank-6 하나의 +108이 아니다.
우리가 묻는 질문은 “그 모델이 더 좋은가”가 아니라 **“그 모델의 오차가 우리와 충분히 다른가”**다.

### +100에 필요한 K

| 목표 혼합 이득 | 필요한 `K` | 의미 |
|---:|---:|---|
| +10 | 117.86 | 관측된 기존 최대 K와 비슷 |
| +15 | 141.27 | 기록 갱신 가능 |
| +20 | 163.61 | 제출 2회 사용 최소선 |
| +30 | 206.69 | 의미 있는 독립 파트너 |
| +50 | 290.06 | 문샷 |
| **+100** | **493.50** | `w_EXP021≈0.450` |

### 제출 전에 하는 무료 게이트

공개 repo의 누락 OOF를 아래 순서로 재현한다. 최종 ZIP은 아직 만들지 않는다.

1. `train_exp019_r_full_residual.py`
2. `train_exp019_histgb_residual.py`
3. `train_exp019_team_eb_ensemble.py`
4. `train_exp020_pitcher_count_eb_atop_team.py`
5. `train_exp020_low_rank_pitcher_context_eb.py`

원 기록 실행시간은 합계 약 20분이고 이 PC 예상은 25~45분이다. 2022/2023/2024 OOF의 row order와
target equality를 먼저 확인한 뒤 현 frozen champion OOF와 `d`, `K`, `w*`, gain을 계산한다.

LB 1회 사용 gate:

- 세 폴드 중 2개 이상에서 analytic blend gain ≥+20.
- 보수적 `K_2025` ≥150.
- exact 외부 OOF metric 재현.
- 50:50 결합 bundle의 singleton/permutation QA와 runtime <600초.

gate를 통과하면 endpoint 재제출은 생략한다. 외부 direct LB 1043.607이 이미 있으므로
`champion×EXP021` 50:50 한 번으로

```text
K = 4 × [S50 − 0.5 × (1092.808353586 + 1043.6074197937)]
```

를 계산한다.

판정:

- `K<117.9`: 기대 이득 +10 미만, 즉시 종료.
- `117.9≤K<163.6`: 기록 갱신용일 뿐, +100 트랙에서는 종료.
- `K≥163.6`: 계산된 `w*`를 두 번째 슬롯에 제출.
- `K≥290.1`: +50 문샷 생존.
- `K≥493.5`: 이 한 축만으로 +100 경로 성립.

## 문샷 2 — 두 번째 독립 endpoint

EXP-021의 `K`가 충분하지 않을 때 같은 모델의 작은 변형은 쓰지 않는다. 다음 두 후보만 허용한다.

### 2-A. mk-isos EXP-071 exact 재현

외부 direct LB는 1053.8615519684, 챔피언 대비 `d=-38.946802`다. 모델 파일이 Git에 없어 재학습
경로를 복원해야 한다. +100에 필요한 `K`는 **474.70**, +30은 189.91, +15는 125.84다.
EXP-021과 OOF correction 상관이 너무 높으면 같은 endpoint로 보고 제출하지 않는다.

### 2-B. 2023 이후 F-regime 보호형 expert

raw F 모델을 다시 학습하는 것이 아니다. 챔피언의 R 예측을 완전히 고정하고, 과거 OOF residual에서
학습한 강수축 correction을 F행에만 적용한다. public recent-F 계열이 실패했으므로 gate를 높게 둔다.

- discovery에서 구조·수축·weight를 잠금.
- untouched confirmation에서 global-equivalent raw ≥+35.
- equal-mean shape ≥+25.
- pitcher cluster bootstrap 2.5% >0.
- F 개선이 R의 0 correction을 훼손하지 않음.

이 조건을 못 넘으면 F/R offset 같은 소형 보정으로 축소하지 않고 문샷 트랙에서 종료한다.

## 문샷 3 — 허가된 새 정보원

`row_id`가 날짜·경기·투구순번을 결정적으로 암호화하는지 train에서 감사하는 것은 가능하다.
배포 조건은 모두 필요하다.

1. 운영진의 **구체적 서면 허용**.
2. train에서 99.9% 이상 deterministic decode.
3. 평가 행 하나만으로 계산.
4. strict next-season raw ≥+30, shape ≥+20.

하나라도 없으면 제출 슬롯을 0회 쓴다. test 행 정렬·차분·누적은 어떤 답변과도 별개로 금지다.

## 9-slot adaptive 표

아래는 예약표다. 앞 gate가 실패하면 다음 칸을 억지로 소진하지 않는다.

| 슬롯 | 질문 | 실행 조건 | 결과에 따른 행동 |
|---:|---|---|---|
| 1 | champion×EXP021 50:50 | exact OOF gate 통과 | `K` 계산 |
| 2 | EXP021 analytic `w*` | `K≥163.6` | `K≥493.5`면 +100 주력 |
| 3 | EXP071 또는 F expert endpoint | strict high gate | endpoint가 +20 또는 예상 blend +20이어야 생존 |
| 4 | slot3×champion 50:50 | endpoint 생존 | 두 번째 `K` 계산 |
| 5 | slot3 analytic `w*` | 측정 gain ≥+20 | 최적 2-way 후보 확정 |
| 6 | 승인된 row-id/new-info endpoint | 서면 허용+strict gate | 아니면 사용 안 함 |
| 7 | new-info×champion 50:50 | endpoint 생존 | 세 번째 `K` 계산 |
| 8 | 최상 endpoint 3-way blend | 모든 pair에서 `d+K>0`, 예상 +15 이상 | 닫힌 식/작은 simplex로 비율 고정 |
| 9 | 최종 QA/실행오류 reserve | SHA·환경·행독립 확인 | 최종 winner 또는 미사용 |

## 소형 후보의 위치

아래는 +100 카드가 아니라 문샷이 잡힌 뒤 마지막 결합용이다.

| 후보 | 외부/내부 근거 | 우리 기대 |
|---|---|---:|
| joint rank-6 atop champion | 외부 rolling +5.2/+8.9/+19.5 | +0~10 |
| EXP-063 uncertain-window | 외부 direct +4.52 | +0~5 |
| EXP-072 pitcher AR(1) | 외부 direct +5.58 | +0~3 |
| EXP-064 stable-cell | 외부 direct +1.06 | +0~1.5 |
| F/R global offset | 2024 differential oracle +2.13 | +0~5 |

소형 축을 각각 LB에 던지지 않는다. strict temporal에서 독립 양수인 성분만 묶고, 묶은 결과를 다시
검증한 뒤 슬롯 1회를 고려한다.

## 검토했지만 9회 전략에서 뺀 기발한 경로

서로 겹치지 않는 4개 구간에 `±δ`를 제출하면 8회로 각 구간의 LB 잔차 평균을 대수적으로 얻고,
9번째에 최적 결합할 수 있다. 하지만 다음 이유로 채택하지 않는다.

- 2024 현 챔피언에서 4-group 최선의 전역보정 초과 상한은 **+7.61**뿐이었다.
- 12 count 상태까지 정답을 보고 맞춰도 초과 상한은 +10.59였다.
- 일반 모델 혼합 승인보다 훨씬 강한 LB 피드백 사용이라 별도 규정 확인이 필요하다.

즉 슬롯 수에는 정확히 맞지만 +100을 설명하지 못하고 규정 위험만 높다.

## 중단 규칙

- IT300, HT, season-weight bundle, hand delta, T13, rookie expert, FM, NN, TrackMan,
  conditional head는 direct LB 또는 강한 local death이므로 조합으로 재등판시키지 않는다.
- 새 endpoint의 offline global gain이 +20 미만이면 LB 제출 금지.
- +50 근거가 없는 후보는 “진출 카드”가 아니라 “기록 갱신 카드”로 명시한다.
- 서로 다른 endpoint probe가 3개 연속 실패하면 더 이상의 LB 탐색을 중단하고 챔피언을 보존한다.

## 현실적인 확률 판단

EXP-021이 `K≈494`를 보일 사전확률은 낮다. 기존 최대 관측 `K≈117.5`의 네 배가 필요하다.
따라서 이 계획은 +100을 보장하는 계획이 아니라, **9회 안에 +100이 가능한 유일한 수학적 조건을
가장 싼 순서로 확인하고 불가능한 가지를 즉시 자르는 계획**이다.
