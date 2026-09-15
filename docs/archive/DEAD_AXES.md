# 사망 축과 재개 조건

이 문서는 같은 실패를 다시 돌리지 않기 위한 목록이다. “축 전체”라는 표현보다 exact variant와
죽인 범위를 우선한다.

## Direct-LB 사망 — 가장 강한 반증

| exact variant | direct 결과 | 죽인 범위 | 재개 조건 |
|---|---:|---|---|
| hand-delta 67 features | HGB 대비 약 −22.2 | 단순 과거 hand target table 피처 | 보호형 hierarchy처럼 구조가 달라야 함; 그 구조는 V18로 이미 채택 |
| CatBoost 16seed | −0.95 vs 8seed | 동일 모델 시드 수 8→16 | 새 함수공간 없이 재시도 금지 |
| lr.04 + season weight bundle | −27.61 | 두 변경의 결합 | lr 단독 direct 반증으로 오기 금지; local도 개선 없음 |
| TrackMan direct physical | 약 −7.81 | 평균 물리량 직접 피처 | 새 행별 정보원+strict 양수 없으면 금지 |
| IT300 | −19.01 | tree count 하향과 그 위 단순 조합 | 어떤 V18 결합도 제출 금지 |
| scale 1.15 | −3.15 vs s=1 baseline | 해당 scale | 정점 s≈1.06 채택, 재탐색 금지 |

## Strong local/temporal 사망

- target taxonomy: M4, C6, MC7, 9-class 확장.
- categorical: 팀 조합, count×hand/baseout, forced one-hot, CTR1, high-card player categorical.
- learners: depthwise, lossguide, EBM, full ExtraTrees partner, ResMLP/TabM 계열.
- residual: V18 child 7종, 우리 5신호 보호잔차, pooled temporal ExtraTrees adapter.
- cold start: rookie expert, P01 harness, HGB no-pitcher, batter-ID 제거.
- matchup: 직접 pair table, low-rank pitcher-batter FM.
- physical: release consistency, distillation, arrival-region, average physical paths.
- prev-window: denominator 복원 자체는 성공했지만 workload residual은 실패.

상세 수치와 근거는 [EXPERIMENT_LEDGER.md](EXPERIMENT_LEDGER.md)를 본다.

## 죽지 않은 exact 질문

아래는 인접 실험이 실패했지만 exact variant는 남아 있다.

| 질문 | 왜 아직 exact-dead가 아닌가 | 현재 우선순위 |
|---|---|---|
| mk-isos EXP-021 full composite | 우리 데이터/챔피언과의 OOF `K` 미측정 | 최우선 |
| joint pitcher×24 context rank-6 | 단순 pitcher×count child와 구조가 다름 | 2순위 |
| x2 V14 3-domain ET | 공개 코드·OOF 없음, 우리 ET는 proxy | 재현 불가 |
| exact reverse-only head | conditional head만 실행 | 매우 낮음 |
| `exp/99` 파일의 RK arm | 팀 harness 결과는 있으나 그 파일 artifact 없음 | 매우 낮음 |
| full CAT5 no-pitcher | exp/86 결과 미완료 | 매우 낮음 |
| EXP-063/064/072 | 외부 direct 소액 양수, 우리 exact 미실행 | 문샷 뒤 |

## 규정 때문에 닫힌 것

- test 행끼리 집계·정렬·누적·분포 추정.
- 현재 투구의 TrackMan 측정값 사용.
- 외부 데이터/API.
- `row_id` 시간 피처: 운영진이 식별자 외 용도를 서면 허용하기 전.
- LB로 여러 세그먼트의 잔차 평균을 역산하는 calibration tomography: 모델 혼합 승인보다 강한
  행위이므로 별도 서면 승인 전.

## 재제안 체크리스트

사망 축을 다시 열려면 아래 중 하나가 문서에 명시돼야 한다.

1. 기존 실패와 다른 **새 정보원**이 생겼다.
2. 기존 실패가 묻지 않은 **다른 함수공간**이며 차이를 수식으로 쓸 수 있다.
3. 실패가 thread/seed/serve-skew 때문에 무효였다는 재현 증거가 있다.
4. 외부 direct LB 양수와 실행 가능한 코드가 있고, 우리 strict OOF gate를 통과했다.

“조금 다른 hyperparameter”나 “여러 실패축 합치기”는 재개 조건이 아니다.
