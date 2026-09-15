# 분류형 실험 장부

최종 갱신: 2026-08-30 PDT / 2026-08-31 KST

이 문서는 현재 판단용 정본이다. 시간순 세부 과정은 `HANDOFF.md`, 원시 수치는 각 `lab/` 결과를
참조한다. 표의 “사망”은 적힌 **exact variant**에만 적용한다.

## 증거와 상태

| 등급 | 뜻 |
|---|---|
| E0 | 아이디어만 있음 |
| E1 | 데이터 감사·복원 가능성 확인 |
| E2 | 단일시드, 오라클, 사후 진단 |
| E3 | paired 다중시드 local 검증 |
| E4 | 과거 discovery에서 잠근 뒤 다음 연도 confirmation |
| E5 | 우리 팀의 direct Public LB |
| X5 | 외부 팀의 direct Public LB |

우선순위는 `E5 > E4 > E3 > E2 > E1 > E0`이다. E5 음수는 같은 축의 양수 local을 덮는다.

상태는 다음만 사용한다.

- `CHAMPION`: 현재 최고 제출에 포함.
- `ADOPTED`: 챔피언 계보에 포함됐거나 direct LB 양수.
- `LB-DEAD`: direct LB에서 반증.
- `LOCAL-DEAD`: exact local gate 실패.
- `AUDIT`: 데이터 사실만 확인.
- `UNRESOLVED`: exact 질문이 아직 남음.
- `BLOCKED-RULE`: 규정 답변 전 배포 금지.

## 1. 챔피언 계보와 direct LB

| 단계 | 단일 변화 | Public LB | 직전 기준 대비 | 등급 | 판정 |
|---|---|---:|---:|---|---|
| 운영진 RF | baseline | 549.51193 | — | E5 | 기준선 |
| HGB65 | 상황·누적 피처 GBDT | 830.322760105 | +280.81 | E5 | ADOPTED |
| HGB/CB w=.3 | 이종 혼합 | 864.3312059823 | +34.01 | E5 | 역사적 승자 |
| HGB/CB w=.5 | CB 비중 증가 | 880.5655581163 | +16.23 | E5 | 역사적 승자 |
| CatBoost 8seed | CB 단독 | 898.61863054 | +18.05 | E5 | ADOPTED |
| CatBoost 16seed | 시드 수 2배 | 897.6644306949 | −0.95 | E5 | LB-DEAD |
| CS76 | 현 시즌 진행분 복원 | 990.957167531 | +92.34 | E5 | ADOPTED |
| CS79 | 구종 관련 3피처 | 993.6345481775 | +2.68 | E5 | ADOPTED |
| target5 | 5-class 후 P(success) | 1033.9866361712 | +40.35 | E5 | ADOPTED |
| lr.04+season weight | 두 변경 묶음 | 1006.3794404841 | −27.61 | E5 | bundle LB-DEAD |
| team categorical | 팀 ID를 categorical 처리 | 1018.1353 | 약 +24.50 vs CS79 | E5 | ADOPTED |
| target5+team_cat | 두 검증축 결합 | 1059.0501189623 | +25.06 vs target5 | E5 | ADOPTED |
| IT300 | 추론 트리 500→300 | 1040.0366 | −19.01 | E5 | LB-DEAD |
| shift −.012 | 전역 중심 이동 probe | 약 1065.26 | +6.21 | E5 | probe |
| shift −.0066 | 중심 정점 | 약 1076.81 | +17.76 | E5 | ADOPTED |
| scale 1.15 | scale bracket | 약 1073.66 | −3.15 | E5 | probe |
| affine s=1.06 | 중심+scale | 약 1081.67 | +22.62 vs 1059 | E5 | ADOPTED |
| V18 γ=.30 | frozen hand+pressure residual | **1092.808353586** | 약 +11.14 | E5 | **CHAMPION** |

주요 근거: `HANDOFF.md` §1, §1.17, §103~105와 `lab/chain_log.txt`.

## 2. 백본·학습기·앙상블

| exact variant | 질문 | 결과 | 등급 | 상태 | 근거/주의 |
|---|---|---|---|---|---|
| HGB65 | 비선형 tabular 기준선 | LB 830.32 | E5 | ADOPTED | `submit4.zip` |
| CatBoost 8seed | categorical·seed ensemble | LB 898.62 | E5 | ADOPTED | 현재 계보 |
| CatBoost 16seed | 시드 증가 | LB −0.95 | E5 | LB-DEAD | 시드 수 증가 일반론까지 죽이지 않으나 재시도 가치 없음 |
| HGB partner | 현 CB와 혼합 | 2025 endpoint가 CB보다 약 −68, 최적 이득 +.21 | E5 | LB-DEAD | `lab/77_result.txt` |
| XGBoost partner | 이종 다양성 | 독립시드 순증 약 +1 | E3 | LOCAL-DEAD | 진출 카드 아님 |
| LightGBM partner | 이종 다양성 | HGB 시드보다 다양성 작음 | E3 | LOCAL-DEAD | |
| OVA CatBoost | MultiClass 대신 one-vs-all | 기대 혼합 +4 이하 | E3 | LOCAL-DEAD | `lab/78_result.txt` |
| depthwise/lossguide | 트리 성장 정책 | −82.9/−88.1 | E3 | LOCAL-DEAD | `lab/69_result.txt` |
| l2/rsm/min-data 변형 | 정규화 | 무효 또는 음수 | E3 | LOCAL-DEAD | 축별 exact 값은 lab/69 |
| EBM | 강한 이종모델 | `d=-412.7, K=315.2, w*=0` | E3 | LOCAL-DEAD | `lab/83_result.txt` |
| ExtraTrees full partner | no-ID 이종모델 | `d≈-332, K≈446, w*=0` | E3 | LOCAL-DEAD | 완성모델 기준 |
| ResMLP | DL 문샷 | 2seed `d=-389, K=335, w*=0`; 시드분산 6.6× | E3 | LOCAL-DEAD | `lab/101` 결과, TabM 포함 DL 종료 |
| has_time=True | ordered CTR 계산 변경 | paired −16.5/−23.0, 평균 −19.7 | E3 | LOCAL-DEAD | **LB 결과가 아님** |
| conditional binary head | target5 조건부 성공 head 교체 | seed42 raw −19.79, shape −10.16 | E2 | LOCAL-DEAD | reverse-only exact head는 별도 미실행 |
| mk-isos EXP-021 full backbone | LGB/HGB residual+Team EB+rank6 SVD 독립모델 | exact OOF 재현; 2024 proxy d=-56.920, K=175.413, w*=.3378, gain +20.011 | X5/E3 | **OFFLINE-SCREEN-PASS** | 8seed exact 아님; endpoint build+QA 1회 자격만 |

## 3. ASOF·숨은 상태 복원

| exact variant | 복원/질문 | 결과 | 등급 | 상태 | 근거 |
|---|---|---|---|---|---|
| current-season pitcher/batter | 누적값−전 시즌 종료 상수 | direct +92.3 | E5 | ADOPTED | CS76 |
| pitchmix current-season | 구종 누적값 차분 | direct +2.7 | E5 | ADOPTED | CS79 |
| success/middle/reverse label | 연속 asof 상태 차분 | 99.9~100% 복원 | E1 | ADOPTED | target5 기반 |
| prev1/3/5 denominator | rate 공통분모 역산 | 2022~24 exact n 81.9~84.2%, coverage 100% | E1 | AUDIT | `lab/111_prev_window_denominator_audit.txt` |
| decoded workload residual | 복원 n·reliability로 next-year residual | 2023 gamma 0, .25 raw −1.52 | E4 | LOCAL-DEAD | 2024 미개봉 |
| row_id temporal code | 날짜·경기·순번 암호 여부 | exact 사용 미감사 | E0 | BLOCKED-RULE | 운영진 서면 허용 전 피처 금지 |

핵심 구분: “분모를 복원했다”는 데이터 감사 성공이지, 그 분모가 예측력을 준다는 뜻은 아니다.

## 4. 타깃 구조·hazard 분해

| exact variant | 결과 | 등급 | 상태 | 죽인 범위 |
|---|---:|---|---|---|
| target5 | direct +40.35 | E5 | ADOPTED | — |
| M4 | paired −11.8 | E3 | LOCAL-DEAD | 해당 4-class mapping |
| C6 | seed42 −16.7 | E2 | LOCAL-DEAD | 해당 6-class mapping |
| MC7 | 2seed 평균 −22.8 | E3 | LOCAL-DEAD | 7-class와 9-class 확장 우선순위 |
| reverse/breaking current-season CS feature 확장 | cs_rev −19.4, cs_rev_brk −12.6 | E3 | LOCAL-DEAD | binary head가 아님 |
| conditional-first `P(S|not R)` | raw −19.79 | E2 | LOCAL-DEAD | 한 head 교체 방식 |
| reverse-only binary head | 미실행 | E0 | UNRESOLVED-low | oracle qR 보정이 본점수 −35.62라 후순위 |

## 5. categorical·상황 상호작용

| exact variant | 결과 | 등급 | 상태 | 비고 |
|---|---:|---|---|---|
| pitcher/batter team as categorical | direct 약 +24.5 | E5 | ADOPTED | 저카디널 13팀 |
| 팀×팀/팀×역할 categorical | paired −20.2 | E3 | LOCAL-DEAD | `lab/93_A` |
| count×hand + base/out categorical | paired −10.1 | E3 | LOCAL-DEAD | `lab/93_B` |
| forced one-hot | paired −6.0 | E3 | LOCAL-DEAD | `lab/93_OH` |
| `max_ctr_complexity=1` | paired −4.3 | E3 | LOCAL-DEAD | `lab/93_CTR1` |
| 선수 ID high-card categorical | 약 −160~−174 | E3 | LOCAL-DEAD | exact pair categorical도 금지 우선 |
| simple pitcher×count residual child | 작게 양수이나 Holm/부트스트랩 탈락 | E4 | LOCAL-DEAD | low-rank 24-vector와는 다른 질문 |
| pitcher×inning/baseout/stint | LOO weight 0 | E4 | LOCAL-DEAD | `lab/107`, `lab/108` |
| batter×pitcher_hand residual child | LOO weight 0 | E4 | LOCAL-DEAD | 단순 child 기준 |
| pitcher×24(count×batter_hand) rank-6 SVD | 외부 +5.2/+8.9/+19.5, 우리 exact 미측정 | X5/E0 | **UNRESOLVED** | joint 저차원 구조 |

## 6. 보호형 residual·라우팅

| exact variant | 결과 | 등급 | 상태 | 근거 |
|---|---|---|---|---|
| V18 protected hierarchy | strict 2022/23/24 양수, local 2024 affine 포함 +22.2, LB +11.14 | E5 | CHAMPION | exp/103~105 |
| V18 hand/detail 분리 | exact 평균 −2.43 | E4 | LOCAL-DEAD | exp/106 |
| V25 hierarchy child 7종 | LOO/Holm 생존 0 | E4 | LOCAL-DEAD | exp/107~108 |
| 우리 5신호 protected residual | 전 LOO weight 0 | E4 | LOCAL-DEAD | exp/109 |
| pooled temporal ExtraTrees adapter | 2022→23 gamma 0 | E4 | LOCAL-DEAD | exp/110, 2024 미개봉 |
| rookie expert | 신인행 `d=-51.75, K=42.32, w*=0` | E3 | LOCAL-DEAD | exp/97~98 |
| HGB65 no-pitcher rookie router | 2024 신인행 d=-39.84, K=43.63, opt +0.08 | E3 | LOCAL-DEAD | whole-fold 2024 no-pitcher −12.4; no-both −6.7은 별도 |
| low-rank pitcher-batter FM | 전체 +1.65, CI 0 포함, unseen pair 음수 | E3 | LOCAL-DEAD | `lab/codex_fm_result.txt` |
| x2 V14 exact 3-domain ET | 외부 약 +20, 코드/OOF 비공개 | X5 | UNRESOLVED-unreproducible | 우리 pooled ET와 exact 다름 |
| mk EXP-063 uncertain-window | 외부 direct +4.52 | X5 | UNRESOLVED-small | 현 챔피언 중복 가능 |
| mk EXP-064 stable-cell EB | 외부 direct +1.06 | X5 | UNRESOLVED-small | |
| mk EXP-072 pitcher AR(1) | 외부 direct +5.58 | X5 | UNRESOLVED-small | CS/V18 중복 가능 |

## 7. 출력 보정·혼합 기하

| exact variant | 결과 | 등급 | 상태 | 근거 |
|---|---:|---|---|---|
| global shift −.0066 | direct +17.8 vs 1059 | E5 | ADOPTED | 2차식 정점 |
| affine scale 1.06 | 누적 direct +22.6 | E5 | ADOPTED | 현재 챔피언 포함 |
| Platt/isotonic 과거 방식 | 연도 전이 약 −150.9 | E4 | LOCAL-DEAD | `archive/results/calib_*` |
| V18 gamma bracket | 현재 .30 추가 상방 약 +.01~+.32 | E5+algebra | CLOSED | 제출 취소 |
| F/R separate offset | local 2024 차등 oracle 약 +2.13 over global | E2 | UNRESOLVED-small | 규정 상태 별도 확인 |
| 4-group LB tomography | 2024 best tested 4-group differential +7.61; +100 불가 | E2 | REJECT | 규정도 불명확 |
| 12 count offsets | 2024 differential +10.59 | E2 | REJECT | 9회로 측정 불가, +100 아님 |

혼합 공식은 후보 B의 챔피언 A 대비 단독 차이를 `d=S_B-S_A`, 다양성을 `K`라 할 때

```text
S(w) = S_A + d·w + K·w(1-w)
w*   = clip((d+K)/(2K), 0, 1)
gain = (d+K)^2/(4K), 단 K>-d
```

이다. 단독 점수만 다른 모델은 가치가 없다. `K`가 함께 커야 한다.

## 8. TrackMan·물리

| exact variant | 결과 | 등급 | 상태 |
|---|---:|---|---|
| train↔TrackMan key audit | train key 존재 79.7%, 유일 조인 54.9%, pitcher ID 일치 100% | E1 | AUDIT |
| pitcher/batter entity map | 투구량 coverage 97.2%/95.5% | E1 | AUDIT |
| direct physical features | direct LB 약 −7.81 | E5 | LB-DEAD |
| CS 위 physical | 순증 +.2 | E3 | LOCAL-DEAD |
| release consistency | 세 번 실패 | E3 | LOCAL-DEAD |
| distillation | 초기 +37.3은 thread 교란, 정합 재실행 +.5 | E3 | LOCAL-DEAD |
| arrival-region model | 설명 증가 약 +.004 | E3 | LOCAL-DEAD |
| mk-isos physical | 외부 direct −1.21/−3.91 | X5 | 외부 교차반증 |

물리 정보를 새로 쓰려면 기존 평균 물리량 재가공이 아니라, 현재 피처가 전혀 담지 못하는 새 행별
정보원과 strict temporal 증거가 필요하다.

## 9. 신인·콜드스타트·no-ID

| exact variant | 결과 | 등급 | 상태 | 주의 |
|---|---:|---|---|---|
| rookie expert | 최적 혼합 0 | E3 | LOCAL-DEAD | 신규 투수 전용 모델 가설 반증 |
| team P01 `f_rk_ratedev` harness | seed42 −24.2, seed7 +16.6, 평균 −3.8 | E3 | LOCAL-DEAD | `exp/99` 파일 자체가 아니라 팀 harness 실행 |
| `exp/99_microfeat.py` local script | 결과 artifact 없음 | E0 | UNRESOLVED-low | 동일 P01 반증 때문에 재실행 가치 낮음 |
| full CAT5 no-pitcher exp/86 | 완료 결과 없음 | E0 | UNRESOLVED-low | 인접 no-ID 모델들은 강한 음수 |
| batter-ID 제거 | 선택시드 양수 후 독립시드 전부 반전 | E3 | LOCAL-DEAD | |

## 10. 배포·재현성

| 검사 | 결과 | 상태 |
|---|---|---|
| server env | Python 3.11 / numpy 1.26.4 / pandas 2.0.3 / CatBoost 1.2.10 | 고정 |
| candidate ZIP CRC/구조 | PASS | 고정 |
| 245,789행 fake server | 약 10초, 10분 제한 이내 | PASS |
| batch/reverse/permutation/duplicate/split/singleton | 모두 max diff 0 | PASS |
| 현재 챔피언 SHA | `ED1B...8557` | 제출 전 재확인 |
| IT300+V18 | `archive/rejected/`로 격리 | DO NOT SUBMIT |

## 11. 미해결 후보 우선순위

| 우선 | exact 질문 | 조건부 상방 | 비용 | 다음 게이트 |
|---:|---|---:|---:|---|
| 1 | mk-isos EXP-021 full hetero endpoint | 2024 proxy 혼합 +20.011; +100에는 실제 2025 K≈494 필요 | OOF 완료; bundle 미생성 | endpoint build/runtime/행독립 QA 후 사용자 결정 |
| 2 | pitcher×24 context rank-6을 champion residual 위에 직접 적용 | +0~10, 외부 local 상한 +19.5 | 수분~1시간 | strict 3-fold +15 |
| 3 | EXP-063 uncertain-window residual | +0~5 | 1~3시간 | strict 다음연도 +12 |
| 4 | EXP-072 AR(1) | +0~3 | 1~2시간 | V18/CS effect 상관과 strict fold |
| 5 | EXP-064 stable-cell EB | +0~1.5 | 수분 | 마지막 결합용 |
| 차단 | row_id 시간 구조 | +20~80 추측뿐 | 감사 20분 | **운영진 서면 허용** |

공개 EXP-021의 `935.81→1043.61(+107.80)`은 약한 외부 백본을 통째로 바꾼 composite 결과다.
rank-6 단독 +107로 기록하거나 우리 1092 위 +107로 외삽하지 않는다.
