# 제출 장부

실제 평가 서버에 제출한 것만 기록한다. local 후보, 빌드만 한 ZIP, 외부팀 점수는 이 표에 넣지 않는다.
새 행은 append-only이며 과거 점수를 수정할 때는 정정 행을 추가한다.

## 현재 최고

| ZIP | SHA256 | Public LB | runtime | 상태 |
|---|---|---:|---:|---|
| `candidate_affine_opt.zip` | `E772CA86DAF99209A50AB9E29F030EE680C0770520D98FF5ABE9C3FD4832EC65` | **1116.5770907872** | 미기록 | SCORE CHAMPION / 자체 구현 |

## 실제 제출 이력

| 순서 | ZIP/설명 | parent | 단일 변화 | Public LB | parent 대비 | 판정 |
|---:|---|---|---|---:|---:|---|
| 0 | 운영진 RF | — | baseline | 549.51193 | — | 기준 |
| 1 | hand delta 67 | HGB65 | hand tables | 약 808.09 | −22.2 | 철회 |
| 2 | `submit4.zip` | RF | HGB65 | 830.322760105 | +280.81 | 채택 |
| 3 | `submit5.zip` | HGB/CB endpoints | w=.3 | 864.3312059823 | +34.01 | 역사 |
| 4 | `submit6.zip` | 동일 | w=.5 | 880.5655581163 | +16.23 | 역사 |
| 5 | `submit8.zip` | 동일 | CB w=1 | 898.61863054 | +18.05 | 채택 |
| 6 | `submit9.zip` | submit8 | 16 seeds | 897.6644306949 | −0.95 | 기각 |
| 7 | `submit10.zip` | submit8 | CS76 | 990.957167531 | +92.34 | 채택 |
| 8 | `submit12.zip` | submit10 | pitchmix 3 | 993.6345481775 | +2.68 | 채택 |
| 9 | `submit13.zip` | CS79/CB65 | 50:50 K probe | 975.5048373847 | — | 진단 |
| 10 | `submit14.zip` | CS79 | target5 | 1033.9866361712 | +40.35 | 채택 |
| 11 | `submit16.zip` | target5 | lr.04+season weight | 1006.3794404841 | −27.61 | 기각 |
| 12 | team_cat | CS79 | team categorical | 1018.1353 | +24.50 | 채택축 |
| 13 | RMSE variant | team_cat | loss variant | 약 1017.66 | −.47 | 기각 |
| 14 | TrackMan physical | team_cat | physical | 약 1010.32 | −7.81 | 기각 |
| 15 | `champion_target5_teamcat.zip` 계열 | target5 | team categorical | 1059.0501189623 | +25.06 | 채택 |
| 16 | `submit18_it300.zip` | 1059 champion | ntree_end=300 | 1040.0366 | −19.01 | **기각** |
| 17 | `submit19_it500_m012.zip` | 1059 champion | shift −.012 | 약 1065.26 | +6.21 | probe |
| 18 | `submit20_it500_m0066.zip` | 1059 champion | shift −.0066 | 약 1076.81 | +17.76 | 채택 |
| 19 | `submit21_s115.zip` | shift champion | scale 1.15 | 약 1073.66 | −3.15 | probe |
| 20 | `submit22_s106.zip` | shift champion | scale 1.06 | 약 1081.67 | +4.86 | 채택 |
| 21 | `candidate_v18g030.zip` | submit22 | frozen V18 γ=.30 | **1092.808353586** | 약 +11.14 | **CHAMPION** |
| 22 | `candidate_champ55.zip` | candidate_v18g030 / EXP021 strict | EXP021 weight .45 diversity probe | **1113.1142204091** | +20.3058668231 | 기존 제출 보존 / 추론파일 재작성 대기 |
| 23 | `candidate_exp021_w0356557.zip` | candidate_v18g030 / EXP021 strict | analytic optimum EXP021 weight .3565571695 | **1114.6116846973** | +1.4974642882 | 기존 제출 보존 / 예측값 exact 일치 |
| 24 | `probe_affine_s096.zip` | `candidate_exp021_w0356557.zip` | 고정 중심 `.44`, affine scale `.96` | **1109.4232092263** | **−5.1884754710** | **기각 / 챔피언 유지** |
| 25 | `candidate_affine_opt.zip` | `candidate_exp021_w0356557.zip` | LB 3점으로 복원한 affine 정점 (`r=0.4607`) | **1116.5770907872** | **+1.9654060899** | **채택 / 새 챔피언** |

과거 ZIP의 SHA가 원장에 남지 않은 경우 추측해 채우지 않는다. `archive/submissions/`에서 재계산할 수
있는 것만 후속 감사로 보강한다.

## 최초 9회 예약표 — 4회 사용, 5회 남음

슬롯 번호는 “무조건 제출할 순서”가 아니라 gate를 통과한 질문에만 배정한다.

| 잔여 슬롯 | 예약 질문 | 제출 전 필수값 | ZIP | SHA | LB | 결정 |
|---:|---|---|---|---|---:|---|
| 1 | EXP-021 diversity probe | strict OOF gate, endpoints | `candidate_champ55.zip` | `32D7F275AFD34134ECD34891E67B3128B20D817732364507B3F4FBE2F1630FAF` | 1113.1142204091 | K=171.50015, w*=.356557 |
| 2 | EXP-021 analytic `w*` | measured K=171.50015 | `candidate_exp021_w0356557.zip` | `C080DA7E82B1707E091501B9BA0D58BCD18CBEF9B73A975DBCEE3A87C0F330B` | 1114.6116846973 | **예측 exact / 축 종료** |
| 3 | affine scale `.96` 단일 probe | 2024 exact OOF +5.415 | `probe_affine_s096.zip` | `A860F30CF4B4A73E2CD700F574B98EA2A23B66DC082B4F7AAFDE1E27FBBA8A27` | 1109.4232092263 | **−5.1884754710 / 기각** |
| 4 | affine 해석적 정점 | 복원한 2차식의 `r=0.4607` | `candidate_affine_opt.zip` | `E772CA86DAF99209A50AB9E29F030EE680C0770520D98FF5ABE9C3FD4832EC65` | 1116.5770907872 | **+1.9654060899 / 새 챔피언** |
| 5 | rank-6 atop champion | strict 3-fold +15 | | | | |
| 6 | 소형 residual 묶음 | 각 성분 strict 양수, 결합 재검증 | | | | |
| 7 | 규정 승인된 새 정보원 | 서면 답변+offline gate | | | | |
| 8 | 상위 후보 multiway blend | endpoint/K matrix로 사전 계산 | | | | |
| 9 | 최종 확인/비상 reserve | best artifact SHA 재확인 | | | | |

`probe_affine_s096.zip` 제출 시각은 **2026-09-01 02:01:14 KST**, 실행시간은 **14초**다.

## 새 제출 행 양식

```text
timestamp(KST):
slot:
question:
zip:
sha256:
parent zip / sha:
single change:
offline evidence + gate:
row-independence/runtime:
Public LB:
delta versus parent:
decision:
```
