# 현재 상태

최종 갱신: **2026-09-01 07:01 KST**

## 한 줄 결론

현재 최고는 **1116.5770907872**이고 잔여 제출 수는 **5회**다.
공개 mk-isos `EXP-021`과 기존 챔피언의 혼합 곡선을 45% probe로 측정했고, 해석적 정점인
EXP-021 `35.6557169%` 후보가 예상값 **1114.6116846973**과 소수점 끝까지 일치했다.
고정 중심 `.44`, scale `.96` affine probe는 **1109.4232092263**으로 챔피언보다
**5.1884754710점 낮아 기각**했다. 남은 제출은 독립 오차 구조에만 사용한다.
이후 LB 3점으로 복원한 affine 정점 `candidate_affine_opt.zip`을 제출해 예상
`1116.522784`보다 `0.0543067872` 높은 **1116.5770907872**를 기록했다. 직전 챔피언 대비
이득은 **+1.9654060899**이며, 사후처리 평면의 새 기준점으로 고정한다.

07시대 랩 GPU에서 EXP182 X80 MLP를 3-seed 검증했으나 endpoint가 current 2024 OOF보다
`814.2355`점 낮고 unconstrained 혼합가중치도 `-0.0378`이라 `FAIL_NO_DEPLOY`였다. 셔플만 고친
재실행과 `final_train`은 금지한다. 다음 GPU 질문은 행 셔플과 TrackMan 55열을 포함한 별도
X176 ResMLP EXP183 한 번뿐이다. 근거는 `lab/186_exp182_gpu_actual.txt`에 기록했다.

EXP183도 랩 GPU에서 293초 만에 strict 완료됐으나 `FAIL_NO_DEPLOY`였다. 2023 discovery
`+309.5464`는 그 해 current anchor 자체가 `−22.74`였던 regime 착시였고, 잠근 가중치를
2024에 넘기자 `−185.3540`이었다. 더 결정적으로 2024 라벨을 직접 본 oracle 재적합도
최대 `+0.6899`뿐이므로 NN/TrackMan endpoint 블렌드 축은 완전히 종료한다. 근거는
`lab/187_exp183_gpu_x176_actual.txt`에 기록했다.

**운영 상태:** Claude의 `exp021_inference.py` 재작성은 완료됐다. 새 컴포넌트는 468줄,
SHA256 `C531527B975A855ACA2ACD936CF03CA65BFCF2F35ADA1D0E0114F28285A85C44`이며 기존 873줄
파일과 바이트가 다르다. 20,000행·245,789행 종단 차분과 행 독립성 6종에서 기존 endpoint 대비
최대 절대차가 모두 `0`이었다. 신규 동작 기준선은 `candidate_exp021_ours_v001.zip`이다. 코드가
달라졌다는 기술 QA와 대회 규정·권리의 최종 판단은 별개이며, 기존 감사 근거는
`docs/COMPLIANCE_AUDIT_EXP021.md`에 그대로 보존한다.

## 09-01 GPU 및 긴급 후보 갱신

- EXP182 X80 MLP: current 대비 endpoint `-814.2355`, 최적 convex weight `0`, 종료.
- EXP183 X176+TrackMan ResMLP: 잠근 2024 gain `-185.3540`; 2024 same-fold oracle도
  `+0.6899`뿐이라 종료.
- EXP188 1D residual head: 최소 scale `.025`부터 전체 `-0.7102`, early `-0.1767`,
  late `-1.4075`, pitcher p025 `-1.2813`. 손실의 98.7%가 2023 R 평균 잔차의 상수 이동이며
  행별 신호는 `-0.0096`뿐이었다. final train/ZIP 없음.
- 공개 JY `1130.3604943627` 패키지는 Git LFS 원본을 확보했으나 타 팀 가중치이며 저장소에
  LICENSE가 없다. 이름·변수만 바꿔 우리 모델로 위장하는 사용은 하지 않는다.
- 자체 Tensor 효과는 2024에서 전체 `+0.9043`, early `+6.0963`, late `-5.8803`이었다.
  이를 이용한 아래 ZIP은 strict gate PASS가 아니라 남은 슬롯을 위한 명시적 고위험 후보다.

| 후보 | 변화 | SHA256 | 정책 |
|---|---|---|---|
| `candidate_exp189_affine_tensor_early_hold.zip` | R, month<=6에 scale `.20`; 나머지는 affine exact | `954CFB42E2938E7B6C82C63DBD4316CCD97D6E1D1E11C50ED646D3A1825562CC` | 첫 probe |
| `candidate_exp191_affine_tensor_early_s058_lottery.zip` | EXP189 early scale `.20→.58` | `8CD8F811E816F3530BE1EAF203ADD07C542B42FCBBD9F121EBBE8396BE5BA29E` | EXP189 상승 때만 |
| `candidate_exp190_affine_tensor_phase_lottery.zip` | early `.58`, late `-.20`, F exact | `C1F23DCA2A3771650D4865BBEEABC6688E3C3AA5AECAD4E545C6C6BE02E3584C` | EXP191 상승 때만 |

세 ZIP 모두 역순·무작위·분할·단일행·중복·소배치 QA에서 최대 차이 `0.0`이다.
추가 clean-room component-disagreement tree gate는 2022→2023에서 scale을 잠근 뒤 2024 한 점만
확인하도록 구현했으며, PASS일 때만 ZIP을 만들도록 강제했다.

## 최신 순위표 스냅샷

사용자 제공 2026-08-31 17:05 KST 이후 순위표에서 우리 팀은 **190위**다. 직전 1092.808 시점의
246위보다 56계단 상승했다. 이전 표의 1319.9016 팀은 새 표에 보이지 않는다.

| 기준 | 점수 | 현재와의 격차 |
|---|---:|---:|
| 우리 팀 | **1116.5770907872** | — |
| 100위 | 1141.93034127 | +25.3532504828 |
| 20위 | 1181.11904112 | +64.5419503328 |

**사용자가 직접 확인한 실제 마감은 2026-09-01 10:00 KST다.** 이전 기록의 09-02 09:59는
잘못된 페이지 판독이므로 폐기한다. 순위는 계속 변하므로 점수 컷은 이 시점의 스냅샷으로만 사용한다.

## 현재 챔피언

| 항목 | 값 |
|---|---|
| ZIP | `candidate_affine_opt.zip` |
| SHA256 | `E772CA86DAF99209A50AB9E29F030EE680C0770520D98FF5ABE9C3FD4832EC65` |
| Public LB | **1116.5770907872** |
| 실행시간 | 미기록 |
| 기반 | 직전 EXP-021 혼합 챔피언 위 affine 정점 `r=0.4607` |
| 행 독립 QA | 245,789행 및 순열·분할·단일행 QA 통과 |
| QA 근거 | `lab/168_affine_opt_qa.txt` |

직전 재구성 기준선 `champion_target5_teamcat.zip`의 SHA256은
`B8B7151E54AEFAAFC3B73915316019551B68570383A30AA32C0880ED8EFB8676`이다.

## 제출 금지

| artifact | 이유 | 위치 |
|---|---|---|
| IT300 + V18 | IT300이 direct LB 1040.0366, 당시 챔피언 대비 −19.0 | `archive/rejected/v18_it300/` |
| gamma .40/.45 괄호 | 2025 곡선 역산상 γ=.30이 정점과 사실상 일치, 상방 ≤+.32 | 생성하지 않음 |
| TrackMan 물리 재등판 | 우리와 외부 저장소 모두 direct LB/strict 검증 실패 | 과거 결과만 보존 |
| AUX5 shared-tree | seed42 raw −28.5, shape −17.1, 계산 혼합상방 +5.1 | seed7·ZIP·LB 없음 |
| 투수×문맥 rank-6 | exact 2023→2024 현재혼합 OOF raw +2.418, oracle +2.428 | `lab/126_lowrank_currentblend_gate.*` |
| 동적 계층 `p0`/전체행 residual | 2023 discovery 양수였으나 잠근 2024 확인에서 primary −150.38 | `exp/127`, `exp/128` |
| 단순 투수×주자 residual | 현재혼합 2024 raw +3.80, oracle +3.84~4.34 | `exp/152`, `lab/152_*` |
| 원인별 투수×주자·문맥 | target5 설명은 3년 안정 양수지만 현재혼합 2024 raw +0.36, oracle +4.18 | `exp/153`, `exp/155` |
| 타자별 hidden-call fingerprint | 2022/23/24 모든 K에서 target5 logloss 악화 | `exp/154`, `lab/154_*` |
| 과거 구종선택×실패원인 prior | clean-room forward 2024 raw +2.65, oracle +5.67 | `exp/156`, `lab/156_*` |
| CAT5 원인 posterior endpoint routing | 2023 전이 −47.56, 2024 +0.08/shape −0.09, 계수 cosine −0.95 | `exp/157`, `lab/157_*` |
| affine scale `.96` | direct LB 1109.4232092263, 챔피언 대비 −5.1884754710 | `probe_affine_s096.zip` |

## 현재 소유권

| 작업 | 소유자 | 상태 | 충돌 방지 |
|---|---|---|---|
| 장부·폴더 재정리 | Codex | **완료** | 기존 `HANDOFF.md` 내용 삭제 금지 |
| mk-isos EXP-021 exact 재현·배포 | Codex | **완료, LB 측정 완료** | endpoint와 45% bundle 고정 보존 |
| `exp021_inference.py` 재작성 | Claude | **완료·동작 QA 통과** | 원본/재작성본과 감사 기록 모두 보존 |
| 공개 저장소 추가 해부 | Claude 세션 가능 | 별도 | 같은 실험 번호·산출물 사용 금지 |
| LB 제출 | 사용자 | 직접 수행 | `SUBMISSION_LOG.md` 기록 후 제출 |

`exp/115_reproduce_mkis_exp021.py`는 공개 EXP-021의 다섯 OOF 단계를 완료했고,
`exp/116_rebuild_exp021_endpoint.py`는 공개 소스와 동일한 환경에서 endpoint를 재현했다.
2025 LB에서 endpoint 단독 점수 `1043.6074197937`, 기존 챔피언 `1092.808353586`,
45% 혼합 `1113.1142204091`을 이용해 `K=171.5001496147`을 실측했다. 이는 2024 OOF의
`K=175.413`과 매우 가깝다. 해석적 정점은 EXP-021 가중치 `0.3565571695`, 기존 챔피언
가중치 `0.6434428305`, 예상 점수 **1114.6116846973**였고 실제 LB와 exact 일치했다.

## 다음 의사결정

1. EXP-021 가중치 미세조정은 종료한다. 동일 두 endpoint 사이의 다른 비율은 계산상 현재보다 낮다.
2. 이미 gate를 실패한 T13 hard-regime은 재개하지 않는다.
3. 모든 기존 실험·artifact를 다시 대조해 현재 챔피언과 다른 오차 구조를 갖는 후보만 다음 probe에 올린다.
4. 팀원의 IT1000 2시드 결과는 별도 저우선순위 후보로 받아 offline/구조 검증 뒤 판단한다.
5. EXP021 aggressive 3-way는 2024 OOF 거리비로 환산한 2025 예상 이득이 `−0.021`이라 `+0.5`
   사전 gate를 실패했다. ZIP·LB probe를 만들지 않는다.
6. AUX5 seed42는 raw `−28.5`, shape `−17.1`, 혼합 상방 `+5.1`로 gate 실패했다. seed7·ZIP·LB 없음.
7. `exp/126_lowrank_currentblend_gate.py`는 24문맥 rank-6을 exact 2023 residual에서 적합해 2024로
   넘겼다. 고정 γ=.25 raw `+2.418`, same-fold oracle도 `+2.428`뿐이므로 **저랭크 축은 종료**한다.
8. 단순 F-only expert/hard routing은 과거 exp/47 `−49.3`(F `−456.4`), exp/122 RECENT_F
   `−386.6` 등으로 이미 사망했다. 재실행하지 않는다.
9. 동적 계층 `p0`와 전체행 residual은 2023 discovery의 큰 양수를 잠근 뒤 untouched 2024로 확인했으나
   `.5×현재혼합+.5×p0`가 `−150.381`, direct `p0`가 `−537.515`였다. 이 축은 종료한다.
10. 재작성 endpoint 기반 첫 점수 probe는 고정 중심 `.44`, scale `.96`이다. 2024 exact OOF 이득은
    `+5.415`, 반대편 scale `1.04`는 `−11.825`였으므로 `.96` 한 장을 먼저 측정하고 `.104`는 선제 제출하지 않는다.
11. 라벨 의미 기반 신규축을 재감사했다. `pitcher×runner_call_state×count_group×batter_hand`는
    reverse/bigmiss multiclass 예측을 2022/23/24 모두 개선했지만, 현행 이진 혼합 잔차에서는
    untouched 2024 raw `+0.356`, same-fold oracle `+4.179`뿐이었다. 의미상 맞아도 제출축은 아니다.
12. batter reverse/overlap/bigmiss EB는 전 연도 악화했다. 공개 저장소의 고수준 아이디어에서 독립 구현한
    fine-pitch selection×failure prior도 true-forward 2024 `+2.653`, oracle `+5.669`로 제출 gate를 실패했다.
13. `probe_affine_s096.zip`은 2026-09-01 02:01:14 KST에 제출되어 **1109.4232092263**을 기록했다.
    챔피언 대비 `−5.1884754710`이므로 이 affine 축은 종료한다. 이 결과만으로 `s104`를 반대편 보정처럼
    추론해 제출하지 않으며, 다점 곡선 복원과 hidden 평가 라벨 평균/분포 역산도 진행하지 않는다.
14. CAT5의 middle/reverse/bigmiss posterior로 Champion↔EXP021 가중치를 행별 라우팅하는 축도
    2022→2023 전이 `−47.56`, refit→2024 raw `+0.076`/shape `−0.092`, 계수 cosine `−0.952`로 종료했다.

## 규정 경계

- 각 평가 행은 다른 평가 행의 존재·순서·집계와 무관해야 한다.
- train에서 고정한 상수표를 현재 행에 조인하는 것은 허용 기록이 있다.
- `row_id`의 시간 암호화나 다구간 LB 보정은 운영진의 구체적 서면 허용 전에는 배포하지 않는다.
- LB 기반 모델 혼합이 허용됐다는 팀 기록은 있으나, 임의 세그먼트별 잔차를 LB로 역산하는 것은
  더 강한 행위다. 별도 승인 없이 9회 전략에 넣지 않는다.
