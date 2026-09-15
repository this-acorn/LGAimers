
# 2026 LG Aimers 9기 — 투구 제구 성공 확률 예측 (dou 브랜치)

## 최고 점수 제출물: `submit_target5_teamcat.zip` — 리더보드 **1059.0501189623**

dou 원본(submit14)의 5클래스 타겟 분해{성공/미들/리버스/미들∩리버스/빅미스} + team_cat의
`pitcher_team_id`/`batter_team_id` cat_features 스태킹, CatBoost 8시드 멀티클래스 앙상블.

```
submit_target5_teamcat.zip
├── script.py           추론 코드 (claude/script_v8_target5_teamcat.py 기준, 피처 로직의 단일 원본, 행 독립 연산만 사용)
├── requirements.txt    numpy/pandas/sklearn/joblib 서버 기본 + catboost==1.2.10
└── model/model.pkl     CatBoost 8시드 멀티클래스 + 배포 상수표 + 구종/미들/리버스 성향표
```

## 핵심 아이디어

1. **시즌 진행분 복원 (실전 +92.3)**
   `asof_*` 누적 통계에서 "전 시즌 종료 시점 누적(train에서 계산한 상수표)"을 빼서
   각 행 시점의 **현 시즌 순수 성적**을 행 독립적으로 복원. 운영진 공식 Q&A에서 허용 확인.
2. **구종 라벨 복원 (+2.7)**
   같은 투수의 연속 행 asof 차분으로 train 투구별 구종을 복원(train 내 사용은 공식 허용),
   투수×카운트 구종 성향표와 P(패스트볼|상황) 모델을 만들어 피처로 사용.
   추론 시 현재 투구의 실제 구종은 사용하지 않음.
3. **team_cat_features (+24.51)**
   `pitcher_team_id`/`batter_team_id`는 이미 원본 47피처 안에 raw로 존재하던 컬럼인데, 이걸 CatBoost의
   `cat_features`로 추가 지정(3개→5개)만 해도 크게 개선됨. 팀 코드는 13개뿐이라 저카디널리티라서,
   이미 기각된 `cb_cat`(선수 ID 800명대 고카디널리티 범주형, -173.9로 실패)과 달리 과적합 위험이 적었던
   것으로 판단. 새 컬럼 추가 없이 기존 컬럼의 처리 방식만 바꾼 실험.
4. **target5 — 5클래스 타겟 분해 (dou 원본 아이디어, +15.6) + team_cat 스태킹 (+40.91 최종)**
   `control_success`(이진)를 성공/미들/리버스/미들∩리버스/빅미스 5클래스로 세분화해 멀티클래스로
   학습하고, 제출값은 `P(class0=성공)`만 사용. 미들/리버스 라벨은 공식 `asof_pitcher_middle_rate`/
   `asof_pitcher_reverse_rate`를 train 내에서 차분 복원(구종 라벨 복원과 동일 기법)해서 만듦. 원본은
   팀원 dou가 비공개 저장소에서 검증(LB 1033.99)했고, 이 저장소는 그 로직을 공식 데이터 설명서
   기준으로 재구성한 뒤 team_cat의 cat_features를 추가로 스태킹함 — 두 개선축이 서로 겹치지 않는다는
   가설이 그대로 확인되어(단독 성과 +15.6, +24.51 → 합산 +40.91) 신규 팀 최고 기록(1059.05) 달성.
   상세 배경: `claude/target5_teamcat_decision_2026-08-26.md`.

## 재현 순서

```
exp/41_season_progress.py             시즌 진행분 로컬 검증 (2024 폴드)
exp/45_label_recon.py                 구종 라벨 복원 + 피처 검증
exp/48_train_cs79.py                  CS79(79피처) 배포 학습 (venv: numpy 1.26.4 / pandas 2.0.3 / catboost 1.2.10)
exp/49_build_submit12.py              CS79 zip 빌드 + 가짜 서버 245,789행 검증
claude/62_team_cat_feature_test.py    team_cat_features 로컬 2-way 검증 (baseline vs +team cat_features)
exp/64_train_team_cat.py              team_cat 배포 학습 (script_v5_team_cat.py를 script.py로 교체 후 실행)
exp/65_build_submit_team_cat.py       team_cat zip 빌드 + 검증 → submit_team_cat.zip
exp/73_train_target5_teamcat.py       target5_teamcat 배포 학습 (script_v8_target5_teamcat.py를 script.py로 교체 후 실행, ~3~4시간)
exp/74_build_submit_target5_teamcat.py  target5_teamcat zip 빌드 + 검증 → submit_target5_teamcat.zip  ★ 현재 최고 기록
```

## 점수 이력

549.51(베이스라인) → 830.32 → 898.62(CatBoost) → 990.96(시즌 진행분) → 993.63(구종 라벨 복원) →
*(CS83 989.48, 기각)* → *(CS93 973.48, 기각)* → 1018.1353(team_cat_features) →
*(rmse 1017.66, -0.47, 동률·미채택)* → *(trackman_phys 1010.32, -7.81, 기각)* →
**1059.0501189623(target5_teamcat = 5클래스 타겟 분해 + team_cat 스태킹, 현재 팀 최고 기록)**

## 문서 안내

- `claude/hackathon_roadmap.md` — 대회 일정·규정·EDA 핵심 발견·trackman_history 조사 결과
- `claude/eda_feature_engineering_handoff.md` — 팀원 코드와 무관한 자체 EDA·피처 탐색 기록
- `claude/model_experiment_log.md` / `claude/model_experiment_log_table.md` — LightGBM 트랙 실험 기록(역사적 참고용) + 팀원 아이디어 브레인스토밍
- `claude/team_pipeline_sync_2026-08-25.md` — CatBoost(dou 브랜치) 파이프라인 채택 이후 모든 실험(CS83/CS93/team_cat/rmse/hand_cat/trackman_phys/target5_teamcat)의 1차 참고 문서, 최신 현황 갱신 위치
- `claude/target5_teamcat_decision_2026-08-26.md` — 현재 최고 기록(target5_teamcat, 1059.0501189623)의 배경·재구성 로직·결과 상세
- `claude/rejected_ideas_worth_revisiting_2026-08-27.md` — 기각/보류됐지만 다시 시도할 가치가 있는 후보 정리(CatBoost 하이퍼파라미터 튜닝, global shift 재보정 등, 다음 우선순위)
