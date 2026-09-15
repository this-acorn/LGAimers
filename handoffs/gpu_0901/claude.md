# 프로젝트: LG Aimers 9기 - 투구 제구 성공 확률 예측 AI 해커톤

## 대회 개요
- **주제**: 투구 직전까지의 정보만으로 해당 투구의 제구 성공 확률(`control_success`=1일 확률)을 예측
- **주최**: LG AI연구원 / 주관: 데이콘
- **일정**: 2026.08.05 ~ 2026.09.02 09:59 (리더보드 제출 마감 09.01)
- **형식**: 코드 제출(submit.zip) 대회 — 노트북/스크립트 캡처 제출이 아니라 실제 추론 코드가 격리된 서버에서 재실행됨

---

## 데이터 구조

```
open.zip
├── baseline_submit.zip   # 베이스라인 코드 예시
└── data/
    ├── train.csv              # 1,475,092행 x 49컬럼 (control_success 포함)
    ├── test.csv               # 형식 확인용 5건 샘플, 48컬럼 (실제 평가 시 245,789행으로 서버에서 교체)
    ├── sample_submission.csv  # 제출 양식 (row_id, control_success)
    └── trackman_history.csv   # 2019~2024년 Trackman 로그, 1,793,078행 x 30컬럼 (보조 피처용, 1:1 결합 아님)
```

### train.csv / test.csv 컬럼 그룹
1. **기본 식별자/경기 정보**: row_id, season, game_month, game_dayofweek, inning, top_bottom, game_type
2. **카운트/점수 상황**: balls_before, strikes_before, outs_before, run_top_before, run_bot_before, run_total_before, score_diff_home, score_diff_pitcher_team
3. **주자/상황 중요도**: runner_on_1b/2b/3b, num_runners_on, base_state, home_win_expectancy, away_win_expectancy, li
4. **선수/팀**: pitcher_id, batter_id, pitcher_hand, batter_hand, pitcher_team_id, batter_team_id
5. **`asof_*` 과거 이력 피처** (투구 직전까지의 정보로 사전 계산됨, 그대로 사용 가능):
   - 투수: asof_pitcher_n, asof_pitcher_success_rate, asof_pitcher_reverse_rate, asof_pitcher_middle_rate, asof_pitcher_ball_rate, asof_pitcher_strike_rate
   - 투수 최근 경기: asof_pitcher_prev{1,3,5}_game_success_rate, asof_pitcher_prev{1,3,5}_game_middle_rate
   - 타자: asof_batter_n, asof_batter_success_rate, asof_batter_middle_rate
   - 구종 믹스: asof_pitcher_pitchmix_n, asof_pitcher_fastball_rate, asof_pitcher_breaking_rate, asof_pitcher_offspeed_rate
   - ⚠️ 표본 0인 경우 rate 컬럼이 결측 → cold-start 처리(smoothing/fallback)는 자유 설계

### target: `control_success`
- 1 = 제구 성공, 0 = 제구 실패 (아래 3가지가 실패로 정의)
  1. 스트라이크존 가운데 부근
  2. 스트라이크존에서 크게 벗어남
  3. 포수 요구 방향과 반대

### trackman_history.csv
- 2019~2024만 포함 (2025 없음). train/test와 1:1 결합 테이블 아님.
- 구종/구속/회전수/무브먼트 등 투구 특성 → 투수 단위 요약 피처 생성용 참고 자료로만 사용

---

## ⚠️ 매우 중요한 제약사항 (반드시 코드에 반영)

### 1. 평가 데이터 행 간 독립성 원칙 (규칙 위반 시 실격)
`test.csv`의 각 행은 **독립적으로** 예측해야 함. 다른 행을 이용한 아래와 같은 방식 **절대 금지**:
- test.csv 내부 행 기반 선수별/팀별/월별 **누적 통계**
- test.csv 내부 **빈도/분포** 통계
- test.csv 내부 **target encoding**
- test.csv **행 순서 기반** rolling/expanding feature
- 평가 데이터 전체를 보고 만든 **사후 보정값**

→ **허용되는 것은 `asof_*` 컬럼처럼 각 행의 투구 직전 시점까지의 과거 기록만으로 계산된 피처뿐.** 새로운 피처를 만들 때도 반드시 "그 투구 시점 이전 정보만" 사용해야 함 (train.csv나 trackman_history.csv 기반으로 사전 계산 후 test에 조인하는 방식은 OK, test 자체의 분포/집계를 쓰는 건 NG).

### 2. 사용 금지 정보
- 현재 투구 이후 확정 정보, 실제 위치/코스, 실제 판정·결과, 실제 구종, 실제 Trackman 측정값
- 2025년 Trackman 데이터
- 위에서 말한 test.csv 내부 집계/누적/빈도/rolling/target encoding

### 3. 외부 데이터/모델 제한
- **외부 데이터 사용 금지**: 대회 공식 데이터(train/test/trackman_history) 외 어떤 외부 데이터도 사용 불가
- **사전학습 모델**: MIT/Apache 2.0 등 최소 비상업 이용 허용 라이선스로 가중치가 "공개"된 모델만 사용 가능
- **외부 API 금지**: OpenAI API, Gemini API 등 원격 서버 기반 API 사용 불가. 모든 연산은 로컬 환경에서 재현 가능해야 함

---

## 평가 지표: Brier Skill Score

```
Brier Score = mean((p_i - y_i)^2)                      # p_i: 예측 확률, y_i: 실제 0/1
r = mean(y_i)                                            # 전체 평가 데이터 평균 성공률 (비공개)
평균 제구율 Brier Score = r × (1 - r)                     # 베이스라인 (단순 평균 예측)
Score = max(0, 100000 × (1 - Brier Score / 평균 제구율 Brier Score))
```

- 이 지표는 사실상 **MSE**와 동일한 형태 (0/1 타겟에 대한 확률 예측의 제곱오차 평균)
- **낮은 Brier Score = 높은 최종 Score.** 확률 보정(calibration)이 핵심 — 자신 있게 틀리면 페널티가 큼
- **모델링 전략 힌트**:
  - `objective='regression'`(MSE) 또는 `binary` + calibration(Platt scaling, isotonic regression) 둘 다 시도해볼 만함
  - Brier Score를 직접 최적화하려면 회귀형 MSE objective가 지표와 가장 직접적으로 부합
- **LG Aimers 9기 수료 조건**: Phase1 이수 + Public Score 549.51 이상 (베이스라인 추론 코드를 운영진 환경에서 실행했을 때 기준)

---

## 제출 형식 (submit.zip 코드 제출)

```
submit.zip
├── model/              # 모델 가중치 저장 디렉토리 (예: model.pt)
├── script.py           # 추론 실행 코드 (평가 서버가 자동 실행)
└── requirements.txt    # pip install -r requirements.txt로 설치 가능해야 함
```

- 평가 서버가 자동으로 `data/`(실제 테스트 데이터, 읽기전용)와 `output/`(결과 저장용) 디렉토리를 추가
- `script.py`는 `output/submission.csv`를 반드시 생성해야 함 (row_id, control_success 컬럼)
- **최상위 폴더가 추가로 있으면 구조 불일치로 설치 오류** → zip 압축 시 구조 정확히 일치시킬 것

### 제약 조건 (반드시 지킬 것)
| 항목 | 제한 |
|---|---|
| 전체 추론 시간 | ≤ 10분 (245,789개 샘플) |
| 패키지 설치 시간 | ≤ 10분 |
| 제출 파일 용량 | ≤ 10GB (압축해제 후 최대 32GB) |
| 인터넷 연결 | 패키지 설치 외 완전 차단 (오프라인 실행) |
| 서버 사양 | 6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM, Ubuntu 22.04.5, CUDA 12.8, Python 3.11.15 |
| 일일 제출 횟수 | 최대 5회 (단, 설치 오류는 횟수 미반영, 코드 실행 오류는 횟수 반영됨) |
| 사용 언어 | Python만 가능 |

### 평가 서버 기본 설치 패키지 (버전 명시된 건 requirements.txt에 넣지 말고 그대로 사용 권장)
```
torch==2.7.1+cu128, pandas==2.0.3, numpy==1.26.4, scipy==1.15.3,
scikit-learn==1.8.0, joblib==1.5.3, threadpoolctl==3.6.0, narwhals==2.21.2,
transformers==4.46.3, accelerate==1.9.0, sentencepiece==0.1.99,
regex==2023.12.25, tqdm==4.66.4, loguru==0.7.2, pyyaml==6.0.1, rich==13.7.1
```
- 시스템 패키지: git, build-essential, python3.11, libblas3, liblapack3, libomp-dev, cmake, ninja-build 등 (LightGBM/XGBoost 빌드에 필요한 것들 대부분 포함)
- ⚠️ LightGBM, XGBoost, CatBoost 등은 기본 목록에 없으므로 requirements.txt에 명시 필요 (인터넷 설치 단계에서만 다운로드 가능, 추론 시엔 인터넷 없음)

---

## 개발 시 유의사항 (Claude Code 작업 가이드)

1. **로컬 개발용 train.csv/trackman_history.csv는 대용량**(각 100만 행 이상) → pandas 기본 로드 시 메모리/속도 고려, 필요시 dtype 최적화나 청크 처리 고려
2. **로컬 test.csv는 5행 샘플뿐** → 실제 평가는 245,789행. 로컬에서는 train.csv를 자체 train/valid split해서 검증할 것 (test.csv로 검증 불가)
3. **피처 엔지니어링 시** train.csv 기반으로 만든 as-of 통계는 반드시 "해당 시점 이전 데이터만" 사용하는 방식으로 구현 (예: groupby + expanding/shift, 시간순 정렬 후 누적 계산). test.csv에 적용할 때도 동일한 로직으로 "test 행 자체의 정보"가 아니라 "train + trackman_history 등 사전에 계산 가능한 값"을 조인하는 구조로 설계
4. **cold-start(표본 0) 결측치 처리 전략**을 설계에 포함 (전체 평균으로 fallback, 베이지안 스무딩 등)
5. **추론 시간 10분 제한** 염두에 두고 모델 크기/앙상블 개수 결정 (L4 GPU 1장, CPU 6코어 환경)
6. **최종 제출 전 반드시 submit.zip 구조를 로컬에서 재현 테스트** (가짜 data/ 폴더 만들어서 script.py가 정상적으로 output/submission.csv를 생성하는지 확인)
