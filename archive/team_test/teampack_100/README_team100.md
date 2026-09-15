# 팀원용 짝지음 실험 패키지 (exp/100) — 08-30 KST

챔피언(하민 target5_teamcat, LB 1059.05)의 로컬 2024 폴드 하네스 그대로, **설정 하나만 바꾼 팔**을
여러분 기계에서 돌려 주세요. 새 피처·새 데이터 없음 → 규칙 문제 없음. 결과 txt + npy 만 보내주시면 됩니다.

## 0. 준비 (10분)
```
대회 data/ 폴더 (train.csv, test.csv)  ← 여러분이 이미 갖고 있는 것
pip install catboost==1.2.10 pandas numpy pyarrow      (python 3.10~3.12 아무거나. 배포용이 아니라 실험용)
이 패키지의 폴더 구조를 그대로 두고, 그 루트에서 실행:
  <루트>/data/train.csv, data/test.csv
  <루트>/exp/100_champ_arms.py, exp/common.py
  <루트>/submit12_src/script.py            (하민 v8 스크립트 = 피처 정의의 단일 원본)
  <루트>/lab/89_cat5_probs_seed42.npy, 89_cat5_probs_seed7.npy   (우리 기계 기준선, thread14)
  <루트>/lab/90_analysis_2024.parquet      (세그먼트 리포트용, 없어도 돌아감)
```
★ **노트북이면 충전기 꽂고** `powercfg /change standby-timeout-dc 0` (배터리 절전으로 2시간 학습이 죽습니다 — 우리가 오늘 10시간 잃음).

## 1. 스레드 수 정하기 (중요)
CatBoost 는 **스레드 수가 달라지면 결과가 시드 바뀐 만큼 달라집니다**(실측). 그래서
같은 기계에서 `--threads N` 을 **모든 실행에 똑같이** 주세요. N = 물리 코어 수 정도 (예: 8).
우리 기준선(lab/89)은 thread14 라 여러분 기계와 직접 비교하면 잡음 σ≈21 입니다 → 아래처럼 **자기 기준선을 먼저** 만드세요.

## 2. 실행 순서 (팔당 약 1.5~2.5시간, 코어 수에 따라)
```
# (1) 스모크 3분 — 환경 확인
PYTHONIOENCODING=utf-8 python -u exp/100_champ_arms.py --arm BASE --threads 8 --smoke

# (2) 자기 기준선 (챔피언 그대로, 시드 42/7)  → lab/100_BASE_probs_seed{42,7}.npy
PYTHONIOENCODING=utf-8 python -u exp/100_champ_arms.py --arm BASE --threads 8  > lab/100_BASE_result.txt

# (3) 배정된 팔  (자동으로 (2)의 기준선과 짝지음)
PYTHONIOENCODING=utf-8 python -u exp/100_champ_arms.py --arm HT --threads 8    > lab/100_HT_result.txt
```
IT 팔(학습곡선)은 기준선이 필요 없습니다 — it=500 단계가 곧 기준선입니다:
```
PYTHONIOENCODING=utf-8 python -u exp/100_champ_arms.py --arm IT --threads 8 --seeds 42  > lab/100_IT_result.txt   (약 2배 시간)
# 곡선이 평평하지 않고 정점이 500 에서 ±100 이상 떨어져 있으면 --seeds 7 도 추가
```

## 3. 팔 설명
| 팔 | 바꾸는 것 | 질문 |
|---|---|---|
| BASE | 없음 | 여러분 기계의 기준선 (시드 42/7) |
| HT | `has_time=True` | CatBoost 범주형 CTR 을 "그 행 이전 행만"으로 계산(시간순). 추론 때 쓰는 ≤2024 표와 성격을 맞춤 |
| IT | `iterations=1000` + 50회마다 채점 | 학습 길이 곡선. it=500(현행)이 정점인가? 자기검증: it=500 점수가 기준선 시드와 일치해야 함 |
| C6 | 타겟 6클래스 | (우리 기계에서 실행 중) 빅미스를 브레이킹/비브레이킹으로 분할 |

## 4. 보내주실 것
- `lab/100_<팔>_result.txt` 전체 (맨 아래 6줄 표 + "세그먼트 paired" 줄이 핵심)
- `lab/100_<팔>_probs_seed*.npy`, `lab/100_BASE_probs_seed*.npy` (혼합/세그먼트 재분석용, 각 5MB)
- 사용한 `--threads` 값과 기계 사양(코어 수)

## 5. 판정 눈높이 (우리가 합니다)
paired 평균 **+8 이상, 두 시드 동부호, 산포 < 3** 이면 배포 후보. σ(시드 1개) ≈ 15 이라 +5 는 잡음과 구분이 안 됩니다.
F(퓨처스) 세그먼트는 σ 가 3배라 단독 판정 근거로 쓰지 않습니다.

## 6. 하지 말 것
- 이 스크립트/결과를 대회에 제출하지 마세요 (제출물은 배포 파이프라인에서만 빌드).
- 두 학습을 동시에 돌리지 마세요 (스레드 수 고정이 깨집니다).
- 스크립트 안의 파라미터를 임의로 바꾸지 마세요 — 바꾸면 짝지음이 깨져 비교 불가.
