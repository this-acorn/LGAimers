# EXP183 GPU X176 robust-z ResMLP

이 파일은 기존 `exp182`의 80피처 모델을 다시 돌리는 작업이 아닙니다. 공식 데이터에서 아래 176열을 독립 재구성합니다.

```text
공통 raw/one-hot/derived 68
+ 과거 시즌만 쓴 platoon EB 4
+ 직전 시즌 lookup 기반 season-form 8
+ TrackMan 물리 55 + count 27 + 투수 역할 5 + TrackMan coverage 1
+ fit-fold ID log-frequency/unseen 8
= X176 -> robust median/IQR z + finite mask -> 352 inputs
```

공개 `LA9elephantmiracle` 커밋 `57783aa88ccc30915a57abe887c8b8125408d483`의 MIT 코드 명세를 사용하되, 그 저장소의 데이터 파생 파일, LUT, OOF, 예측, 체크포인트, ZIP은 읽지 않습니다. `run_arm.py`도 필요하지 않습니다. 재배포할 때 `reference/LA9elephantmiracle/LICENSE`를 함께 보존해야 합니다.

## 필요 파일

GPU 저장소 루트 기준:

```text
data/train.csv
data/trackman_history.csv
lab/179_current_2023.npy
lab/179_current_2024.npy
reference/LA9elephantmiracle/LICENSE
reference/LA9elephantmiracle/cowork/cw/v17/src/common.py
reference/LA9elephantmiracle/performance_tracking/models/sj_stdmlp/prep_mlp.py
```

고정 anchor SHA256:

```text
179_current_2023.npy  ee3aa1e2e777025a503790cfdc690aa5e5ed21d6b2ea97607e051cfec894a112
179_current_2024.npy  e78b168244675adea4bd6a3d56f021a9376cd4b78c428d536a9b53b02fcebe9b
```

현재 handoff 폴더에는 2023 anchor까지 복사되어 있습니다. 데이터는 대회 파일이라 Git에 올리지 않습니다.

Python 의존성은 `numpy`, `pandas`, CUDA 대응 `torch`, `scipy`입니다. `scipy`는 공식 TrackMan 경기·투수 Hungarian 1:1 매칭에 필요합니다.

## 1. 환경 확인과 smoke

저장소 루트에서 실행합니다.

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python exp/183_gpu_x176_resmlp.py smoke --device cuda --smoke-rows 4000
```

Smoke는 학습하지 않으며 X176/352 계약, robust-z, 네트워크 forward만 확인합니다. 비싼 TrackMan 매칭은 빈 lookup으로 대체합니다. 성공 시 `SMOKE_PASS_NO_TRAINING`이 출력됩니다.

## 2. 고정 validation 한 번 실행

```bash
python -u exp/183_gpu_x176_resmlp.py validate \
  --device cuda --amp \
  --seeds 0 1 2 --width 384 --depth 3 --dropout 0.30 \
  --batch 1024 --epochs 8 --lr 0.002 --weight-decay 0.0003 \
  2>&1 | tee lab/183_gpu_x176_live.log
```

PowerShell에서는 마지막 줄만 다음처럼 바꿉니다.

```powershell
python -u exp\183_gpu_x176_resmlp.py validate --device cuda --amp 2>&1 |
  Tee-Object -FilePath lab\183_gpu_x176_live.log
```

`validate`의 모델 명세는 코드에서 고정되어 있어 다른 seed/폭/깊이/epoch로 실행하면 시작 전에 중단합니다.

자동 순서:

1. 공식 `train.csv`와 `trackman_history.csv`만 읽습니다. `test.csv`는 읽지 않습니다.
2. `<=2022 -> 2023` 세 seed의 마지막 epoch 예측을 만듭니다.
3. 2023 라벨만으로 logit scale/intercept와 현 champion 혼합계수를 적합하고 `lock_2023.json`을 먼저 기록합니다.
4. 동일 명세로 `<=2023 -> 2024`를 새로 학습합니다.
5. 2023 보정과 혼합계수를 한 자리도 바꾸지 않고 2024에 적용합니다.
6. early/late와 pitcher-cluster bootstrap, batch-vs-singleton 행 독립성을 계산합니다.

학습만 AMP를 사용하고, 검증·행 독립성 예측은 공개 final builder와 동일하게 fp32로 고정합니다.

출력은 기본적으로 `lab/183_gpu_x176/`에 저장됩니다.

```text
prep_2023.npz / prep_2024.npz
state_2023_seed{0,1,2}.pt / state_2024_seed{0,1,2}.pt
endpoint_2023_seed*.npy / endpoint_2024_seed*.npy
endpoint_*_ensemble_raw.npy
endpoint_*_ensemble_calibrated.npy
candidate_*_locked.npy
lock_2023.json
validation_report.json
```

엄격 PASS 조건은 다음과 같습니다.

```text
2023 analytic locked blend gain >= 8
2024 transferred locked blend gain >= 8
2024 early gain > 0
2024 late gain > 0
pitcher bootstrap p025 > 0
모든 seed의 batch-vs-singleton max abs <= 1e-5
```

하나라도 실패하면 `FAIL_NO_DEPLOY`이며 final training, 테스트 추론, ZIP 제작, 제출을 진행하지 않습니다. 2024 결과를 본 뒤 파라미터를 바꿔 재실행하는 것도 금지합니다. 기존 출력 폴더가 있으면 스크립트가 덮어쓰지 않고 중단하는 이유입니다.

## 현재 로컬 QA

- Python 3.11 `py_compile`: 통과
- CPU smoke 4,000행: `raw=(4000,176)`, `network=(4000,352)`, finite output 통과
- 기존 `exp/182_gpu_stdmlp_handoff.py`: 수정하지 않음
- 학습 및 `test.csv` 읽기: 수행하지 않음
