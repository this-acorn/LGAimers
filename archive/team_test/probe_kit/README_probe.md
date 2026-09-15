# 프로브 키트 v3 — 이종 모델 4인 분산 측정

## 목적 한 줄

우리 챔피언(CatBoost 5클래스, LB 1033.99)과 **비슷하게 잘하면서 다르게 틀리는** 모델을
찾는다. 이길 필요 없다 — 혼합 이득에서 다양성 K가 크면 30점 져도 가치가 있다.
단, EBM 실측이 보여줬듯 **K가 아무리 커도 격차 d가 −K보다 깊으면 이득은 0**이다
(EBM: K=315인데 d=−413 → w*=0). 목표는 d를 −30 안쪽으로 가져오는 것.

## 폴더 내용

```
X_train_sub.parquet   학습 40만 행 × 79피처  (전원 동일한 행 — 비교 가능성의 핵심)
y_train_sub.npy       이진 타겟 (0/1) — 참고용
y5_train_sub.npy      ★ 5클래스 타겟 (0성공/1미들만/2리버스만/3미들∩리버스/4빅미스, -1=결측)
X_valid.parquet       검증 253,507행 × 79피처 (2024 시즌 전체)
y_valid.npy           검증 이진 타겟 (채점용)
mc04_ref.npy          챔피언(2시드 평균)의 검증 예측 — d·K 계산 기준
score_probe.py        채점기 v3 — 이 출력이 곧 보고 양식
```

## ★ 규칙 3줄 (어기면 결과 비교 불가)

```
① 5클래스로 학습하라. 챔피언의 +36이 타겟에서 왔다 — 이진으로 배우면 그만큼 손해를
   안고 비교당한다. y5_train_sub.npy 사용, -1 행은 제외, 제출 확률 = P(클래스 0).
② 전처리 통계(정규화 평균·표준편차 등)는 40만 train에서만 fit 하라.
   early stopping이 필요하면 40만 안에서 10%를 떼서 쓰고, X_valid는 마지막 예측에만.
③ 게이트는 전부 score_probe가 출력하는 ★로컬 혼합 이득★ 기준이다 — 단독 점수가 아니다!
   단독으로 져도 다르게 틀리면 통과할 수 있다. 시드 42는 예비 필터:
     · 혼합 이득 +8 이상 (단일시드면 보수 보정 K−27 이득까지 +8) → 시드 7 확정 추가
     · 원시 혼합 이득만 +8 → 회색지대, 시간 남으면 추가
     · w* = 0 → 즉시 기각 보고
   최종 판정은 2시드 평균 (기준 mc04_ref도 2시드 평균이라 대칭).
```

## 공통 절차

1. `X_train_sub` + `y5_train_sub`(−1 제외)로 학습, 시드 42.
2. `X_valid` 전체에 **P(클래스 0 = 성공)** 예측 → `preds_<모델>_seed42.npy` (float32, 행 순서 유지).
3. `python score_probe.py preds_<모델>_seed42.npy 모델명` 실행 —
   결과가 **`report_모델명_1seed.txt`로 자동 저장**됩니다. **그 txt + npy를 단톡에** (복사 불필요).
4. 규칙 ③의 게이트에 따라 시드 7 추가 시
   `python score_probe.py preds_seed42.npy preds_seed7.npy 모델명` 으로 재채점 →
   `report_모델명_2seed.txt` + npy 재전송.
5. ⚠ **프로브를 대회에 제출하지 마세요.** 슬롯은 팀 공유이고, 40만 행 프로브 점수는
   정보값이 없으며, 제출 zip은 서버 버전이 맞춰진 검증 파이프라인에서만 빌드합니다.
6. 마감: **08-27 (수) 22:00 KST**.

점수 눈높이: 40만 행 학습이라 절대값이 눌린다(같은 조건 CatBoost가 40만에서 713.6,
전체 122만에서 830.9). **전체학습 보정 +117 안팎은 취합 때 내가 적용** — d·K만 보면 된다.
X_valid의 행을 섞거나 X_valid로 집계를 내는 것은 금지 (대회 행 독립성과 같은 이유).

## 후보들의 성격 — 승자는 예상이 아니라 계산된 혼합 이득으로 정한다

```
no-ID CatBoost   단독 격차 d가 얕을 가능성 최고 (검증된 계열). 관건은 K —
                 같은 계열의 실측 K는 시드간 27 ~ 인접피처셋 70 ~ 큰 피처셋 차이 191
                 사이 어딘가다. ID 두 개는 중요도 최상위라 191 쪽에 가까울 수도 있다.
RealMLP / FT     큰 K를 만들 가능성 최고. d가 −60 안쪽이면 no-ID를 역전할 수 있다.
ExtraTrees       K는 기대되나 d가 깊어질 위험 (EBM처럼 죽는 경로).
```

## 담당 배정

### A — 두(랩실 GPU): RealMLP 먼저 → 끝나면 FT-Transformer

**1순위 RealMLP:**
```bash
pip install pytabkit
```
```python
import pandas as pd, numpy as np
from pytabkit import RealMLP_TD_Classifier
CAT = ["top_bottom", "game_type", "base_state"]
X = pd.read_parquet("X_train_sub.parquet")
y5 = np.load("y5_train_sub.npy")
Xv = pd.read_parquet("X_valid.parquet")
for c in CAT: X[c] = X[c].astype("category"); Xv[c] = Xv[c].astype("category")
ok = y5 >= 0
m = RealMLP_TD_Classifier(random_state=42, device="cuda")
m.fit(X[ok], y5[ok])                     # 5클래스 학습
p = m.predict_proba(Xv)[:, 0]            # 클래스 0 = 성공
np.save("preds_realmlp_seed42.npy", p.astype("float32"))
```
메타튜닝 기본값 그대로(내부 검증 분할을 스스로 함 — 규칙 ② 충족).

**2순위 FT-Transformer** (RealMLP 제출 후):
```bash
pip install pytorch_tabular
```
`FTTransformerConfig`, continuous 76개(`fillna(-999)`) + categorical 3개, 5클래스,
`batch_size=4096`, `max_epochs=20`, early stopping은 train 내부 10% 분할로.
**설정을 기록해서 함께 보낼 것** (embedding 차원·층수·lr·batch·멈춘 epoch) —
트랜스포머는 설정에 민감해서 "기본값 하나 돌려보고 기각"이면 판정이 아니라 미측정이 된다.

### B — 팀원 (CPU): ExtraTrees
```python
import pandas as pd, numpy as np
from sklearn.ensemble import ExtraTreesClassifier
CAT = ["top_bottom", "game_type", "base_state"]
X = pd.read_parquet("X_train_sub.parquet")
y5 = np.load("y5_train_sub.npy")
Xv = pd.read_parquet("X_valid.parquet")
for c in CAT:
    codes = {v: i for i, v in enumerate(pd.concat([X[c], Xv[c]]).unique())}
    X[c] = X[c].map(codes); Xv[c] = Xv[c].map(codes)
X = X.fillna(-999); Xv = Xv.fillna(-999)
ok = y5 >= 0
m = ExtraTreesClassifier(n_estimators=600, min_samples_leaf=20, max_features=0.6,
                         n_jobs=-1, random_state=42)
m.fit(X[ok], y5[ok])
np.save("preds_extratrees_seed42.npy", m.predict_proba(Xv)[:, 0].astype("float32"))
```
(범주 코드표를 train+valid 합쳐 만드는 건 값 목록일 뿐 통계가 아니라 누수 아님)

### C — 팀원 (CPU): CatBoost **no-ID** 프로브
```bash
pip install catboost
```
```python
import pandas as pd, numpy as np
from catboost import CatBoostClassifier, Pool
CAT = ["top_bottom", "game_type", "base_state"]
DROP = ["pitcher_id", "batter_id"]
X = pd.read_parquet("X_train_sub.parquet").drop(columns=DROP)
y5 = np.load("y5_train_sub.npy")
Xv = pd.read_parquet("X_valid.parquet").drop(columns=DROP)
ok = y5 >= 0
m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.04,
                       l2_leaf_reg=10.0, loss_function="MultiClass",
                       thread_count=8, random_seed=42,
                       verbose=False, allow_writing_files=False)
m.fit(Pool(X[ok], y5[ok], cat_features=CAT))
np.save("preds_noid_seed42.npy",
        m.predict_proba(Pool(Xv, cat_features=CAT))[:, 0].astype("float32"))
```
왜 다시 재나: 예전에 **챔피언 후보**로는 기각됐지만(−6.7/−12.4) **파트너** 질문은 잰 적이
없다. 단독 격차가 얕을 가능성이 네 후보 중 가장 높고, CatBoost 계열이라 2025 전이가
증명돼 있다. 혼합 이득이 최고일지는 K에 달렸다 — 그래서 잰다.

### D — 내 기계: EBM ✕ 완료·기각
배포형 d=−413 → w*=0 / 낙관 상한(전체학습 보정 +117)으로도 d≈−295 → 이득 +0.3.

먼저 끝난 사람은 새 모델 대신 **자기 모델의 시드 7**을 돌려주세요 (짝지음 확증용).

## 판정과 그 다음

- 2시드 로컬 혼합 이득 **+15 이상 → 승격**: 전체 데이터 재학습(내 기계, 5h)
  → **단독 1회 제출로 2025 전이 실측**(HGB가 로컬 무해·실전 −68이었던 전례 때문에 필수)
  → 리더보드 실측 d로 혼합비 재계산 → 혼합 제출.
- 로컬 선별은 로컬 K로. 2025 추정에 쓰는 **K 전이율 0.614는 실측 1쌍(CB65↔CS79)에서
  나온 가정**이라 채점기가 0.4/0.614/0.8 민감도를 병기한다.
  **실제 혼합비는 리더보드 숫자로만** 계산한다.
- 기대치: 완벽한 파트너여도 2025 혼합 상한 **+29**. 이 축은 +10~25짜리다.

## 다른 후보들의 우선순위가 낮은 이유 (영구 사망이 아니라 남은 5일 예산 기준)

| 후보 | 판정 |
|---|---|
| LightGBM / XGBoost / HGB | 실측 — LightGBM은 사실상 CatBoost의 9번째 시드(K≈0), HGB는 2025 전이 −68 |
| OneVsAll·손실함수 변경 | 실측 — 같은 함수로 수렴 (K=12) |
| DART | 메커니즘은 시드평균과 다르지만(학습 중 트리 드롭) 수확 대상이 같은 분산 축. 정규화 팔 전멸 + 숙주(XGB/LGBM) 실측 약세 → 예산 밖 |
| RuleFit | GBDT의 자기 규칙 압축 → d 음수 확정 + 교사와 강상관이라 K 작음 (증류 +0.5 전례) |
| kNN / NCA | 79차원 국소평균의 d가 −200급으로 추정(자체 논거). EBM 사례가 보여주듯 깊은 d는 K로 못 살린다 → 예산 제외. **실측 확정 사망은 아님** |
| FM (ID 포함) | 선수 잠재벡터는 CTR과 다른 물건이지만 과거에 얼어붙는 위험은 같고, 그 계열 실측이 −160.9 / 전이 0.14 → 예산 밖 |
| FM (ID 제외) | EBM 실패로 저차 가법류 기대치가 내려갔다 — **증명은 아니지만** 5일 예산에서 제외 |
| TabPFN 계열 | 147만 행 규모 밖 |
| RNN/LSTM/시계열 | 테스트 행 독립성 규칙과 충돌. prev1/3/5가 이미 합법 요약 |
| GNN | ≤2024 그래프 = 냉동 표 문제로 회귀 |
| AutoML | 이 키트가 같은 정찰을 더 통제된 방식으로 함 |
