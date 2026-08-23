"""
LG Aimers 9기 - 투구 제구 성공 확률 예측
[01] 전략 수립용 진단 스크립트

목적: 모델링 전에 반드시 답해야 할 질문들을 데이터로 확인한다.
  Q1. train/test는 시간 분할인가? (train=2019~2024, test=2025 가설 검증) ->
  Q2. 제구 성공률 r은 시즌별로 안정적인가? (전역 calibration 리스크)
  Q3. asof_* 피처의 결측(cold-start) 비율은 얼마나 되는가?
  Q4. 시즌 간 pitcher_id / batter_id 신규 등장 비율은? (분포 시프트 크기)
  Q5. trackman의 ID 체계가 메인 데이터와 연결 가능한가?
  Q6. 단순 규칙들의 BSS 상한선은 어디쯤인가? (신호 크기 감각 잡기)

실행: python 01_diagnose.py
"""

import numpy as np
import pandas as pd

DATA_DIR = "data"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)


# --------------------------------------------------------------------------
# 평가 지표 (대회 공식 산식 그대로)
# --------------------------------------------------------------------------
def brier(p, y):
    """Brier Score = mean((p - y)^2). 낮을수록 좋음."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def dacon_score(p, y):
    """
    대회 공식 점수.
    Score = max(0, 100000 * (1 - Brier / (r*(1-r))))
    주의: r은 '평가 대상 y의 평균'으로 계산된다. 검증 시에도 동일하게 valid의 y 평균을 쓴다.
    """
    y = np.asarray(y, dtype=float)
    r = y.mean()
    base = r * (1.0 - r)
    return max(0.0, 100000.0 * (1.0 - brier(p, y) / base))


# --------------------------------------------------------------------------
# 로딩 (1.47M행 x 49컬럼 → dtype 지정으로 메모리 절약)
# --------------------------------------------------------------------------
def load_train():
    df = pd.read_csv(f"{DATA_DIR}/train.csv")
    # BOM 제거 (test.csv 헤더에 \ufeff 확인됨 → train도 동일 가능성)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]

    # 메모리 다이어트: 정수형 다운캐스트
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")

    print(f"[load] train shape = {df.shape}")
    print(f"[load] memory      = {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    return df


# --------------------------------------------------------------------------
# Q1 + Q2. 시즌 구성과 시즌별 성공률
# --------------------------------------------------------------------------
def q1_q2_season_structure(df):
    print("\n" + "=" * 70)
    print("Q1/Q2. 시즌 구성 및 시즌별 제구 성공률")
    print("=" * 70)

    g = df.groupby("season").agg(
        n_rows=("control_success", "size"),
        success_rate=("control_success", "mean"),
        n_pitcher=("pitcher_id", "nunique"),
        n_batter=("batter_id", "nunique"),
    )
    g["brier_of_constant"] = g["success_rate"] * (1 - g["success_rate"])
    print(g.round(6))

    print(f"\n전체 평균 성공률 r = {df['control_success'].mean():.6f}")
    print(f"전체 상수예측 Brier = {df['control_success'].mean() * (1 - df['control_success'].mean()):.6f}")

    # 핵심 판정: test(245,789행 = 2025)와 대응되는 구조인가?
    print("\n--- 가설 검증 ---")
    print("가설: train=2019~2024(6시즌), test=2025(1시즌, 245,789행)")
    print(f"  train 시즌 목록          : {sorted(df['season'].unique())}")
    print(f"  train 행수 / 6           : {len(df) / 6:,.0f}   (test 245,789와 비교)")
    print(f"  시즌당 평균 행수         : {g['n_rows'].mean():,.0f}")
    if 2025 in g.index:
        print("  ⚠️ train에 2025가 포함되어 있음 → 시간분할 가설 재검토 필요")
    else:
        print("  ✅ train에 2025 없음 → 시간분할(2025 홀드아웃) 가설과 일치")

    # 성공률 추세: 마지막 시즌 기준으로 얼마나 흔들리는가
    rates = g["success_rate"]
    drift = rates.max() - rates.min()
    print(f"\n시즌별 성공률 최대-최소 차이 = {drift:.6f}")
    r_all = df["control_success"].mean()
    r_last = rates.iloc[-1]
    delta = abs(r_last - r_all)
    loss = 100000 * (delta ** 2) / (r_all * (1 - r_all))
    print(f"전체평균 vs 최종시즌 차이 δ = {delta:.6f}")
    print(f"  → 상수예측을 전체평균으로 했을 때 예상 점수 손실 ≈ {loss:.1f}점")
    print("  (이 값이 100점 이상이면 시즌 드리프트 보정이 매우 중요함)")

    return g


# --------------------------------------------------------------------------
# Q3. asof_* 결측(cold-start) 비율
# --------------------------------------------------------------------------
def q3_missing(df):
    print("\n" + "=" * 70)
    print("Q3. asof_* 피처 결측률 (cold-start 규모)")
    print("=" * 70)

    asof_cols = [c for c in df.columns if c.startswith("asof_")]
    miss = df[asof_cols].isna().mean().sort_values(ascending=False)
    print((miss * 100).round(3).to_string())

    # 결측 여부가 타겟과 관계있는지 → 결측 자체가 신호일 수 있음
    print("\n--- 결측 그룹 vs 비결측 그룹의 성공률 (결측 자체가 신호인가?) ---")
    for c in ["asof_pitcher_success_rate", "asof_batter_success_rate",
              "asof_pitcher_prev1_game_success_rate", "asof_pitcher_fastball_rate"]:
        if c not in df.columns:
            continue
        m = df[c].isna()
        if m.sum() == 0:
            continue
        print(f"  {c:45s} 결측 {m.mean()*100:5.2f}% | "
              f"결측시 y={df.loc[m, 'control_success'].mean():.4f} | "
              f"정상시 y={df.loc[~m, 'control_success'].mean():.4f}")

    # 표본 수 구간별 성공률 → 스무딩 강도 설계 근거
    print("\n--- asof_pitcher_n 구간별 성공률 (베이지안 스무딩 설계용) ---")
    bins = [-1, 0, 10, 50, 200, 1000, 5000, 10**9]
    labels = ["0", "1-10", "11-50", "51-200", "201-1000", "1001-5000", "5000+"]
    tmp = df.assign(_bin=pd.cut(df["asof_pitcher_n"], bins=bins, labels=labels))
    print(tmp.groupby("_bin", observed=True)["control_success"]
             .agg(["size", "mean"]).round(5).to_string())


# --------------------------------------------------------------------------
# Q4. 시즌 간 신규 선수 비율 (분포 시프트)
# --------------------------------------------------------------------------
def q4_new_players(df):
    print("\n" + "=" * 70)
    print("Q4. 시즌 간 신규 선수 등장 비율 (2025 test의 cold-start 규모 추정)")
    print("=" * 70)

    seasons = sorted(df["season"].unique())
    for i in range(1, len(seasons)):
        s = seasons[i]
        past = df[df["season"] < s]
        cur = df[df["season"] == s]

        for col in ["pitcher_id", "batter_id"]:
            known = set(past[col].unique())
            is_new = ~cur[col].isin(known)
            print(f"  season {s} | {col:11s} | 신규 선수 비율(행 기준) = {is_new.mean()*100:5.2f}%"
                  f" | 신규 선수 수 = {cur.loc[is_new, col].nunique()}")
        print()

    print("→ 마지막 시즌의 신규 비율이 2025 test에서도 비슷하게 나타날 것으로 예상.")
    print("  이 비율이 높으면 pitcher_id를 카테고리 피처로 직접 쓰는 건 위험함.")


# --------------------------------------------------------------------------
# Q5. trackman ID 연결 가능성 -->이미 없는걸로 알고있는데
# --------------------------------------------------------------------------
def q5_trackman_link(df):
    print("\n" + "=" * 70)
    print("Q5. trackman_history ID 체계 연결 가능성")
    print("=" * 70)

    tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", nrows=500_000)
    tm.columns = [c.replace("\ufeff", "").strip() for c in tm.columns]
    print(f"[load] trackman(부분) shape = {tm.shape}")
    print(f"  season 범위 : {sorted(tm['season'].unique())}")

    main_p = set(df["pitcher_id"].unique())
    tm_p = set(tm["pitcher_trackman_id"].dropna().unique())
    inter = main_p & tm_p

    print(f"\n  train pitcher_id 고유값        : {len(main_p)}")
    print(f"  trackman pitcher_trackman_id   : {len(tm_p)}")
    print(f"  교집합                          : {len(inter)}")
    print(f"  trackman 기준 커버율            : {len(inter)/max(len(tm_p),1)*100:.1f}%")

    if len(inter) / max(len(tm_p), 1) > 0.5:
        print("  ✅ ID 체계가 공유될 가능성 높음 → 투수별 구질(stuff) 피처 조인 가능")
    else:
        print("  ⚠️ ID가 직접 연결되지 않음 → 다른 키(팀+손+시즌 등)로 간접 매칭 필요")
        print("     또는 trackman은 '구종군별 평균 구질' 같은 집계 피처로만 활용")

    # ID 값 범위 비교 (같은 인코딩 체계인지 힌트)
    print(f"\n  train pitcher_id  범위: {df['pitcher_id'].min()} ~ {df['pitcher_id'].max()}")
    print(f"  trackman pitcher  범위: {tm['pitcher_trackman_id'].min()} ~ {tm['pitcher_trackman_id'].max()}")


# --------------------------------------------------------------------------
# Q6. 단순 규칙들의 점수 (신호 크기 감각 잡기)
# --------------------------------------------------------------------------
def q6_simple_baselines(df):
    print("\n" + "=" * 70)
    print("Q6. 단순 예측 규칙들의 점수 (시간분할 검증: 마지막 시즌을 valid로)")
    print("=" * 70)

    seasons = sorted(df["season"].unique())
    valid_season = seasons[-1]
    tr = df[df["season"] < valid_season]
    va = df[df["season"] == valid_season]
    print(f"  train seasons = {seasons[:-1]} ({len(tr):,}행)")
    print(f"  valid season  = {valid_season} ({len(va):,}행)\n")

    y_va = va["control_success"].values
    r_tr = tr["control_success"].mean()
    r_va = y_va.mean()
    print(f"  train r = {r_tr:.6f} | valid r = {r_va:.6f} | 차이 = {abs(r_tr-r_va):.6f}\n")

    cands = {}

    # (a) train 평균으로 상수 예측 → 시즌 드리프트 손실만 반영됨
    cands["const(train mean)"] = np.full(len(va), r_tr)

    # (b) valid 자신의 평균 → 정의상 정확히 0점 (이론적 하한 확인용)
    cands["const(valid mean)*"] = np.full(len(va), r_va)

    # (c) 투수 as-of 성공률 그대로 사용 (결측은 train 평균으로)
    p = va["asof_pitcher_success_rate"].fillna(r_tr).values
    cands["asof_pitcher_rate (raw)"] = np.clip(p, 0.01, 0.99)

    # (d) 투수 as-of 성공률을 train 평균 쪽으로 강하게 수축(shrink)
    #     → 신호가 약한 문제에서 수축이 왜 중요한지 확인
    for w in [0.1, 0.2, 0.3, 0.5]:
        p_raw = va["asof_pitcher_success_rate"].fillna(r_tr).values
        cands[f"asof_pitcher shrink w={w}"] = r_tr + w * (p_raw - r_tr)

    # (e) 베이지안 스무딩: (성공수 + a*prior) / (n + a)
    for a in [50, 200, 500]:
        n = va["asof_pitcher_n"].fillna(0).values
        rate = va["asof_pitcher_success_rate"].fillna(r_tr).values
        p_sm = (rate * n + r_tr * a) / (n + a)
        cands[f"asof_pitcher bayes a={a}"] = p_sm

    rows = []
    for name, p in cands.items():
        rows.append({"rule": name, "brier": brier(p, y_va), "score": dacon_score(p, y_va)})
    res = pd.DataFrame(rows).sort_values("score", ascending=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n※ (*) valid mean 상수는 정의상 0점 — 이 지표가 '평균만 맞추면 0점'임을 보여줌")
    print("※ shrink/bayes 계열이 raw보다 높으면 → 수축·스무딩이 필수라는 강한 증거")


# --------------------------------------------------------------------------
def main():
    df = load_train()

    q1_q2_season_structure(df)
    q3_missing(df)
    q4_new_players(df)
    try:
        q5_trackman_link(df)
    except Exception as e:
        print(f"\n[Q5 skip] trackman 로딩 실패: {e}")
    q6_simple_baselines(df)

    print("\n" + "=" * 70)
    print("진단 완료. 위 결과를 바탕으로 검증 전략과 피처 설계를 확정할 것.")
    print("=" * 70)


if __name__ == "__main__":
    main()