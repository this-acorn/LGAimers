# -*- coding: utf-8 -*-
"""
[21] 외부 리뷰 주장 검증 — 순수 진단, 모델 학습 없음

검증 대상 (전부 사실 주장이므로 데이터로 판정 가능):
  A. 범주형 처리: CAT 3개의 카디널리티 / base_state가 runner 3개로 복원 가능한가
  B. 선수 ID: 카디널리티, 표본 분포, ★ID가 시간순 배정인가(=수치 분할에 의미가 있나)
  C. TTO 버그: cumcount()가 타석이 아니라 투구를 센다는 주장 재현 + 올바른 TTO로 재계산
  D. trackman 키: (tid, season) 키 생성 방식이 2025에 0개 키를 만드는가

★ 한 번의 로드로 전부 처리한다 (train.csv 351MB — 병렬 에이전트 4개가
  같은 파일을 4번 읽는 것보다 단일 공유 로드가 엄격히 낫다)
"""
import time
import numpy as np
import pandas as pd

T0 = time.time()


def log(m=""):
    print(m, flush=True)


def tick(m):
    print(f"  [{time.time()-T0:6.0f}s] {m}", flush=True)


COLS = ['season', 'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
        'runner_on_1b', 'runner_on_2b', 'runner_on_3b', 'base_state', 'top_bottom',
        'game_type', 'inning', 'outs_before', 'num_runners_on', 'run_top_before',
        'run_bot_before', 'game_month', 'game_dayofweek', 'asof_pitcher_n',
        'asof_pitcher_prev1_game_success_rate', 'control_success']

d = pd.read_csv('data/train.csv', usecols=COLS, encoding='utf-8-sig')
d.columns = [c.replace('﻿', '').strip() for c in d.columns]
tick(f"로드 {d.shape}")

# =====================================================================
log("\n" + "=" * 86)
log("A. 범주형 처리 — '범주형을 하나도 범주형으로 안 본다'는 주장의 실제 피해 규모")
log("=" * 86)

for c in ['top_bottom', 'game_type', 'base_state']:
    vals = d[c].unique()
    log(f"  CAT '{c}': {d[c].nunique()}개 값  {sorted(map(str, vals))[:10]}")

# base_state가 runner 3개 flag로 완전히 복원되는가 (=> 순서 인코딩 피해 0)
key = (d.runner_on_1b.astype(int) * 4 + d.runner_on_2b.astype(int) * 2
       + d.runner_on_3b.astype(int))
red = d.groupby(key)['base_state'].nunique()
log(f"\n  base_state 중복성 검사: runner 3-flag 조합별 base_state 고유값 수")
log(f"    {red.to_dict()}")
log(f"    → 전부 1이면 base_state는 이미 있는 3개 이진컬럼으로 100% 복원 가능"
    f"  (판정: {'완전 중복 — 인코딩 피해 0' if (red == 1).all() else '중복 아님'})")

log(f"\n  top_bottom: 2값 → 수치 분할과 범주 분할이 수학적으로 동일 (피해 0)")

# =====================================================================
log("\n" + "=" * 86)
log("B. 선수/팀 ID — 수치 취급이 실제로 해로운가")
log("=" * 86)

for c in ['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id']:
    log(f"  {c}: {d[c].nunique():,}개 고유값  범위 [{d[c].min()}, {d[c].max()}]")

rows_per_p = d.groupby('pitcher_id').size()
log(f"\n  투수별 행수 분포: p10={rows_per_p.quantile(.1):.0f} "
    f"p50={rows_per_p.median():.0f} p90={rows_per_p.quantile(.9):.0f} "
    f"max={rows_per_p.max():,}")
log(f"    HGB max_bins=255 / 투수 {d.pitcher_id.nunique()}명 → ID축은 255구간으로 뭉개짐")

# ★ 핵심: ID가 시간순(데뷔순) 배정이면 수치 분할 = '경력 단계' 분할이라 의미가 있다
first_season = d.groupby('pitcher_id')['season'].min()
c1 = np.corrcoef(first_season.index.to_numpy(float), first_season.to_numpy(float))[0, 1]
first_season_b = d.groupby('batter_id')['season'].min()
c2 = np.corrcoef(first_season_b.index.to_numpy(float),
                 first_season_b.to_numpy(float))[0, 1]
log(f"\n  ★ corr(pitcher_id, 첫 등장 시즌) = {c1:+.3f}")
log(f"  ★ corr(batter_id,  첫 등장 시즌) = {c2:+.3f}")
log(f"    |corr|이 크면 → ID 수치 분할이 사실상 '데뷔 시기/경력' 분할이라 무의미하지 않음")
log(f"    |corr|이 0에 가까우면 → 리뷰 주장대로 임의 구간 자르기 = 순수 과적합 위험")

# =====================================================================
log("\n" + "=" * 86)
log("C. TTO 버그 검증 — cumcount()가 타석이 아니라 투구를 세는가")
log("=" * 86)

d = d.sort_values(['pitcher_id', 'asof_pitcher_n'], kind='mergesort').reset_index(drop=True)
pid = d.pitcher_id.to_numpy()
daykey = (d.season.to_numpy() * 10000 + d.game_month.to_numpy() * 100
          + d.game_dayofweek.to_numpy()).astype('int64')
p1 = np.nan_to_num(d.asof_pitcher_prev1_game_success_rate.to_numpy('float64'), nan=-1.0)
newp = np.r_[True, pid[1:] != pid[:-1]]
newg = newp | (daykey != np.r_[0, daykey[:-1]]) | (np.abs(p1 - np.r_[0.0, p1[:-1]]) > 1e-12)
d['gid'] = np.cumsum(newg)

# 세그먼트 내 투구 번호 (exp/17의 pc와 동일)
seg_start = np.flatnonzero(newg)
seg_len = np.diff(np.r_[seg_start, len(d)])
d['pc'] = np.arange(len(d)) - np.repeat(seg_start, seg_len)
d['seg'] = np.repeat(seg_len, seg_len)

# (1) 버그 버전 — exp/17_ttop_probe.py 그대로
d['tto_bug'] = d.groupby(['pitcher_id', 'gid', 'batter_id'], sort=False).cumcount() + 1

# (2) 올바른 버전 — 타석 경계를 먼저 잡고, 타석 단위로 센다
bat = d.batter_id.to_numpy()
gid = d.gid.to_numpy()
new_pa = np.r_[True, (gid[1:] != gid[:-1]) | (bat[1:] != bat[:-1])]
d['pa_id'] = np.cumsum(new_pa)
first = d.loc[new_pa, ['pa_id', 'pitcher_id', 'gid', 'batter_id']].copy()
first['tto'] = first.groupby(['pitcher_id', 'gid', 'batter_id'], sort=False).cumcount() + 1
d = d.merge(first[['pa_id', 'tto']], on='pa_id', how='left')
tick("TTO 재계산 완료")

log(f"\n  [버그판] tto_bug 분포 (clip4):")
log(str(d.tto_bug.clip(upper=4).value_counts().sort_index().to_dict()))
log(f"  [정상판] tto     분포 (clip4):")
log(str(d.tto.clip(upper=4).value_counts().sort_index().to_dict()))
log(f"\n  타석당 평균 투구수 = {len(d)/d.pa_id.nunique():.2f}  "
    f"(버그판이 이 값을 TTO로 착각하고 있었다)")

log(f"\n  ── 선발(그 경기 60구+) · 이닝 고정 후 '올바른 TTO' 효과 ──")
st = d[d.seg >= 60]
for inn in [4, 5, 6]:
    s = st[st.inning == inn]
    gg = s.groupby(s.tto.clip(upper=4))['control_success'].agg(['mean', 'size'])
    gg = gg[gg['size'] > 2000]
    if len(gg) >= 2:
        txt = "  ".join(f"TTO{i}:{r['mean']:.4f}(n={int(r['size']):,})"
                        for i, r in gg.iterrows())
        log(f"   {inn}회  {txt}")

# 행 독립 대리지표 est_tto 의 정확도
bat_runs = np.where(d.top_bottom.to_numpy() == 'T',
                    d.run_top_before.to_numpy(), d.run_bot_before.to_numpy())
d['est_pa'] = (3 * (d.inning.to_numpy() - 1) + d.outs_before.to_numpy()
               + d.num_runners_on.to_numpy() + bat_runs)
d['est_tto'] = np.minimum(d.est_pa // 9 + 1, 4)
st = d[d.seg >= 60]
log(f"\n  ── 행 독립 대리지표 정확도 ──")
log(f"   전체:   corr(est_tto, 참 TTO) = {np.corrcoef(d.est_tto, d.tto)[0,1]:+.4f}")
log(f"   선발만: corr(est_tto, 참 TTO) = {np.corrcoef(st.est_tto, st.tto)[0,1]:+.4f}")
log(f"   선발만: corr(inning,  참 TTO) = {np.corrcoef(st.inning, st.tto)[0,1]:+.4f}"
    f"   ← inning 단독 대비 이득이 있어야 새 정보")
acc = (st.est_tto == st.tto.clip(upper=4)).mean()
log(f"   선발만: est_tto 정확 일치율 = {acc*100:.1f}%")

# =====================================================================
log("\n" + "=" * 86)
log("D. trackman 키 생성 — 2025에 키가 0개가 되는가")
log("=" * 86)

FIRST = ["DOO_BEA", "HAN_EAG", "KIA_TIG", "KIW_HER", "KT_WIZ", "LG_TWI",
         "LOT_GIA", "NC_DIN", "SAM_LIO", "SK_WYV", "SSG_LAN"]
tm = pd.read_csv('data/trackman_history.csv',
                 usecols=['season', 'pitcher_trackman_id', 'pitcher_team'],
                 encoding='utf-8-sig')
tm.columns = [c.replace('﻿', '').strip() for c in tm.columns]
tm1 = tm[tm.pitcher_team.isin(FIRST)]
tick(f"trackman 1군 {len(tm1):,}행 / 시즌 {sorted(tm1.season.unique())}")

match = pd.read_csv('lab/trackman_match_v3.csv')
strict = match[match['round'] <= 2]
log(f"  매칭 strict(round<=2) {len(strict)}명 / 전체 {len(match)}명")

tm_keys = set(map(tuple, tm1[['pitcher_trackman_id', 'season']].drop_duplicates().to_numpy()))
pid2tid = dict(zip(strict.pid, strict.tid))

# train에서 각 (매칭된 투수, 시즌) 조합이 trackman 키를 갖는가
tp = d[d.pitcher_id.isin(pid2tid)][['pitcher_id', 'season']].drop_duplicates()
tp['tid'] = tp.pitcher_id.map(pid2tid)
tp['has_key'] = [(t, s) in tm_keys for t, s in zip(tp.tid, tp.season)]
# 그 시즌 이전에 trackman 이력이 있는가 (있으면 '쓸 수 있었는데 버려진' 경우)
prior_seasons = {}
for t, s in tm_keys:
    prior_seasons.setdefault(t, []).append(s)
tp['has_past'] = [any(ss < s for ss in prior_seasons.get(t, []))
                  for t, s in zip(tp.tid, tp.season)]

log(f"\n  매칭 투수의 (투수,시즌) 조합 {len(tp):,}개 중")
log(f"    현재 로직으로 키 생성됨(그 시즌 trackman 행 존재): "
    f"{tp.has_key.sum():,} ({tp.has_key.mean()*100:.1f}%)")
lost = tp[(~tp.has_key) & tp.has_past]
log(f"    ★ 과거 이력은 있는데 키가 없어 버려짐: {len(lost):,} "
    f"({len(lost)/len(tp)*100:.1f}%)  ← 리뷰가 지적한 손실분")
log(f"\n  시즌별 키 생성률:")
for s, g in tp.groupby('season'):
    log(f"    {s}: {g.has_key.mean()*100:5.1f}%  (n={len(g)})")
log(f"\n  ★ 2025: trackman 데이터 자체가 없음 → (tid, 2025) 키 0개 "
    f"→ 현재 방식이면 배포 시 100% NaN (로컬 검증에선 안 보이는 결함)")

log(f"\n총 소요 {time.time()-T0:.0f}초")
log("=" * 86)
