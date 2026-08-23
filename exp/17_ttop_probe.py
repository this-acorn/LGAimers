# -*- coding: utf-8 -*-
"""
TTOP(타순 일순 패널티) 축이 살아있는지 상한부터 잰다.
train.csv에서 '그 경기 안 실제 투구 수 / 타자별 상대 횟수'를 복원해
신호가 있는지 본다.  ※ 진단용. test에는 이 방식을 쓸 수 없다(행 독립성).
"""
import time, numpy as np, pandas as pd
T0 = time.time()
def log(m): print(f"  [{time.time()-T0:6.0f}s] {m}", flush=True)

cols = ['row_id','season','game_month','game_dayofweek','inning','outs_before','num_runners_on',
        'top_bottom','run_top_before','run_bot_before','pitcher_id','batter_id','asof_pitcher_n',
        'asof_pitcher_prev1_game_success_rate','control_success']
d = pd.read_csv('data/train.csv', usecols=cols)
d.columns = [c.replace('﻿','') for c in d.columns]
log(f"로드 {d.shape}")

d = d.sort_values(['pitcher_id','asof_pitcher_n'], kind='mergesort').reset_index(drop=True)
pid = d.pitcher_id.to_numpy()

# 경기 경계: 날짜키가 바뀌거나 prev1_game_success_rate가 바뀌면 새 경기
daykey = (d.season.to_numpy()*10000 + d.game_month.to_numpy()*100 + d.game_dayofweek.to_numpy()).astype('int64')
p1 = np.nan_to_num(d.asof_pitcher_prev1_game_success_rate.to_numpy('float64'), nan=-1.0)
newp = np.r_[True, pid[1:] != pid[:-1]]
newg = newp | (daykey != np.r_[0, daykey[:-1]]) | (np.abs(p1 - np.r_[0.0, p1[:-1]]) > 1e-12)
gid = np.cumsum(newg)
d['gid'] = gid
log("경기 세그먼트 완료")

# 세그먼트 내 투구 인덱스 + 세그먼트 크기
seg_start = np.flatnonzero(newg)
seg_len = np.diff(np.r_[seg_start, len(d)])
pc = np.arange(len(d)) - np.repeat(seg_start, seg_len)
d['pc'] = pc
d['seg'] = np.repeat(seg_len, seg_len)

print(f"\n세그먼트 {len(seg_start):,}개  |  투구수 분위 "
      f"p25={np.percentile(seg_len,25):.0f} p50={np.percentile(seg_len,50):.0f} "
      f"p75={np.percentile(seg_len,75):.0f} p95={np.percentile(seg_len,95):.0f} max={seg_len.max()}")

print("\n" + "="*88)
print("F. 참값 within-game 투구수 → 성공률")
print("="*88)
b = pd.cut(d.pc, [-1,14,29,44,59,74,89,9999],
           labels=['0-14','15-29','30-44','45-59','60-74','75-89','90+'])
print(d.groupby(b, observed=True).control_success.agg(['mean','size']))

print("\n" + "="*88)
print("G. 선발(그 경기 60구+)만 · 이닝 고정 후 투구수 효과  ← 순수 피로도")
print("="*88)
st = d[d.seg >= 60]
print(f"  선발 행수 {len(st):,}")
for inn in range(2, 8):
    s = st[st.inning == inn]
    if len(s) < 5000: continue
    q = pd.qcut(s.pc, 3, labels=['적','중','많'], duplicates='drop')
    gg = s.groupby(q, observed=True).control_success.agg(['mean','size'])
    if len(gg) == 3:
        lo, hi = gg['mean'].iloc[0], gg['mean'].iloc[2]
        se = np.sqrt(0.25/gg['size'].iloc[0] + 0.25/gg['size'].iloc[2])
        print(f"   {inn}회  적 {lo:.4f}  중 {gg['mean'].iloc[1]:.4f}  많 {hi:.4f}"
              f"   차 {hi-lo:+.4f}  ({(hi-lo)/se:+.1f}σ)  n={len(s):,}")

print("\n" + "="*88)
print("H. 타자 기준 TTO — 그 경기에서 이 투수를 몇 번째 상대? (참값)")
print("="*88)
d['tto'] = d.groupby(['pitcher_id','gid','batter_id'], sort=False).cumcount() + 1
g = d.groupby(d.tto.clip(upper=4)).control_success.agg(['mean','size'])
print(g)
print("\n  선발(60구+)만:")
st = d[d.seg >= 60]
g2 = st.groupby(st.tto.clip(upper=4)).control_success.agg(['mean','size'])
print(g2)
print("\n  선발 + 이닝 고정 (TTO는 이닝과 강하게 얽혀 있으므로 이게 진짜 검정):")
for inn in [4,5,6]:
    s = st[st.inning == inn]
    gg = s.groupby(s.tto.clip(upper=4)).control_success.agg(['mean','size'])
    gg = gg[gg['size'] > 2000]
    if len(gg) >= 2:
        txt = "  ".join(f"TTO{i}:{r['mean']:.4f}(n={int(r['size']):,})" for i, r in gg.iterrows())
        print(f"   {inn}회  {txt}")

print("\n" + "="*88)
print("I. 행 독립 대리지표 est_bf 가 참값 pc 를 얼마나 맞히나")
print("="*88)
bat_runs = np.where(d.top_bottom.to_numpy() == 'T',
                    d.run_top_before.to_numpy(), d.run_bot_before.to_numpy())
est_bf = 3*(d.inning.to_numpy()-1) + d.outs_before.to_numpy() + d.num_runners_on.to_numpy() + bat_runs
d['est_bf'] = est_bf
st = d[d.seg >= 60]
print(f"  전체:   corr(est_bf, 참값 pc) = {np.corrcoef(d.est_bf, d.pc)[0,1]:.4f}")
print(f"  선발만: corr(est_bf, 참값 pc) = {np.corrcoef(st.est_bf, st.pc)[0,1]:.4f}")
print(f"  선발만: corr(inning , 참값 pc) = {np.corrcoef(st.inning, st.pc)[0,1]:.4f}   ← inning 단독 대비 이득이 있어야 의미")
print(f"  선발만: corr(est_bf//9+1, 참값 TTO) = {np.corrcoef(st.est_bf//9+1, st.tto)[0,1]:.4f}")
print(f"\n  총 {time.time()-T0:.0f}초")
