# -*- coding: utf-8 -*-
"""
[96] exp/93 팔 결과 종합표 — 아침 검토용 (학습 없음, 1초)

lab/93_summary.txt (팔당 한 줄) + lab/93_{arm}_result.txt 의 game_type 줄을 읽어
HANDOFF §1.9 기준(시드평균/proj8 이득, paired 산포, shape, 벌점, d/K/혼합이득)으로 한 표에 모은다.
실행: PYTHONIOENCODING=utf-8 python -u exp/96_review_table.py
"""

import glob
import os
import re
import numpy as np

BASE_SOLO = {42: 851.2, 7: 856.5}
BASE_SEEDMEAN = float(np.mean(list(BASE_SOLO.values())))
BASE_ENS, BASE_PROJ8, BASE_PEN = 873.9, 889.0, 37.0
DESC = {"A": "team 조합3 (matchup/p·b team_role)", "B": "count_hand + base_out", "C": "A+B",
        "M4": "4클래스 (3→미들만 병합)", "OH": "one_hot_max_size=16 (CTR 없음)", "CTR1": "max_ctr_complexity=1"}

rows = []
if os.path.exists("lab/93_summary.txt"):
    for line in open("lab/93_summary.txt", encoding="utf-8"):
        kv = dict(re.findall(r"(\w+)=(\[[^\]]*\]|\S+)", line))
        arm = kv["arm"]
        solo = [float(x) for x in kv["solo"].strip("[]").split(",")]
        pairs = [float(x) for x in kv["paired"].strip("[]").split(",")]
        ens, sm, p8, pen = float(kv["ens"]), float(kv["seedmean"]), float(kv["proj8"]), float(kv["pen"])
        d, K = float(kv["d"]), float(kv["K"])
        bg = (d + K) ** 2 / (4 * K) if abs(d) < K else max(d, 0.0)
        gt = ""
        rp = f"lab/93_{arm}_result.txt"
        if os.path.exists(rp):
            m = re.search(r"game_type별 2시드 이득: (.*)$", open(rp, encoding="utf-8").read(), re.M)
            gt = m.group(1).strip() if m else ""
        rows.append((arm, solo, pairs, ens, sm, p8, pen, float(kv["shape"]), d, K, bg, gt))

print(f"기준 cat5 (하민 챔피언 로컬 재현): 시드 {BASE_SOLO[42]} / {BASE_SOLO[7]}  시드평균 {BASE_SEEDMEAN:.1f}  2시드 {BASE_ENS}  proj8 {BASE_PROJ8}  벌점 {BASE_PEN}")
print()
hdr = f"{'팔':5s} {'설명':34s} {'시드42':>7s} {'시드7':>7s} {'paired평균':>9s} {'산포':>5s} {'시드평균이득':>10s} {'proj8이득':>9s} {'shape':>6s} {'벌점':>5s} {'d':>6s} {'K':>6s} {'혼합이득':>7s}  game_type"
print(hdr)
print("-" * len(hdr))
for arm, solo, pairs, ens, sm, p8, pen, sh, d, K, bg, gt in rows:
    print(f"{arm:5s} {DESC.get(arm, ''):34s} {solo[0]:7.1f} {solo[1]:7.1f} {np.mean(pairs):+9.1f} {np.std(pairs):5.2f} "
          f"{sm - BASE_SEEDMEAN:+10.1f} {p8 - BASE_PROJ8:+9.1f} {sh:+6.1f} {pen:5.1f} {d:+6.1f} {K:6.1f} {bg:+7.1f}  {gt}")
skipped = [f for f in glob.glob("lab/93_*_SKIPPED.txt")]
for f in skipped:
    print(f"\n[{os.path.basename(f)}] " + open(f, encoding="utf-8").read().strip().splitlines()[0])
print("\n판정 규칙(§1.9/§2-1b): 주축 = 시드평균 이득·proj8 이득 + paired 산포. 산포가 작으면 +7 도 유의, 크면 +15 도 잡음.")
print("전이 위험(§1.13 세 축): 냉동 표(cat CTR 은 타겟 통계표) / 편향 vs 분산 / 레짐 의존.")
