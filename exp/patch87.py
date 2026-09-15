# -*- coding: utf-8 -*-
"""exp/66 -> exp/87 : cat_features 확장 축 측정 (하민 +24.51 발견의 연장)."""
import io
s = io.open("exp/66_mc_lr04.py", encoding="utf-8").read()
_o = s.index('"""'); s = s[s.index('"""', _o+3)+3:]

HEAD = '''# -*- coding: utf-8 -*-
"""
[87] cat_features 확장 — 하민의 +24.51 발견을 축으로 확장 측정 (전체 데이터, 2시드 짝지음)

발견 (팀원 하민, 08-27 LB 확정):
  submit14(우리, 1033.99)와 target5_teamcat(1059.05)의 코드 실질 차이는 단 하나 —
      TEAM_CAT = ["pitcher_team_id", "batter_team_id"] 를 cat_features 로 지정
  피처 79개/CS/구종/5클래스/it500-d6-lr0.08-l2=10/8시드 전부 동일. **+25.06**.

  우리 원장의 "범주형 처리 개선 = 피해 0" 판정은 틀렸다 — 선수 ID(800명대 고카디널리티,
  cb_cat -173.9)만 테스트하고 **팀 ID(12개, 저카디널리티)** 는 본 적이 없었다.

가설: CatBoost 는 cat_features 들의 **조합(CTR combination)** 을 자동 생성한다
  (max_ctr_complexity 기본 4). 팀 ID 추가의 이득은 team x base_state, team x game_type
  같은 조합에서 나왔을 가능성이 크다. 그렇다면 저카디널리티 컬럼을 더 넣으면
  조합이 더 늘어난다 -> 더 오를 수도, 과적합으로 무너질 수도 있다. 측정한다.

★ 기준 설정을 하민 것(lr0.08)에 맞춘다 — 팀 최고 기록 위에 쌓는 게 목적이므로.
  in-run 베이스라인(cat3)을 함께 돌려 컬럼 순서 교란 없이 짝지음한다.

팔 (79피처/멀티클래스5/it500-d6-lr0.08-l2=10/thread14/시드[42,7] 전부 고정):
  cat3   현행 3개                                    <- in-run 베이스라인
  cat5   +pitcher_team_id +batter_team_id            <- 하민 (LB +24.51 / 하민 로컬 +2.0)
  cat8   cat5 +balls_before +strikes_before +outs_before
  cat12  cat8 +pitcher_hand +batter_hand +inning +game_month
  (season 은 제외 — 드리프트 축이라 레짐 베팅 위험)

판독:
  · cat5 가 로컬에서 +2 근처면 하네스가 이 축을 재현하는 것 -> 상대 순위 신뢰 가능
  · 단 이 축은 하민 실측이 로컬 +2.0 -> 실전 +24.51 (12배). 로컬 크기로 실전을 추정하지 마라.
    로컬은 **방향과 순위**만 본다. 절대 크기는 실전에서 다시 잰다.

실행: PYTHONIOENCODING=utf-8 python -u exp/87_catfeat.py   (~5.5시간)
"""'''
s = HEAD + s
s = s.replace("learning_rate=0.04", "learning_rate=0.08")

TAIL = '''# --------------------------------------------------------------------------
FEATS = BASE + ENG18 + s12.CS_FEATS + s12.PT_FEATS
m_tr = tr["_cls"].to_numpy() >= 0
y_cls = tr["_cls"].to_numpy()[m_tr]
log(f"멀티클래스 학습 행: {m_tr.sum():,} / {len(tr):,}   피처 {len(FEATS)}")

C3 = list(s12.CAT)
ARMS = [
    ("cat3", C3),
    ("cat5", C3 + ["pitcher_team_id", "batter_team_id"]),
    ("cat8", C3 + ["pitcher_team_id", "batter_team_id",
                   "balls_before", "strikes_before", "outs_before"]),
    ("cat12", C3 + ["pitcher_team_id", "batter_team_id",
                    "balls_before", "strikes_before", "outs_before",
                    "pitcher_hand", "batter_hand", "inning", "game_month"]),
]


def build(d, cats):
    """피처 순서를 고정한 채 cats 만 문자열로. 팀ID는 float 표기 방지 위해 int64 경유."""
    out = pd.DataFrame(index=d.index)
    for c in FEATS:
        if c in cats:
            v = d[c]
            out[c] = (v.fillna(-1).astype("int64").astype(str)
                      if pd.api.types.is_numeric_dtype(v) else v.astype(str))
        else:
            out[c] = d[c]
    return out


r_ = float(y_va.mean())
DEN = r_ * (1 - r_)
res = {}
for name, cats in ARMS:
    Xtr = build(tr[m_tr], cats)
    Xva = build(va, cats)
    ptr = Pool(Xtr, y_cls, cat_features=cats)
    pva = Pool(Xva, cat_features=cats)
    del Xtr, Xva
    preds, solo = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(**CB_PRM, random_seed=sd).fit(ptr)
        p = m.predict_proba(pva)[:, 0]
        preds.append(p)
        solo.append(raw_score(p, y_va))
        tick(f"{name} seed={sd}  {solo[-1]:8.1f}  ({time.time()-t0:.0f}s)")
        del m
    del ptr, pva
    ens = np.mean(preds, axis=0)
    np.save(f"lab/87_{name}.npy", ens.astype("float32"))
    e2 = raw_score(ens, y_va)
    ms = float(np.mean(solo))
    p8 = 1.75 * e2 - 0.75 * ms
    pen = 100000 * (ens.mean() - r_) ** 2 / DEN
    res[name] = (e2, ms, p8, pen, solo, ens)
    log(f"  -> {name:6s} 2시드 {e2:7.1f}  시드평균 {ms:7.1f}  proj8 {p8:7.1f}  벌점 {pen:5.1f}")
    log("")

log("=" * 92)
log("판정 — cat_features 확장 (in-run 베이스라인 cat3 대비)")
log("=" * 92)
b2, bm, bp = res["cat3"][0], res["cat3"][1], res["cat3"][2]
log(f"  {'팔':7s} {'2시드':>8s} {'시드평균':>9s} {'proj8':>8s} {'벌점':>6s} "
    f"{'paired 차이':>22s}")
for k, (e2, ms, p8, pen, solo, _) in res.items():
    pr = [solo[i] - res["cat3"][4][i] for i in range(len(solo))]
    ps = " / ".join(f"{v:+.1f}" for v in pr)
    log(f"  {k:7s} {e2:8.1f} {ms:9.1f} {p8:8.1f} {pen:6.1f}   {ps:>14s} "
        f"(평균 {np.mean(pr):+.1f}, 산포 {np.std(pr):.2f})")
log("")
log("  ※ 하민 실측: 이 축은 로컬 +2.0 -> 실전 +24.51 (12배). 로컬 크기로 실전 추정 금지.")
log("     로컬은 **방향과 순위**만. 최선 팔을 전체 배포 후 실전에서 다시 잰다.")
best = max(res, key=lambda k: res[k][1])
log(f"  시드평균 기준 최선: {best}  (cat3 대비 {res[best][1] - bm:+.1f})")
log("=" * 92)
'''
s = s[:s.index("FEATS = BASE + ENG18")] + TAIL
io.open("exp/87_catfeat.py", "w", encoding="utf-8", newline="").write(s)
import ast; ast.parse(io.open("exp/87_catfeat.py", encoding="utf-8").read())
print("exp/87_catfeat.py 생성 (syntax OK)")
