# -*- coding: utf-8 -*-
"""2022->2023 discovery of a row-independent season-boundary form state.

For each pitcher, freeze the mean current-anchor residual over his final 300
regular-season pitches in 2022, shrink it, and decay it according to the
current row's own career-count advance beyond the frozen 2022 endpoint.  F
rows are untouched because their 2022->2023 regime discontinuity is known.
No other validation/test row is consulted and 2024 is not evaluated.
"""

from __future__ import annotations
import importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT_JSON=ROOT/"lab"/"173_boundary_form_decay_discovery.json"
OUT_TXT=ROOT/"lab"/"173_boundary_form_decay_discovery.txt"
TAIL=300
ALPHA=300.0
TAUS=(200.0,500.0,1000.0)
SCALES=(0.25,0.50,1.00)

def lm(path,name):
    s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def score(p,y):
    r=y.mean(); return 100000*(1-np.mean((p-y)**2)/(r*(1-r)))
def sg(b,c,y,m): return float(score(c[m],y[m])-score(b[m],y[m]))
def boot(b,c,y,p,draws=5000):
    z=(b-y)**2-(c-y)**2
    g=pd.DataFrame({'p':p,'s':z,'n':1}).groupby('p').agg({'s':'sum','n':'sum'})
    ss,nn=g.s.to_numpy(),g.n.to_numpy(); rng=np.random.default_rng(173); v=np.empty(draws)
    for i in range(draws):
        k=rng.integers(0,len(g),len(g)); v[i]=ss[k].sum()/nn[k].sum()
    v*=100000/(y.mean()*(1-y.mean()))
    return {'clusters':len(g),'p025':float(np.quantile(v,.025)),'median':float(np.median(v)),
            'p975':float(np.quantile(v,.975)),'prob_positive':float(np.mean(v>0))}

def main():
    e158=lm(ROOT/'exp'/'158_robust_conditional_tensor_eb.py','e158_173')
    e155=lm(ROOT/'exp'/'155_cause_runner_currentblend_gate.py','e155_173')
    cols=['season','game_month','game_type','pitcher_id','asof_pitcher_n','control_success']
    f=pd.read_csv(ROOT/'data'/'train.csv',usecols=cols,encoding='utf-8-sig',low_memory=False)
    f=f[f.season.isin([2022,2023])].reset_index(drop=True)
    rows={z:f[f.season.eq(z)].reset_index(drop=True) for z in (2022,2023)}
    y={z:rows[z].control_success.to_numpy(float) for z in rows}
    base={z:e158.load_current_baseline(e155,rows[z],z) for z in rows}
    src=rows[2022].copy(); src['resid']=y[2022]-base[2022]
    src=src[src.game_type.eq('R')].sort_values(['pitcher_id','asof_pitcher_n'])
    src=src.groupby('pitcher_id',sort=False,group_keys=False).tail(TAIL)
    src['resid']-=src.resid.mean()
    tab=src.groupby('pitcher_id').resid.agg(['sum','size']).reset_index()
    tab['state']=tab['sum']/(tab['size']+ALPHA)
    end=rows[2022].groupby('pitcher_id').asof_pitcher_n.max().rename('n_end').reset_index()
    tab=tab.merge(end,on='pitcher_id',how='left')
    v=rows[2023][['pitcher_id','asof_pitcher_n']].copy(); v['_i']=np.arange(len(v))
    v=v.merge(tab[['pitcher_id','state','n_end']],on='pitcher_id',how='left',sort=False,validate='m:1').sort_values('_i')
    state=v.state.fillna(0).to_numpy(float); gap=np.maximum(v.asof_pitcher_n.to_numpy(float)-v.n_end.fillna(v.asof_pitcher_n).to_numpy(float),0)
    R=rows[2023].game_type.eq('R').to_numpy(); F=~R; early=rows[2023].game_month.le(6).to_numpy(); allm=np.ones(len(v),bool)
    results=[]; effects={}
    for tau in TAUS:
        e=state*np.exp(-gap/tau); e[F]=0; effects[tau]=e
        for scale in SCALES:
            c=np.clip(base[2023]+scale*e,0,1)
            results.append({'tau':tau,'scale':scale,'coverage':float(v.state.notna().mean()),
                'gain':sg(base[2023],c,y[2023],allm),'F':sg(base[2023],c,y[2023],F),
                'R':sg(base[2023],c,y[2023],R),'early':sg(base[2023],c,y[2023],early),
                'late':sg(base[2023],c,y[2023],~early),'effect_std':float((scale*e).std())})
    best=max(results,key=lambda q:q['gain']); c=np.clip(base[2023]+best['scale']*effects[best['tau']],0,1)
    bs=boot(base[2023],c,y[2023],rows[2023].pitcher_id.to_numpy())
    passed=best['gain']>=8 and best['R']>0 and best['early']>0 and best['late']>0 and bs['p025']>0
    out={'experiment':173,'discovery':'2022 tail residual -> decayed 2023 state','reads_2024':False,
         'tail':TAIL,'alpha':ALPHA,'results':results,'best':best,'bootstrap':bs,
         'gate_pass':bool(passed),'status':'PASS_MAY_OPEN_2024' if passed else 'FAIL_DO_NOT_OPEN_2024'}
    OUT_JSON.write_text(json.dumps(out,indent=2)+"\n",encoding='utf-8')
    lines=['EXP173 season-boundary form decay (2022 -> 2023)',f'best={best}',f'bootstrap={bs}',f"FINAL: {out['status']}"]
    OUT_TXT.write_text('\n'.join(lines)+'\n',encoding='utf-8'); print('\n'.join(lines),flush=True)
if __name__=='__main__': main()
