"""Leakage-safe, context-adjusted pitcher pressure profiles.

All target-year features are computed solely from earlier seasons. Psychological
response is treated as a proxy: pitcher residual after removing observable game
context, not as a direct measurement of mental state.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CONDITIONS=('high_li','extreme_li','traffic','risp','two_out_risp','three_ball','full_count','late_high_li','close_high_li','compound_crisis')

def context_frame(d):
    r=d.reset_index(drop=True);balls=pd.to_numeric(r.balls_before,errors='coerce').fillna(0).astype(int);strikes=pd.to_numeric(r.strikes_before,errors='coerce').fillna(0).astype(int);outs=pd.to_numeric(r.outs_before,errors='coerce').fillna(0).astype(int);inning=pd.to_numeric(r.inning,errors='coerce').fillna(0);li=pd.to_numeric(r.li,errors='coerce').fillna(0);score=pd.to_numeric(r.score_diff_pitcher_team,errors='coerce').fillna(0);runners=pd.to_numeric(r.num_runners_on,errors='coerce').fillna(0);risp=r.runner_on_2b.eq(1)|r.runner_on_3b.eq(1)
    x=pd.DataFrame(index=r.index);x['count']=balls.astype(str)+'-'+strikes.astype(str);x['outs']=outs.astype(str);x['base']=r.base_state.astype(str);x['hand']=r.pitcher_hand.astype(str)+'-'+r.batter_hand.astype(str);x['inning_bin']=pd.cut(inning,[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str);x['li_bin']=pd.cut(li,[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str);x['home']=r.top_bottom.astype(str).eq('T').astype(int).astype(str);x['score_bin']=pd.cut(score,[-np.inf,-3,-1,1,3,np.inf],labels=False).fillna(-1).astype(int).astype(str)
    c=pd.DataFrame({'high_li':li>=1.5,'extreme_li':li>=3,'traffic':runners>0,'risp':risp,'two_out_risp':(outs==2)&risp,'three_ball':balls==3,'full_count':(balls==3)&(strikes==2),'late_high_li':(inning>=7)&(li>=1.5),'close_high_li':(score.abs()<=1)&(li>=1.5),'compound_crisis':(li>=1.5)&risp&(balls>=2)},index=r.index).astype('int8')
    return x,c

def context_adjusted_residual(history,alpha=400.):
    """Leave-one-out smoothed context expectation for each historical row."""
    h=history.reset_index(drop=True);ctx,_=context_frame(h);y=h.control_success.to_numpy(float);prior=float(np.mean(y));levels=[['count','outs','base','hand','inning_bin','li_bin','home'],['count','outs','base','hand'],['count','outs','base'],['count','hand']]
    estimates=[];reliabilities=[]
    temp=ctx.copy();temp['_y']=y
    for keys in levels:
        agg=temp.groupby(keys,dropna=False)['_y'].agg(['sum','count']);idx=pd.MultiIndex.from_frame(ctx[keys]);s=agg['sum'].reindex(idx).to_numpy();n=agg['count'].reindex(idx).to_numpy();rate=(s-y+alpha*prior)/(np.maximum(n-1,0)+alpha);estimates.append(rate);reliabilities.append(np.maximum(n-1,0)/(np.maximum(n-1,0)+alpha))
    # Reliability-weighted ensemble of detailed and backed-off contexts.
    E=np.column_stack(estimates);W=np.column_stack(reliabilities);expected=(E*(.25+W)).sum(1)/(.25+W).sum(1)
    return y-expected,expected

def build_profiles(history,alpha_context=400.,alpha_pitcher=300.):
    h=history.reset_index(drop=True);residual,_=context_adjusted_residual(h,alpha_context);_,cond=context_frame(h);pid=h.pitcher_id.astype(str);global_res=float(residual.mean());overall=pd.DataFrame({'pid':pid,'res':residual}).groupby('pid').res.agg(['sum','count']);overall_rate=(overall['sum']+alpha_pitcher*global_res)/(overall['count']+alpha_pitcher);out=pd.DataFrame(index=overall.index);out['psych_ctx_history_log_n']=np.log1p(overall['count'])
    season_effects={}
    for name in CONDITIONS:
        mask=cond[name].to_numpy(bool);frame=pd.DataFrame({'pid':pid[mask],'res':residual[mask]});agg=frame.groupby('pid').res.agg(['sum','count']);n=agg['count'].reindex(out.index).fillna(0);rate=(agg['sum'].reindex(out.index).fillna(0)+alpha_pitcher*overall_rate)/(n+alpha_pitcher);effect=rate-overall_rate;out[f'psych_ctx_{name}_effect']=effect;out[f'psych_ctx_{name}_reliability']=n/(n+alpha_pitcher)
        per=[]
        for _,idx in h.groupby('season').groups.items():
            ii=np.asarray(list(idx));m=cond.loc[ii,name].to_numpy(bool);sub_pid=pid.iloc[ii].to_numpy()[m];sub_res=residual[ii][m];a=pd.DataFrame({'pid':sub_pid,'res':sub_res}).groupby('pid').res.mean();per.append(a.reindex(out.index))
        S=pd.concat(per,axis=1) if per else pd.DataFrame(index=out.index);sign_consistency=np.abs(np.sign(S).mean(1)).fillna(0);out[f'psych_ctx_{name}_stability']=sign_consistency;out[f'psych_ctx_{name}_stable_effect']=effect*sign_consistency
    return out.reset_index(names='pitcher_id')

def attach_context_adjusted_psych(raw,history,alpha_context=400.,alpha_pitcher=300.):
    profile=build_profiles(history,alpha_context,alpha_pitcher);_,cond=context_frame(raw);x=pd.DataFrame({'pitcher_id':raw.pitcher_id.astype(str).to_numpy()}).merge(profile,on='pitcher_id',how='left',sort=False).drop(columns='pitcher_id').fillna(0)
    for name in CONDITIONS:
        active=cond[name].to_numpy(float);x[f'psych_ctx_{name}_active_effect']=x[f'psych_ctx_{name}_effect']*active;x[f'psych_ctx_{name}_active_stable_effect']=x[f'psych_ctx_{name}_stable_effect']*active;x[f'psych_ctx_{name}_active_reliability']=x[f'psych_ctx_{name}_reliability']*active
    return x.astype('float32')

def attach_saved_context_psych(raw,profile):
    _,cond=context_frame(raw);x=pd.DataFrame({'pitcher_id':raw.pitcher_id.astype(str).to_numpy()}).merge(profile,on='pitcher_id',how='left',sort=False).drop(columns='pitcher_id').fillna(0)
    for name in CONDITIONS:
        active=cond[name].to_numpy(float);x[f'psych_ctx_{name}_active_effect']=x[f'psych_ctx_{name}_effect']*active;x[f'psych_ctx_{name}_active_stable_effect']=x[f'psych_ctx_{name}_stable_effect']*active;x[f'psych_ctx_{name}_active_reliability']=x[f'psych_ctx_{name}_reliability']*active
    return x.astype('float32')
