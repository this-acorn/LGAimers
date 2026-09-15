"""Row-local baseball context features, with a dedicated outs/runners pressure block."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_context_pressure_features(raw: pd.DataFrame) -> pd.DataFrame:
    r=raw.reset_index(drop=True);x=pd.DataFrame(index=r.index)
    balls=pd.to_numeric(r.balls_before,errors='coerce').fillna(0);strikes=pd.to_numeric(r.strikes_before,errors='coerce').fillna(0);outs=pd.to_numeric(r.outs_before,errors='coerce').fillna(0);inning=pd.to_numeric(r.inning,errors='coerce').fillna(0);li=pd.to_numeric(r.li,errors='coerce').fillna(0).clip(0,10);runners=pd.to_numeric(r.num_runners_on,errors='coerce').fillna(0);score=pd.to_numeric(r.score_diff_pitcher_team,errors='coerce').fillna(0)

    # T means away team bats, hence the pitcher is the home-team pitcher.
    x['pitcher_is_home']=r.top_bottom.astype(str).eq('T').astype('int8')
    x['pitcher_win_expectancy']=np.where(x.pitcher_is_home.eq(1),r.home_win_expectancy,r.away_win_expectancy)/100
    x['expectancy_tension']=4*x.pitcher_win_expectancy*(1-x.pitcher_win_expectancy)
    x['home_pressure']=x.pitcher_is_home*li;x['away_pressure']=(1-x.pitcher_is_home)*li

    # Recent trajectory x game importance. These are proxies, not literal psychology.
    recent=r[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce');career=pd.to_numeric(r.asof_pitcher_success_rate,errors='coerce')
    x['recent_shock']=(recent.iloc[:,0]-recent.iloc[:,2]).fillna(0);x['recent_level']=(recent.mean(1)-career).fillna(0);x['recent_acceleration']=(recent.iloc[:,0]-2*recent.iloc[:,1]+recent.iloc[:,2]).fillna(0)
    phase=np.select([inning<=3,inning<=6],['early','middle'],default='late')
    x['bad_form_high_li']=np.maximum(-x.recent_level,0)*li;x['good_form_high_li']=np.maximum(x.recent_level,0)*li
    x['bad_form_early']=np.maximum(-x.recent_level,0)*(inning<=3);x['good_form_late']=np.maximum(x.recent_level,0)*(inning>=7)

    # Platoon advantage: same handedness favors pitcher, opposite favors batter.
    ph=pd.to_numeric(r.pitcher_hand,errors='coerce');bh=pd.to_numeric(r.batter_hand,errors='coerce');x['same_hand_matchup']=(ph==bh).astype('int8');x['opposite_hand_matchup']=(ph!=bh).astype('int8');x['left_pitcher']=(ph==1).astype('int8');x['rare_left_opposite']=x.left_pitcher*x.opposite_hand_matchup

    # Only prior pitch-mix is legal; current pitch type is unknown.
    fast=pd.to_numeric(r.asof_pitcher_fastball_rate,errors='coerce');brk=pd.to_numeric(r.asof_pitcher_breaking_rate,errors='coerce');off=pd.to_numeric(r.asof_pitcher_offspeed_rate,errors='coerce');mix_n=pd.to_numeric(r.asof_pitcher_pitchmix_n,errors='coerce').fillna(0)
    x['fastball_rate']=fast;x['breaking_rate']=brk;x['offspeed_rate']=off;x['pitchmix_reliability']=mix_n/(mix_n+100);x['fastball_threeball']=fast*(balls==3);x['breaking_twostrike']=brk*(strikes==2);x['offspeed_twostrike']=off*(strikes==2);x['breaking_platoon']=brk*x.same_hand_matchup;x['offspeed_platoon']=off*x.opposite_hand_matchup

    # Exact pitch count/times-through-order is unavailable; inning is a weak fatigue proxy.
    x['fatigue_proxy']=np.maximum(inning-5,0);x['fatigue_high_li']=x.fatigue_proxy*li;x['fatigue_runners']=x.fatigue_proxy*runners

    # Dedicated high-resolution outs/runners block.
    risp=(r.runner_on_2b.eq(1)|r.runner_on_3b.eq(1)).astype('int8');loaded=r.base_state.astype(str).eq('123').astype('int8')
    x['risp']=risp;x['bases_loaded']=loaded;x['two_outs']=(outs==2).astype('int8');x['zero_outs']=(outs==0).astype('int8');x['runner_pressure']=runners*(3-outs);x['risp_out_pressure']=risp*(3-outs);x['loaded_out_pressure']=loaded*(3-outs);x['outs_li']=outs*li;x['runners_li']=runners*li;x['risp_li']=risp*li;x['loaded_li']=loaded*li;x['two_out_risp']=x.two_outs*risp;x['zero_out_risp']=x.zero_outs*risp;x['two_out_full_count']=x.two_outs*((balls==3)&(strikes==2));x['compound_crisis']=li*(1+runners)*(1+loaded)*(1+(balls==3))/(1+outs)

    x['home_count_context']=x.pitcher_is_home.astype(str)+'|'+balls.astype(int).astype(str)+'-'+strikes.astype(int).astype(str)
    x['platoon_count_context']=x.same_hand_matchup.astype(str)+'|'+balls.astype(int).astype(str)+'-'+strikes.astype(int).astype(str)
    x['outs_base_context']=outs.astype(int).astype(str)+'|'+r.base_state.astype(str)
    x['outs_runner_count_context']=outs.astype(int).astype(str)+'|'+runners.astype(int).astype(str)
    x['crisis_context']=pd.cut(li,[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str)+'|'+x.outs_base_context
    form=pd.Series(np.select([x.recent_level<-.03,x.recent_level>.03],['bad','good'],default='flat'),index=x.index)
    x['form_phase_context']=form+'|'+pd.Series(phase,index=x.index)
    return x.replace([np.inf,-np.inf],np.nan)

CAT_CONTEXT=['home_count_context','platoon_count_context','outs_base_context','outs_runner_count_context','crisis_context','form_phase_context']
