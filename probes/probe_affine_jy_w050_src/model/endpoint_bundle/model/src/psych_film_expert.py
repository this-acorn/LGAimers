"""Pressure-gated FiLM expert: psychology modulates context, never acts alone."""
from __future__ import annotations
import torch
from torch import nn
import numpy as np
import pandas as pd

class PsychFiLMExpert(nn.Module):
    def __init__(self, base_dim, psych_dim, pressure_dim, width=96, dropout=.15):
        super().__init__();self.base=nn.Sequential(nn.Linear(base_dim,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width),nn.GELU());self.film=nn.Sequential(nn.Linear(psych_dim,width),nn.GELU(),nn.Linear(width,width*2));self.delta=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width//2,1));self.gate=nn.Sequential(nn.Linear(pressure_dim+2,width//2),nn.GELU(),nn.Linear(width//2,1))
    def forward(self,base_x,psych_x,pressure_x,psych_reliability,pressure_prior):
        h=self.base(base_x);gamma,beta=self.film(psych_x).chunk(2,dim=1);gamma=.20*torch.tanh(gamma);beta=.20*torch.tanh(beta);mod=h*(1+gamma)+beta;delta=.05*torch.tanh(self.delta(mod).squeeze(1));gate_in=torch.cat([pressure_x,psych_reliability[:,None],pressure_prior[:,None]],1);learned=torch.sigmoid(self.gate(gate_in).squeeze(1));gate=learned*pressure_prior*psych_reliability;return delta*gate,delta,gate

class PsychRegimeFiLMExpert(nn.Module):
    """Sparse situation regimes + TabM-style cheap ensemble heads."""
    def __init__(self,base_dim,psych_dim,pressure_dim,width=128,n_regimes=9,n_heads=4,dropout=.15):
        super().__init__();self.width=width;self.n_regimes=n_regimes;self.n_heads=n_heads
        self.base=nn.Sequential(nn.Linear(base_dim,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,width),nn.GELU())
        self.regime=nn.Sequential(nn.Linear(psych_dim+pressure_dim,width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,n_regimes))
        self.regime_film=nn.Parameter(torch.zeros(n_regimes,2,width));nn.init.normal_(self.regime_film,std=.02)
        # Parameter-efficient head-specific affine adapters; backbone is shared.
        self.head_scale=nn.Parameter(torch.ones(n_heads,width));self.head_bias=nn.Parameter(torch.zeros(n_heads,width))
        self.heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width//2),nn.GELU(),nn.Dropout(dropout),nn.Linear(width//2,1)) for _ in range(n_heads)])
        self.gate=nn.Sequential(nn.Linear(pressure_dim+n_regimes+3,width//2),nn.GELU(),nn.Linear(width//2,1))
    def forward(self,base_x,psych_x,pressure_x,psych_reliability,pressure_prior):
        h=self.base(base_x);routing_logits=self.regime(torch.cat([psych_x,pressure_x],1));raw_regime=torch.softmax(routing_logits,1)
        # Top-2 sparse routing prevents every situation expert from averaging together.
        _,top=torch.topk(raw_regime,k=2,dim=1);mask=torch.zeros_like(raw_regime).scatter_(1,top,1);sparse=raw_regime*mask;sparse=sparse/sparse.sum(1,keepdim=True).clamp_min(1e-6)
        # Unreliable profiles are routed toward explicit neutral expert 0.
        neutral=torch.zeros_like(sparse);neutral[:,0]=1;regime=psych_reliability[:,None]*sparse+(1-psych_reliability[:,None])*neutral
        gamma=torch.einsum('br,rd->bd',regime,self.regime_film[:,0]);beta=torch.einsum('br,rd->bd',regime,self.regime_film[:,1]);mod=h*(1+.20*torch.tanh(gamma))+.20*torch.tanh(beta)
        members=[]
        for j,head in enumerate(self.heads):members.append(.05*torch.tanh(head(mod*self.head_scale[j]+self.head_bias[j]).squeeze(1)))
        member_delta=torch.stack(members,1);delta=member_delta.mean(1);disagreement=member_delta.std(1,unbiased=False)
        entropy=-(regime.clamp_min(1e-6).log()*regime).sum(1)
        confidence=(1-entropy/torch.log(torch.tensor(float(self.n_regimes),device=regime.device))).clamp(0,1)
        gate_in=torch.cat([pressure_x,regime,psych_reliability[:,None],pressure_prior[:,None],confidence[:,None]],1);learned=torch.sigmoid(self.gate(gate_in).squeeze(1));gate=learned*pressure_prior*psych_reliability*(.5+.5*confidence)
        return delta*gate,delta,gate,regime,disagreement,member_delta

class PsychSituationInteractionEncoder(nn.Module):
    """Separate latent mental-state and situation towers with low-rank interaction."""
    def __init__(self,base_dim,psych_dim,situation_dim,width=96,latent=32,dropout=.12):
        super().__init__()
        block=lambda n:nn.Sequential(nn.Linear(n,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,latent),nn.LayerNorm(latent),nn.GELU())
        self.base=block(base_dim);self.psych=block(psych_dim);self.situation=block(situation_dim)
        self.interaction=nn.Sequential(nn.Linear(latent*4,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout))
        self.delta=nn.Sequential(nn.Linear(width,48),nn.GELU(),nn.Linear(48,1))
        self.failure=nn.Sequential(nn.Linear(width,48),nn.GELU(),nn.Linear(48,3))
        self.gate=nn.Sequential(nn.Linear(width+3,48),nn.GELU(),nn.Linear(48,1))
    def forward(self,base_x,psych_x,situation_x,reliability,pressure):
        b=self.base(base_x);p=self.psych(psych_x);s=self.situation(situation_x)
        h=self.interaction(torch.cat([b,p*s,torch.abs(p-s),p+s],1))
        raw=.04*torch.tanh(self.delta(h).squeeze(1));aux=self.failure(h)
        confidence=torch.sigmoid(self.gate(torch.cat([h,reliability[:,None],pressure[:,None],torch.linalg.vector_norm(p-s,dim=1,keepdim=True)],1)).squeeze(1))
        gate=confidence*reliability*(.25+.75*pressure)
        return raw*gate,aux,gate,p,s

class PsychFailureController(nn.Module):
    """Three overlapping failure experts modulated by a separate psych tower."""
    def __init__(self,base_dim,psych_dim,situation_dim,width=96,latent=32,dropout=.12):
        super().__init__()
        block=lambda n:nn.Sequential(nn.Linear(n,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout),nn.Linear(width,latent),nn.GELU())
        self.base=block(base_dim);self.psych=block(psych_dim);self.situation=block(situation_dim)
        self.shared=nn.Sequential(nn.Linear(latent*3,width),nn.LayerNorm(width),nn.GELU(),nn.Dropout(dropout))
        self.failure_heads=nn.ModuleList([nn.Linear(width,1) for _ in range(3)])
        self.psych_offsets=nn.ModuleList([nn.Sequential(nn.Linear(latent*2,32),nn.GELU(),nn.Linear(32,1)) for _ in range(3)])
        self.failure_to_delta=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,1))
        self.gate=nn.Sequential(nn.Linear(width+5,32),nn.GELU(),nn.Linear(32,1))
    def forward(self,base_x,psych_x,situation_x,reliability,pressure):
        b=self.base(base_x);p=self.psych(psych_x);s=self.situation(situation_x);h=self.shared(torch.cat([b,s,b*s],1))
        raw=torch.cat([head(h) for head in self.failure_heads],1)
        po=torch.cat([head(torch.cat([p,p*s],1)) for head in self.psych_offsets],1)
        # Psychology can only perturb, not replace, the observable failure risks.
        offsets=.75*torch.tanh(po)*reliability[:,None]*(.25+.75*pressure[:,None]);logits=raw+offsets
        risks=torch.sigmoid(logits);delta=.04*torch.tanh(self.failure_to_delta(torch.cat([risks,offsets],1)).squeeze(1))
        disagreement=risks.std(1,unbiased=False);confidence=torch.sigmoid(self.gate(torch.cat([h,reliability[:,None],pressure[:,None],disagreement[:,None],risks.mean(1,keepdim=True),offsets.abs().mean(1,keepdim=True)],1)).squeeze(1))
        gate=confidence*(.35+.65*reliability)*(.30+.70*pressure)
        return delta*gate,logits,gate,p,s

def _num(d,c,fill=0):return pd.to_numeric(d[c],errors='coerce').fillna(fill).to_numpy(np.float32)

def production_arrays(d,base_prediction,psych):
    effect=[c for c in psych if c.endswith(('stable_effect','active_stable_effect'))];relcols=[c for c in psych if c.endswith('active_reliability')];px=psych[effect].to_numpy(np.float32);rel=np.clip(psych[relcols].max(axis=1).to_numpy(np.float32),0,1)
    balls=_num(d,'balls_before');strikes=_num(d,'strikes_before');outs=_num(d,'outs_before');inning=_num(d,'inning');li=np.clip(_num(d,'li'),0,10);runners=_num(d,'num_runners_on');score=_num(d,'score_diff_pitcher_team');risp=((_num(d,'runner_on_2b')>0)|(_num(d,'runner_on_3b')>0)).astype(np.float32);recent=np.c_[_num(d,'asof_pitcher_prev1_game_success_rate',.52),_num(d,'asof_pitcher_prev3_game_success_rate',.52),_num(d,'asof_pitcher_prev5_game_success_rate',.52)]
    bx=np.c_[base_prediction,_num(d,'asof_pitcher_success_rate',.52),recent,recent.mean(1)-_num(d,'asof_pitcher_success_rate',.52),_num(d,'asof_batter_success_rate',.52),_num(d,'asof_pitcher_fastball_rate'),_num(d,'asof_pitcher_breaking_rate'),_num(d,'asof_pitcher_offspeed_rate'),(_num(d,'pitcher_hand')==_num(d,'batter_hand')).astype(np.float32)];qx=np.c_[balls,strikes,outs,inning,li,runners,score,risp,(balls==3),(strikes==2),(inning>=7),(abs(score)<=1)].astype(np.float32);prior=np.clip((np.log1p(li)/np.log(11))*(.30+.25*runners+.20*risp+.15*(balls==3)+.10*(inning>=7)),0,1).astype(np.float32);return bx.astype(np.float32),px,qx,rel,prior

def apply_saved_regime(raw,base_prediction,profile,asset_path,state_path):
    from src.context_adjusted_psych import attach_saved_context_psych
    psych=attach_saved_context_psych(raw,profile);bx,px,qx,rel,prior=production_arrays(raw,base_prediction,psych);a=np.load(asset_path);bx=np.nan_to_num((bx-a['base_mean'])/a['base_std']);px=np.nan_to_num((px-a['psych_mean'])/a['psych_std']);qx=np.nan_to_num((qx-a['pressure_mean'])/a['pressure_std']);model=PsychRegimeFiLMExpert(bx.shape[1],px.shape[1],qx.shape[1],width=128,n_regimes=9,n_heads=4,dropout=.15);model.load_state_dict(torch.load(state_path,map_location='cpu',weights_only=True));model.eval();out=[]
    with torch.no_grad():
        for st in range(0,len(raw),8192):
            c,*_=model(torch.from_numpy(bx[st:st+8192]),torch.from_numpy(px[st:st+8192]),torch.from_numpy(qx[st:st+8192]),torch.from_numpy(rel[st:st+8192]),torch.from_numpy(prior[st:st+8192]));out.append(c.numpy())
    return np.concatenate(out)*float(a['scale'])
