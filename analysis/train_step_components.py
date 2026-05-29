"""Attribute the ~460 ms training step to forward / backward / optimizer(+clip),
and test how much the two 5.65M-param FC heads drive backward+optimizer by
comparing the full model vs a variant with a tiny fully-convolutional policy head.
Read-only; synthetic batch (component timing is data-independent)."""
from __future__ import annotations
import json, os, time
import torch
from torch import nn
from hexo_models.dense_cnn.architecture import Model1Network
from hexo_models.dense_cnn.losses import model1_loss
from hexo_models.dense_cnn.constants import BOARD_AREA

H=(1,4,8); BS=256; DEV=torch.device("cuda")
torch.backends.cudnn.benchmark=True


def synth_batch():
    g=torch.Generator().manual_seed(0)
    b={"input":torch.randn(BS,13,41,41,generator=g),
       "policy":torch.softmax(torch.randn(BS,BOARD_AREA,generator=g),dim=1),
       "opp_policy":torch.softmax(torch.randn(BS,BOARD_AREA,generator=g),dim=1),
       "value":torch.randn(BS,generator=g).clamp(-1,1)}
    for h in H:
        b[f"stvalue_{h}"]=torch.randn(BS,generator=g).clamp(-1,1)
        b[f"stvalue_{h}_mask"]=torch.ones(BS)
    out={}
    for k,v in b.items():
        out[k]=v.to(DEV,memory_format=torch.channels_last) if k=="input" else v.to(DEV)
    return out


def measure(model, tag):
    model=model.to(DEV,memory_format=torch.channels_last); model.train()
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scaler=torch.amp.GradScaler("cuda",enabled=True)
    batch=synth_batch()
    nparams=sum(p.numel() for p in model.parameters())

    def fwd():
        b=dict(batch); inp=b.pop("input")
        with torch.autocast(device_type="cuda",enabled=True):
            out=model(inp); loss,_=model1_loss(out,b,policy_weight=1.0,value_weight=1.0,opp_policy_weight=0.25,short_term_value_weight=0.25)
        return loss
    # warm
    for _ in range(20):
        opt.zero_grad(set_to_none=True); l=fwd(); scaler.scale(l).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    N=80
    # forward only
    t0=time.perf_counter()
    for _ in range(N):
        opt.zero_grad(set_to_none=True); l=fwd(); torch.cuda.synchronize()
    t_fwd=(time.perf_counter()-t0)/N*1000
    # forward+backward
    t0=time.perf_counter()
    for _ in range(N):
        opt.zero_grad(set_to_none=True); l=fwd(); scaler.scale(l).backward(); torch.cuda.synchronize()
    t_fb=(time.perf_counter()-t0)/N*1000
    # full (fwd+bwd+unscale+clip+step+update)
    t0=time.perf_counter()
    for _ in range(N):
        opt.zero_grad(set_to_none=True); l=fwd(); scaler.scale(l).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); torch.cuda.synchronize()
    t_full=(time.perf_counter()-t0)/N*1000
    r={"tag":tag,"params_M":round(nparams/1e6,2),"forward_ms":round(t_fwd,1),
       "backward_ms":round(t_fb-t_fwd,1),"opt_clip_ms":round(t_full-t_fb,1),"full_ms":round(t_full,1)}
    print(json.dumps(r)); return r


class ConvPolicyHead(nn.Module):
    def __init__(self,ch):
        super().__init__(); self.c1=nn.Conv2d(ch,ch,3,padding=1); self.r=nn.ReLU(inplace=True); self.c2=nn.Conv2d(ch,1,1)
    def forward(self,x): return self.c2(self.r(self.c1(x))).flatten(1)


def main():
    out={}
    out["full_model"]=measure(Model1Network(channels=64,blocks=4,short_term_value_horizons=H),"full_model_FC_heads")
    # variant: replace both FC policy heads with tiny fully-conv heads
    m2=Model1Network(channels=64,blocks=4,short_term_value_horizons=H)
    m2.policy_head=ConvPolicyHead(64); m2.opp_policy_head=ConvPolicyHead(64)
    out["conv_head_variant"]=measure(m2,"conv_policy_heads")
    json.dump(out,open(os.path.join(os.path.dirname(__file__),"train_step_components_summary.json"),"w"),indent=2)
    print("wrote train_step_components_summary.json")


if __name__=="__main__":
    main()
