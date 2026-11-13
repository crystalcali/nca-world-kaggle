"""NCA World v1: local vs world-predictor vs tile-attention ablation.
Synthetic action-conditioned GUI benchmark. NOT a general-world model.
"""
import argparse, copy, json, os, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def generate(batch, frames, size, rng):
    H=W=size
    rgb=np.empty((batch,frames,3,H,W),dtype=np.float32)
    masks=np.zeros((batch,frames,1,H,W),dtype=np.float32)
    actions=np.zeros((batch,frames-1),dtype=np.float32)
    for b in range(batch):
        ui=float(rng.uniform(.14,.30)); bg=float(rng.uniform(.035,.095))
        x=int(rng.integers(25,size-16)); y=int(rng.integers(17,size-14))
        v=int(rng.choice((-1,1)))
        base=np.zeros((3,H,W),dtype=np.float32);base[:]=bg
        base[:,:,:16]=ui; base[:,:7,:]=ui+.09
        base[:,9:14,3:12]=np.array([.10,.35,.49],dtype=np.float32)[:,None,None]
        base[:,22:27,3:12]=np.array([.10,.28,.40],dtype=np.float32)[:,None,None]
        base[:,8:-3,16:18]=np.array([.35,.36,.40],dtype=np.float32)[:,None,None]
        base[:,size-5:size-3,18:size-3]=.26
        for t in range(frames):
            if t>0:
                if t>1 and rng.random()<.25:
                    actions[b,t-1]=1.;v=-v
                x+=v
                if x<=21 or x>=size-7:
                    v=-v; x=max(21,min(size-7,x))
            image=base.copy()
            ys=slice(max(8,y-2),min(H-5,y+3))
            xs=slice(max(20,x-2),min(W-4,x+3))
            image[:,ys,xs]=np.array([.96,.19,.09],dtype=np.float32)[:,None,None]
            masks[b,t,0,ys,xs]=1.
            rgb[b,t]=image
    return torch.from_numpy(rgb),torch.from_numpy(actions),torch.from_numpy(masks)

class Encoder(nn.Module):
    def __init__(self,latent):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(6,24,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(24,32,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(32,48,3,stride=2,padding=1),nn.GELU(),
            nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(48,latent),nn.LayerNorm(latent))
    def forward(self,a,b):return self.net(torch.cat((a,b),dim=1))

class VideoNCA(nn.Module):
    def __init__(self,mode,channels=20,latent=48,inner_steps=3,tile_size=8):
        super().__init__()
        assert mode in ('local','world','world_tiles')
        self.mode=mode;self.channels=channels;self.inner_steps=inner_steps;self.tile_size=tile_size
        self.init_hidden=nn.Conv2d(6,channels-3,1)
        self.perception=nn.Conv2d(channels,channels,3,padding=1)
        self.encoder=Encoder(latent) if mode!='local' else None
        self.pred=nn.GRUCell(1,latent) if mode!='local' else None
        self.teacher=copy.deepcopy(self.encoder) if self.encoder is not None else None
        if self.teacher is not None:
            for p in self.teacher.parameters():p.requires_grad_(False)
        self.broadcast=nn.Linear(latent,channels) if mode!='local' else None
        if mode=='world_tiles':
            self.tile_norm=nn.LayerNorm(channels)
            self.attn=nn.MultiheadAttention(channels,num_heads=4,batch_first=True)
        dim=channels*2+1+(channels if mode!='local' else 0)+(channels if mode=='world_tiles' else 0)
        self.rule=nn.Sequential(nn.Conv2d(dim,64,1),nn.GELU(),nn.Conv2d(64,channels,1))
        nn.init.zeros_(self.rule[-1].weight);nn.init.zeros_(self.rule[-1].bias)
    @torch.no_grad()
    def update_teacher(self,decay=.99):
        if self.teacher is None:return
        for p,q in zip(self.encoder.parameters(),self.teacher.parameters()):q.lerp_(p,1.-decay)
    def initialize(self,prev,cur):
        return torch.cat((torch.logit(cur.clamp(.02,.98)),
                          self.init_hidden(torch.cat((prev,cur),dim=1))),dim=1)
    def tick(self,state,z,action):
        B,C,H,W=state.shape
        stimulus=state.new_zeros(B,1,H,W)
        stimulus[:,:,11:14,5:8]=action[:,None,None,None]
        shared=self.broadcast(z)[:,:,None,None].expand(-1,-1,H,W) if z is not None else None
        for _ in range(self.inner_steps):
            parts=[state,self.perception(state),stimulus]
            if shared is not None:parts.append(shared)
            if self.mode=='world_tiles':
                pooled=F.avg_pool2d(state,kernel_size=self.tile_size)
                th,tw=pooled.shape[-2:]
                tokens=self.tile_norm(pooled.flatten(2).transpose(1,2))
                msg,_=self.attn(tokens,tokens,tokens,need_weights=False)
                tile_map=msg.transpose(1,2).reshape(B,C,th,tw)
                parts.append(F.interpolate(tile_map,size=(H,W),mode='nearest'))
            state=state+self.rule(torch.cat(parts,dim=1))/self.inner_steps
        return state
    def forward(self,images,actions,horizon):
        s=self.initialize(images[:,0],images[:,1])
        z=self.encoder(images[:,0],images[:,1]) if self.encoder else None
        preds=[];aux=[]
        for i in range(horizon):
            action=actions[:,i+1]
            if z is not None:z=self.pred(action[:,None],z)
            s=self.tick(s,z,action)
            preds.append(torch.sigmoid(s[:,:3]))
            if z is not None and self.training:
                with torch.no_grad():
                    target=self.teacher(images[:,i+1],images[:,i+2])
                aux.append(F.mse_loss(z,target))
        out=torch.stack(preds,dim=1)
        return out,(torch.stack(aux).mean() if aux else out.new_zeros(()))

def loss_fn(pred,target,masks):
    return ((pred-target).abs()*(1.+45.*masks)).mean()

@torch.no_grad()
def evaluate(model,device,seed,horizon,samples=64,size=64):
    model.eval()
    im,actions,masks=generate(samples,horizon+2,size,np.random.default_rng(seed))
    im,actions,masks=im.to(device),actions.to(device),masks.to(device)
    pred,_=model(im,actions,horizon);true=im[:,2:]
    mae=(pred-true).abs().mean().item()
    panel=(pred[:,:,:,:8,:16]-true[:,:,:,:8,:16]).abs().mean().item()
    foreground=(((pred-true).abs()*masks[:,2:]).sum()/(masks[:,2:].sum()*3+1e-6)).item()
    heat=F.relu(pred[:,:,0,15:,18:]-pred[:,:,1,15:,18:])
    ys=torch.arange(15,size,device=device)[None,None,:,None]
    xs=torch.arange(18,size,device=device)[None,None,None,:]
    mass=heat.sum(dim=(2,3)).clamp(min=1e-6)
    px=(heat*xs).sum(dim=(2,3))/mass;py=(heat*ys).sum(dim=(2,3))/mass
    gm=masks[:,2:,0]
    gx=(gm*torch.arange(size,device=device)[None,None,None,:]).sum(dim=(2,3))/gm.sum(dim=(2,3)).clamp(min=1e-6)
    gy=(gm*torch.arange(size,device=device)[None,None,:,None]).sum(dim=(2,3))/gm.sum(dim=(2,3)).clamp(min=1e-6)
    err=torch.sqrt((px-gx).square()+(py-gy).square())
    flipped=(actions[:,1:horizon+1].sum(dim=1)>0)
    ierr=err[flipped].mean().item() if flipped.any() else None
    return {'mae':round(mae,6),'panel_mae':round(panel,6),
            'foreground_mae':round(foreground,6),
            'ball_position_error_px':round(err.mean().item(),3),
            'intervention_position_error_px':round(ierr,3) if ierr is not None else None}

def run(args):
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    device=torch.device('cuda:0' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('device',device,'gpus',torch.cuda.device_count(),flush=True)
    Path(args.out).mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    results={};loss_history={}
    for mode in ('local','world','world_tiles'):
        torch.manual_seed(args.seed)
        model=VideoNCA(mode,channels=args.channels,latent=args.latent,inner_steps=args.inner_steps).to(device)
        params=sum(p.numel() for p in model.parameters() if p.requires_grad)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.lr,weight_decay=1e-5)
        losses=[];start=time.monotonic()
        for step in range(args.steps):
            model.train()
            frames,acts,masks=generate(args.batch,args.horizon+2,args.size,rng)
            frames,acts,masks=frames.to(device),acts.to(device),masks.to(device)
            opt.zero_grad(set_to_none=True)
            pred,latent=model(frames,acts,args.horizon)
            reconstruction=loss_fn(pred,frames[:,2:],masks[:,2:])
            loss=reconstruction+args.latent_weight*latent
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step();model.update_teacher()
            losses.append(float(loss.detach()))
            if (step+1)%args.log_every==0 or step==0:
                print(mode,step+1,'/',args.steps,'loss',round(loss.item(),5),
                      'pixel',round(reconstruction.item(),5),'latent',round(latent.item(),5),flush=True)
        short=evaluate(model,device,args.seed+1000,args.horizon,size=args.size)
        long=evaluate(model,device,args.seed+2000,args.eval_horizon,size=args.size)
        results[mode]={'parameters':params,'training_seconds':round(time.monotonic()-start,1),
                       'loss_start':round(losses[0],5),'loss_last_10':round(float(np.mean(losses[-10:])),5),
                       'test_short':short,'test_long':long}
        print('EVAL',mode,json.dumps(results[mode]),flush=True)
        loss_history[mode]=losses
        torch.save({'mode':mode,'weights':model.state_dict(),'settings':vars(args)},
                   Path(args.out)/(mode+'.pt'))
        del model,opt
        if device.type=='cuda':torch.cuda.empty_cache()
    report={'experiment':'nca-world-v1','scope':'synthetic UI with remote action','device':str(device),
            'settings':vars(args),'results':results,
            'caveat':'Small, non-parameter-matched exploratory baselines; not real-world video understanding.'}
    (Path(args.out)/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    (Path(args.out)/'losses.json').write_text(json.dumps(loss_history))
    print('FINAL_RESULTS_JSON='+json.dumps(report,ensure_ascii=False),flush=True)
    return report

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--horizon',type=int,default=6)
    p.add_argument('--eval-horizon',type=int,default=12)
    p.add_argument('--channels',type=int,default=20)
    p.add_argument('--latent',type=int,default=48)
    p.add_argument('--inner-steps',type=int,default=3)
    p.add_argument('--lr',type=float,default=7e-4)
    p.add_argument('--latent-weight',type=float,default=.05)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--out',default='runs/v1')
    p.add_argument('--log-every',type=int,default=25)
    a=p.parse_args()
    if a.pilot:
        a.steps=30;a.horizon=4;a.eval_horizon=8;a.batch=4;a.log_every=10
    if a.size<40 or a.size%8: p.error('size must be a multiple of 8 and >=40')
    return a

if __name__=='__main__':
    run(parse_args())
