"""SNDS with the revised smooth, strongly convex, gradient-anchored construction."""
from dataclasses import asdict, dataclass
import copy
import math
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .data import load_motion, training_samples, demonstration_ids


@dataclass(frozen=True)
class SNDSConfig:
    motion: str = 'Sine'
    n_demos: int = 5
    demo_selection: str = 'cover_starts'
    samples_per_demo: int = 200
    epochs: int = 500
    seed: int = 0
    learning_rate: float = .001
    decay_rate: float = .01
    quadratic_margin: float = .01
    method: str = 'snds'

    def validate(self):
        for name, low, high in [('n_demos',1,7),('samples_per_demo',20,1000),('epochs',1,10000),('seed',0,2**31-1)]:
            value=getattr(self,name)
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f'{name} must be an integer in [{low}, {high}]')
        for name, low, high in [('learning_rate',1e-6,.1),('decay_rate',1e-6,10),('quadratic_margin',1e-6,10)]:
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not low<=value<=high:
                raise ValueError(f'{name} must be finite and in [{low}, {high}]')
        if self.method!='snds':raise ValueError('SNDS requires method=snds')
        if self.demo_selection not in ('first','cover_starts'):raise ValueError('Unknown demonstration selection')


def smooth_leaky(x):
    # Smooth convex nondecreasing activation, without a linear Softplus cutoff.
    return .01*x+.99*torch.logaddexp(torch.zeros_like(x),10*x)/10


class RawPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([nn.Linear(a,b) for a,b in zip((2,256,256,256),(256,256,256,2))])
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self,x):
        for layer in self.layers[:-1]:x=smooth_leaky(layer(x))
        return self.layers[-1](x)


class ConvexPotential(nn.Module):
    def __init__(self):
        super().__init__()
        self.W=nn.ParameterList([nn.Parameter(torch.empty(n,2)) for n in (64,64,1)])
        self.U=nn.ParameterList([nn.Parameter(torch.empty(64,64)),nn.Parameter(torch.empty(1,64))])
        self.bias=nn.ParameterList([nn.Parameter(torch.zeros(n)) for n in (64,64,1)])
        for weight in (*self.W,*self.U):nn.init.kaiming_uniform_(weight,a=5**.5)

    def forward(self,x):
        z=smooth_leaky(F.linear(x,self.W[0],self.bias[0]))
        z=smooth_leaky(F.linear(x,self.W[1])+.01*F.linear(z,F.softplus(self.U[0]),self.bias[1]))
        return F.linear(x,self.W[2])+F.linear(z,F.softplus(self.U[1]),self.bias[2])


class RevisedSNDS(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.raw=RawPolicy(); self.q=ConvexPotential()
        self.margin=config.quadratic_margin; self.decay=config.decay_rate
        self.double()

    def potential(self,x):
        zero=torch.zeros((1,2),dtype=x.dtype,device=x.device,requires_grad=True)
        q0=self.q(zero)
        g0=torch.autograd.grad(q0.sum(),zero,create_graph=True)[0]
        # Keep both anchors in the parameter graph during training.
        return self.q(x)-q0-(x*g0).sum(1,keepdim=True)+self.margin*x.square().sum(1,keepdim=True)

    def forward(self,x):
        raw=self.raw(x)-self.raw(torch.zeros_like(x[:1]))
        v=self.potential(x)
        gradient=torch.autograd.grad(v.sum(),x,create_graph=True)[0]
        return project(x,raw,v,gradient,self.decay)


def project(x,raw,v,gradient,decay):
    target=(x==0).all(1,keepdim=True)
    denominator=torch.where(target,torch.ones_like(v),gradient.square().sum(1,keepdim=True))
    # Exact positive part and denominator off target; no epsilon or clipping.
    output=raw-gradient*F.relu((gradient*raw).sum(1,keepdim=True)+decay*v)/denominator
    return torch.where(target,torch.zeros_like(output),output)


class NeuralPolicy:
    """Frozen inference; fixed-parameter anchors are cached, not retrained."""
    def __init__(self,network):
        self.network=network.eval()
        for p in network.parameters():p.requires_grad_(False)
        zero=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
        q0=network.q(zero)
        self.q0=q0.detach()
        self.g0=torch.autograd.grad(q0.sum(),zero)[0].detach()
        with torch.no_grad():self.raw0=network.raw(zero).detach()

    def components(self,points):
        x=torch.tensor(np.atleast_2d(points),dtype=torch.float64,requires_grad=True)
        q=self.network.q(x)
        g=torch.autograd.grad(q.sum(),x)[0]-self.g0+2*self.network.margin*x
        v=q.detach()-self.q0-(x*self.g0).sum(1,keepdim=True)+self.network.margin*x.square().sum(1,keepdim=True)
        with torch.no_grad():
            raw=self.network.raw(x)-self.raw0
            velocity=project(x,raw,v,g,self.network.decay)
        return v.detach().numpy()[:,0],g.detach().numpy(),velocity.numpy()

    def predict(self,points):
        points=np.atleast_2d(points)
        if not len(points):return np.empty((0,2))
        return np.concatenate([self.components(points[i:i+256])[2] for i in range(0,len(points),256)])

    def potential(self,points):
        points=np.atleast_2d(points)
        values=[]
        with torch.no_grad():
            for i in range(0,len(points),512):
                x=torch.tensor(points[i:i+512],dtype=torch.float64)
                values.append((self.network.q(x)-self.q0-(x*self.g0).sum(1,keepdim=True)+self.network.margin*x.square().sum(1,keepdim=True)).numpy()[:,0])
        return np.concatenate(values) if values else np.empty(0)

    def save(self,path):torch.save(self.network.state_dict(),path)

    @classmethod
    def load(cls,path,config):
        cfg=SNDSConfig(**config);cfg.validate()
        with torch.random.fork_rng():
            network=RevisedSNDS(cfg)
        network.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        return cls(network)


def audit(policy):
    xx,yy=np.meshgrid(np.linspace(-2,2,41),np.linspace(-2,2,41))
    theta=np.linspace(0,2*np.pi,32,endpoint=False)
    rings=np.concatenate([r*np.column_stack([np.cos(theta),np.sin(theta)]) for r in (1e-4,.01,.1)])
    points=np.vstack([np.column_stack([xx.ravel(),yy.ravel()]),rings,[[0.,0.]]])
    parts=[policy.components(points[i:i+256]) for i in range(0,len(points),256)]
    v,g,f=[np.concatenate([part[i] for part in parts]) for i in range(3)]
    residual=(g*f).sum(1)+policy.network.decay*v
    slack=v-policy.network.margin*(points**2).sum(1)
    if not all(np.isfinite(a).all() for a in (v,g,f,residual,slack)):raise RuntimeError('Nonfinite SNDS audit')
    tolerance=1e-9
    values=dict(audit_points=len(points),audit_tolerance=tolerance,minimum_v=float(v.min()),
        minimum_lower_bound_slack=float(slack.min()),lower_bound_violations=int((slack < -tolerance).sum()),
        negative_v_count=int((v < -tolerance).sum()),maximum_decay_residual=float(residual.max()),
        decay_violations=int((residual > tolerance).sum()),target_gradient_norm=float(np.linalg.norm(g[-1])),
        origin_velocity_norm=float(np.linalg.norm(f[-1])))
    return values,dict(points=points,potential=v,gradient=g,velocity=f,decay_residual=residual,lower_bound_slack=slack)


def train(config,callback=lambda message:None):
    config.validate();torch.set_num_threads(1)
    data=load_motion(config.motion)
    x,u=training_samples(data,config.n_demos,config.samples_per_demo,config.demo_selection)
    train_ids=demonstration_ids(data,config.n_demos,config.demo_selection)
    heldout_ids=[i for i in range(len(data['demos'])) if i not in train_ids]
    started=time.perf_counter()
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        network=RevisedSNDS(config)
    tx,tu=torch.tensor(x,dtype=torch.float64),torch.tensor(u,dtype=torch.float64)
    optimizer=torch.optim.Adam(network.parameters(),lr=config.learning_rate)
    scheduler=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.,end_factor=.01,total_iters=config.epochs)
    generator=torch.Generator().manual_seed(config.seed+1000)
    history=[];best=float('inf');best_epoch=0
    for epoch in range(config.epochs):
        for ids in torch.randperm(len(tx),generator=generator).split(128):
            optimizer.zero_grad(set_to_none=True)
            prediction=network(tx[ids].detach().requires_grad_(True))
            loss=(prediction-tu[ids]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite SNDS loss at epoch {epoch+1}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),.5,error_if_nonfinite=True)
            optimizer.step()
        scheduler.step()
        if epoch%25==0 or epoch==config.epochs-1:
            squared=0.
            for ids in torch.arange(len(tx)).split(256):
                prediction=network(tx[ids].detach().requires_grad_(True))
                squared+=float((prediction-tu[ids]).square().sum().detach())
            score=squared/(2*len(tx))
            if score<best:
                best,best_epoch,best_weights=score,epoch+1,copy.deepcopy(network.state_dict())
            history.append(dict(epoch=epoch+1,training_mse=score,learning_rate=scheduler.get_last_lr()[0]))
            callback(f'SNDS epoch {epoch+1}/{config.epochs} · training MSE {score:.5g}')
    network.load_state_dict(best_weights)
    policy=NeuralPolicy(network)
    callback('Checking anchored Lyapunov values and strict decay')
    diagnostics,arrays=audit(policy)
    warnings=[]
    if diagnostics['negative_v_count'] or diagnostics['decay_violations'] or diagnostics['lower_bound_violations']:
        warnings.append('Sampled SNDS Lyapunov checks exceed the numerical tolerance; inspect audit.npz.')
    metrics=dict(status='trained',warnings=warnings,certificate_verified=False,
        elapsed_seconds=time.perf_counter()-started,history=history,best_epoch=best_epoch,
        training_demo_ids=train_ids,heldout_demo_ids=heldout_ids,fit_samples=len(x),**diagnostics)
    for label,ids in [('training',train_ids),('heldout',heldout_ids)]:
        metrics[label+'_velocity_mse']=float(np.mean(np.concatenate([(policy.predict(data['demos'][i][0])-data['demos'][i][1])**2 for i in ids]))) if ids else None
    return dict(config=asdict(config),policy=policy,metrics=metrics,audit=arrays,
        data_fingerprint=data['fingerprint'],training_data=dict(positions=x,velocities=u))
