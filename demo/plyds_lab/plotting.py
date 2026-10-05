"""Vector-field figures with the saved ID/OOD rollout protocol."""
import os
from .data import ROOT,load_motion
os.environ.setdefault("MPLBACKEND","Agg")
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from .solver import predict

ID_COLOR="#128a87"
OOD_COLOR="#b44687"


def view_bounds(protocol):
    bounds=np.array(protocol["plot_bounds"])
    center=bounds.mean(axis=0);half=float(np.ptp(bounds,axis=0).max())/2
    return np.vstack([center-half,center+half])


def draw_panel(ax,model,evaluation,title=None):
    bounds=view_bounds(evaluation["protocol"])
    xx,yy=np.meshgrid(np.linspace(bounds[0,0],bounds[1,0],23),np.linspace(bounds[0,1],bounds[1,1],23))
    xy=np.column_stack([xx.ravel(),yy.ravel()])
    velocity=predict(model,xy)
    speed=np.linalg.norm(velocity,axis=1)
    direction=np.divide(velocity,speed[:,None],out=np.zeros_like(velocity),where=speed[:,None]>1e-12)
    ax.quiver(xy[:,0],xy[:,1],direction[:,0],direction[:,1],color="#acb8c7",alpha=.75,
              pivot="mid",scale=31,width=.0025,zorder=0)
    for i,(x,u) in enumerate(load_motion(model["config"]["motion"])["demos"]):
        training=i in model["metrics"]["training_demo_ids"]
        ax.plot(x[:,0],x[:,1],color="#202c3e",lw=1.15,alpha=.8 if training else .4,
                linestyle="-" if training else "--",zorder=3)
    for record in evaluation["rollouts"]:
        path=record["path"]
        ood=record["group"] in ("ood_ring","ood_trajectory")
        color=OOD_COLOR if ood else '#596b91' if record['group']=='along_trajectory' else ID_COLOR
        ax.plot(path[:,0],path[:,1],color=color,alpha=.48,lw=.8,linestyle="--" if ood else "-",zorder=2)
        ax.scatter(*record["position"],color=color,s=8,marker="^" if ood else "o",alpha=.8,zorder=4)
    ax.scatter(0,0,marker="*",s=85,color="#0e1727",zorder=6)
    ax.set_xlim(bounds[:,0]);ax.set_ylim(bounds[:,1]);ax.set_aspect("equal",adjustable="box")
    groups=evaluation["groups"]
    ax.set_title((title or model["config"]["motion"])+f"\nID {groups['id_all']['target_rate']:.0%} · OOD {groups.get('ood_all',groups['ood_ring'])['target_rate']:.0%}",fontsize=11)
    ax.tick_params(labelsize=8);ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.1)


def legend(fig):
    fig.legend(handles=[Line2D([0],[0],color="#202c3e",label="Demonstrations (held-out dashed)"),
        Line2D([0],[0],color=ID_COLOR,marker="o",markersize=4,label="ID starts + full rollouts"),
        Line2D([0],[0],color=OOD_COLOR,linestyle="--",marker="^",markersize=4,label="OOD starts + full rollouts"),
        Line2D([0],[0],color="#acb8c7",marker=">",label="Vector-field direction"),
        Line2D([0],[0],color="#0e1727",marker="*",linestyle="None",label="Target")],
        loc="outside lower center",ncol=3,frameon=False,fontsize=10)


def save_single(model,evaluation,directory):
    fig,ax=plt.subplots(figsize=(8,7),layout="constrained")
    cfg=model["config"]
    title=f"{cfg['motion']} · SNDS · CPU" if cfg.get('method')=='snds' else f"{cfg['motion']} · policy {cfg['policy_degree']} / Lyapunov {cfg['lyapunov_degree']}"
    draw_panel(ax,model,evaluation,title)
    legend(fig)
    fig.savefig(directory/"vector-field.png",dpi=150)
    fig.savefig(directory/"vector-field.pdf")
    plt.close(fig)


def save_grid(items,path,title,ncols=5):
    rows=(len(items)+ncols-1)//ncols
    fig,axes=plt.subplots(rows,ncols,figsize=(4.4*ncols,4.5*rows+1),layout="constrained",squeeze=False)
    for ax,item in zip(axes.ravel(),items):
        draw_panel(ax,item[0],item[1],item[2] if len(item)>2 else None)
    for ax in axes.ravel()[len(items):]:ax.set_visible(False)
    fig.suptitle(title,fontsize=19)
    legend(fig)
    fig.savefig(path.with_suffix(".png"),dpi=170)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
