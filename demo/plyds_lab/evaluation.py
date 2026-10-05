"""Fixed, motion-specific starting sets shared across policies and baselines."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from .data import ROOT, load_motion
from .solver import predict

PROTOCOL_VERSION = "lasa-rollouts-v4"
PROTOCOL_ROOT = ROOT / "evaluation_protocols" / PROTOCOL_VERSION
GROUPS = ("id_trajectory", "ood_ring", "ood_trajectory")


def _v2_protocol_for(motion):
    # Frozen generator for historical anchors and unchanged OOD coordinates.
    PROTOCOL_VERSION = "lasa-rollouts-v2"
    PROTOCOL_ROOT = ROOT / "evaluation_protocols" / PROTOCOL_VERSION
    data = load_motion(motion)
    PROTOCOL_ROOT.mkdir(parents=True,exist_ok=True)
    path = PROTOCOL_ROOT / (motion+".json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["data_fingerprint"] != data["fingerprint"]:
            raise ValueError("Dataset changed: create a new protocol version instead of replacing starts")
        return saved
    points = np.concatenate([x for x,u in data["demos"]])
    length = float(np.linalg.norm(np.ptp(points,axis=0)))
    centers = np.array([x[0] for x,u in data["demos"]])
    seed = int.from_bytes(hashlib.sha256((PROTOCOL_VERSION+motion).encode()).digest()[:4],"big")
    rng = np.random.default_rng(seed)
    starts = []
    def add(group,xy,reference,**extra):
        starts.append(dict(id=f"{group}-{sum(s['group']==group for s in starts):02d}",
                           group=group,position=np.asarray(xy).tolist(),reference_demo=reference,**extra))
    for i,(x,u) in enumerate(data["demos"]): add("demo_start",x[0],i)
    for demo_id,center in enumerate(centers):
        phase=2*np.pi*demo_id/len(centers)
        for group,radius,count in [("near_start",.02*length,2),("id_ring",.05*length,2),("ood_ring",.0625*length,4)]:
            for angle in phase+np.linspace(0,2*np.pi,count,endpoint=False):
                add(group,center+radius*np.array([np.cos(angle),np.sin(angle)]),demo_id)
    for _ in range(12):
        demo_id=int(rng.integers(len(data["demos"])))
        x=data["demos"][demo_id][0]
        sample=int(rng.integers(int(.1*len(x)),int(.85*len(x))))
        add("along_trajectory",x[sample],demo_id,sample_index=sample)
    all_points=np.vstack([points,np.array([s["position"] for s in starts])])
    lower,upper=all_points.min(axis=0),all_points.max(axis=0)
    pad=.07*np.max(upper-lower)
    protocol=dict(version=PROTOCOL_VERSION,motion=motion,seed=seed,data_fingerprint=data["fingerprint"],
        starts=starts,centers=centers.tolist(),motion_extent=length,
        near_start_radius=.02*length,id_ring_radius=.05*length,ood_ring_radius=.0625*length,
        id_radius_fractions=[.02,.05],id_groups=["demo_start","near_start","id_ring"],
        ood_radius_multiplier=1.25,plot_bounds=[(lower-pad).tolist(),(upper+pad).tolist()],
        integrator=dict(method="RK45",horizon=60.0,max_step=.25,rtol=1e-6,atol=1e-8,
                        target_radius=.01,escape_radius=8.0,max_rhs_evaluations=20000),
        interpretation="ID: exact demonstration starts and offsets of 2% and 5% of the normalized motion bounding-box diagonal, around every demonstration start. OOD: radius 6.25%, 25% larger than the outer ID radius. Along-trajectory starts are a separate group. These geometric labels are not a statistical support guarantee.")
    path.write_text(json.dumps(protocol,indent=2)+"\n",encoding="utf-8")
    return protocol


def _v3_protocol_for(motion):
    PROTOCOL_VERSION = "lasa-rollouts-v3"
    PROTOCOL_ROOT = ROOT / "evaluation_protocols" / PROTOCOL_VERSION
    data=load_motion(motion)
    PROTOCOL_ROOT.mkdir(parents=True,exist_ok=True)
    path=PROTOCOL_ROOT/(motion+'.json')
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved['data_fingerprint']!=data['fingerprint']:
            raise ValueError('Dataset changed: create a new protocol version')
        return saved
    previous=_v2_protocol_for(motion)
    seed=int.from_bytes(hashlib.sha256((PROTOCOL_VERSION+motion).encode()).digest()[:4],'big')
    rng=np.random.default_rng(seed)
    starts=[]
    for anchor in previous['starts']:
        if anchor['group']!='along_trajectory':continue
        angle=float(rng.uniform(0,2*np.pi))
        fraction=float(rng.uniform(.01,.02))
        offset=previous['motion_extent']*fraction*np.array([np.cos(angle),np.sin(angle)])
        starts.append(dict(id=f'id_trajectory-{len(starts):02d}',group='id_trajectory',
            position=(np.array(anchor['position'])+offset).tolist(),
            reference_demo=anchor['reference_demo'],sample_index=anchor['sample_index'],
            anchor_position=anchor['position'],offset_fraction=fraction))
    # Copy records exactly, not just their radius or random seed.
    starts.extend(s for s in previous['starts'] if s['group']=='ood_ring')
    protocol=dict(version=PROTOCOL_VERSION,motion=motion,seed=seed,
        data_fingerprint=data['fingerprint'],starts=starts,id_groups=['id_trajectory'],
        id_offset_fractions=[.01,.02],anchor_source='lasa-rollouts-v2',
        ood_source='lasa-rollouts-v2',ood_ring_radius=previous['ood_ring_radius'],
        centers=previous['centers'],motion_extent=previous['motion_extent'],
        plot_bounds=previous['plot_bounds'],integrator=previous['integrator'],
        interpretation='ID: the 12 saved along-trajectory anchors, each perturbed by a seeded random direction and an offset magnitude uniformly between 1% and 2% of the normalized motion bounding-box diagonal. No demonstration-start ID group. The 28 OOD records are copied exactly from v2 (6.25% radius about demonstration starts). These are geometric labels, not statistical support estimates.')
    path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    return protocol


def protocol_for(motion):
    data=load_motion(motion)
    PROTOCOL_ROOT.mkdir(parents=True,exist_ok=True)
    path=PROTOCOL_ROOT/(motion+'.json')
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved['data_fingerprint']!=data['fingerprint']:
            raise ValueError('Dataset changed: create a new protocol version')
        return saved
    previous=_v3_protocol_for(motion)
    seed=int.from_bytes(hashlib.sha256((PROTOCOL_VERSION+motion).encode()).digest()[:4],'big')
    rng=np.random.default_rng(seed)
    anchors=[s for s in previous['starts'] if s['group']=='id_trajectory']
    xy=np.array([s['anchor_position'] for s in anchors])
    # Spread eight anchor locations through the motion, independent of policy.
    chosen=[0]
    while len(chosen)<8:
        distances=np.linalg.norm(xy[:,None]-xy[chosen][None,:],axis=2).min(axis=1)
        distances[chosen]=-1
        chosen.append(int(np.argmax(distances)))
    starts=list(previous['starts'])
    for index,anchor_id in enumerate(chosen):
        anchor=anchors[anchor_id]
        angle=float(rng.uniform(0,2*np.pi));fraction=float(rng.uniform(.05,.0625))
        position=xy[anchor_id]+previous['motion_extent']*fraction*np.array([np.cos(angle),np.sin(angle)])
        starts.append(dict(id=f'ood_trajectory-{index:02d}',group='ood_trajectory',
            position=position.tolist(),reference_demo=anchor['reference_demo'],
            sample_index=anchor['sample_index'],anchor_position=anchor['anchor_position'],
            offset_fraction=fraction,source_anchor_id=anchor['id']))
    points=np.vstack([np.concatenate([x for x,u in data['demos']]),np.array([s['position'] for s in starts])])
    low,high=points.min(axis=0),points.max(axis=0);pad=.07*np.max(high-low)
    # Extend the old view only if needed to include new outer starts.
    bounds=np.array(previous['plot_bounds'])
    bounds[0]=np.minimum(bounds[0],low-pad);bounds[1]=np.maximum(bounds[1],high+pad)
    protocol=dict(previous,version=PROTOCOL_VERSION,seed=seed,starts=starts,
        previous_protocol='lasa-rollouts-v3',ood_groups=['ood_ring','ood_trajectory'],
        ood_trajectory_offset_fractions=[.05,.0625],plot_bounds=bounds.tolist(),
        interpretation='ID: same 12 trajectory starts with 1-2% offsets as v3. OOD: same 28 ring starts plus eight spatially spread trajectory anchors perturbed by 5-6.25% of the normalized motion bounding-box diagonal. All starts are deterministic per motion; geometric labels are not statistical support estimates.')
    path.write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    return protocol


def _arc_samples(path,n=100):
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
    distance,indices=np.unique(distance,return_index=True)
    if len(indices)<2:return path[:1]
    target=np.linspace(0,distance[-1],n)
    return np.column_stack([np.interp(target,distance,path[indices,axis]) for axis in range(2)])


def evaluate(model,callback=lambda message:None,velocity_fn=None):
    plan=protocol_for(model["config"]["motion"])
    data=load_motion(plan["motion"])
    trees=[cKDTree(x) for x,u in data["demos"]]
    params=plan["integrator"]
    records=[]
    for index,start in enumerate(plan["starts"]):
        if index%10==0:callback(f"Rolling out {index+1}/{len(plan['starts'])} fixed starts")
        calls=0
        def field(t,x):
            nonlocal calls
            calls+=1
            if calls>params["max_rhs_evaluations"]:raise RuntimeError("RHS evaluation budget exceeded")
            velocity=(predict(model,x) if velocity_fn is None else np.asarray(velocity_fn(x[None])))[0]
            if np.shape(velocity)!=(2,):raise RuntimeError("Policy must return an N-by-2 velocity array")
            if not np.isfinite(velocity).all():raise RuntimeError("Nonfinite vector field")
            return velocity
        def target(t,x):return np.linalg.norm(x)-params["target_radius"]
        def escape(t,x):return params["escape_radius"]-np.linalg.norm(x)
        target.terminal=True;target.direction=-1
        escape.terminal=True;escape.direction=-1
        initial=np.array(start["position"])
        failure=None
        try:
            if np.linalg.norm(initial)<=params["target_radius"]:
                path=initial[None];ts=np.array([0.]);status="target"
            else:
                solution=solve_ivp(field,(0,params["horizon"]),initial,
                    events=[target,escape],max_step=params["max_step"],rtol=params["rtol"],atol=params["atol"])
                path,ts=solution.y.T,solution.t
                status="target" if len(solution.t_events[0]) else "escaped" if len(solution.t_events[1]) else "horizon" if solution.success else "integration_failed"
        except RuntimeError as error:
            path,ts=initial[None],np.array([0.]);status="integration_failed";failure=str(error)
        reference_distance=float(trees[start["reference_demo"]].query(_arc_samples(path))[0].mean())
        records.append(dict(**start,status=status,path=path,times=ts,final_distance=float(np.linalg.norm(path[-1])),
            duration=float(ts[-1]),mean_reference_distance=reference_distance,rhs_calls=calls,failure=failure))
    groups={}
    for group in (*GROUPS,"id_all","ood_all","all"):
        selected=[r for r in records if group=="all" or (group=="id_all" and r["group"] in plan["id_groups"]) or (group=="ood_all" and r["group"] in plan["ood_groups"]) or r["group"]==group]
        groups[group]=dict(count=len(selected),target_count=sum(r["status"]=="target" for r in selected),
            target_rate=sum(r["status"]=="target" for r in selected)/len(selected),
            mean_reference_distance=float(np.mean([r["mean_reference_distance"] for r in selected])),
            mean_final_distance=float(np.mean([r["final_distance"] for r in selected])))
    return dict(protocol=plan,rollouts=records,groups=groups)


def evaluate_policy(motion,velocity_fn,callback=lambda message:None):
    """Evaluate any baseline callable N-by-2 -> N-by-2 on the exact same starts."""
    return evaluate(dict(config=dict(motion=motion)),callback,velocity_fn=velocity_fn)
