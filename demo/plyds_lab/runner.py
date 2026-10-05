"""Training, evaluation, and portable numerical artifacts in one run."""
from dataclasses import asdict
from datetime import datetime,timezone
import json
import importlib.metadata
import platform
from pathlib import Path
import time
import numpy as np
from .data import ROOT,load_motion
from .solver import train,predict
from .evaluation import evaluate
from .plotting import save_single,view_bounds
from .lyapunov import sample_lyapunov

RUNS=ROOT/"runs"/"plyds-lab"


def run(config,callback=print,output=None):
    started=time.perf_counter()
    if getattr(config,'method','plyds')=='snds':
        from .neural import train as train_snds
        model=train_snds(config,callback)
    else:model=train(config,callback)
    return save_evaluated(model,callback,output,started)


def save_evaluated(model,callback=print,output=None,started=None,source_run=None):
    started=time.perf_counter() if started is None else started
    config=model['config']
    evaluation=evaluate(model,callback)
    output=Path(output) if output else RUNS/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"-"+config['motion'])
    output.mkdir(parents=True,exist_ok=False)
    neural=config.get('method')=='snds'
    if neural:
        model['policy'].save(output/'model.pt')
        np.savez_compressed(output/'audit.npz',**model['audit'])
        np.savez_compressed(output/'training-data.npz',**model['training_data'])
    else:
        np.savez_compressed(output/"model.npz",coefficients=model["coefficients"],policy_basis=model["policy_basis"],
            vcoeff=model["vcoeff"],lyapunov_basis=model["lyapunov_basis"],q=model["q"],r=model["r"],certificate_basis=model["certificate_basis"])
    trajectories={}
    for i,record in enumerate(evaluation["rollouts"]):
        trajectories[f"path_{i}"]=record["path"];trajectories[f"times_{i}"]=record["times"]
    np.savez_compressed(output/"rollouts.npz",**trajectories)
    bounds=view_bounds(evaluation["protocol"])
    xx,yy=np.meshgrid(np.linspace(bounds[0,0],bounds[1,0],27),np.linspace(bounds[0,1],bounds[1,1],27))
    points=np.column_stack([xx.ravel(),yy.ravel()])
    serial_rollouts=[]
    for record in evaluation["rollouts"]:
        small={k:v for k,v in record.items() if k not in ("path","times")}
        idx=np.unique(np.linspace(0,len(record["path"])-1,min(250,len(record["path"]))).astype(int))
        small["path"]=record["path"][idx].tolist();serial_rollouts.append(small)
    summary=dict(config=config,metrics=model["metrics"],groups=evaluation["groups"],
        protocol=evaluation["protocol"],rollouts=serial_rollouts,
        demonstrations=[x[::4].tolist() for x,u in load_motion(config['motion'])["demos"]],
        field=dict(points=points.tolist(),velocities=predict(model,points).tolist()),
        lyapunov=sample_lyapunov(model,bounds),
        run_id=output.name,data_fingerprint=model["data_fingerprint"],view_bounds=bounds.tolist(),
        implementation="snds-revised-v1" if neural else "plyds-lab-v4",source_run=source_run,python=platform.python_version(),
        packages={name:importlib.metadata.version(name) for name in ("numpy","scipy","cvxpy","scs","pyLasaDataset")+(('torch',) if neural else ())})
    callback("Rendering vector field and saving numerical results")
    save_single(model,evaluation,output)
    summary["total_elapsed_seconds"]=time.perf_counter()-started
    (output/"result.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"protocol.json").write_text(json.dumps(evaluation["protocol"],indent=2)+"\n",encoding="utf-8")
    callback(f"Saved {output}")
    return model,evaluation,summary,output


def load_run(directory):
    directory=Path(directory)
    summary=json.loads((directory/"result.json").read_text(encoding="utf-8"))
    if summary['config'].get('method')=='snds':
        from .neural import NeuralPolicy
        model=dict(policy=NeuralPolicy.load(directory/'model.pt',summary['config']))
        for key,file in [('audit','audit.npz'),('training_data','training-data.npz')]:
            with np.load(directory/file,allow_pickle=False) as archive:model[key]={k:archive[k] for k in archive.files}
    else:
        with np.load(directory/"model.npz",allow_pickle=False) as archive:
            model={k:archive[k] for k in archive.files}
    model.update(config=summary["config"],metrics=summary["metrics"],data_fingerprint=summary['data_fingerprint'])
    with np.load(directory/"rollouts.npz",allow_pickle=False) as archive:
        records=[dict(r,path=archive[f"path_{i}"],times=archive[f"times_{i}"]) for i,r in enumerate(summary["rollouts"])]
    evaluation=dict(protocol=summary["protocol"],groups=summary["groups"],rollouts=records)
    return model,evaluation,summary
