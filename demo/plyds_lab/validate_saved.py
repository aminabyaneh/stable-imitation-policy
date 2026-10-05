"""Independently check archived policies, certificate identities and start reuse."""
import json
from pathlib import Path
import sys
import numpy as np
from .runner import load_run
from .polynomial import features,gradient_features


maximum=0.0
count=0
start_sets={}
for argument in sys.argv[1:]:
    for path in Path(argument).rglob("result.json"):
        model,evaluation,summary=load_run(path.parent)
        version=summary['protocol']['version']
        assert len(evaluation["rollouts"])=={'lasa-rollouts-v1':55,'lasa-rollouts-v2':75,'lasa-rollouts-v3':40,'lasa-rollouts-v4':48}[version]
        assert len(evaluation['rollouts'])==len(summary['protocol']['starts'])
        points=np.random.default_rng(41).uniform(-.5,.5,(20,2))
        field=features(points,model["policy_basis"])@model["coefficients"]
        grad=np.column_stack([gradient_features(points,model["lyapunov_basis"],axis)@model["vcoeff"] for axis in range(2)])
        z=features(points,model["certificate_basis"])
        residual=np.sum(grad*field,axis=1)+model["config"]["epsilon_decay"]*np.sum(points*points,axis=1)+np.einsum("ni,ij,nj->n",z,model["r"],z)
        maximum=max(maximum,float(np.abs(residual).max()))
        motion=(model["config"]["motion"],version)
        starts=json.dumps(summary["protocol"]["starts"],sort_keys=True)
        assert motion not in start_sets or start_sets[motion]==starts
        start_sets[motion]=starts
        assert np.isfinite(model["coefficients"]).all()
        count+=1
assert count>0
assert maximum<1e-5
print(f"Verified {count} saved policies and shared start sets; maximum sampled identity residual {maximum:.3g}")
