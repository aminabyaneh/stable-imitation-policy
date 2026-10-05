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
        for record,start in zip(evaluation['rollouts'],summary['protocol']['starts']):
            assert len(record['path'])==len(record['times'])>0
            assert np.isfinite(record['path']).all() and np.isfinite(record['times']).all()
            assert np.all(np.diff(record['times'])>0)
            np.testing.assert_array_equal(record['path'][0],start['position'])
            if record['status']=='target':
                assert np.linalg.norm(record['path'][-1])<=summary['protocol']['integrator']['target_radius']+1e-8
        points=np.random.default_rng(41).uniform(-.5,.5,(20,2))
        if model['config'].get('method')=='snds':
            from .neural import audit
            import torch
            torch.set_num_threads(1)
            checks,arrays=audit(model['policy'])
            assert checks['negative_v_count']==checks['decay_violations']==checks['lower_bound_violations']==0
            assert checks['origin_velocity_norm']==0
            for key,value in arrays.items():np.testing.assert_allclose(value,model['audit'][key],atol=1e-12)
            np.testing.assert_allclose(model['policy'].predict(summary['field']['points']),summary['field']['velocities'],atol=1e-12)
            from .lyapunov import sample_lyapunov
            heat=summary['lyapunov']
            np.testing.assert_allclose(sample_lyapunov(model,heat['bounds'],heat['resolution'])['values'],heat['values'],atol=1e-12)
            residual=np.maximum(arrays['decay_residual'],0)
        else:
            field=features(points,model["policy_basis"])@model["coefficients"]
            grad=np.column_stack([gradient_features(points,model["lyapunov_basis"],axis)@model["vcoeff"] for axis in range(2)])
            z=features(points,model["certificate_basis"])
            residual=np.sum(grad*field,axis=1)+model["config"]["epsilon_decay"]*np.sum(points*points,axis=1)+np.einsum("ni,ij,nj->n",z,model["r"],z)
            assert np.isfinite(model["coefficients"]).all()
        maximum=max(maximum,float(np.abs(residual).max()))
        motion=(model["config"]["motion"],version)
        starts=json.dumps(summary["protocol"]["starts"],sort_keys=True)
        assert motion not in start_sets or start_sets[motion]==starts
        start_sets[motion]=starts
        count+=1
assert count>0
assert maximum<1e-5
print(f"Verified {count} saved policies and shared start sets; maximum sampled certificate residual {maximum:.3g}")
