"""Revised theorem, training gradients, model reload, and API contract checks."""
from dataclasses import asdict
import uuid
import unittest
import numpy as np
import torch
from .neural import SNDSConfig, RevisedSNDS, NeuralPolicy, project, train, audit
from .server import parse_config
from .lyapunov import sample_lyapunov
from .data import ROOT


class NeuralChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(13)
        self.config=SNDSConfig(epochs=2,n_demos=1,samples_per_demo=20)
        self.net=RevisedSNDS(self.config)

    def test_strong_convexity_anchors_and_projected_decay(self):
        zero=torch.zeros((1,2),dtype=torch.float64,requires_grad=True)
        v=self.net.potential(zero)
        np.testing.assert_allclose(v.detach(),0,atol=1e-14)
        np.testing.assert_allclose(torch.autograd.grad(v.sum(),zero)[0],0,atol=1e-14)
        for point in ([0.,0.],[.7,-1.2],[-1.,2.]):
            x=torch.tensor(point,dtype=torch.float64)
            h=torch.autograd.functional.hessian(lambda z:self.net.potential(z[None]).sum(),x)
            self.assertGreaterEqual(torch.linalg.eigvalsh(h).min(),2*self.config.quadratic_margin-1e-10)
        metrics,_=audit(NeuralPolicy(self.net))
        for key in ['negative_v_count','decay_violations','lower_bound_violations','origin_velocity_norm']:
            self.assertEqual(metrics[key],0)

    def test_projection_is_exact_near_goal_and_leaves_feasible_raw_unchanged(self):
        x=torch.tensor([[0.,0.],[1e-7,0.],[1.,1.]],dtype=torch.float64)
        v=x.square().sum(1,keepdim=True);g=2*x
        raw=torch.tensor([[4.,2.],[1.,0.],[-1.,-1.]],dtype=torch.float64)
        f=project(x,raw,v,g,.01)
        np.testing.assert_array_equal(f[0],np.zeros(2))
        np.testing.assert_array_equal(f[2],raw[2])
        self.assertLessEqual(float(((g*f).sum(1,keepdim=True)+.01*v).max()),1e-20)

    def test_cached_inference_and_saved_weights_match_training_network(self):
        points=np.array([[0.,0.],[.1,.2],[-1.,.4],[1e-6,-1e-6]])
        x=torch.tensor(points,dtype=torch.float64,requires_grad=True)
        expected=self.net(x).detach().numpy()
        potential=self.net.potential(x).detach().numpy().ravel()
        policy=NeuralPolicy(self.net)
        np.testing.assert_allclose(policy.predict(points),expected,rtol=1e-7,atol=1e-12)
        np.testing.assert_allclose(policy.potential(points),potential,atol=1e-12)
        cache=ROOT/'.cache';cache.mkdir(exist_ok=True)
        path=cache/(uuid.uuid4().hex+'.pt')
        try:
            policy.save(path)
            loaded=NeuralPolicy.load(path,asdict(self.config))
            np.testing.assert_array_equal(loaded.predict(points),policy.predict(points))
        finally:path.unlink(missing_ok=True)
        model=dict(config=asdict(self.config),policy=policy)
        heat=sample_lyapunov(model,[[-1,-1],[1,1]],11)
        self.assertEqual(heat['source'],'saved neural potential')
        self.assertAlmostEqual(heat['values'][5][5],0,places=12)

    def test_parameter_gradient_includes_anchors(self):
        x=torch.tensor([[.2,.7],[-.8,.4]],dtype=torch.float64,requires_grad=True)
        weight=self.net.q.W[0]
        loss=self.net.potential(x).sum()
        derivative=torch.autograd.grad(loss,weight)[0][0,0].item()
        original=weight[0,0].item();eps=1e-5
        with torch.no_grad():weight[0,0]=original+eps
        upper=self.net.potential(x).sum().item()
        with torch.no_grad():weight[0,0]=original-eps
        lower=self.net.potential(x).sum().item()
        self.assertAlmostEqual(derivative,(upper-lower)/(2*eps),places=7)

    def test_cpu_training_and_api_validation(self):
        model=train(self.config)
        self.assertTrue(np.isfinite(model['metrics']['heldout_velocity_mse']))
        self.assertEqual(next(model['policy'].network.parameters()).device.type,'cpu')
        self.assertEqual(parse_config({'method':'snds'}).method,'snds')
        self.assertEqual(parse_config({}).policy_degree,4)
        for payload in [[],{'method':'other'},{'method':'snds','policy_degree':4},
                        {'method':'snds','epochs':True},{'method':'snds','decay_rate':0},
                        {'method':'snds','quadratic_margin':float('nan')},
                        {'method':'snds','motion':'missing'}]:
            with self.assertRaises((ValueError,TypeError)):parse_config(payload)


if __name__=='__main__':unittest.main()
