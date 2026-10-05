"""Math, protocol and persistence checks for the research interface."""
import json
import unittest
import numpy as np
from .data import load_motion,demonstration_ids
from .evaluation import protocol_for,evaluate_policy,_v2_protocol_for,_v3_protocol_for
from .polynomial import basis,features,gradient_features,derivative_policy_map,derivative_v_map
from .solver import TrainingConfig,train,predict
from .server import safe_run_path
from .lyapunov import sample_lyapunov


class PolynomialChecks(unittest.TestCase):
    def test_coefficient_maps_match_analytic_derivative(self):
        fp,vp,cp=basis(3),basis(4),basis(6,constant=True)
        v=np.array([1 if p in ((2,0),(0,2)) else .2 if p==(2,2) else 0 for p in vp])
        f=np.zeros((len(fp),2));f[fp.index((1,0)),0]=-1;f[fp.index((0,1)),1]=-1
        points=np.random.default_rng(4).normal(size=(40,2))
        expected=-2*np.sum(points**2,axis=1)-.8*points[:,0]**2*points[:,1]**2
        a=derivative_policy_map(v,vp,fp,cp)@f.T.reshape(-1)
        b=derivative_v_map(f,fp,vp,cp)@v
        np.testing.assert_allclose(features(points,cp)@a,expected,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a,b,atol=1e-12)

    def test_full_basis_includes_mixed_terms(self):
        self.assertIn((1,5),basis(6));self.assertIn((5,1),basis(6))
        self.assertNotIn((0,0),basis(6))


class ProtocolChecks(unittest.TestCase):
    def test_id_perturbs_saved_trajectory_anchors_and_ood_is_unchanged(self):
        first=protocol_for("Sine");second=protocol_for("Sine")
        self.assertEqual(json.dumps(first,sort_keys=True),json.dumps(second,sort_keys=True))
        self.assertEqual(len(first["starts"]),48)
        previous=_v2_protocol_for('Sine')
        self.assertEqual([s for s in first['starts'] if s['group']=='ood_ring'],
                         [s for s in previous['starts'] if s['group']=='ood_ring'])
        anchors=[s for s in previous['starts'] if s['group']=='along_trajectory']
        ids=[s for s in first['starts'] if s['group']=='id_trajectory']
        self.assertEqual(len(ids),12)
        self.assertEqual(set(s['group'] for s in first['starts']),{'id_trajectory','ood_ring','ood_trajectory'})
        for start,anchor in zip(ids,anchors):
            self.assertEqual(start['anchor_position'],anchor['position'])
            data_point=load_motion('Sine')['demos'][start['reference_demo']][0][start['sample_index']]
            np.testing.assert_array_equal(data_point,start['anchor_position'])
            fraction=np.linalg.norm(np.array(start['position'])-data_point)/first['motion_extent']
            self.assertGreaterEqual(fraction,.01)
            self.assertLessEqual(fraction,.02)
            self.assertAlmostEqual(fraction,start['offset_fraction'])

    def test_extra_ood_preserves_v3_and_uses_eight_distinct_trajectory_anchors(self):
        plan=protocol_for('Sine');previous=_v3_protocol_for('Sine')
        self.assertEqual(plan['starts'][:40],previous['starts'])
        extra=plan['starts'][40:]
        self.assertEqual(len(extra),8)
        self.assertEqual(len(set(s['source_anchor_id'] for s in extra)),8)
        for start in extra:
            self.assertEqual(start['group'],'ood_trajectory')
            anchor=load_motion('Sine')['demos'][start['reference_demo']][0][start['sample_index']]
            np.testing.assert_array_equal(anchor,start['anchor_position'])
            fraction=np.linalg.norm(np.array(start['position'])-anchor)/plan['motion_extent']
            self.assertGreaterEqual(fraction,.05)
            self.assertLessEqual(fraction,.0625)
            self.assertAlmostEqual(fraction,start['offset_fraction'])

    def test_coverage_split_is_disjoint_and_covers_multimodal_starts(self):
        data=load_motion("Multi_Models_1")
        ids=demonstration_ids(data,5,"cover_starts")
        self.assertEqual(ids,[0,4,6,5,1])
        self.assertEqual(sorted(set(range(7))-set(ids)),[2,3])

    def test_dataset_is_not_mutated(self):
        data=load_motion("Sine");before=data["fingerprint"]
        self.assertFalse(data["demos"][0][0].flags.writeable)
        self.assertEqual(before,load_motion("Sine")["fingerprint"])

    def test_other_baseline_uses_same_starts_and_reaches_target(self):
        result=evaluate_policy("Sine",lambda points:-points)
        self.assertEqual(result["protocol"]["starts"],protocol_for("Sine")["starts"])
        self.assertEqual(result["groups"]["all"]["target_count"],48)
        self.assertEqual(result['groups']['id_all']['count'],12)
        self.assertEqual(result['groups']['ood_ring']['count'],28)
        self.assertEqual(result['groups']['ood_trajectory']['count'],8)
        self.assertEqual(result['groups']['ood_all']['count'],36)
        self.assertEqual(result['groups']['ood_all']['target_count'],36)
        self.assertEqual(result['groups']['id_trajectory']['count'],12)
        # Analytic linear decay time is log(||x0|| / target radius).
        record=result["rollouts"][0]
        expected=np.log(np.linalg.norm(record["position"])/.01)
        self.assertAlmostEqual(record["duration"],expected,places=4)


class TrainingChecks(unittest.TestCase):
    def test_heatmap_uses_polynomial_and_preserves_negative_values(self):
        vp=basis(4);v=np.zeros(len(vp));v[vp.index((2,0))]=1;v[vp.index((0,2))]=1;v[vp.index((2,2))]=.3
        h=sample_lyapunov(dict(lyapunov_basis=vp,vcoeff=v),[[-2,-2],[2,2]],11)
        a=np.linspace(-2,2,11);xx,yy=np.meshgrid(a,a)
        np.testing.assert_allclose(h['values'],xx**2+yy**2+.3*xx**2*yy**2)
        self.assertEqual(h['values'][5][5],0)
        v=-v
        negative=sample_lyapunov(dict(lyapunov_basis=vp,vcoeff=v),[[-2,-2],[2,2]],11)
        self.assertLess(negative['minimum'],0)
        np.testing.assert_allclose(negative['values'],-np.array(h['values']))

    def test_real_scs_fit_obeys_equilibrium_and_polynomial_identity(self):
        model=train(TrainingConfig(motion="Sine",policy_degree=2,n_demos=1,samples_per_demo=40))
        np.testing.assert_array_equal(predict(model,[[0,0]]),np.zeros((1,2)))
        points=np.random.default_rng(7).uniform(-1,1,(70,2))
        grad=np.column_stack([gradient_features(points,model["lyapunov_basis"],a)@model["vcoeff"] for a in range(2)])
        z=features(points,model["certificate_basis"])
        residual=np.sum(grad*predict(model,points),axis=1)+.001*np.sum(points**2,axis=1)+np.einsum('ni,ij,nj->n',z,model["r"],z)
        self.assertLess(np.abs(residual).max(),1e-5)
        self.assertFalse(model["metrics"]["certificate_verified"])

    def test_invalid_configuration_and_path_are_rejected(self):
        for config in [TrainingConfig(n_demos=0),TrainingConfig(policy_degree=10),
                       TrainingConfig(lyapunov_degree=4,learn_lyapunov=False)]:
            with self.assertRaises(ValueError):config.validate()
        with self.assertRaises(ValueError):safe_run_path("../../../../Thesis.tex")


if __name__=="__main__":unittest.main()
