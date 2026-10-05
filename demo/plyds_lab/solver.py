"""Convex SOS subproblems and local alternating Lyapunov search, using SCS."""
from dataclasses import dataclass, asdict
import time
import cvxpy as cp
import numpy as np
from .data import load_motion, training_samples, demonstration_ids
from .polynomial import basis, features, gradient_features, gram_map, derivative_policy_map, derivative_v_map


@dataclass(frozen=True)
class TrainingConfig:
    motion: str = "Sine"
    policy_degree: int = 4
    lyapunov_degree: int = 2
    n_demos: int = 5
    demo_selection: str = "first"
    learn_lyapunov: bool = False
    alternating_steps: int = 3
    samples_per_demo: int = 200
    epsilon_v: float = 0.001
    epsilon_decay: float = 0.001
    regularization: float = 1e-7
    solver_tolerance: float = 1e-7
    max_iterations: int = 50000
    solver_time_limit: float = 20.0

    def validate(self):
        if not 1 <= self.policy_degree <= 8: raise ValueError("Policy degree must be 1..8")
        if self.lyapunov_degree not in (2,4,6): raise ValueError("Lyapunov degree must be 2, 4 or 6")
        if not 1 <= self.n_demos <= 7: raise ValueError("Demonstrations must be 1..7")
        if self.demo_selection not in ("first","cover_starts"):raise ValueError("Unknown demonstration selection")
        if not 1 <= self.alternating_steps <= 6: raise ValueError("Alternating steps must be 1..6")
        if not 20 <= self.samples_per_demo <= 1000: raise ValueError("Samples/demo must be 20..1000")
        if self.lyapunov_degree > 2 and not self.learn_lyapunov:
            raise ValueError("Higher Lyapunov degrees require learned SOS mode")
        if not (0 < self.epsilon_v < 1 and 0 < self.epsilon_decay <= 0.1):
            raise ValueError("Invalid positive certificate margins")
        if not (0 <= self.regularization <= 1 and 1e-9 <= self.solver_tolerance <= 1e-3):
            raise ValueError("Invalid regularization or solver tolerance")
        if not 100 <= self.max_iterations <= 200000 or not 1 <= self.solver_time_limit <= 60:
            raise ValueError("Solver budget out of bounds")


def _solve(problem, matrices, config, label, callback, history):
    callback(f"Solving {label} with SCS")
    start = time.perf_counter()
    problem.solve(solver="SCS", eps_abs=config.solver_tolerance, eps_rel=config.solver_tolerance,
                  max_iters=config.max_iterations, time_limit_secs=config.solver_time_limit,
                  verbose=False, warm_start=False)
    record = dict(phase=label, status=problem.status, elapsed_seconds=time.perf_counter()-start,
                  objective=problem.value, iterations=problem.solver_stats.num_iters)
    if problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE) or any(m.value is None for m in matrices):
        record["numerically_accepted"] = False
        history.append(record)
        raise RuntimeError(f"{label}: {problem.status}")
    record["max_constraint_residual"] = max(float(np.max(c.violation())) for c in problem.constraints)
    record["minimum_gram_eigenvalue"] = min(float(np.linalg.eigvalsh((m.value+m.value.T)/2).min()) for m in matrices)
    record["numerically_accepted"] = bool(record["max_constraint_residual"] <= 5e-5 and
        record["minimum_gram_eigenvalue"] >= -5e-5 and all(np.isfinite(m.value).all() for m in matrices))
    history.append(record)
    if not record["numerically_accepted"]:
        raise RuntimeError(f"{label}: residual/PSD checks failed ({record['max_constraint_residual']:.2g})")
    return record


def train(config: TrainingConfig, callback=lambda message: None):
    config.validate()
    started = time.perf_counter()
    data = load_motion(config.motion)
    x, u = training_samples(data, config.n_demos, config.samples_per_demo,config.demo_selection)
    train_ids=demonstration_ids(data,config.n_demos,config.demo_selection)
    heldout_ids=[i for i in range(len(data["demos"])) if i not in train_ids]
    fp = basis(config.policy_degree)
    vp = basis(config.lyapunov_degree)
    # Degree of grad(V).f; odd leading terms must be zero for global negativity.
    max_derivative_degree = config.policy_degree + config.lyapunov_degree - 1
    rp = basis(max_derivative_degree//2)
    cpowers = basis(max_derivative_degree, constant=True)
    rmap = gram_map(rp, cpowers)
    decay = np.array([config.epsilon_decay if m in ((2,0),(0,2)) else 0.0 for m in cpowers])
    vcoeff = np.array([1.0 if m in ((2,0),(0,2)) else 0.0 for m in vp])
    qp = basis(config.lyapunov_degree//2)
    qmap = gram_map(qp, vp)
    vbase = np.array([config.epsilon_v if m in ((2,0),(0,2)) else 0.0 for m in vp])
    qvalue = np.zeros((len(qp),len(qp)))
    for i, m in enumerate(qp):
        if m in ((1,0),(0,1)): qvalue[i,i] = 1-config.epsilon_v
    phi = features(x, fp)
    coeff = np.zeros((len(fp),2))
    coeff[fp.index((1,0)),0] = -1
    coeff[fp.index((0,1)),1] = -1
    history, warnings = [], []

    def certificate_step(current_coeff, number):
        q = cp.Variable((len(qp),len(qp)), PSD=True)
        r = cp.Variable((len(rp),len(rp)), PSD=True)
        vc = vbase + qmap @ cp.vec(q,order="C")
        derivative = derivative_v_map(current_coeff,fp,vp,cpowers)
        constraints = [derivative @ vc + decay + rmap @ cp.vec(r,order="C") == 0,
                       vc[vp.index((2,0))]+vc[vp.index((0,2))] == 2,
                       cp.trace(q) <= 100]
        # Encourage demonstrated velocities to descend the next certificate.
        directional = sum(gradient_features(x,vp,axis)*u[:,axis,None] for axis in range(2))
        violations = cp.pos(directional @ vc + 0.05*np.sum(u*u,axis=1))
        objective = cp.sum_squares(violations)/len(x) + 1e-6*cp.sum_squares(q)
        _solve(cp.Problem(cp.Minimize(objective),constraints),[q,r],config,
               f"certificate {number}",callback,history)
        return np.array(vc.value),q.value.copy()

    best = None
    rounds = config.alternating_steps if config.learn_lyapunov else 1
    for round_id in range(rounds):
        if config.learn_lyapunov:
            try:
                proposed_v, proposed_q = certificate_step(coeff,round_id+1)
            except RuntimeError as error:
                warnings.append(str(error)); proposed_v, proposed_q = vcoeff,qvalue
        else:
            proposed_v, proposed_q = vcoeff,qvalue
        c = cp.Variable((len(fp),2))
        r = cp.Variable((len(rp),len(rp)),PSD=True)
        derivative = derivative_policy_map(proposed_v,vp,fp,cpowers)
        constraints = [derivative @ cp.vec(c,order="F") + decay + rmap @ cp.vec(r,order="C") == 0]
        objective = cp.sum_squares(phi @ c-u)/(2*len(x)) + config.regularization*cp.sum_squares(c)
        try:
            record = _solve(cp.Problem(cp.Minimize(objective),constraints),[r],config,
                            f"policy {round_id+1}",callback,history)
        except RuntimeError as error:
            if best is None: raise
            warnings.append(str(error)); break
        coeff, vcoeff, qvalue = c.value.copy(),proposed_v,proposed_q
        score = float(np.mean((phi @ coeff-u)**2)+config.regularization*np.sum(coeff*coeff))
        if best is None or score < best["score"]:
            best = dict(coefficients=coeff.copy(),vcoeff=vcoeff.copy(),q=qvalue.copy(),r=r.value.copy(),
                        score=score,status=record["status"],selected_phase=record["phase"])
    if best is None: raise RuntimeError("No numerically acceptable policy")
    # Verify the accepted pair, not an unaccepted final certificate iterate.
    residual = derivative_policy_map(best["vcoeff"],vp,fp,cpowers) @ best["coefficients"].T.reshape(-1)
    residual += decay + rmap @ best["r"].reshape(-1)
    metrics = dict(coefficient_residual=float(np.max(np.abs(residual))),
        q_min_eigenvalue=float(np.linalg.eigvalsh(best["q"]).min()),
        r_min_eigenvalue=float(np.linalg.eigvalsh(best["r"]).min()),
        origin_velocity_norm=0.0,certificate_verified=False,
        status=best["status"],selected_phase=best["selected_phase"],warnings=warnings,
        elapsed_seconds=time.perf_counter()-started,history=history,
        fit_samples=len(x),training_demo_ids=train_ids,heldout_demo_ids=heldout_ids)
    for label, demos in [("training",[data["demos"][i] for i in train_ids]),("heldout",[data["demos"][i] for i in heldout_ids])]:
        metrics[label+"_velocity_mse"] = float(np.mean(np.concatenate([
            (features(dx,fp) @ best["coefficients"]-du)**2 for dx,du in demos]))) if demos else None
    return dict(config=asdict(config),metrics=metrics,policy_basis=fp,lyapunov_basis=vp,
                certificate_basis=rp,data_fingerprint=data["fingerprint"],**best)


def predict(model, points):
    return features(points,model["policy_basis"]) @ model["coefficients"]
