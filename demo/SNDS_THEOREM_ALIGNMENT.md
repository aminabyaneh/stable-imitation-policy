# SNDS in the motion notebook

The browser's **SNDS** option implements the
`stability_proposition` in the thesis SNDS chapter. Its source hashes are in
`snds-theorem-source.json`. This implementation is in `plyds_lab/neural.py`;
the historical `src/` and `exp/` implementations remain unchanged.

## Construction

Coordinates are normalized and the target is zero. For a convex scalar ICNN
q, raw policy h, quadratic margin m > 0, and decay rate c > 0:

```text
V(x) = q(x) − q(0) − ∇q(0)ᵀx + m ‖x‖²
f₀(x) = h(x) − h(0)
p(x) = ∇V(x)
f(x) = f₀(x) − p(x) max(p(x)ᵀf₀(x) + cV(x), 0) / ‖p(x)‖²   (x ≠ 0)
f(0) = 0
```

The thesis parameter δ equals **2m**. Convexity gives V ≥ m‖x‖²,
V(0) = 0, ∇V(0) = 0 and ∇²V ≥ 2mI. The centered raw field is zero
at the goal, and the exact projection enforces ∇Vᵀf ≤ −cV in exact arithmetic.
Under the theorem's smoothness assumptions this defines a locally Lipschitz
field, unique forward-complete solutions, and global asymptotic stability.
It does not imply that every finite-horizon numerical rollout reaches a
chosen target ball, or that Euclidean distance always decreases.

| Requirement | Implementation |
| --- | --- |
| Convex C² ICNN | Two 64-unit hidden layers; nonnegative hidden weights through Softplus; unrestricted input weights |
| Smooth convex nondecreasing activation | 0.01z + 0.99 log(1 + exp(10z))/10, evaluated using `logaddexp` without a linear cutoff |
| C¹ raw field | Three 256-unit hidden layers with the same smooth activation |
| Value and gradient anchoring | Both anchors recomputed with parameter gradients throughout training; cached only for frozen inference |
| Positive strong-convexity margin and decay | Validated positive m and c; both default to 0.01 |
| Exact correction | Exact positive part and denominator off target; explicit zero at target; no denominator epsilon |

The projection has derivative kinks at its switching surface. Replacing its
positive part with a smooth approximation, removing the gradient anchor, or
adding a denominator epsilon would change the stated construction.

## Training and evidence

Training uses CPU float64, Adam, minibatches of 128, gradient norm clipping
at 0.5, and a linear learning-rate schedule from 0.001 to 0.00001. The default
is 3,000 epochs, seed 0, five demonstrations selected to cover their starting
regions, and 200 evenly spaced samples per demonstration. The checkpoint with
the lowest full training MSE at the recorded checkpoints is retained; held-out
demonstrations do not select the checkpoint.

Both methods use the same LASA normalization and immutable v4 evaluation
starts: 12 ID and 36 OOD rollouts. Held-out velocity MSE uses all samples of
the remaining demonstrations. Training uses velocity regression, not the
historical rollout-training objective. These results are therefore a new
experiment, not a reproduction of the published SNDS figures.

Each run saves CPU weights (`model.pt`), fit samples (`training-data.npz`),
full RK45 paths and times (`rollouts.npz`), the fixed protocol, PNG/PDF figures,
and JSON settings, package versions, metrics and the actual neural V grid.
The saved `audit.npz` contains V, ∇V, f, ∇Vᵀf+cV, and V−m‖x‖² at 1,778
points: a 41×41 grid on [−2,2]², three 32-point near-goal rings, and the goal.
Violation counts use an absolute tolerance of 1e-9. The blue asinh heatmap
retains negative numerical values. Saved models can be reopened and checked
with `python -m plyds_lab.validate_saved runs/plyds-lab`.

**Floating-point calculations and sampled checks are not a rigorous global
certificate.** Cancellation near the anchor and numerical integration error
remain possible; every run records `certificate_verified: false`. The theorem
concerns the fixed continuous-time field, not a discrete integrator or a robot
with tracking error. A finite rollout failure remains in the score denominator.

## Verified demo run (2026-10-05)

The browser completed a Sine run with the defaults above on Windows, Python
3.10.22 and PyTorch 2.0.1+cpu. Training took 759 seconds; training, evaluation
and saving together took 857 seconds on this machine.

| Measurement | Result |
| --- | ---: |
| Held-out velocity MSE | 0.01308643 |
| ID target reached | 12/12 |
| OOD ring target reached | 18/28 |
| OOD trajectory target reached | 8/8 |
| Remaining trajectories | 10 reached the horizon; no integration failures or escapes |
| Maximum sampled ∇Vᵀf+cV | 2.45e-14 |
| Lower-bound / decay violations at 1e-9 tolerance | 0 / 0 |

Run ID: `20261005T152354931690Z-Sine`. The 15 automated tests passed, including
the existing PLYDS regression tests. Both methods trained through the browser;
SNDS reloading, bookmark restoration, plot toggles and downloads were checked.
Reloaded weights reproduced the saved field, Lyapunov grid and audit arrays.
The [screenshot](snds-preview.jpg) shows this result. Numerical run artifacts
remain in the local ignored run directory, rather than being bundled in Git.
