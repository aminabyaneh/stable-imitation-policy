# PLYDS / SNDS motion notebook

A local CPU trainer and browser demo for all 30 LASA handwriting motions.
Choose **PLYDS** (SCS polynomial fitting) or **SNDS · revised theorem** (neural
policy with a smooth, gradient-anchored Lyapunov function), then the motion
and 1–7 demonstrations. PLYDS exposes polynomial degrees; SNDS exposes epochs,
seed, learning rate, positive decay rate and quadratic margin. Click
**Train model** to fit, integrate, and save a new experiment.

The plot includes a vector field, a blue Lyapunov heatmap with equal-value
contours, demonstrations, and independently switchable ID/OOD rollouts.
Metrics show held-out velocity error, target-reaching rates, and elapsed time.
Saved experiments can be reopened without retraining. The page URL retains the
selected run for bookmarking on the same machine.

## Quick start

Use **Python 3.10**; the tested interpreter is 3.10.22. The demo has its own
environment, independent of the repository's historical Conda environment.

From the repository root, on macOS/Linux:

```sh
cd demo
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m plyds_lab serve --port 8765
```

On Windows PowerShell:

```powershell
cd demo
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m plyds_lab serve --port 8765
```

Open **http://127.0.0.1:8765/**. Keep the terminal running; Ctrl+C stops the
server. Use `--port 8766` if another copy is already running.
The server binds only to loopback. The first page has no saved models until
you train one; LASA data comes with `pyLasaDataset`. No MOSEK, GPU, Node.js,
or thesis checkout is required.

`requirements.txt` pins direct dependencies; `requirements-lock.txt` also
pins the tested transitive dependencies. CVXPY may install other solver
packages as dependencies, but this trainer explicitly selects SCS.

## Reproducible evaluation

The current protocol is **lasa-rollouts-v4**, with 48 full rollouts per motion:

| Group | Starts | Definition |
| --- | ---: | --- |
| ID | 12 | Saved along-trajectory anchors, perturbed by 1–2% position offsets |
| OOD rings | 28 | Existing starts at radius 6.25% around demonstration starts |
| OOD along trajectories | 8 | Spatially spread trajectory anchors, perturbed by 5–6.25% |

Percentages use the diagonal of the bounding box of the normalized motion.
Directions and offset magnitudes are seeded per motion, independent of policy,
degree, solver, or training split. The combined OOD score covers all 36 OOD
starts, with both subgroup scores retained. These are geometric test groups,
not estimates of statistical support.

Each demonstration is endpoint-centred; positions and velocity labels are
independently divided by their maximum vector norms. Rollout time is therefore
normalized, not physical robot time. RK45 runs to target radius 0.01 or horizon
60, with escape radius 8 and an RHS budget of 20,000. Failures stay in the score
denominator. Arrows show direction, not speed. Heatmap values come from the
actual saved Lyapunov polynomial or neural potential; the labeled asinh colour scale preserves
negative numerical values rather than clipping them.

Motion-specific JSON plans live in `evaluation_protocols/`. Existing plans are
included; other motions are generated deterministically on first use. Keep
those files for comparisons with future baselines. Historical protocol versions
are preserved. New definitions require a new version, never edited old starts.

For another baseline in the same normalized coordinates:

```python
from plyds_lab.evaluation import evaluate_policy
evaluation = evaluate_policy("Sine", lambda points: baseline.predict(points))
```

The callable accepts and returns an N-by-2 array.

## Command line and tests

Run from `demo/` with the environment activated (or use its Python executable):

```sh
python -m plyds_lab train --motion Sine --policy-degree 4 --demos 5
python -m plyds_lab train --method snds --motion Sine --epochs 3000 --seed 0 --demos 5
python -m plyds_lab train --motion Multi_Models_1 --policy-degree 6 --lyapunov-degree 6 --demos 5 --cover-starts
python -m plyds_lab batch
python -m plyds_lab complexity --cover-starts
python -m unittest plyds_lab.test_lab plyds_lab.test_neural -v
```

`--cover-starts` selects demonstrations by deterministic farthest-start
coverage. PLYDS defaults to the first demonstrations; SNDS defaults to coverage
selection. SNDS trains on CPU in float64; the UI shows live epoch progress.
Hyperparameter selection
on held-out demonstrations is validation, not an untouched final test.

Each run gets a new directory under `runs/plyds-lab/`, containing `model.npz`,
full adaptive `rollouts.npz`, `result.json`, `protocol.json`, and PNG/PDF vector
field figures. SNDS instead saves `model.pt`, plus `audit.npz` and
`training-data.npz`. Both methods share this archive and evaluation protocol.
The JSON also stores the Lyapunov grid, configurations, package
versions, numerical diagnostics, and decimated paths for display. Runs and
virtual environments are Git-ignored; copy results explicitly when sharing.

## Relationship to the updated theorem

**PLYDS constraints follow the corrected scalar SOS theorem. Returned
floating-point solutions are not rigorously certified.**
See [the theorem-to-code comparison](THEOREM_ALIGNMENT.md) for the matching
conditions and the differences in polynomial bases, objective, and search.
Every result explicitly records `certificate_verified: false`.

SNDS follows the revised smooth, strongly convex, gradient-anchored construction
with exact strict-decay projection. See [SNDS theorem alignment](SNDS_THEOREM_ALIGNMENT.md)
for the equations, architecture, numerical audit, and limits of the guarantee.

The demo is self-contained in `plyds_lab/` and does not call the original
`src/` or `exp/` solvers. Those historical implementations have not been
converted to the updated theorem by this addition.
