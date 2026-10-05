# Updated theorem and the demo

Checked on **2026-10-05** against the current local thesis working tree,
including its uncommitted revisions: `past/plyds/methodology.tex`, theorem
`theorem:ds_stability_sos` (Global SOS certificate), and the proof and
verification discussion in `past/plyds/supplements.tex`. Exact source hashes
are recorded in [theorem-source.json](theorem-source.json); this is a dated
comparison, not a claim that future edits automatically update the code.

## What matches

The corrected theorem assumes a polynomial vector field with `f(0) = 0`, a
scalar `V(0) = 0`, and strictly positive margins such that both

```text
V(e) - epsilon_V ||e||²
-grad(V(e))ᵀ f(e) - epsilon_D ||e||²
```

are SOS. Exact satisfaction gives forward completeness and global asymptotic
stability of the continuous-time field.

| Theorem condition | Demo implementation |
| --- | --- |
| Target equilibrium | `solver.py` omits the constant policy monomial, imposing `f(0)=0`. Data are centred on the endpoint. |
| One scalar positive, radially unbounded V | Learned mode uses `epsilon_V ||e||² + z_Vᵀ Q z_V`, with a complete nonconstant monomial basis and a PSD variable Q. Default fixed mode uses `V=||e||²`, a special case. |
| Strict full-field decay | `derivative_policy_map` and `derivative_v_map` in `polynomial.py` include both components of `grad(V)ᵀ f`. The solver imposes the identity `grad(V)ᵀ f + epsilon_D ||e||² + z_Dᵀ R z_D = 0` with a PSD variable R. |
| Every coefficient, including zeros | All monomials through `policy_degree + lyapunov_degree - 1` are matched; odd highest-degree terms are constrained to zero. |
| Positive margins | Both margins default to 0.001 and are validated as positive. |
| Bilinear joint search | Learned mode alternates convex SCS certificate and policy subproblems. It does not treat the joint problem as one convex SDP or claim a global optimum. |

## Deliberate differences from the manuscript optimization

- The demo uses direct coefficients in a **complete total-degree policy basis**,
  whereas the manuscript first defines a compact coordinate-power quadratic
  matrix representation and explicitly allows a complete basis as an extension.
  The demo's degree controls mean actual polynomial degree, not basis degree
  alpha or beta. Lyapunov degree is twice its Gram basis degree.
- The derivative Gram basis uses degree `floor((d_f+d_V-1)/2)`, sized to the
  derivative polynomial. The manuscript supplies a larger sufficient basis.
  In an exact PSD representation, higher square degrees cannot cancel, so
  omitting those structurally zero degrees is compatible with the theorem.
- Policy regression uses L2 regularization of direct coefficients, with no
  adjustable L1 term. It is not the same elastic-net objective on the manuscript's
  policy matrices.
- Learned-V search begins with `f=-e`; the certificate step minimizes a convex
  demonstration-descent penalty while enforcing certificate feasibility. It
  normalizes the quadratic trace of V to 2 and bounds `trace(Q)` by 100.
  These choices restrict/search the feasible class; the theorem does not prescribe
  them. The best accepted training objective is retained across rounds.
- The demo implements planar LASA experiments, not arbitrary-dimensional robot
  deployment or a reproduction of all historical paper results.

## What is not established

The numerical acceptance gate allows coefficient/constraint residuals up to
`5e-5` and Gram eigenvalues as low as `-5e-5`. These are practical diagnostics,
**not exact PSD or identity verification**. In particular, a small high-degree
coefficient error can dominate the quadratic margin far from the target.
Some accepted solutions also have small negative Gram eigenvalues.

There is no exact reconstruction, interval proof, or global residual bound
preserving the margins. Consequently every run records
`certificate_verified: false`; neither `optimal` status nor successful ID/OOD
rollouts is a theorem-backed global certificate. Rigorous certificate validation
remains necessary before claiming the theorem holds for a returned model.
The theorem also does not automatically certify RK45 discretization or a robot.

The original repository's `src/` and `exp/` code is preserved separately.
This alignment assessment applies to **`demo/plyds_lab/` and the current local
browser demo**, not automatically to those historical implementations.
