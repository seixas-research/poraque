# Theory

## Density-functional theory

The Hohenberg–Kohn theorems replace the $3N$-dimensional many-body
wavefunction by the electron density. The first states that the ground-state
density determines the external potential up to a constant, so for a fixed
electron number

$$
V_\mathrm{ext}(\mathbf r) \;\longleftrightarrow\; \rho_0(\mathbf r).
$$

The second gives a universal functional $F[\rho]$ such that
$E[\rho]=F[\rho]+\int V_\mathrm{ext}\rho\,\mathrm d^3r$ is minimised by the
ground-state density.

This is the formal footing of the `ext2chg` model: its target is not a fitted
correlation but a single-valued function whose existence and uniqueness is a
theorem.

```{note}
The map is conditional on the electron number $N$. In a dataset where $N$
follows from the geometry a network can infer it; for charged systems or mixed
chemistry, $N$ must become an explicit input.
```

## Kohn–Sham theory

Since $F[\rho]$ is unknown and its kinetic part is badly approximated by
explicit density functionals, Kohn and Sham introduce non-interacting
electrons with the same density:

$$
\left(-\tfrac12\nabla^2 + v_\mathrm{s}[\rho]\right)\psi_i = \varepsilon_i\psi_i,
\qquad
v_\mathrm{s} = V_\mathrm{ext} + v_H[\rho] + v_{xc}[\rho],
$$

solved self-consistently. Two features shape the architecture: the map is
**non-local** (the Hartree term is a $1/|\mathbf r-\mathbf r'|$ convolution) and
**self-consistent** (the density is a fixed point).

## Kinetic energy density

The quantity stored in `TAUCAR` is the positive-definite form

$$
\tau(\mathbf r) = \tfrac12\sum_i^\mathrm{occ} f_i |\nabla\psi_i(\mathbf r)|^2 .
$$

An alternative using the Laplacian integrates to the same $T_s$ but differs
pointwise by $\tfrac14\nabla^2\rho$ and is not sign-definite. The distinction
matters: the constraints below apply to the positive-definite form.

### Exact properties

| Property | Statement |
| --- | --- |
| Non-negativity | $\tau \ge 0$ |
| Hoffmann-Ostenhof bound | $\tau \ge \tau_\mathrm{vW} = \lvert\nabla\rho\rvert^2/8\rho$ |
| Uniform-gas limit | $\tau \to C_\mathrm{TF}\rho^{5/3}$ as $\nabla\rho\to0$ |
| Coordinate scaling | $T_s[\rho_\lambda] = \lambda^2 T_s[\rho]$ |

The decomposition $\tau = \tau_\mathrm{vW} + \tau_\mathrm{P}$ with
$\tau_\mathrm{P}\ge0$ defines the **Pauli term** — the part arising purely from
Fermi statistics, and what every orbital-free kinetic functional is really
trying to approximate.

## Orbital-free DFT

Removing the orbitals entirely and minimising $E[\rho]$ under
$\int\rho=N$ gives the Euler–Lagrange equation

$$
\frac{\delta T_s}{\delta\rho}(\mathbf r) + V_\mathrm{ext}(\mathbf r)
  + v_H[\rho](\mathbf r) + v_{xc}[\rho](\mathbf r) = \mu ,
$$

with $\mu$ constant in space. One scalar equation replaces the $N$ coupled
Kohn–Sham equations, and the cost becomes near-linear in system size.

Everything in it is available: $V_\mathrm{ext}$ is Model 1's input, $\rho$ its
output, $v_H$ is computed exactly by FFT, and $\delta T_s/\delta\rho$ follows
from autograd through Model 2. Since $\mu$ is a constant, subtracting the cell
average eliminates it, leaving a residual that must vanish for the true
ground-state density — a **label-free** training signal.

```{warning}
The residual is a *consistency* condition, not a *correctness* one. With both
models free to adapt, the pair can drive it to zero without either being
right. Joint training is safe only with the data terms retained and dominant.
```

## Why the classical functionals fail

| Functional | Regime | Failure |
| --- | --- | --- |
| Thomas–Fermi | uniform gas | no shell structure; does not bind molecules |
| von Weizsäcker | one orbital | exact only for a single nodeless orbital |
| TF + $\lambda$vW | gradient expansion | not reliable across chemistry |

On a $d$-band metal Thomas–Fermi attains a *negative* coefficient of
determination — as a pointwise predictor of $\tau$ it is worse than the
constant mean. That is the gap a learned functional is meant to close.

## Electrostatics on a periodic grid

Both $V_\mathrm{ext}$ and $v_H$ are evaluated in reciprocal space, where the
Coulomb kernel is diagonal:

$$
v_H(\mathbf G) = \frac{4\pi e^2 \rho(\mathbf G)}{G^2},
\qquad v_H(\mathbf 0) \equiv 0 .
$$

The $\mathbf G = 0$ component diverges for any charged distribution. In a
neutral periodic crystal that divergence cancels between the electron–electron,
electron–ion and ion–ion terms; the standard convention is to zero each term's
$\mathbf G=0$ component individually, which adds a uniform neutralising
background and leaves potentials defined up to a constant.
