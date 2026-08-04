# Neural operators

## Two models, one system

| Task | Map | Meaning |
| --- | --- | --- |
| `ext2chg` | $V_\mathrm{ext}\mapsto\rho$ | the Hohenberg–Kohn map |
| `chg2tau` | $\rho\mapsto\tau$ | the kinetic energy density functional |

The first learns an object guaranteed to exist by theorem. The second learns
the only term of the DFT total energy with no accurate explicit density
functional — it *is* an orbital-free kinetic functional, since
$T_s[\rho]=\int\tau[\rho]\,\mathrm d^3r$.

## Why a Fourier neural operator

A conventional network maps fixed-size vectors, which is fatal here: every
material has a different grid. A **neural operator** learns a map between
function spaces and is discretised only for evaluation, so one set of weights
serves every grid.

One Fourier layer is

$$
v_{\ell+1}(\mathbf r) = \sigma\Big( W v_\ell(\mathbf r) + b
  + \mathcal F^{-1}\big[R_\ell(\mathbf G)\cdot\mathcal F[v_\ell](\mathbf G)\big](\mathbf r)\Big),
$$

with $R_\ell$ a learned complex multiplier truncated to the lowest modes. Three
properties match the physics:

* **Periodicity is exact, not learned.** The FFT imposes Born–von Kármán
  boundary conditions, which a crystal already obeys.
* **The coupling is global in one layer.** The Hohenberg–Kohn map is non-local
  — the Hartree kernel $4\pi e^2/G^2$ is *diagonal* in exactly the basis the
  layer works in.
* **Derivatives are free and exact.** $\nabla\to i\mathbf G$ and
  $\nabla^2\to-G^2$ hold exactly for band-limited fields, and the layer is
  already computing FFTs.

## Grids that differ between materials

Three mechanisms deliver grid independence:

1. **Dynamic mode truncation.** Weights are allocated for `modes` coefficients
   per axis; each forward pass uses `min(modes, available)` and leaves the rest
   untouched.
2. **Resolution-invariant normalisation.** Both transforms use
   `norm="forward"`, so coefficients approximate *continuous* Fourier-series
   coefficients and their magnitude does not drift with $N$. Under the default
   convention a $120^3$ field would enter a layer with amplitudes ~125× those
   of a $24^3$ field.
3. **Shape-bucketed batching.** Samples of different shape cannot be stacked.
   Rather than pad — which wastes compute and injects fictitious vacuum into
   the FFT — materials are grouped by shape and batches drawn within a group.

`GroupNorm` is used throughout: batch statistics are meaningless when every
sample has a different spatial extent.

## Universal training

**One model is trained on the combined data of all structures.** Batches are
drawn across materials, so a gradient step generally mixes several.

```bash
python scripts/run_train.py --config configs/train_config.yaml   # mode: universal
```

This writes a single unified checkpoint, `models/poraque_models.pth`,
holding both operators under the keys `ext2chg` and `chg2tau`.

```{warning}
With no structure held out, the reported metrics are **training fit**. They
show the model can represent the data; they are not a generalisation estimate.
Use `--mode leave_one_out` for that, or name structures in
`training.holdout`.
```

## Constraints enforced by construction

A soft penalty trades accuracy against constraint satisfaction and achieves
neither. Where a constraint can be built into the parameterisation it should
be — it then holds for every weight configuration, with nothing to tune.

For `chg2tau`, the exact decomposition $\tau=\tau_\mathrm{vW}+\tau_\mathrm{P}$
with $\tau_\mathrm{P}\ge 0$ (Hoffmann-Ostenhof) gives

$$
\tau_\theta = \tau_\mathrm{vW}[\rho] + s\,\mathrm{softplus}\big(f_\theta(\rho)\big).
$$

The bound becomes structural, and the network stops re-deriving
$|\nabla\rho|^2/8\rho$ — roughly 31 % of $\tau$ on the reference data, now
supplied analytically. Measured against a matched unconstrained baseline this
improves every fold and trains ~18 % *faster*.

```{note}
The bound is a theorem for all-electron densities; `CHGCAR` and `TAUCAR` are
pseudo quantities, so verify it with
{py:func}`~poraque.ml.heads.pauli_bound_violation` before enabling the head on
a new dataset.
```

## Hardware

`resolve_device` prefers CUDA, then Apple Metal, then CPU, and falls back with
a warning rather than raising. Metal required three workarounds, all covered by
regression tests:

* **no float64, no `linalg.det`** — the cell metric is evaluated on the host in
  double precision and only the $3\times3$ result moved to the device;
* **no complex `einsum`** — the spectral contraction is expressed in real
  arithmetic, which is numerically identical;
* **strided complex views are silently wrong** — operands are made contiguous
  before use, without which results are 40–90 % off with no error raised.

| Grid | CPU | MPS | speed-up |
| --- | --- | --- | --- |
| 32³ | 42.0 ms | 19.2 ms | 2.19× |
| 48³ | 124.3 ms | 38.3 ms | 3.24× |
| 64³ | 262.2 ms | 45.9 ms | 5.71× |

## Reporting

Every run writes loss curves, field cross-sections and parity plots, and
assembles them with the metrics into a typeset PDF under `reports/`. The
report is built in a temporary directory and only the PDF is moved out, so no
`.tex` or auxiliary files are left behind.
