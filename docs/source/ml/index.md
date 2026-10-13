# Neural operators

## Four maps, one system

| Task | Map | Meaning |
| --- | --- | --- |
| `ext2pseudo` | $V_\mathrm{ext}\mapsto\rho$ | the Hohenberg–Kohn map |
| `pseudo2tau` | $\rho\mapsto\tau$ | the kinetic energy density functional |
| `ext2chg` | $V_\mathrm{ext}\mapsto(\tilde\rho,\rho^a_{ij})$ | the pseudo-density **and** every atom's PAW augmentation occupancies |
| `chg2tau` | $(\tilde\rho,\rho^a_{ij})\mapsto\tau$ | the kinetic energy density from both |

The first learns an object guaranteed to exist by theorem. The second learns
the only term of the DFT total energy with no accurate explicit density
functional — it *is* an orbital-free kinetic functional, since
$T_s[\rho]=\int\tau[\rho]\,\mathrm d^3r$.

The other two sit beside that chain rather than in it, and exist because a
`CHGCAR` is not only a grid. The one-center PAW terms inside the augmentation
spheres have no representation on the plane-wave mesh, and together with the
pseudo-density they are what an all-electron density is reconstructed from.
`ext2chg` predicts them — an FNO for the grid, a readout at each atom over one
band of $|\mathbf G|$, and an equivariant head per element — and `chg2tau`
reads them, written into extra input channels by an equivariant
{class}`~poraque.ml.paw.SiteWriter` beside the density. What the one predicts is
exactly what the other reads, which is the route this project takes to the
total energy. In a configuration they are the two sections, `ext2chg:` and
`chg2tau:`, and the occupancies are a block inside each: `occupancies: false`
trains the grid alone, which is the chain's pair of maps. The
{ref}`configuration reference <paw>` describes both operators and their keys.

`ext2pseudo` and `ext2chg` can also be handed the structure explicitly, as
superpositions of isolated atoms beside $V_\mathrm{ext}$
({ref}`ext2chg.structure_input <structure-input>`): scalar fields, so the
operator stays equivariant, and functions of the geometry alone, so inference
rebuilds them from the structure as it rebuilds the potential.

## Why a Fourier neural operator

A conventional network maps fixed-size vectors, which is fatal here: every
material has a different grid. A **neural operator** learns a map between
function spaces and is discretized only for evaluation, so one set of weights
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
2. **Resolution-invariant normalization.** Both transforms use
   `norm="forward"`, so coefficients approximate *continuous* Fourier-series
   coefficients and their magnitude does not drift with $N$. Under the default
   convention a $120^3$ field would enter a layer with amplitudes ~125× those
   of a $24^3$ field.
3. **Shape-bucketed batching.** Samples of different shape cannot be stacked.
   Rather than pad — which wastes compute and injects fictitious vacuum into
   the FFT — materials are grouped by shape and batches drawn within a group.
   One optimizer step per batch then weighs a structure by the inverse of its
   bucket's batch size; `training.samples_per_step` accumulates gradients
   across buckets so that every structure weighs the same.

`GroupNorm` is used throughout: batch statistics are meaningless when every
sample has a different spatial extent.

## Training

**One model per task is trained across all the structures**, never one per
material. Batches are drawn across materials, so a gradient step generally
mixes several.

```bash
poraque-train --config configs/train_simple.yaml
```

A fifth of the structures is held back for validation
(`training.validation.fraction`), so the reported score is genuinely held out.
The run writes a single unified checkpoint in Poraquê's own **`.poraque`**
format — `models/<run.name>/<run.name>.poraque` — holding both operators, under
the internal task names (`ext2pseudo` and `pseudo2tau`, or `ext2chg` and
`chg2tau` with the occupancies on). A [fine-tuned](../fine_tuning/index.md)
model is written beside it as `<run.name>_finetuned.poraque`, never over the
top of it.

K-fold cross-validation is the only variation on this:

```bash
poraque-train --config configs/train_simple.yaml --kfold 5
```

```{warning}
With `training.validation.fraction: 0` nothing is held out and the reported metrics are
**training fit**. They show the model can represent the data; they are not a
generalization estimate. That setting is for the final artifact, which uses all
the data — quote a `--kfold` run for the number.
```

## Constraints enforced by construction

A soft penalty trades accuracy against constraint satisfaction and achieves
neither. Where a constraint can be built into the parameterization it should
be — it then holds for every weight configuration, with nothing to tune.

For `pseudo2tau`, the exact decomposition $\tau=\tau_\mathrm{vW}+\tau_\mathrm{P}$
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

The parity plot is drawn over **every structure** of each split, training and
validation side by side, each panel titled with how many structures and voxels
it holds and labeled with its relative $L^2$, $R^2$, MAE and RMSE. A held-out
density visibly wider about the identity line is the generalization gap: the
same story the aggregate table tells, read off the figure instead of inferred
from two numbers. Under `--kfold` the one panel is every structure as its own
fold held it out, which no single fold can show.

Every structure cannot mean every voxel. A 97-cell set at $48^3$ is eleven
million pairs and a Materials Project set is orders of magnitude past that, so
the figure never holds them. Each structure contributes a random sample of
10 000 voxels, all of them when its grid is smaller, and the sample is folded
into a histogram the moment it is drawn and then discarded. The histogram is
sparse and fine — 0.005 decades a bin on log axes — and only its occupied bins
are stored, so memory follows the spread of the values rather than the number
of structures; the 200 bins of the figure are summed from it at the end. The
draw is seeded from `run.seed` and the structure's position, so a rerun
draws the same voxels.

The four numbers on each panel are computed from running sums over the sampled
voxels, not read back off the bins, so they are exact for that sample. They pool
the split's voxels into one set, which weights a structure by its share of the
sample — equally, for every grid larger than it — and is therefore not the
table's mean of per-structure errors; read the two together rather than
expecting them to agree.

The panels share their axis limits, their bin edges and one color scale, and
each is normalized to the share of *its own* sampled voxels, because the
training split is several times the validation one and raw counts would paint it
denser for no physical reason. A spin-polarized run draws a second figure for
the magnetization, on linear axes since $m$ changes sign.
