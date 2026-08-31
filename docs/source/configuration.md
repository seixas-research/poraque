# Configuration

A run is defined by `configs/train.yaml`: one top-level key and four
sections. Every key is optional: a file states only what the run does
differently, and anything omitted takes its default.

`configs/train.yaml` is the clean starting point;
`configs/train_complete_and_commented.yaml` is the same run with every
choice explained. This page documents every key either can contain.

## How a value is resolved

Three sources, highest priority first:

1. an explicit command-line flag,
2. the value in the YAML file,
3. the built-in default.

That ordering is what lets one committed config be swept from the shell without
being edited or copied:

```bash
poraque-train --config configs/train.yaml --epochs 500
```

```{note}
Unknown keys are **rejected**, not ignored. A typo such as `learn_rate` would
otherwise be silently dropped and the run would quietly use the default,
producing results that do not match the file that appears to describe them. The
error message names the valid keys for that section.
```

The *resolved* configuration — defaults, file and overrides merged — is written
beside the results as `<json-stem>_config.yaml`. That is the file worth
keeping: it records what actually ran.

(task-section)=
## `task` — what is trained, and what it is called

| Key | Default | Meaning |
| --- | --- | --- |
| `type` | `all` | which map to train: `ext2chg`, `chg2tau`, or `all` for both in sequence |
| `name` | `poraque_models` | stem of every file the run writes |

```yaml
task:
  type: ext2chg
  name: mp_ag_au_pt
```

`name` is the one string every artefact is built from:

| | |
| --- | --- |
| `models/<name>.pfno` | the weights |
| `reports/<name>_report.pdf` | the PDF report |
| `results/plots/<name>/` | the figures |

A `task: all` run trains two models and so writes two reports, which are then
`<name>_ext2chg_report.pdf` and `<name>_chg2tau_report.pdf` — one name cannot
serve both. Cross-validation writes `<name>_kfold_report.pdf`, and a fine-tune
writes its weights to `<name>_finetuned.pfno` rather than over the general
model it specialises.

```{important}
Two runs that share a `name` share their output files, and the second wins.
Change it for anything worth keeping beside the last run — a different
chemical space, resolution, or set of physics weights.
```

The bare string this key used to be is still accepted and means the same as
setting `type` alone:

```yaml
task: ext2chg          # identical to {type: ext2chg}, with the default name
```

## `data` — where the fields come from

| Key | Default | Meaning |
| --- | --- | --- |
| `train_paths` | `null` | list of dataset directories, which may mix layouts; falls back to `root` |
| `root` | `data/vasp` | single dataset directory, used when `train_paths` is `null` |
| `source` | `auto` | layout of each path: `auto`, `vasp`, `bulk` or `prepared` |
| `cache` | `data/cache` | where downsampled copies are written |
| `pattern` | `struct` | prefix identifying subdirectories of a `vasp` path — a prefix, not a glob |
| `code` | `auto` | DFT code, or `auto` to detect it from the files present |
| `resolution` | `32` | longest grid axis after spectral downsampling |
| `potcar_dir` | `null` | POTCAR library, used where the data ships no pseudopotentials |
| `sigma` | `null` | Gaussian pseudo-ion width in Å, where the Gaussian model is reached |
| `gaussian_blur` | `null` | Gaussian blur width in Å applied to the computed potential |
| `blur_method` | `spectral` | `spectral` or `ndimage` |

See {doc}`data/index` for the layouts `source` recognises and for training
across a mixture of them.

### `resolution`

Fields are reduced to this many points along their longest axis before
training. The reduction is a Fourier truncation — the exact band-limited
projection for a plane-wave field — so periodicity is preserved and the
electron count is unchanged to machine precision.

Cost scales as the cube: raising 32 → 64 is roughly eight times the memory and
time. Grid *shapes* are reduced in proportion, so a 120×128×128 calculation
becomes 30×32×32 rather than being forced square.

```{note}
There is no key for supplying an external potential. `EXTCAR` is **always**
computed by {class}`poraque.fields.ExternalPotential` from the `POTCAR` tables,
which reproduces a reference potential to a relative error of order $10^{-5}$.
Any `EXTCAR` present in a source directory is ignored — the training input must
be exactly what the pipeline produces at inference time, when no such file
exists. Standard VASP does not write one anyway.
```

(potcar-dir-and-the-gaussian-fallback)=
### `potcar_dir` and the Gaussian fallback

The external potential needs pseudopotentials, and not every dataset ships
them. A Materials Project download has a structure and a density and nothing
else; archived local runs often have their `POTCAR`s stripped for licensing.
`potcar_dir` points at a **library** — one subdirectory per species, as VASP
distributes them:

```yaml
data:
  potcar_dir: /opt/vasp/potpaw_PBE       # <dir>/Ag/POTCAR, <dir>/Pt/POTCAR, …
```

`.gz` and `.Z` entries are read directly, and a flat `POTCAR.Ag` / `Ag.POTCAR`
layout is recognised too. A run that has its own `POTCAR` ignores the library
entirely — it is a fallback, not an override.

Which construction results is the **most consequential single property of a
training set**, because the two are different physical quantities rather than
different approximations to one:

| | $V_\mathrm{ext}$ | residual vs. a reference `EXTCAR` |
| --- | --- | --- |
| pseudopotentials available | tabulated local pseudopotential (VASP's `POTION`) | $2\times10^{-5}$ relative $L_2$ |
| none available | Gaussian pseudo-ion model, $f_s(G)=e^{-G^2\sigma^2/2}$ | order $10^{-1}$ |

Measured on the Ag–Pt–Pt Materials Project set, the two constructions differ
from each other by **0.38 relative $L_2$**.

Without the library the map learned is *model potential* $\to$ *DFT density*.
That is well-posed and self-consistent — inference builds the very same
potential — but it is not VASP's $V_\mathrm{ext}$, and a model trained that way
is not comparable with one trained on tabulated potentials. It also means a
mixed dataset spanning both constructions is training on two quantities under
one name; {class}`~poraque.data.dataset.MixedFieldDataset` warns when that
happens, and setting `potcar_dir` is what makes the warning go away *correctly*.

```{warning}
Point at the library that generated the data. The Materials Project uses the
VASP PBE set (`PAW_PBE`); a `PAW_LDA` library would build a potential the
densities were never computed from — worse than the Gaussian fallback, because
it looks exact.
```

Missing or unreadable entries are **not fatal**. Each such species warns once
and falls back to the Gaussian model for the structures that contain it, so a
library covering four elements of five still buys the exact potential for
everything that does not involve the fifth. The run log names the construction
per source, and the cache directory records it (`res32_potcar`), so a tabulated
cache and a Gaussian one can never be confused for each other.

`sigma` sets the Gaussian width and is reached only where the Gaussian model
is. `null` derives it per species from the pseudopotential core radius where
one is known and uses 0.5 Å otherwise — an MP download supplies no `RCORE`, so
it is 0.5 Å there. Set it explicitly if a reference `EXTCAR` lets you fit it.

### `gaussian_blur` and `blur_method`

The blur is applied *on top of* the tabulated pseudopotential, never in place
of it.

```{warning}
`ndimage` uses `scipy.ndimage.gaussian_filter`, which blurs along *grid* axes
and is therefore anisotropic in Cartesian space unless the cell is orthogonal —
on a face-centred cubic cell the two methods differ by 2–6%. `spectral`
multiplies by $e^{-G^2\sigma^2/2}$ using the true reciprocal metric and stays
isotropic on any cell.
```

Measured at σ = 0.15 Å and `resolution: 32`, with one structure held out and
everything else fixed, blurring changed the held-out error by 0.7% and the
training time by 0.4% — neither meaningful, since the grid already band-limits
the field. Leave it off unless working at higher resolution.

The cache directory name encodes these choices (`res32_blur0.15spec`), so
changing them cannot silently reuse a cache built with different settings.

## `model` — the operator's shape

| Key | Default | Meaning |
| --- | --- | --- |
| `width` | `16` | channel width of the Fourier layers |
| `modes` | `8` | retained Fourier modes per axis |
| `n_layers` | `3` | number of Fourier layers |
| `projection_channels` | `64` | hidden width of the output projection |
| `activation` | `silu` | `silu`, `gelu`, `relu`, `tanh`, or `kan` |
| `kan_setup` | `null` | the KAN variant and its hyperparameters; read only when `activation: kan` |
| `use_coordinates` | `true` | append three fractional-coordinate input channels |
| `cell_conditioning` | `true` | condition every layer on lattice descriptors (FiLM) |
| `embedding_dim` | `32` | width of that cell embedding |
| `mode_selection` | `fixed` | `fixed` mode index, or `physical` at constant $G_\mathrm{max}$ |
| `g_max` | `null` | cutoff wavevector in Å⁻¹; required by `mode_selection: physical` |
| `pauli_residual` | `false` | structural $\tau\ge\tau_\mathrm{vW}$ for `chg2tau` |
| `pauli_scale` | `null` | initial Pauli scale in eV/Å³; `null` fits it from the training split |
| `learn_pauli_scale` | `true` | optimise that scale alongside the backbone |

### `width`, `modes`, `n_layers`

These three set the capacity. The parameter count is dominated by the spectral
weights — $4w^2m^3$ complex numbers per layer for width $w$ and $m$ modes — so
`modes` is by far the expensive knob: doubling it multiplies the spectral
parameters by eight.

`modes` is a *capacity limit*, not a requirement. On a grid too coarse to
supply that many modes, fewer are used automatically, so one configuration
serves materials whose grids differ in size.

### `activation` and `kan_setup`

`silu`, `gelu`, `relu` and `tanh` are stateless: one fixed function, no
parameters, nothing to learn. `kan` is the Kolmogorov-Arnold family, where
**each channel learns its own** function, applied elementwise to every voxel of
that channel. Which one, and how it is parameterised, comes from `kan_setup`:

```yaml
model:
  activation: kan
  kan_setup:
    variant: cheby            # bspline | cheby | rbf | rational
    degree: 6                 # cheby only
```

The block is read **only** when `activation: kan`; given beside a stateless
activation it warns rather than being silently ignored, and an unknown key
inside it raises. Omitting `variant` gives `bspline`, the original KAN paper's
own parameterisation.

Six of the seven hyperparameters are read by one variant each — `grid_size`,
`spline_order` and `grid_range` by `bspline` (and the first and third by `rbf`,
which reuses the same fixed-grid design), `degree` by `cheby`,
`rational_num_degree` and `rational_den_degree` by `rational`. The seventh,
`use_base`, applies to all four: `true` keeps each channel's $w_c\,\mathrm{silu}(x)$
base term, so every variant starts close to `silu` and only departs from it as
training moves the learned coefficients; `false` gives a "pure" KAN with no
fixed nonlinearity mixed in at all.

Cost at `width: 16` on CPU, relative to a stateless activation: `cheby` 1.36×,
`rational` 1.54×, `rbf` 2.00×, `bspline` 5.50×. Measure on your own hardware
before committing to a long run.

### `use_coordinates` and `cell_conditioning`

`use_coordinates` appends the three fractional coordinates as extra input
channels, giving the network a positional reference the field alone does not
carry.

`cell_conditioning` feeds lattice descriptors through an embedding of width
`embedding_dim` and uses it to modulate each layer's activations (FiLM).
Without it the operator sees the field but not the cell it lives in, and cannot
distinguish two materials whose grids agree while their lattice vectors do not.
Keep both on unless deliberately ablating them.

### `mode_selection` and `g_max`

`fixed` keeps the lowest `modes` indices on every material. Because the spacing
of reciprocal-lattice points depends on the cell size, that means a *different
physical band* for each material.

`physical` instead keeps every mode below a fixed wavevector $G_\mathrm{max}$,
retaining $\lfloor G_\mathrm{max} L_i / 2\pi \rfloor$ modes along axis $i$, so
every material contributes the same band of physics. Prefer it when cell sizes
vary widely. It requires `g_max`.

### `pauli_residual` and its scale

For `chg2tau`, the model predicts

$$\tau = \tau_\mathrm{vW}[\rho] + s\,\mathrm{softplus}(f_\theta(\rho)),$$

so the Hoffmann–Ostenhof bound $\tau \ge \tau_\mathrm{vW}$ holds by
construction and the network learns only the Pauli term. Against a matched
baseline it improved every fold — 18.3% in mean relative $L^2$ — while training
17.8% *faster*, because the network no longer has to re-derive
$|\nabla\rho|^2/8\rho$, roughly 31% of $\tau$. Recommended.

```{warning}
The bound is a theorem for all-electron densities, whereas `CHGCAR` and
`TAUCAR` hold pseudo quantities. Check it on new data before enabling the head —
the training log reports any reference points that violate it.
```

## `training` — how it is fitted

| Key | Default | Meaning |
| --- | --- | --- |
| `valid_fraction` | `0.2` | fraction of structures held out for validation; `0` uses every structure |
| `enable_kfold` | `false` | run K-fold cross-validation instead; ignores `valid_fraction` |
| `k_folds` | `5` | number of folds, capped at the structure count |
| `eval_epoch` | `10` | evaluate and log every N epochs |
| `early_stopping` | `100` | stop after N epochs without validation improvement; `0` disables |
| `epochs` | `300` | passes over the training set |
| `batch_size` | `4` | maximum samples per grid-shape bucket |
| `learning_rate` | `0.002` | AdamW step size |
| `weight_decay` | `0.0001` | AdamW weight decay |
| `scheduler` | `cosine` | cosine decay, or `null` for a constant rate |
| `grad_clip` | `1.0` | global gradient-norm clip; `0` disables |
| `seed` | `0` | weight initialisation, batch order and fold shuffling |
| `device` | `auto` | `auto`, `cuda`, `mps` or `cpu` |
| `loss` | `relative_l2` | `relative_l2` or `sobolev` |
| `sobolev_weight` | `0.1` | gradient-term weight when `loss: sobolev` |
| `physics` | all `0.0` | physics-informed loss weights |

### `valid_fraction`, `enable_kfold`, `k_folds`

There is **one** training protocol and **one** variation on it.

By default a run trains a single model per task, holding back `valid_fraction`
of the structures for validation. That is all there is to it — no mode to
select, no structures to name.

`enable_kfold` swaps that for K-fold cross-validation: each fold trains a fresh
model on *K*−1 groups and scores it on the group held out. It answers a
different question — whether the architecture generalises — and produces *K*
models rather than one to deploy. `valid_fraction` is ignored, since the folds
define the splits. Setting `k_folds` to the number of structures gives
leave-one-out.

```{tip}
Splitting is always at the *structure* level: whole materials move together. A
voxel-level split would place the same crystal on both sides, and since
neighbouring voxels are strongly correlated the score would look excellent
while saying nothing about transfer to a new material.
```

```{important}
`valid_fraction` defaults to **0.2**, so an ordinary run reports a genuine
held-out score and `early_stopping` is active. The trade-off is that the model
has then seen only 80% of the data. For the final deployable artefact set
`valid_fraction: 0` — accepting that its own metrics become a training fit —
and quote a separate `--kfold` run for the generalisation number.
```

```{warning}
With `valid_fraction: 0` the reported metrics are **training fit**, carrying no
generalisation claim. On the reference dataset they are about four times better
than the cross-validated score. Quote the cross-validated numbers.
```

### `eval_epoch`

Evaluate and log every this many epochs. Validation is computed *only* on those
epochs, so raising it on a large validation set is a genuine speed-up rather
than only a quieter log. The final epoch always reports, so a run never ends
without a current number.

```text
  progress (every 10 epochs):
    train loss: mean PhysicsInformedLoss per batch   |   val rel L2: held-out error, physical units
          epoch     train loss     val rel L2
    -----------------------------------------
         10/300        0.31745        0.34118
         20/300        0.18902        0.21447  *
```

### `early_stopping`

Stop after this many epochs without an improvement in the **validation** error,
and restore the best weights seen:

```text
         30/300        0.03173        0.03659  *
         ...
         38/300        0.02249        0.06568
    stopped early at epoch 38: no improvement in 8 epochs (best 0.03659 at epoch 30)

  trained 38/300 epochs in 12.0 s   loss 0.8599 -> 0.0225
```

```{warning}
It needs a validation split (`valid_fraction > 0`). With nothing held
out there is only the training loss, which falls monotonically by construction
and so can never signal that training should stop — asking for early stopping
anyway **warns** rather than silently doing nothing and leaving you believing
the run was protected.

The shipped default holds a fifth of the structures out (`valid_fraction:
0.2`), so early stopping is active out of the box. It goes inactive only if you
set `valid_fraction: 0` to train on every structure, and the run says so rather
than appearing to be protected.
```

Patience is counted in *epochs* but checked only on the epochs where validation
is computed, so a value below `eval_epoch` behaves like `eval_epoch`.

The best weights are restored on exit, so the operator you get is the best one
*measured* rather than merely the last one reached — stopping partway down a
degrading curve would otherwise hand back the degraded model. `history` records
`best_epoch`, `best_error` and `stopped_early`.

The `*` marks an epoch that improved on the best validation score. Only epochs
on which validation was actually measured can do so — a checkpoint is written
against a measured score, never an assumed one.

### `batch_size`

Capped *per grid-shape bucket*. Materials whose grids differ in shape cannot be
stacked into one tensor, so they are grouped by shape and batches drawn within
a group. A bucket holding a single material yields batches of one however large
this is set. The training log prints the buckets it found.

### `loss` and `sobolev_weight`

`relative_l2` normalises the error per sample, so materials whose fields differ
by orders of magnitude contribute equally and the reported value reads directly
as a fraction.

`sobolev` adds a relative gradient term weighted by `sobolev_weight`. Worth
enabling when the derivative matters — $\tau_\mathrm{vW}$ depends on
$\nabla\rho$, so gradient noise is amplified in low-density regions.

**The validation column follows the objective.** With `loss: sobolev` the
progress table reports `val rel H1` rather than `val rel L2`, and the number
behind it is the objective's own data term evaluated on the held-out set —
`rel L2` plus `sobolev_weight` times the relative $L^2$ of the gradient:

```text
    train loss: mean PhysicsInformedLoss per batch   |   val rel H1: held-out error, physical units
    val rel H1 = rel L2 + 0.1 x the relative L2 of the gradient, matching the
    objective; the final per-structure table below reports plain rel L2
          epoch     train loss     val rel H1
```

That is not only a label. Early stopping and the checkpoint are decided on this
number, and while it read `val rel L2` a Sobolev run was selecting the model
that minimised a functional it was not training on. An $H^1$ error is a larger
number than the $L^2$ of the same prediction, so the two columns are not
comparable — which is why the **per-structure table at the end of a run reports
relative $L^2$ whatever the objective was**. That is the number to quote, and
the one that lets two runs with different losses be put side by side.

### `physics`

| Weight | Constraint it penalises the violation of | Shipped |
| --- | --- | --- |
| `electron_count_weight` | $\int\rho\,d\mathbf{r} = N$ (`ext2chg`) | `0.1` |
| `positivity_weight` | non-negativity of the predicted field | `0.0` |
| `von_weizsacker_weight` | $\tau \ge \tau_\mathrm{vW}$ (`chg2tau`) | `0.0` |
| `euler_lagrange_weight` | orbital-free stationarity residual (`ext2chg`) | `0.0` |

All four default to zero in the code, so the objective is the plain supervised
baseline until one is enabled deliberately. Each term is dimensionless, so a
single weight can serve a heterogeneous dataset.

#### Charge conservation

`electron_count_weight` is the one term the shipped configs switch on, at
`0.1`:

$$
\mathcal{L}_{N} = \left\langle \left(
\frac{\int\hat\rho\,d^3r - \int\rho\,d^3r}{\int\rho\,d^3r}
\right)^{2} \right\rangle ,
$$

the squared relative error between the integral of the **predicted** density
and the integral of the **reference** one — the valence electron count. Two
things make it the first constraint to reach for:

*It needs no labels.* The reference density is in every batch, so $N$ is its
own integral. That is what lets it work on an archive that publishes $\rho$ and
nothing else.

*It fixes what the data term cannot.* A relative $L^2$ is indifferent to a
percent of charge spread thinly through the interstitial region, and a total
energy is not. Across a chemical space where $N$ runs from ten to ninety
electrons per cell, nothing else pins the scale per material.

At `0.1` the constraint sits an order of magnitude below the data term, which
is the intended balance: the physics guides, the data decides. The training log
reports one number, `train loss`: the **total** objective the optimiser stepped
on, data fidelity plus every weighted constraint. That total is the only
quantity comparable between two runs — the individual terms are reported by the
loss unweighted, so they do not sum to it.

```{warning}
Introduce them one at a time against a measured baseline, and only once the
data term has converged — a physics residual evaluated at random initialisation
is noise. A badly scaled constraint degrades accuracy while looking principled.
```

Where a constraint can be imposed structurally, prefer that: `pauli_residual`
enforces $\tau\ge\tau_\mathrm{vW}$ exactly, whereas `von_weizsacker_weight`
only discourages violating it.

## `output` — what is written

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | `models` | parent of the run folder; `null` disables **all** output |
| `checkpoint` | `true` | write `<name>.pfno` |
| `write_log` | `true` | write `log/`: the log, the metrics JSON, the resolved config |
| `plot_figures` | `true` | render `plots/` |
| `write_pdf_report` | `true` | typeset `report/` |
| `log`, `json` | `null` | override the two log paths; `null` derives them from `task.name` |
| `plot_format` | `png` | `png`, `pdf` or `svg` |
| `dpi` | `200` | raster resolution for saved figures |
| `save_raw_plot_data` | `false` | write the numbers behind each figure beside it |

Everything a run writes lives under `<root>/<name>/` — the weights, `log/`,
`plots/` and `report/` together. `root` is the only path setting and
[`task.name`](task-section) is the only thing that distinguishes two runs, so
"delete this experiment" and "send me that model" are one directory each rather
than a collection of fragments by filename.

### `save_raw_plot_data`

A figure is an argument someone will later want to make differently — in a
journal's colours, at a journal's size, with two runs on one axis. Re-running
the model to recover the numbers is the expensive way to do that, so with this
on each figure method writes what it drew, beside what it drew:

```text
plots/ext2chg_loss_curves.png       plots/ext2chg_loss_curves.csv
plots/ext2chg_parity.png            plots/ext2chg_parity.csv
                                    plots/ext2chg_parity_bin_edges.csv
plots/ext2chg_s000_field_slice.png  plots/ext2chg_s000_field_slice.npz
```

The sidecar shares the **stem** of its figure, so the pairing is a property of
the directory listing rather than of a convention to remember. Tabular figures
get a CSV; the slice figure gets a compressed `.npz`, because what it draws are
three 2D arrays and a CSV of a 108×108 grid is neither smaller nor easier to
read back.

- **The loss curve** is one row per epoch: `epoch`, `train_loss`, and the
  validation column *named for the norm it holds* (`val_rel_l2`, or
  `val_rel_h1` for a `loss: sobolev` run). Epochs on which validation was not
  measured leave that cell **empty** rather than interpolated — filling them
  would put points in the file the run never measured. There is no physics
  column: `train_loss` is the total the optimiser stepped on, and the per-term
  breakdown is not recorded.
- **The parity plot** is one row per *occupied* bin: `split`, `reference`,
  `prediction`, `count`, `density`. Long format, because that pivots straight
  back into a `pcolormesh`; occupied bins only, because a 200-bin grid is
  40 000 cells of which a few thousand carry anything. The bin edges go in
  their own file, since a log axis makes them unrecoverable from the centres.
- **The slice figure** stores the three panels as arrays — `reference`,
  `prediction`, `error` — with the colour limits the figure used. The error is
  *stored*, not left to be recomputed: a reader who reconstructs it and gets
  something else then knows the file is inconsistent, rather than trusting
  their own subtraction.

It needs `plot_figures`. The data is written by the figure methods as they
draw, which is what guarantees the file and the image show the same numbers
rather than two independent computations of them.

The resolved configuration is written next to `json` with the suffix
`_config.yaml`. PDF reports are assembled in a temporary directory that is
removed afterwards, so no `.tex`, `.aux` or `.log` files are left behind.

## Worked examples

Train the deployable models:

```yaml
task:     {type: all, name: poraque_models}
model:    {pauli_residual: true}
training: {valid_fraction: 0, epochs: 200}
```

Measure generalisation:

```yaml
task:     {type: all, name: poraque_models_cv}
training: {enable_kfold: true, k_folds: 5}
```

Two runs over the same data with different physics, kept side by side:

```yaml
task:     {type: ext2chg, name: agaupt_baseline}
training: {physics: {electron_count_weight: 0.0}}
```

```yaml
task:     {type: ext2chg, name: agaupt_charge_conserving}
training: {physics: {electron_count_weight: 0.1}}
```

A fast smoke test:

```yaml
data:     {resolution: 20}
model:    {width: 8, modes: 4, n_layers: 2, projection_channels: 16}
training: {epochs: 10}
```

Higher fidelity, if the hardware allows:

```yaml
data:     {resolution: 64}
model:    {width: 32, modes: 12, n_layers: 4}
training: {epochs: 400, batch_size: 2}
```
