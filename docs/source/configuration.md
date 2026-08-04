# Configuration

A run is defined by `configs/train_config.yaml`: one top-level key and four
sections. Generate a copy with every default written out explicitly:

```bash
python scripts/run_train.py --write-config configs/train_config.yaml
```

This page documents every key that file can contain.

## How a value is resolved

Three sources, highest priority first:

1. an explicit command-line flag,
2. the value in the YAML file,
3. the built-in default.

That ordering is what lets one committed config be swept from the shell without
being edited or copied:

```bash
python scripts/run_train.py --config configs/train_config.yaml --epochs 500
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

## Top level

| Key | Default | Meaning |
| --- | --- | --- |
| `task` | `all` | which map to train: `ext2chg`, `chg2tau`, or `all` for both in sequence |

## `data` — where the fields come from

| Key | Default | Meaning |
| --- | --- | --- |
| `root` | `data/vasp` | directory holding one subdirectory per material |
| `cache` | `data/cache` | where downsampled copies are written |
| `pattern` | `struct` | prefix identifying those subdirectories — a prefix, not a glob |
| `code` | `auto` | DFT code, or `auto` to detect it from the files present |
| `resolution` | `32` | longest grid axis after spectral downsampling |
| `gaussian_blur` | `null` | Gaussian blur width in Å applied to the computed potential |
| `blur_method` | `spectral` | `spectral` or `ndimage` |

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
| `n_layers` | `4` | number of Fourier layers |
| `projection_channels` | `48` | hidden width of the output projection |
| `activation` | `gelu` | `gelu`, `relu`, `silu` or `tanh` |
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
| `mode` | `universal` | `universal` or `leave_one_out` |
| `holdout` | `null` | structure names excluded from universal training |
| `enable_kfold` | `false` | run K-fold cross-validation; takes precedence over `mode` |
| `k_folds` | `5` | number of folds, capped at the structure count |
| `epochs` | `200` | passes over the training set |
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

### `mode`, `holdout`, `enable_kfold`, `k_folds`

These four decide *what question the run answers*, and are the settings most
worth getting right.

- **`universal`** trains **one** model per task on the combined data of every
  structure and saves a single checkpoint. This is the deployable artefact.
  Poraquê never trains a separate model per material.
- **`holdout`** is a list of structure names excluded from that training and
  scored separately, which turns the reported number into a generalisation
  estimate.
- **`enable_kfold`** trains a fresh model per fold on *K*−1 groups and scores it
  on the group held out. Produces a generalisation estimate with a spread — but
  no single model to ship.
- **`leave_one_out`** is the same protocol with *K* equal to the number of
  structures.

Both are worth running: cross-validation says whether the architecture
generalises, `universal` produces the model that uses all the data.

```{warning}
With `mode: universal` and `holdout: null`, nothing is held out and the
reported metrics are **training fit**, carrying no generalisation claim. On the
reference dataset they are four to five times better than the cross-validated
score. Quote the cross-validated numbers.
```

Splitting is always at the *structure* level: whole materials move together. A
voxel-level split would place the same crystal on both sides, and since
neighbouring voxels are strongly correlated the score would look excellent
while saying nothing about transfer to a new material.

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

### `physics`

| Weight | Constraint it penalises the violation of |
| --- | --- |
| `electron_count_weight` | $\int\rho\,d\mathbf{r} = N$ (`ext2chg`) |
| `positivity_weight` | non-negativity of the predicted field |
| `von_weizsacker_weight` | $\tau \ge \tau_\mathrm{vW}$ (`chg2tau`) |
| `euler_lagrange_weight` | orbital-free stationarity residual (`ext2chg`) |

All four default to zero, so the objective is the plain supervised baseline
until one is enabled deliberately. Each term is dimensionless, so a single
weight can serve a heterogeneous dataset.

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
| `log` | `logs/fno_training.log` | human-readable training log |
| `json` | `logs/fno_training.json` | metrics and full loss history |
| `checkpoint_dir` | `models` | model weights; `null` disables checkpointing |
| `plot_dir` | `results/plots` | figures; `null` disables plotting |
| `report_dir` | `reports` | PDF reports; `null` disables them |
| `plot_format` | `png` | `png`, `pdf` or `svg` |
| `dpi` | `160` | raster resolution for saved figures |

The resolved configuration is written next to `json` with the suffix
`_config.yaml`. PDF reports are assembled in a temporary directory that is
removed afterwards, so no `.tex`, `.aux` or `.log` files are left behind.

## Worked examples

Train the deployable models:

```yaml
task: all
model:    {pauli_residual: true}
training: {mode: universal, epochs: 200}
```

Measure generalisation:

```yaml
task: all
training: {enable_kfold: true, k_folds: 5}
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
