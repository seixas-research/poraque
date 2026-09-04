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

(enable-blocks)=
## Optional features are blocks

Every feature that can be switched off is one key, with its switch **inside**
the settings it governs:

```yaml
<group>:
  <feature>:
    enable: <switch>
    <setting>: <value>
```

There are four: `model.equivariant`, `training.physics_informed`,
`symbolic.physics` and `fine_tuning`. A setting inside a block is read **only
when `enable` is on**; naming one beside a switch that is off warns rather than
acting, and an unknown one raises, naming the block it was in.

```{note}
Two of these were a scalar beside a separately-named `<feature>_setup` group
until 26.9.8, and the two halves drifted in exactly the way that layout invites:
`equivariant: false` above a populated `equivariant_setup` reads at a glance as
an equivariant run and is not one, and nothing brought the switch and the
settings into the same field of view. The old spellings now raise and name
their replacement — including the bare `equivariant: true`, which is a *type*
error rather than an unknown key, so it carries its own message showing the
block to write instead.

`symbolic.enable_symbolic_distillation` became `symbolic.enable` in the same
change. It restated its own section in its own name, and at 37 characters was
the longest key in the schema — wide enough to set the width of the
configuration table in the PDF report.
```

The one `_setup` that stays is `model.kan_setup`, and deliberately: it is
selected by `model.activation`, which is a *choice among five* rather than a
switch, so there is no `enable` for it to hold and no boolean it could
contradict.

Blocks are validated when the file is **read**, not when the feature is first
used. That matters more than it sounds: the model is built after the field
cache and the objective after that, so a one-line mistake in either block would
otherwise take minutes of downsampling to report.

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
| `models/<name>.poraque` | the weights |
| `reports/<name>_report.pdf` | the PDF report |
| `results/plots/<name>/` | the figures |

A `task: all` run trains two models and so writes two reports, which are then
`<name>_ext2chg_report.pdf` and `<name>_chg2tau_report.pdf` — one name cannot
serve both. Cross-validation writes `<name>_kfold_report.pdf`, and a fine-tune
writes its weights to `<name>_finetuned.poraque` rather than over the general
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
| `data_paths` | `data/vasp/structures` | the dataset, as a list of directories — see below |
| `cache` | `data/cache` | where downsampled copies are written |
| `pattern` | `""` | prefix filter on the material subdirectories; empty takes all of them |
| `format` | `auto` | `vasp`, or `auto` to detect the code from the files present |
| `resolution` | `32` | longest grid axis after spectral downsampling |
| `potcar_dir` | `null` | POTCAR library, used where the data ships no pseudopotentials |
| `strict_potcar` | `false` | refuse the Gaussian fallback when a configured library cannot serve an element |
| `sigma` | `null` | Gaussian pseudo-ion width in Å, where the Gaussian model is reached |
| `gaussian_blur` | `null` | Gaussian blur width in Å applied to the computed potential |
| `blur_method` | `spectral` | `spectral` or `ndimage` |
| `cache_in_memory` | `auto` | keep decoded fields in RAM between epochs — see below |

(cache-in-memory)=
### `cache_in_memory`

The largest performance setting in the file, and the only one whose effect is
completely invisible from the results.

With it off, **every epoch reopens, decompresses and re-parses every field from
disk**. That produces exactly the same numbers as caching does — and takes
about ten times as long. Measured on a V100 with 115 structures at 32³:
`__getitem__` was 59 % of the training loop, 78.9 s over 483 calls, while the
GPU sat at 2–4 % utilisation waiting on `zlib` and `numpy.array` over a text
generator. Turning the cache on took twenty epochs from **168.0 s to 16.3 s**,
with the validation error identical to five decimals.

It is not a GPU setting. The parse is the same on Apple Silicon and on the CPU;
what an accelerator adds is a fast device left idle by it.

`auto` estimates the decoded size — grid points × itemsize × channels, over the
input and target fields — and enables the cache below 4 GiB, logging what it
decided and what it costs:

```
  field cache         : in RAM, ~61.3 MiB (data.cache_in_memory: auto)
```

Set it to `false` when the process shares its memory with something the
estimate cannot see. What `auto` refuses on its own is the case the old default
was right about: 128³ grids over thousands of materials, where caching silently
would turn a slow run into a dead one.

**Parallel loading is not the alternative it looks like.** See
[`num_workers`](num-workers-and-pin-memory).

### `data_paths`

**One key names the dataset, and every entry in it has the same shape:** a
directory of subdirectories, one per material, each holding that material's
volumetric files.

```yaml
data:
  data_paths:
    - data/vasp/structures      # structure_0000/, structure_0001/, ...
    - data/MP                   # mp-124/, mp-81/, ...
    - data/cache/res32          # a cache from an earlier run
```

That is what a VASP run tree looks like, what `poraque-mp` writes, and what the
cache builder produces — so nothing has to be declared about where a path came
from, and no key says so. Every path is pooled into one dataset.

What *differs* is the **content** of a material's directory, and content is
read, not configured:

| A material's directory holds | Poraquê |
| --- | --- |
| inputs beside the density | computes $V_\mathrm{ext}$ from them — exactly, with a `POTCAR` or `potcar_dir` |
| a density on its own | computes it from the structure the `CHGCAR` carries in its own header |
| an `EXTCAR` already | reads it as it stands (a prepared cache) |

$V_\mathrm{ext}$ is **always computed** for the first two, never read from an
`EXTCAR` that happens to sit in a run directory: at inference time there is no
DFT run to read one from, so the input channel has to be Poraquê's own
arithmetic in both places.

`TAUCAR` is **optional, per material.** A directory that has one joins the
`chg2tau` set; one that does not still joins `ext2chg` and is simply absent
from the other. Nothing is declared for it, and a task with no target anywhere
is skipped with a message rather than failing the run — which is what makes
`type: all` sensible on a Materials Project download, where no $\tau$ exists.

A path holding a **single** material's run directly — its files at the top
level rather than one level down — is read as that one material, so a lone
calculation needs no wrapper directory.

```{note}
Three keys used to answer this one question: `train_paths` (a list), `root`
(a single path used when the list was empty) and `source` (which layout each
path was). All three were removed on 2026-08-31. The last is the one that
mattered: it asked the *config* to declare something the *directory* already
answers, which made "a VASP run" and "a Materials Project download" two
different things to write down when they are the same thing on disk. A config
using any of them now fails and is told what to write instead.
```

See {doc}`data/index` for what each kind of directory contains and for training
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

(strict-potcar)=
#### What the cache records, and `strict_potcar`

A library that cannot be read falls back per element, with a `RuntimeWarning`
on stderr, and the run continues. That is the right default. What was wrong
until 26.9.3 is that nothing downstream recorded it: the cache fingerprint held
`potcar_dir`, which is the path that was **asked for**, and said nothing about
whether it was ever opened.

The consequence was found on Santos Dumont. A control run landed on a node
where the library's filesystem was not mounted, warned six times, trained
against analytic pseudo-ion potentials, and wrote
`cache/res32_potcar/cache_fingerprint.json` byte-identical to one built from
real pseudopotentials — down to the directory being *named* `potcar`. Every
later run reused it silently, including on nodes where the library was
mounted, and the warning never came back. The validation error it produced had
already been quoted as a measurement.

The fingerprint now carries `potcar_source`, per element:

```json
{ "potcar_dir": "/prj/…/POTCARs",
  "potcar_source": {"Ag": "gaussian", "Pt": "library"} }
```

so a cache built during an outage no longer matches a run that can read the
library — it rebuilds rather than being reused — and the resolved mode appears
in the run log beside the rest of the cache line. `null` when no library is
configured, so a purely Gaussian dataset's fingerprint is unchanged.

`strict_potcar: true` turns the fallback into an error naming the elements
the library did not serve. It is off by default and is meant for queues: the
failure shape is `training.strict_device`'s, a job that holds its allocation
and quietly computes something other than what was configured.

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
| `equivariant` | `{enable: false}` | rotation-equivariant radial kernel — one block, see below |
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
    variant: chebyshev            # bspline | chebyshev | rbf | rational
    degree: 6                 # chebyshev only
```

The block is read **only** when `activation: kan`; given beside a stateless
activation it warns rather than being silently ignored, and an unknown key
inside it raises. Omitting `variant` gives `bspline`, the original KAN paper's
own parameterisation.

Six of the seven hyperparameters are read by one variant each — `grid_size`,
`spline_order` and `grid_range` by `bspline` (and the first and third by `rbf`,
which reuses the same fixed-grid design), `degree` by `chebyshev`,
`rational_num_degree` and `rational_den_degree` by `rational`. The seventh,
`use_base`, applies to all four: `true` keeps each channel's $w_c\,\mathrm{silu}(x)$
base term, so every variant starts close to `silu` and only departs from it as
training moves the learned coefficients; `false` gives a "pure" KAN with no
fixed nonlinearity mixed in at all.

Cost at `width: 16` on CPU, relative to a stateless activation: `chebyshev` 1.36×,
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

### `equivariant`

One block — see [Optional features are blocks](#enable-blocks):

```yaml
model:
  use_coordinates: false      # required
  equivariant:
    enable: true
    n_radial: 32
```

Constrains the Fourier multiplier to a **real function of $|\mathbf{G}|$
alone**, which is a convolution with a radial kernel and therefore commutes
with every rotation. The coefficients being real also makes it commute with
inversion, so the operator is equivariant under the full $O(3)$ rather than
the $SE(3)$ the construction was asked for.

| Key | Default | Meaning |
| --- | --- | --- |
| `enable` | `false` | use the radial kernel in place of the dense one |
| `n_radial` | `16` | radial basis functions — the whole capacity of the kernel |
| `g_basis` | `null` | radius in Å⁻¹ the basis spans; `null` takes `g_max`, else `8.0` |
| `spherical_cutoff` | `true` | mask the retained modes to the inscribed sphere |

**Three conditions have to hold together, and each fails silently.** The kernel
being radial is the one everybody thinks of; the other two were found by
measuring.

*The retained set must be a ball, not a box* — `spherical_cutoff: true`. A
radial multiplier over a box of modes is equivariant only under the box's own
symmetry group. A cubic cell with equal mode counts hides this completely,
because the box *is* invariant under the octahedral group; on a tetragonal cell
switching the cutoff off takes the error from 3e-7 to 5e-2.

*`use_coordinates` must be off*, and `enable: true` beside it **raises**. The
three fractional coordinates are not three scalar fields — under a rotation
they turn into each other — and being absolute positions they cost translation
equivariance too.

**The cost is capacity, and it is large.** The dense layer holds
$4C_\mathrm{in}C_\mathrm{out}m_1m_2m_3$ complex numbers; this one holds
$C_\mathrm{in}C_\mathrm{out}R$ real ones — 4 096 against 1 048 576 at
`width 16, modes 8, n_radial 16`. Buy it back with `n_radial` and `width`, not
with `modes`.

**Whether it pays was measured once**, on a V100: 400 epochs on 115 Materials
Project structures at 32³, `width 32 / n_radial 32` reached a validation
relative $L^2$ of 0.047 against 0.173 for a dense model with 28× the
parameters — 4.1× better. The 4× *larger* equivariant arm
(`width 128, n_radial 64`) was **worse**, at 0.059: capacity is not what buys
this, the constraint is. One seed, one dataset, and neither equivariant arm had
converged at 400 epochs, so read it as a reason to try the flag rather than as
a number.

```{note}
An equivariant model is slower per epoch despite having two orders of magnitude
fewer parameters, and that is not a bug. The dense layer *stores* one complex
number per retained mode per channel pair and reads it; the radial layer stores
$C^2R$ real numbers and must **reconstruct** the multiplier at every mode from
them — about $R/2$ times the arithmetic at `n_radial: R`. The constraint trades
memory for recomputation, so parameters and FLOPs move in opposite directions
by construction.
```

```{warning}
NVIDIA's cuequivariance is deliberately **not** used and is not a dependency.
It accelerates arithmetic on irreducible representations of $O(3)$, and a
scalar-field-to-scalar-field operator carries only $\ell=0$: every tensor
product in it is a scalar multiply, and the Fourier layer is a complex `einsum`
diagonal in the mode index, which Clebsch–Gordan cannot express. Profiled by
kernel class on a V100 its share is 0.0 %. Equivariance here is a constraint on
the kernel, not a kernel library.
```

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
| `strict_device` | `false` | abort instead of falling back to the CPU — see below |
| `distributed` | `auto` | `auto` or `off`; multi-GPU over NCCL when the launcher describes a group |
| `distributed_timeout` | `30.0` | minutes before a collective is declared failed |
| `num_workers` | `0` | DataLoader worker processes — read `data.cache_in_memory` first |
| `pin_memory` | `auto` | page-locked staging; `auto` is `true` on CUDA and `false` elsewhere |
| `tf32` | `true` | TensorFloat-32 matmul and convolution; CUDA, Ampere and later |
| `loss` | `relative_l2` | `absolute_l2`, `relative_l2`, `absolute_h1` or `relative_h1` |
| `sobolev_weight` | `0.1` | gradient-term weight for the two `h1` losses |
| `physics_informed` | `{enable: auto}` | the switch and the four constraint weights — one block, see below |

(strict-device)=
### `device` and `strict_device`

`device: auto` prefers CUDA, then Apple Metal, then the CPU. An explicit backend
that is unavailable **warns and falls back to the CPU**, so a configuration
written on a machine with a GPU still runs on a laptop.

That is right on a workstation and expensive in a queue. `strict_device: true`
turns the fallback into an error:

```yaml
training:
  device: cuda
  strict_device: true
```

**Every HPC configuration should set it.** Without it, a job that cannot reach
its GPU takes its place in the queue, warns into a log nobody reads until
afterwards, and trains on the CPU *inside* the GPU allocation until the wall
clock ends. The refusal names the probable cause — no GPU visible, an empty
`CUDA_VISIBLE_DEVICES`, a driver older than the wheel's CUDA runtime, or a
build carrying no kernels for this architecture — and prints the full device
report into the run's log.

That last case is worth stating on its own, because `torch.cuda.is_available()`
cannot detect it: it answers "is there a driver and a device", not "can this
binary generate code for this device". A `+cu130` wheel on a V100 reports
available and then aborts at the first kernel launch, since CUDA 13 dropped
Volta. Poraquê checks the compute capability against the build's own
`arch_list` and refuses before any of that. See
[NVIDIA GPUs](nvidia-gpus).

`strict_device` applies to `device: auto` as well, which is the default and
until 26.9.3 was the case it did *not* cover: the `auto` branch returned the
best available backend before `strict` was consulted, so a configuration that
set `strict_device: true` and left `device` alone — the exact pairing this
section recommends — got no protection at all. It now refuses when `auto`
resolves to the CPU, and names `device: cpu` as the way to ask for CPU
training deliberately.

(num-workers-and-pin-memory)=
### `num_workers` and `pin_memory`

`num_workers` and [`data.cache_in_memory`](cache-in-memory) are
**alternatives, not complements**, and the measurement is unambiguous. Twenty
epochs on 115 structures at 32³, on a V100:

| variant | training time | speedup |
| --- | --- | --- |
| as shipped | 168.0 s | 1.0× |
| `cache_in_memory: true` | **16.3 s** | **10.3×** |
| `num_workers: 4`, no cache | 120.2 s | 1.4× |
| both | 18.8 s | 8.9× |

The validation error was identical in all four, as it must be. Each worker is a
*process* with a cache of its own, so it re-parses the whole dataset on its
first epoch — the parse the cache removes is paid N times instead of once.
Raise `num_workers` only when the data genuinely does not fit in RAM, which is
the case `cache_in_memory: false` exists for.

`pin_memory` is a CUDA transfer optimisation and nothing else: `auto` is `true`
there and `false` everywhere, since it does not help on Metal and some PyTorch
versions warn when asked for it.

### `tf32`

`tf32` keeps float32's range and drops its mantissa to ten bits inside the
tensor cores. It is a real speedup on Ampere and later, and **exactly nothing**
before it — a V100 has no TF32 path — so it is on by default and costs nothing
where it does not apply. Ignored off CUDA.

(no-torch-compile)=
### There is no `compile` key

There was one, for a version, and it was removed in 26.9.3 after being
measured. Setting `compile`, `compile_mode` or `compile_dynamic` now raises and
says why, rather than being accepted and ignored.

**Inductor does not generate code for complex operators.** Every compiled run
said so:

```
torch/_inductor/lowering.py:1917: UserWarning: Torchinductor does not support
code generation for complex operators. Performance may be worse than eager.
```

`SpectralConv3d`'s weights are complex, and it *is* the Fourier neural
operator. So the one part of the model compilation was invoked to fuse is the
part Inductor hands back to eager, while the wrapper and the graph breaks are
still paid for. That is the mechanism behind the numbers — 150 epochs, two
repetitions, one V100:

| arm | s/epoch | first epoch (cold → warm) | val rel L2 |
| --- | --- | --- | --- |
| off | **0.6289** | 3.8 s | 0.280563 |
| `mode="default"` | 0.6785 (+7.9 %) | 199.8 s → 17.6 s | 0.284980 |
| `mode="max-autotune"` | 0.6879 (+9.4 %) | 314.0 s → 20.7 s | 0.272624 |

`TORCH_LOGS=recompiles` emitted nothing, so `dynamic=True` did hold one graph
across the 19 shapes: the shape-bucketing worry was unfounded and the gain
still does not exist. The validation error also moved, reproducibly and
identically across both repetitions, which disqualifies the arms independently
of their timing.

On four ranks it did not merely lose. With `--distributed auto` and 60 epochs,
in the same allocation as an uncompiled four-rank run that finished in 86.6 s,
the compiled run sat **eighteen minutes in the first epoch without printing an
epoch line** and was cancelled. A configuration key whose failure mode is a
silent deadlock inside a queue allocation is the hazard
[`strict_device`](strict-device) exists to remove, pointing the other way.

What was *not* measured, and is worth measuring one day: compiling only the
real-valued submodules — `CellEncoder`, FiLM, the pointwise convolutions, the
projection head, the KAN activations — and leaving `SpectralConv3d` in eager.
That is where the 44.4 % of elementwise GPU time lives. It should come back as
an internal decision about which submodules to wrap, not as a switch a config
file can throw. The condition for re-opening whole-model compilation is
upstream: Inductor gaining complex-operator codegen.

(distributed)=
### `distributed` and `distributed_timeout`

`auto` — the default — forms a `DistributedDataParallel` group over NCCL **when
the environment already describes one**, and does nothing otherwise. A Slurm
step with more than one task, or a `torchrun` launch, is such an environment; a
workstation is not.

It cannot turn a one-process run into a four-GPU one. The launcher decides the
topology and this key decides only whether to believe it, so on a cluster the
setting that matters is in the submission script:

```bash
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4
srun poraque-train --config configs/train.yaml --distributed auto
```

See [Several GPUs, under Slurm](installation.md) for the full script and for
how `MASTER_ADDR` and `MASTER_PORT` are derived from the allocation.

`off` refuses a group inside a multi-task allocation, which is how four
independent single-GPU jobs are run from one submission and how a scaling
result is bisected against a single-device baseline.

Three consequences, all of them visible in the run log:

- The **effective batch size is `batch_size` × the world size**. A four-rank
  run at `batch_size: 10` steps on 40 samples, so it is not the same optimiser
  as the one-rank run it is compared against — halve the learning rate or
  quarter the batch size if the comparison is meant to be of the same
  optimisation.
- **Rank 0 alone writes** the checkpoint, the metrics, the figures and the PDF,
  and alone prints.
- **The batches are split, not the samples.** `ShapeBucketSampler` groups
  materials by grid shape so no padding ever reaches the FFT, and a `DataLoader`
  takes a `sampler` or a `batch_sampler` and never both — so the bucketing runs
  first, identically on every rank, and a real `DistributedSampler` partitions
  the resulting *list of batches*. Each rank gets a unique, non-overlapping
  subset, and every rank gets the same *number* of batches, which is not
  tidiness: DDP all-reduces gradients inside each `backward()`, and a rank that
  runs out of batches first leaves the others in a collective that never
  completes.

`distributed_timeout` is long (30 minutes) on purpose. The first collective
happens after every rank has read its prepared cache, and a cold
parallel-filesystem read of a few hundred densities is minutes; a timeout that
fires there reports itself as a NCCL error and sends the reader looking at the
network.

```{note}
There is no `DataParallel` fallback and no Gloo backend. Both would let a
misconfigured run go quietly slower than a single GPU — the first by
replicating in one process, the second by distributing across CPU cores inside
a GPU allocation. Without CUDA the group is refused with a warning naming the
probable cause, and the run continues on one device.
```

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

That cap is a real ceiling, not a formality: **above the largest bucket's size
the setting stops meaning anything.** In the 115-structure set measured on a
V100 the largest bucket held 70 materials (≈56 after the split), and `bs=64`
and `bs=128` were the same run — identical peak memory, identical error,
identical time. With the field cache on, throughput responded to the batch up
to 16 and was flat thereafter.

`peak_vram_bytes` and `seconds_per_epoch` in the metrics JSON are what a sweep
over this should be read from; before they were recorded there, the only route
to either was sampling `nvidia-smi` from outside the process.

### `loss` and `sobolev_weight`

Four objectives, on two named axes:

| `loss` | normalised per sample | gradients in the objective |
| --- | --- | --- |
| `absolute_l2` | no | no |
| `relative_l2` | yes | no |
| `absolute_h1` | no | yes |
| `relative_h1` | yes | yes |

**Relative** divides each sample's error by its own target norm, so materials
whose fields differ by orders of magnitude contribute equally and the value
reads directly as a fraction. **Absolute** does not, so every *voxel* counts
equally and a denser system contributes proportionally more gradient — right
for a set whose materials are genuinely comparable in magnitude, and usually
wrong for a heterogeneous one, which is why `relative_l2` is the default.

**H1** adds `sobolev_weight` times the same error on the spatial gradients.
Worth enabling when the derivative matters — $\tau_\mathrm{vW}$ depends on
$\nabla\rho$, so gradient noise is amplified in low-density regions. Both
halves are taken in the *same* norm, so the weight is a pure ratio between
values and derivatives rather than also absorbing a change of scale.

**The validation column follows the objective.** The progress table reports
`val abs L2`, `val rel L2`, `val abs H1` or `val rel H1` — whichever is
running — and the number behind it is the objective's own data term evaluated
on the held-out set:

```text
    train loss: mean PhysicsInformedLoss per batch   |   val rel H1: held-out error, physical units
    val rel H1 = rel L2 + 0.1 x the relative L2 of the gradient, matching the
    objective; the final per-structure table below reports plain rel L2
          epoch     train loss     val rel H1
```

That is not only a label. Early stopping and the checkpoint are decided on this
number, and while it read `val rel L2` regardless a gradient-constrained run
was selecting the model that minimised a functional it was not training on. The
four are not comparable with each other — which is why the **per-structure
table at the end of a run reports relative $L^2$ whatever the objective was**.
That is the number to quote, and the one that lets two runs be put side by
side.

```{note}
`loss: sobolev` was renamed on 2026-08-31. It named the gradient term and left
the norm implicit, and offered no unnormalised form at all. It now raises,
naming `relative_h1` (what it did) and `absolute_h1` as its replacements.
```

### `physics_informed`

One block — see [Optional features are blocks](#enable-blocks):

```yaml
training:
  physics_informed:
    enable: auto              # auto | true | false
    electron_count_weight: 0.1
```

| Key | Constraint it penalises the violation of | Shipped |
| --- | --- | --- |
| `enable` | — whether any of the below is evaluated | `auto` |
| `electron_count_weight` | $\int\rho\,d\mathbf{r} = N$ (`ext2chg`) | `0.1` |
| `positivity_weight` | non-negativity of the predicted field | `0.0` |
| `von_weizsacker_weight` | $\tau \ge \tau_\mathrm{vW}$ (`chg2tau`) | `0.0` |
| `euler_lagrange_weight` | orbital-free stationarity residual (`ext2chg`) | `0.0` |

All four weights default to zero in the code, so the objective is the plain
supervised baseline until one is enabled deliberately. Each term is
dimensionless, so a single weight can serve a heterogeneous dataset.

`enable` has three states, and the middle one is the reason it exists:

`auto`
: the default. The run is physics-informed if and only if at least one weight
  is positive — which is what every configuration written before the switch
  existed already meant, so nothing changes for one.

`true`
: **raises** when no weight is set. A run that declares physics-informed
  training and silently optimises the supervised baseline has an entirely
  ordinary loss curve and a report that says otherwise; nothing anywhere would
  contradict it.

`false`
: zeroes the weights and warns if one was live. Not merely a change of
  objective: every constraint acts on *decoded* fields, so with one live the
  loop must copy the reference field to the device on every batch, invert the
  target transform over the prediction inside the autograd graph, invert the
  input transform too, and in delta-density mode copy the baseline across and
  add it back twice. Off, all of that is work whose result is discarded.
  Measured: 2.4 % of a step on Apple Metal and **6.6 % on a V100**, where the
  per-batch copy crosses PCIe.

```{warning}
Not `off`, and not `no`. Anything that is not `auto`, `true` or `false` raises
rather than being coerced — `bool("off")` is `True` in Python, so a config
saying `off` would have switched the constraints **on**.
```

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
| `checkpoint` | `true` | write `<name>.poraque` |
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

### What the report's performance table says

One row per split — `train`, `validation`, `all` — and five columns:

| Column | What it answers |
| --- | --- |
| rel. $L^2$ | The headline error, and the one every other number in this manual is quoted in. |
| rel. $H^1$ | Values *and* gradients. A prediction can match a density pointwise and still be rough, and $\tau_\mathrm{vW}$ depends on $\nabla\rho$ — so a small $L^2$ beside a large $H^1$ is a model whose energetics will disappoint. |
| MAE | The typical voxel, in the field's own units, where a squared error is not readable as a quantity. |
| Max. error | The worst voxel. Means and RMS both hide a single catastrophic site, and near a nucleus is exactly where one occurs. |
| $\lvert\Delta N\rvert$ | Conservation. For `ext2chg` this is the electron-count error in electrons, and it is the metric a pointwise loss controls worst. |

The `train` and `validation` rows read against each other **are** the
generalisation gap. That is the single most useful thing the table says, and it
is why the splits are rows.

```{note}
The rel. $H^1$ here is the textbook relative Sobolev norm, which has no free
parameter — **not** the $H^1$ objective `loss: relative_h1` minimises, whose
value depends on `sobolev_weight` and so cannot be compared between two runs.
Every column is computed without consulting the loss, for exactly that reason.

For a two-channel model every number is the **density** channel; the
magnetisation is measured separately, as `magnetisation_relative_l2` in the
metrics JSON, and the report says so in its caveats.

The table lists *splits*, not structures: it used to print one row per
material, so a 115-material run made six pages of numbers nobody reads
individually. The per-structure figures — and `mse`, `rmse`, `r2`,
`nrmse_range` and `jsd`, which are not typeset — are all in
`log/<name>.json`, which is machine-readable and has no margins.
```

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
  `val_abs_h1` for an `absolute_h1` run). Epochs on which validation was not
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

Two runs over the same data with different physics, kept side by side. The
first is explicit rather than left to `auto`: a baseline that says
`enable: false` cannot be misread later as one whose weights happened
to be zero, and it also skips the per-batch decode the constraints need.

```yaml
task:     {type: ext2chg, name: agaupt_baseline}
training: {physics_informed: {enable: false}}
```

```yaml
task:     {type: ext2chg, name: agaupt_charge_conserving}
training: {physics_informed: {electron_count_weight: 0.1}}
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
