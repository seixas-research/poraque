# Configuration

A run is described by a YAML file of eight sections, each about one thing:

```yaml
run:           # what the run is called, where it writes, its seed
data:          # where the materials are, and how they are cached
ext2chg:       # the density operator: V_ext -> pseudo-density + occupancies
chg2tau:       # the kinetic operator: pseudo-density + occupancies -> tau
training:      # the optimization, shared by both operators
fine_tuning:   # start from a trained checkpoint (off unless present)
runtime:       # the hardware: device, processes, distribution
report:        # figures and the PDF
```

Every key is optional: a file states only what the run does differently, and
anything omitted takes its default. Two files in `configs/` are the starting
points. `configs/train_simple.yaml` is the short one, the platinum set with both
operators, forty-odd lines, only what the run decides.
`configs/train_complete_and_commented.yaml` is the map: every key at its
default, with a line saying what it means. This page documents every key either
can contain, and the measurements behind the defaults that were measured.

## How a value is resolved

Three sources, highest priority first:

1. an explicit command-line flag,
2. the value in the YAML file,
3. the built-in default.

That ordering is what lets one committed config be swept from the shell without
being edited or copied:

```bash
poraque-train --config configs/train_simple.yaml --epochs 500 --seed 1
```

Each flag overrides one dotted path (`--epochs` is `training.epochs`,
`--cache-workers` is `runtime.workers`); `poraque-train --help` lists them
under the section they belong to.

```{note}
Unknown keys are **rejected**, not ignored. A typo such as `learn_rate` would
otherwise be silently dropped and the run would quietly use the default,
producing results that do not match the file that appears to describe them. The
error message names the valid keys for that section. The keys and sections of
the schema this one replaced (2026-10-13) are rejected too, and each error names
where its setting went: `training.valid_fraction` is now
`training.validation.fraction`, `data.cache_workers` is `runtime.workers`, and
so on.
```

The *resolved* configuration, with defaults, file and overrides merged, is
written beside the results as `<name>_config.yaml`. That is the file worth
keeping: it records what actually ran.

(enable-blocks)=
## Operators and blocks: presence is the switch

**An operator is trained when its section is present.** `ext2chg:` and
`chg2tau:` are the two; leave one out and it is not trained. A configuration
with neither raises, since there is nothing to do.

```yaml
ext2chg:            # trained
chg2tau:            # trained, with its defaults
```

Each operator carries its own `model`, because the two maps need different
networks and a single shared block forced the larger one into a corner of the
other.

**A feature that can be switched off is a block whose presence is the switch.**
`false` turns it off, `true` (or the bare key) turns it on with the defaults, a
mapping turns it on with those settings. There is no `enable` key anywhere:

```yaml
ext2chg:
  model:
    equivariant: false        # off
```

```yaml
ext2chg:
  model:
    equivariant: true         # on, with the defaults
```

```yaml
ext2chg:
  model:
    equivariant:              # on, with these
      n_radial: 32
```

The old layout put a switch beside the settings it governed, and a block could
be switched off while still listing settings that read as if they applied; with
the presence as the switch, what the file says is what runs. Blocks are
validated when the file is **read**, not when the feature is first used. That
matters more than it sounds: the model is built after the field cache and the
objective after that, so a one-line mistake in either would otherwise take
minutes of downsampling to report.

The blocks are `ext2chg.structure_input`, `ext2chg.occupancies`,
`chg2tau.occupancies`, `model.equivariant`, `training.physics`'s weights (zero
is off), `fine_tuning` and `fine_tuning.lora`.

**The two names are the vocabulary of configuration; four are the vocabulary of
the code.** `occupancies` is the switch between them. With it on (the default)
`ext2chg` predicts the pseudo-density and every atom's PAW augmentation
occupancies, and `chg2tau` reads them. With `occupancies: false` the operator
trains on the grid alone, which internally is the `ext2pseudo` task (V_ext to
pseudo-density) or the `pseudo2tau` task (pseudo-density to $\tau$). Those two
names remain in logs, figure and file names and checkpoints, and `poraque-train
--task` takes only the configuration's two. The grid-alone pair needs no
`POTCAR`, and a bundle needs exactly that pair to run `poraque-inference` and
the ASE calculator.

## What is fixed

Some settings stopped being choices because one value won every measurement or
the alternative never worked. They are not keys: the cosine schedule, a
gradient-norm clip of 1, the weight initialization seeded by `run.seed`, the
read-out activation following `model.activation`, the Pauli scale fitted on the
training split and learned, an Euler–Lagrange weight of 0, fields held in
float64, the isolated atom's PAW records for a prediction, gzip level 4, pinned
memory on CUDA and not elsewhere, a 30 minute collective timeout, and an asinh
scale fitted rather than fixed. Naming one in a file raises.

(run-section)=
## `run` — what the run is called

| Key | Default | Meaning |
| --- | --- | --- |
| `name` | `poraque_models` | the run's name; everything goes in `<output>/<name>/` |
| `output` | `models` | parent directory; `null` writes **nothing** (no checkpoint, log or figure) |
| `seed` | `42` | the validation draw, the folds, the batch order **and** the initial weights |

```yaml
run:
  name: mp_ag_au_pt
  output: models
  seed: 42
```

Everything a run writes lives under `<output>/<name>/`: the weights
(`<name>.poraque`), `log/` (the log, the metrics JSON and the resolved
configuration), `plots/` and `report/`. `output` is the only path setting and
`name` is the only thing that distinguishes two runs, so "delete this
experiment" and "send me that model" are one directory each rather than a
collection of fragments by filename.

A run that trains both operators writes two reports, `<name>_ext2pseudo_report.pdf`
and `<name>_pseudo2tau_report.pdf` (or `ext2chg` and `chg2tau`, by whether the
occupancies are on): one name cannot serve both. Cross-validation writes
`<name>_kfold_report.pdf`, and a fine-tune writes its weights to
`<name>_finetuned.poraque` rather than over the general model it specializes.

```{important}
Two runs that share a `name` share their output files, and the second wins.
Change it for anything worth keeping beside the last run: a different chemical
space, resolution, or set of physics weights.
```

`seed` is one number for everything that is random. Earlier versions had a
second one for the initial weights, so that a committee of models could differ
in initialization alone; now a different seed also draws a different validation
split and a different batch order, which is a stronger, and different, source of
disagreement. The ensemble tools take it as it is (see
`docs/notes/committee.md`).

## `data` — where the fields come from

| Key | Default | Meaning |
| --- | --- | --- |
| `path` | `data/vasp/structures` | the dataset: one directory, one subdirectory per material — see below |
| `potcar_dir` | `null` | POTCAR library, used where a material ships none |
| `atoms` | `data/vasp/isolated_atoms` | isolated-atom reference: a `poraque-atoms` JSON, or a directory of one-atom-in-a-box runs |
| `spin` | `auto` | train on the magnetization as a second channel — [see below](spin-key) |
| `grid.resolution` | `32` | longest grid axis after spectral downsampling |
| `grid.max_spacing` | `null` | largest grid spacing in Å after downsampling — [see below](max-spacing) |
| `cache.dir` | `data/cache` | where downsampled copies are written |
| `cache.storage` | `text` | `text` (a `CHGCAR`-format file per field) or `hdf5` (one `fields.h5` per material) |
| `cache.compression` | `null` | `null`, `gzip` or `lzf`; `hdf5` only, and it raises on a text cache |
| `cache.in_memory` | `auto` | keep decoded fields in RAM between epochs — [see below](cache-in-memory) |

```yaml
data:
  path: data/vasp/structures
  potcar_dir: /opt/vasp/potpaw_PBE
  grid:
    resolution: 48
  cache:
    storage: hdf5
    compression: gzip
```

### `path`

**One key names the dataset**: a directory of subdirectories, one per material,
each holding that material's volumetric files.

```text
data/vasp/structures/
    structure_0000/     CHGCAR  TAUCAR  POSCAR  POTCAR  INCAR ...
    structure_0001/
    ...
```

That is what a VASP run tree looks like, what `poraque-mp` writes, and what the
cache builder produces, so nothing has to be declared about where a path came
from. A material is a directory with a density in it: a `CHGCAR`, compressed or
not, or a `fields.h5`. To pool several sources, gather them under one directory
(symbolic links do); the set is the directory's content.

What *differs* is the **content** of a material's directory, and content is
read, not configured:

| A material's directory holds | Poraquê |
| --- | --- |
| inputs beside the density | computes $V_\mathrm{ext}$ from them: exactly, with a `POTCAR` or `potcar_dir` |
| a density on its own | computes it from the structure the `CHGCAR` carries in its own header |
| an `EXTCAR` already | reads it as it stands (a prepared cache) |

$V_\mathrm{ext}$ is **always computed** for the first two, never read from an
`EXTCAR` that happens to sit in a run directory: at inference time there is no
DFT run to read one from, so the input channel has to be Poraquê's own
arithmetic in both places.

`TAUCAR` is **optional, per material.** A directory that has one joins the
kinetic operator's set; one that does not still joins the density operator's
and is simply absent from the other. Nothing is declared for it, and an
operator with no target anywhere is skipped with a message rather than failing
the run, which is what makes both sections sensible on a Materials Project
download, where no $\tau$ exists.

A path holding a **single** material's run directly, with its files at the top
level rather than one level down, is read as that one material, so a lone
calculation needs no wrapper directory.

See {doc}`data/index` for what each kind of directory contains and for training
across a mixture of them.

### `grid.resolution`

Fields are reduced to this many points along their longest axis before
training. The reduction is a Fourier truncation, the exact band-limited
projection for a plane-wave field, so periodicity is preserved and the electron
count is unchanged to machine precision.

Cost scales as the cube: raising 32 to 64 is roughly eight times the memory and
time. Grid *shapes* are reduced in proportion, so a 120×128×128 calculation
becomes 30×32×32 rather than being forced square.

```{note}
There is no key for supplying an external potential. `EXTCAR` is **always**
computed by {class}`poraque.fields.ExternalPotential` from the `POTCAR` tables,
which reproduces a reference potential to a relative error of order $10^{-5}$.
Any `EXTCAR` present in a source directory is ignored: the training input must
be exactly what the pipeline produces at inference time, when no such file
exists. Standard VASP does not write one anyway.
```

(max-spacing)=
### `grid.max_spacing`

`resolution` caps the **longest** axis, and the others are reduced in
proportion. That is right for a compact cell and wrong for a long, thin one.
On the CHONPS set a 3 × 4 × 35 Å cell kept 6 and 8 points on its short axes,
0.6 Å spacing, where the median cell had 0.2 Å, and such cells were the
worst-predicted of the validation split by a factor of five. `max_spacing` is
the second bound, in Å: an axis that `resolution` would leave coarser than it
is refined to the smallest FFT-friendly count that meets it, but **never past
its native grid**, since a truncation cannot invent coefficients the
calculation never had. `null`, the default, is `resolution` alone.

The value is part of the cache tag (`res32_h0.25_potcar`) and of the
fingerprint, so a cache built with one bound is never read under another, and
like `resolution` it belongs to the set of keys a `--cache-only` job and the
GPU job after it must agree on.

(potcar-dir-and-the-gaussian-fallback)=
### `potcar_dir` and the Gaussian fallback

The external potential needs pseudopotentials, and not every dataset ships
them. A Materials Project download has a structure and a density and nothing
else; archived local runs often have their `POTCAR`s stripped for licensing.
`potcar_dir` points at a **library**, one subdirectory per species, as VASP
distributes them:

```yaml
data:
  potcar_dir: /opt/vasp/potpaw_PBE       # <dir>/Ag/POTCAR, <dir>/Pt/POTCAR, ...
```

`.gz` and `.Z` entries are read directly, and a flat `POTCAR.Ag` / `Ag.POTCAR`
layout is recognized too. A run that has its own `POTCAR` ignores the library
entirely: it is a fallback, not an override.

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
That is well-posed and self-consistent, since inference builds the very same
potential, but it is not VASP's $V_\mathrm{ext}$, and a model trained that way
is not comparable with one trained on tabulated potentials. It also means a
mixed dataset spanning both constructions is training on two quantities under
one name; {class}`~poraque.data.dataset.MixedFieldDataset` warns when that
happens, and setting `potcar_dir` is what makes the warning go away *correctly*.

```{warning}
Point at the library that generated the data. The Materials Project uses the
VASP PBE set (`PAW_PBE`); a `PAW_LDA` library would build a potential the
densities were never computed from, which is worse than the Gaussian fallback
because it looks exact.
```

Missing or unreadable entries are **not fatal**. Each such species warns once
and falls back to the Gaussian model for the structures that contain it, so a
library covering four elements of five still buys the exact potential for
everything that does not involve the fifth. The width is derived per species
from the pseudopotential core radius where one is known and is 0.5 Å otherwise
(an MP download supplies no `RCORE`).

**The cache records which construction it holds.** A library that cannot be
read falls back per element, and what was wrong until 26.9.3 is that nothing
downstream recorded it. A control run on Santos Dumont landed on a node where
the library's filesystem was not mounted, warned six times, trained against
analytic pseudo-ion potentials, and wrote a fingerprint byte-identical to one
built from real pseudopotentials. Every later run reused it silently, including
on nodes where the library was mounted. The fingerprint now carries
`potcar_source`, per element:

```json
{ "potcar_dir": "/prj/.../POTCARs",
  "potcar_source": {"Ag": "gaussian", "Pt": "library"} }
```

so a cache built during an outage no longer matches a run that can read the
library; it rebuilds rather than being reused. The run log names the
construction per source, and the cache directory records it (`res32_potcar`),
so a tabulated cache and a Gaussian one can never be confused for each other.

```{important}
**The occupancies need the `POTCAR`.** The layout of each element's PAW record
(which projector channels, so how many values) is read from the pseudopotential.
With `ext2chg.occupancies` on, the default, and no `POTCAR` anywhere, the run
stops before training and says to give the data one. A dataset without them,
such as a Materials Project download, trains the grid alone: write
`occupancies: false` in the operator's section.
```

(spin-key)=
### `spin`

| Value | What happens |
| --- | --- |
| `auto` | ask the sources whether their `CHGCAR`s carry a second grid block, and settle the channel count for the whole dataset before the cache is written |
| `true` | demand one: a dataset with no magnetization **raises** |
| `false` | drop it deliberately, and say so in the log |

This is the switch for spin-polarized training, and there is no other. A
`CHGCAR` from an `ISPIN = 2` run stores two grids, the total density $\rho$ and
the magnetization $m = \rho_\uparrow - \rho_\downarrow$; with `spin` resolved
on, both reach the cache, the density operator predicts them as channels 0 and
1 of one output, and the kinetic operator reads both. How many channels there
are is a property of the data, not of the architecture, which is why the key
lives in `data` and why `ext2chg` has no second one: two switches could
disagree, and one of them would then be quietly ignored. An unpolarized member
of a spin set carries $m \equiv 0$, so a mixed archive trains one two-channel
operator rather than two datasets. The cache tag says which it holds
(`res48_potcar_spin`).

The objective is still one number over $(\rho, m)$ together. The training log
says how it divides:

```text
          epoch     train loss     charge        mag     val rel L2
    ---------------------------------------------------------------
            3/6        1.23195    0.78913    0.94600        1.17954  *
            6/6        1.16123    0.73227    0.90121        1.11899  *
```

`charge` is the data term with the prediction substituted into the density
channel alone, the magnetization left at its reference; `mag` is the same the
other way round. Both keep the objective's norm and **its denominator**, the
norm of the whole $(\rho, m)$ target. That choice is forced rather than
convenient: a relative error of $m$ on its own divides by $m$, which is
identically zero on every non-magnetic cell, so it would be undefined on
precisely the materials most datasets are made of. With a shared denominator
the parts of an $L^2$ objective add in quadrature,
$\text{data} = \sqrt{\text{charge}^2 + \text{mag}^2}$ per sample, and the
columns say where the error lives. An $H^1$ objective adds its gradient term
rather than combining in quadrature, so there the columns rank the channels
without summing to anything; the log's legend states which case applies.
`train loss` remains the total the optimizer stepped on, physics terms
included.

Both parts are in the metrics JSON's `history`, as `train_loss_charge` and
`train_loss_magnetization`, and exist only for a two-channel density target; a
one-channel run prints neither column. The magnetization gets its own parity
figure, `<task>_parity_magnetization`, on linear axes because $m$ changes sign,
and its own held-out number, `magnetization_relative_l2`; the report's
performance table stays on the density channel. How the two channels enter the
objective is [`training.spin`](training-spin).

(cache-in-memory)=
### `cache.in_memory`

The largest performance setting in the file, and the only one whose effect is
completely invisible from the results.

With it off, **every epoch reopens, decompresses and re-parses every field from
disk**. That produces exactly the same numbers as caching does, and takes
about ten times as long. Measured on a V100 with 115 structures at 32³:
`__getitem__` was 59 % of the training loop, 78.9 s over 483 calls, while the
GPU sat at 2–4 % utilization waiting on `zlib` and `numpy.array` over a text
generator. Turning the cache on took twenty epochs from **168.0 s to 16.3 s**,
with the validation error identical to five decimals.

It is not a GPU setting. The parse is the same on Apple Silicon and on the CPU;
what an accelerator adds is a fast device left idle by it.

**The cache keeps the normalized sample, not only the field.** Removing the
parse left the transforms: every epoch still ran the input and target
normalizations (an asinh over every voxel of every channel) on the CPU, and
on a fast GPU that, not the operator, was the training loop. Measured on an
RTX 5060 Ti with the CHONPS set at 48³, the GPU sat at about 2 % utilization
with every decoded field already in RAM; memoizing the normalized samples took
an epoch from ~60 s to ~8 s and the GPU to ~99 %, for the same arithmetic. The
memo cannot go stale, because a dataset's transforms are properties: assigning
either one empties it, so no sample normalized by a superseded transform
survives the change.

`auto` estimates what the cache will hold (grid points × itemsize × channels
over the input and target fields, three times over: the decoded fields, the
tensors made from them, and the normalized samples), plus the atomic baseline
in δ-density mode, and enables the cache below 4 GiB, logging what it decided
and what it costs:

```text
  field cache         : in RAM, ~61.3 MiB (data.cache.in_memory: auto)
```

Set it to `false` when the process shares its memory with something the
estimate cannot see. What `auto` refuses on its own is the case the old default
was right about: 128³ grids over thousands of materials, where caching silently
would turn a slow run into a dead one.

**Parallel loading is not the alternative it looks like.** See
[`runtime.loader_workers`](num-workers-and-pin-memory).

### `cache.storage` and `cache.compression`

`text` writes each field as a `CHGCAR`-format file, readable by anything that
reads VASP; `hdf5` writes one chunked `fields.h5` per material, 1.38× smaller at
gzip and exact to $10^{-16}$ relative (text round-trips to $10^{-11}$). HDF5
stores carry the magnetization block and the PAW records, but VASP cannot open
one; `poraque-vasp chgcar` writes it back to text. `compression` applies to
HDF5 only (`gzip` at level 4, or `lzf`, with byte-shuffle on) and raises on a
text cache. Both are part of the cache fingerprint.

### `atoms`

The isolated-atom reference, as a `poraque-atoms` JSON or as a directory of
one-atom-in-a-box runs. It supplies three things: the baseline
$\rho_\mathrm{sup}$ of `ext2chg.target: delta`, the structure channels of
[`structure_input`](structure-input), and the PAW records a prediction is written
with. An element the library does not cover fails before the first epoch. The
shipped library covers platinum only.

## `ext2chg` — the density operator

$V_\mathrm{ext} \to \tilde\rho$ and, with the occupancies on, every atom's
augmentation occupancies. Present: trained.

| Key | Default | Meaning |
| --- | --- | --- |
| `target` | `delta` | `delta` learns $\rho-\rho_\mathrm{sup}$ and adds $\rho_\mathrm{sup}$ back on output; `full` learns $\rho$ |
| `structure_input` | `false` | channels describing the structure, appended after $V_\mathrm{ext}$ — [see below](structure-input) |
| `model` | width 64, layers 4, modes 16 | the network — [see below](model-section) |
| `occupancies` | on | the per-atom PAW head — [see below](#paw); `false` trains the grid alone |

### `target`

`delta`, the default, trains on the residual over a superposition of isolated
atoms ($\delta\rho=\rho-\rho_\mathrm{sup}$; the residual is 0.057 in $L^2$ on the
Pt cells). The operator still returns the **full** density: prediction adds
$\rho_\mathrm{sup}$ back, in a fixed order (decode, add the baseline, clip,
normalize), and every physics term sees the absolute density. The atomic
library is copied into the checkpoint. It needs [`data.atoms`](potcar-dir-and-the-gaussian-fallback)
to cover every element. `full` trains on $\rho$ itself and needs no library.

(structure-input)=
### `structure_input`

A block, off by default:

```yaml
ext2chg:
  structure_input:
    superposition: true    # rho_sup, the delta-density baseline, as a channel
    elements: auto         # one superposition per element of the data, or a list
```

The operator is handed $V_\mathrm{ext}$ and must work out from it, voxel by
voxel, *which* atoms surround a point and how far away they are before it can
say anything about the bonding charge there. The potential carries that
information, but only implicitly, in the shape of each ion's local
pseudopotential, and a network of a few layers spends much of its depth
decoding it. This block states it outright, as extra input channels after the
potential:

| Key | Default | Meaning |
| --- | --- | --- |
| `superposition` | `true` | lead with $\rho_\mathrm{sup}$, the superposition of isolated atoms the δ-density baseline already builds |
| `elements` | `auto` | one channel per element (that element's atoms alone, superposed); `auto` is every element of the training data, a list names them, an empty list gives none |

The per-element channels sum to $\rho_\mathrm{sup}$, and an element absent
from a structure is a channel of zeros. All of them share one asinh scale, the
median of $|\rho_\mathrm{sup}|$, because a scale fitted to a channel that is
mostly zeros would be its noise floor.

Every channel is a scalar field built from radial form factors of the
isolated-atom library, so the operator that reads them stays equivariant under
rotations, inversion and translations, and every one is a function of the
geometry alone: at inference they are rebuilt from the structure exactly as the
potential is, and a caller still hands over a potential and nothing else. The
library is therefore required (`data.atoms`, with `target: delta`; the
combination with `full` raises). The checkpoint records the resolved block,
element list included, because an input-channel count alone cannot say which
channels the weights expect; a structure holding an element outside that list
is refused at inference rather than having its atoms silently vanish from the
per-element channels while remaining in the potential.

Measured on 785 CHONPS cells with the same backbone, the channels took the
held-out relative $L^2$ of $\rho$ from 4.4 % to 3.7 % at epoch 10: one
comparison, early in training, so a reason to try the block rather than a
settled number.

(model-section)=
## `model` — the network of each operator

`ext2chg.model` and `chg2tau.model` have the same keys; they differ in their
defaults, because the two maps differ in difficulty:

| Key | `ext2chg` | `chg2tau` | Meaning |
| --- | --- | --- | --- |
| `width` | `64` | `16` | channel width of the Fourier layers |
| `layers` | `4` | `3` | number of Fourier layers |
| `modes` | `16` | `8` | retained Fourier modes per axis (under `band`, the ceiling) |
| `band` | `null` | `null` | `null`: `modes` on every axis; a number in Å⁻¹: the same physical band in every cell |
| `projection_channels` | `64` | `64` | hidden width of the read-out head |
| `activation` | `silu` | `silu` | `silu`, `gelu`, `relu` or `tanh`, in every layer and in the read-out |
| `cell_conditioning` | `32` | `32` | width of the lattice embedding that modulates every layer (FiLM); `false` drops it |
| `equivariant` | `false` | `false` | rotation-equivariant radial kernel — a block, see below |
| `precision` | `float32` | `float32` | `float32` or `float64`, what the operator computes in |

### `width`, `modes`, `layers`

These three set the capacity. The parameter count is dominated by the spectral
weights, $4w^2m^3$ complex numbers per layer for width $w$ and $m$ modes, so
`modes` is by far the expensive knob: doubling it multiplies the spectral
parameters by eight.

`modes` is a *capacity limit*, not a requirement. On a grid too coarse to
supply that many modes, fewer are used automatically, so one configuration
serves materials whose grids differ in size.

### `activation`

`silu`, `gelu`, `relu` and `tanh` are stateless: one fixed function, no
parameters, nothing to learn, and the same function serves the read-out.
`silu` is the default because it won: learnable Kolmogorov–Arnold activations
were tried against it on this project's Pt data and none beat it. The KAN code,
its tests and its comparison configs now live in `experiments/kan_symbolic`;
`model.kan_setup` raises and names the archive.

### `cell_conditioning`

The lattice, through an embedding of this width, modulates each layer's
activations (FiLM). Without it the operator sees the field but not the cell it
lives in, and cannot distinguish two materials whose grids agree while their
lattice vectors do not. Keep it on unless deliberately ablating it. The three
fractional coordinates are appended to the input as extra channels exactly when
the model is *not* equivariant; there is no key for it, since an equivariant
operator cannot take them (below).

### `band`

`null` keeps the lowest `modes` indices on every material. Because the spacing
of reciprocal-lattice points depends on the cell size, that means a *different
physical band* for each material.

A number keeps every mode below that wavevector, retaining
$\lfloor\texttt{band}\,L_i/2\pi\rfloor$ modes along an axis of length $L_i$ (never
more than `modes`), so every material contributes the same band of physics.
Prefer it when cell sizes vary widely.

**Worth trying rather than only tidier.** Measured at LNCC on 115 Materials
Project structures whose cells span 2.29 to 25.87 Å, a fixed band with a
generous value beat the per-axis count by 20 % in **both** seeds, and was
*faster*, since fewer modes are retained. It reproduced in a second regime
sharing no parameter with the first (32³ / `modes` 8 / band 16, and 64³ /
`modes` 24 / band 24; 20.5 % and 20.3 %). The mechanism looks like uniformity
rather than bandwidth: with a fixed count, cells spanning a factor of 11 in
length retain physical bands spanning a factor of 11, and `band` removes that.

It is not the default (one dataset, one architecture), and it is not a free
switch either. **Too small is much worse than `null`**: a band of 6 starved the
model and lost 55 %. The useful value also moves with the regime (16 in the
first study, 24 in the second, with the margin collapsing to 3 % at 32), so it
belongs to the run rather than to the schema. `poraque-train` reports the modes
retained per structure at startup and, for an equivariant run, the band they
span; read those rather than trusting a number copied from another config.

`band` does not size the radial basis of an equivariant model when it is
`null`: it did until 2026-09-04, silently, and a 90-run architecture study at
LNCC was run through a leftover value because nothing in the log said so. The
basis is `equivariant.g_basis`, and only that (below).

### `equivariant`

A block, off by default:

```yaml
ext2chg:
  model:
    equivariant:
      n_radial: 32
```

Constrains the Fourier multiplier to a **real function of $|\mathbf{G}|$
alone**, which is a convolution with a radial kernel and therefore commutes
with every rotation. The coefficients being real also makes it commute with
inversion, so the operator is equivariant under the full $O(3)$ rather than
the $SE(3)$ the construction was asked for.

| Key | Default | Meaning |
| --- | --- | --- |
| `n_radial` | `16` | radial basis functions: the whole capacity of the kernel |
| `g_basis` | `auto` | radius in Å⁻¹ the basis spans: `auto` or a number |

#### `g_basis: auto` — size the basis from the data

The basis spans $[0, g_\mathrm{basis}]$; the band a *material* retains is
$\min_i 2\pi m_i/L_i$, a property of its own cell. Over 115 Materials Project
cells those ran from 1.94 to 20.96 Å⁻¹, so no constant is right for all of
them, and the two ways to miss cost very differently:

| missing | what happens | measured cost |
| --- | --- | --- |
| too **narrow** | modes beyond the basis are clamped onto its outermost function and share one response | none: 71 % of modes clamped moved the error less than the seed spread |
| too **wide** | radial functions sit at radii no mode reaches | **14 % to 43 %**: a basis of 48 against a median band of 31 orphaned six of sixteen functions |

`auto` takes the **median** band the training split retains, resolved before
the model is built so that the number a checkpoint records is a number. Half
the set is then clamped, deliberately: that is the free half. The median is
also what fits the measurements: the four configurations that performed well
had `g_basis` within a few percent of the median band edge, and the one that
did not was 54 % above it. The run says at startup what the basis spans, the
range and median of the band the training structures retain, and the fraction
of retained modes lying beyond the basis.

**Three conditions have to hold together, and each fails silently.** The kernel
being radial is the one everybody thinks of; the other two were found by
measuring, and the schema now guarantees them rather than asking.

*The retained set must be a ball, not a box.* A radial multiplier over a box of
modes is equivariant only under the box's own symmetry group. A cubic cell with
equal mode counts hides this completely, because the box *is* invariant under
the octahedral group; on a tetragonal cell a box takes the error from 3e-7 to
5e-2. The modes of an equivariant model are always a ball (there used to be a
`spherical_cutoff` key; it is gone, as nobody wanted the broken setting).

*No coordinate channels.* The three fractional coordinates are not three scalar
fields (under a rotation they turn into each other) and being absolute
positions they cost translation equivariance too. An equivariant model does not
append them.

*The coefficients must be real,* which the layer enforces.

**The cost is capacity, and it is large.** The dense layer holds
$4C_\mathrm{in}C_\mathrm{out}m_1m_2m_3$ complex numbers; this one holds
$C_\mathrm{in}C_\mathrm{out}R$ real ones: 4 096 against 1 048 576 at
`width 16, modes 8, n_radial 16`.

**Which does not make `n_radial` the knob to raise first.** `modes` costs
*zero* parameters under equivariance (the coefficient count is $C^2R$ whatever
band is retained), so it is a bandwidth knob paid only in FLOPs. On 92
training materials LNCC measured the error rising monotonically with
`n_radial` across 8, 16, 32 and 64, while raising `modes` from 4 to 16 improved
it **2.7× at identical parameter count**, for 36 % more time per epoch. On a
small training set the radial bank is where the excess capacity accumulates
and the bandwidth is free; expect the ordering to invert as the set grows,
which is untested.

**Whether it pays was measured once**, on a V100: 400 epochs on 115 Materials
Project structures at 32³, `width 32 / n_radial 32` reached a validation
relative $L^2$ of 0.047 against 0.173 for a dense model with 28× the
parameters, 4.1× better. The 4× *larger* equivariant arm
(`width 128, n_radial 64`) was **worse**, at 0.059: capacity is not what buys
this, the constraint is. One seed, one dataset, and neither equivariant arm had
converged at 400 epochs, so read it as a reason to try the block rather than as
a number.

```{note}
An equivariant model is slower per epoch despite having two orders of magnitude
fewer parameters, and that is not a bug. The dense layer *stores* one complex
number per retained mode per channel pair and reads it; the radial layer stores
$C^2R$ real numbers and must **reconstruct** the multiplier at every mode from
them, about $R/2$ times the arithmetic at `n_radial: R`. The constraint trades
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

(paw)=
## `ext2chg.occupancies` — the per-atom PAW head

A block, **on by default**:

```yaml
ext2chg:
  occupancies:
    weight: 3.0
    neighbor_min: 0.6
```

`false` trains the grid alone and needs no `POTCAR`; `true` or the bare key is
the defaults.

| Key | Default | Meaning |
| --- | --- | --- |
| `weight` | `1.0` | the per-atom term against the field objective, in training and in the validation score alike |
| `magnetization_weight` | `1.0` | weight of the magnetization record set in the occupancy loss and score, against the charge set's |
| `readout_band` | `auto` | the band in Å⁻¹ read at each atom; `auto` is 95 % of the coarsest Nyquist frequency over the training and validation grids |
| `neighbor_min` | `2.2` | first of ten neighbor-kernel centers in Å, below the fixed last center at 5.8 Å; 0.6 also covers short CHON bonds |

(`runtime.readout_cache_mib` sets the device memory for the read-out's kernels;
it is a hardware setting, below.)

With the head on, the operator maps $V_\mathrm{ext}$ to two things at once: the
pseudo-density on the grid and every atom's **augmentation occupancies**
$\rho(\ell\ell'LM)$, the one-center PAW terms VASP writes after the grid, which
together with the pseudo-density are what an all-electron density is
reconstructed from. They are not a field. There are 138 per atom for
`PAW_PBE Pt` (projector channels d, d, s, s, p, p, in the POTCAR's order) and
one set per density channel, so a spin-polarized dataset keeps the
magnetization's set as well.

The operator is built as `experiments/paw_occupancies` measured it should be
({mod}`poraque.ml.paw`):

**An FNO for the grid**, sized by the section's `model`. Every other `model`
setting (activation, equivariance, band) applies to it unchanged.

**A readout at each atom**, of the latent field and the predicted density,
onto solid-harmonic Gaussians $r^L Y_{LM}(\hat r)\,e^{-r^2/2s_n^2}$. It is the
decoder half of a graph neural operator, with one constraint that was not
optional. The fields are first restricted to one band
$|\mathbf G| \le$ `readout_band` and resampled to a spacing of $\pi/2G_\mathrm{max}$.
A readout of the DFT density integrated on each grid as it came scored 3 % on
nanoparticles it had seen and 184 % on held-out ones, whose grid spacing
differs; over one band, 4.9 %. A grid too coarse to supply the band **raises**,
at training and at inference.

`auto` takes the band from the validation grids as well as the training ones.
The band is a property of the geometry, not of any target, and a held-out cell
coarser than every training cell would otherwise stop the run at its first
validation pass rather than before the first epoch.

The geometric neighbor features retain ten Gaussian channels of width 0.3 Å,
under a 6 Å cutoff. `neighbor_min` changes their first center, preserving the
original 2.2–5.8 Å range by default. The experiment in `experiments/chon_ext2chg`
tests 0.6–5.8 Å for shorter covalent bonds. Their actual centers are checkpoint
buffers; loading an older checkpoint preserves its features.

**An equivariant head per element.** Each L block of the record is one map
shared across M, gated by an MLP over the invariants, with a linear path beside
it. With `model.equivariant` on, the whole operator rotates its predicted
occupancies with the crystal, measured to 7e-15 in double precision. The layout
of each element's record comes from the POTCAR's `Non local Part`, so a dataset
without POTCARs cannot train it.

**An invertible target.** The records train with L = 0 centered and every
(set, L, pair) block scaled to unit RMS, both fitted on the training split,
stored in the checkpoint and inverted at prediction, and every L block weighs
the same in the loss. On raw records the loss would be 99 % the L = 0 block.

**One weight per record set.** A spin-polarized density carries a second set,
the magnetization's, and with every (set, L) block weighed alike its L > 0
blocks (0.005 % of the occupancy norm on the CHONPS set, and as unpredictable
from the potential as the field magnetization is: 98 % held-out error at
L = 1) were most of the occupancy loss and of the score the best epoch is
chosen on. `magnetization_weight` scales that set against the charge set's;
below one, the charge blocks drive both. The weights are normalized to their
mean, so the default of 1.0 is the unweighted loss to the bit. A one-channel
dataset has no second set and ignores the key. It is not
[`training.spin.magnetization_weight`](training-spin), which weights the
magnetization *channel* of the field objective.

The training log gains two columns: `val occ`, the relative RMS of the
held-out occupancies in physical units, and `occ loss`, the training term on the
held-out split. The best epoch is chosen on
`val rel L2 + weight × occ loss`, the objective itself, on held-out data.
`val occ` is reported rather than selected on because 99 % of the records'
squared norm is the charge set's L = 0 block: it reads a few percent while the
L > 0 blocks are tens of percent wrong. After training, each structure's
`occupancy_relative_rms` is written to the metrics JSON beside a baseline (the
training-mean L = 0 record with every L > 0 block zero), and each split is
broken down by record set and L, in the log and under `occupancies`.

A bundle holding this operator gives `poraque-inference --add-paw` a fourth PAW
source, `model`; `auto` prefers it to the averaged table.

The cache carries the records; a cache built before 16 September 2026 dropped
them, so delete it and let it rebuild. `FieldPairDataset.site_records(i)`
returns one structure's (occupancies, record lengths, atomic numbers and
fractional positions, in the file's atom order) for both operators that carry
them: as the target of `ext2chg` and as the input of [`chg2tau`](chg2tau). A
dataset built for either yields, beside the usual `input` and `target`:

| Key | Shape | |
| --- | --- | --- |
| `paw` | `(B, A, S, L)` | occupancies: batch, atoms, record sets, values |
| `paw_mask` | `(B, A, L)` | a real value of a real atom's record |
| `atom_mask` | `(B, A)` | a real atom |
| `species` | `(B, A)` | atomic numbers, 0 for padding |
| `positions` | `(B, A, 3)` | fractional coordinates |

```{warning}
Not yet supported, and refused before the cache: `training.validation.kfold`
(the transform is fitted per split, not per fold), `fine_tuning` (a pretrained
bundle has no occupancy head to adapt), and an f-channel element, whose records
reach L = 6. Each is fine with `occupancies: false`.
```

(chg2tau)=
## `chg2tau` — the kinetic operator

The kinetic energy density from the pseudo-density **and** every atom's
augmentation occupancies, the on-site part the grid cannot hold, entering as
input. Present: trained, on the materials that carry a `TAUCAR`.

| Key | Default | Meaning |
| --- | --- | --- |
| `head` | `plain` | `plain` predicts $\tau$; `pauli` predicts $\tau_\mathrm{vW}[\rho]+s\,\mathrm{softplus}(f)$ — [see below](pauli) |
| `model` | width 16, layers 3, modes 8 | the network — [see above](model-section) |
| `occupancies` | on | the per-atom input; `false` trains on the grid alone |
| `occupancies.channels` | `8` | field channels each atom's records are written into |

```yaml
chg2tau:
  occupancies:
    channels: 8
```

What `ext2chg` predicts is exactly what `chg2tau` reads, which is the route this
project takes to a total energy:
$V_\mathrm{ext}\to(\tilde\rho,\rho^a_{ij})\to\tau$. The data is the same as for
the grid-alone kinetic map: a material needs a `CHGCAR` with its records and a
`TAUCAR`, and the records come from the input density, read by the same
`site_records`.

The operator is the ordinary FNO with more input channels. A
{class}`~poraque.ml.paw.SiteWriter`, the transpose of the `ext2chg` readout,
writes each atom's normalized record as a field: atom-centered solid-harmonic
Gaussians $r^L Y_{LM}(\hat r)\,e^{-r^2/2s^2}$ at three widths (0.25, 0.45 and
0.8 Å), with coefficients from one learned map per element and L, **shared
across M**. A rotation of the atom's occupancies therefore rotates the kernel
it writes, the written field is a scalar field, and the operator reading it
stays equivariant. The field is built in reciprocal space, where it is exact:
by Hobson's theorem the Fourier transform of a solid harmonic times a Gaussian
is the same solid harmonic of $\mathbf G$ times a Gaussian, so each atom is a
phase $e^{-i\mathbf G\cdot\mathbf R_a}$, the periodic sum over images comes
free with the inverse FFT, and nothing depends on the grid but the band it can
hold. The `channels` written fields are concatenated after the density, and an
{class}`~poraque.ml.paw.OccupancyInputModel` hands the whole stack to
{class}`~poraque.ml.fno.FNO3d`. The records are normalized as `ext2chg`
normalizes its target (the same `OccupancyTransform`, fitted on the training
split and stored in the checkpoint) and their layouts come from the POTCAR, so
here too a dataset without POTCARs cannot train it.

The operator cannot predict from a density by itself:
`FieldOperator.predict(field, sites=...)` needs the structure's records, a
dataset's `site_records(i)` or occupancies, species and positions assembled by
hand, and raises without them.

```{warning}
`chg2tau` with its occupancies is refused under `training.validation.kfold` and
under `fine_tuning`, as `ext2chg` is and for the same reasons. The ASE
calculator does not run it yet.
```

`poraque-inference` chains the density operator into the kinetic one. The two
operators are usually trained apart (the density on whatever archive is at
hand, the kinetic energy density on calculations that wrote a `TAUCAR`), so
`--models` may be repeated, and each task is taken from the first bundle that
holds it:

```bash
poraque-inference Pt2/ --models models/chonps_v2/chonps_v2.poraque \
                       --models models/pt_chg2tau/pt_chg2tau.poraque
```

`--kinetic auto` (the default) runs `chg2tau` whenever a bundle holds it, and
then prefers `ext2chg` for the density, so the pseudo-density and the
occupancies `chg2tau` reads come from one model; `--kinetic pseudo2tau` keeps the
density route. The occupancies are the ones the `ext2chg` operator predicts
for the structure in front of it; with no such operator in `--models`, those of a
reference `CHGCAR` beside the structure; with neither, the run stops and says
what to add.

(pauli)=
### `head: pauli`

The model predicts

$$\tau = \tau_\mathrm{vW}[\rho] + s\,\mathrm{softplus}(f_\theta(\rho)),$$

so the Hoffmann–Ostenhof bound $\tau \ge \tau_\mathrm{vW}$ holds by
construction and the network learns only the Pauli term. The scale $s$ is
fitted on the training split and then learned alongside the backbone; neither
is a key. Against a matched baseline it improved every fold, by 18.3 % in mean
relative $L^2$, while training 17.8 % *faster*, because the network no longer
has to re-derive $|\nabla\rho|^2/8\rho$, roughly 31 % of $\tau$. Recommended,
with one limitation of the present version: it is implemented for the grid
alone, so it needs `occupancies: false`, and with the occupancies on it raises.

```yaml
chg2tau:
  head: pauli
  occupancies: false
```

```{warning}
The bound is a theorem for all-electron densities, whereas `CHGCAR` and
`TAUCAR` hold pseudo quantities. Check it on new data before using the head;
the training log reports any reference points that violate it.
```

## `training` — how it is fitted

One section for both operators.

| Key | Default | Meaning |
| --- | --- | --- |
| `epochs` | `500` | passes over the training set |
| `batch_size` | `4` | maximum samples per grid-shape bucket |
| `samples_per_step` | `0` | structures per optimizer step, accumulated across buckets; `0` steps once per batch |
| `optimizer.name` | `adamw` | `adamw`, `adam` or `sgd`, on a cosine schedule with the gradient norm clipped at 1 |
| `optimizer.learning_rate` | `0.002` | step size |
| `optimizer.weight_decay` | `0.0001` | weight decay |
| `loss.norm` | `relative_l2` | `absolute_l2`, `relative_l2`, `absolute_h1` or `relative_h1` |
| `loss.sobolev_weight` | `0.1` | gradient-term weight; read by the two `h1` norms only |
| `spin.mode` | `joint` | `joint` or `per_channel`: how the two channels of a spin-polarized target enter the data term |
| `spin.magnetization_weight` | `1.0` | weight of the magnetization term; `per_channel` only |
| `spin.magnetization_floor` | `0.01` | floor on its denominator, as a fraction of $\lVert\rho\rVert$; `per_channel` only |
| `validation.fraction` | `0.2` | fraction of structures held out; `0` uses every structure |
| `validation.every` | `10` | evaluate and log every N epochs |
| `validation.patience` | `300` | stop after N epochs without improvement; `0` never stops |
| `validation.kfold` | `null` | K: cross-validate over K folds instead of one split |
| `physics.electron_count` | `0.0` | weight of $\int\rho=N$ (density operator) |
| `physics.positivity` | `0.0` | weight of $\rho,\tau\ge0$ |
| `physics.von_weizsacker` | `0.0` | weight of $\tau\ge\tau_\mathrm{vW}$ (kinetic operator) |

### `validation`: the split, the folds, the stopping

There is **one** training protocol and **one** variation on it.

By default a run trains a single model per operator, holding back
`validation.fraction` of the structures for validation. That is all there is to
it: no mode to select, no structures to name.

`kfold: K` swaps that for K-fold cross-validation (on the command line,
`--kfold K`, or a bare `--kfold` for five): each fold trains a fresh model on
*K*−1 groups and scores it on the group held out. It answers a different
question, whether the architecture generalizes, and produces *K* models rather
than one to deploy. `fraction` is ignored, since the folds define the splits.
Setting K to the number of structures gives leave-one-out; it is capped there.
It cannot be combined with `fine_tuning` (each fold trains from scratch) and is
not available to an operator with its occupancies on.

```{tip}
Splitting is always at the *structure* level: whole materials move together. A
voxel-level split would place the same crystal on both sides, and since
neighboring voxels are strongly correlated the score would look excellent
while saying nothing about transfer to a new material.
```

```{important}
`fraction` defaults to **0.2**, so an ordinary run reports a genuine held-out
score and early stopping is active. The trade-off is that the model has then
seen only 80 % of the data. For the final deployable artifact set
`fraction: 0`, accepting that its own metrics become a training fit, and quote
a separate `kfold` run for the generalization number.
```

```{warning}
With `fraction: 0` the reported metrics are **training fit**, carrying no
generalization claim. On the reference dataset they are about four times better
than the cross-validated score. Quote the cross-validated numbers.
```

**`every`** is how often to evaluate and log. Validation is computed *only* on
those epochs, so raising it on a large validation set is a genuine speed-up
rather than only a quieter log. The final epoch always reports, so a run never
ends without a current number.

```text
  progress (every 10 epochs):
    train loss: mean PhysicsInformedLoss per batch   |   val rel L2: held-out error, physical units
          epoch     train loss     val rel L2
    -----------------------------------------
         10/500        0.31745        0.34118
         20/500        0.18902        0.21447  *
```

**`patience`** stops the run after this many epochs without an improvement in
the **validation** error, and restores the best weights seen:

```text
         30/500        0.03173        0.03659  *
         ...
         38/500        0.02249        0.06568
    stopped early at epoch 38: no improvement in 8 epochs (best 0.03659 at epoch 30)

  trained 38/500 epochs in 12.0 s   loss 0.8599 -> 0.0225
```

```{warning}
It needs a validation split (`fraction > 0`). With nothing held out there is
only the training loss, which falls monotonically by construction and so can
never signal that training should stop; asking for patience anyway **warns**
rather than silently doing nothing and leaving you believing the run was
protected.
```

Patience is counted in *epochs* but checked only on the epochs where validation
is computed, so a value below `every` behaves like `every`. The best weights are
restored on exit, so the operator you get is the best one *measured* rather than
merely the last one reached; stopping partway down a degrading curve would
otherwise hand back the degraded model. `history` records `best_epoch`,
`best_error` and `stopped_early`. The `*` marks an epoch that improved on the
best validation score. Only epochs on which validation was actually measured
can do so: a checkpoint is written against a measured score, never an assumed
one.

### `batch_size`

Capped *per grid-shape bucket*. Materials whose grids differ in shape cannot be
stacked into one tensor, so they are grouped by shape and batches drawn within
a group. A bucket holding a single material yields batches of one however large
this is set. The training log prints the buckets it found.

That cap is a real ceiling, not a formality: **above the largest bucket's size
the setting stops meaning anything.** In the 115-structure set measured on a
V100 the largest bucket held 70 materials (≈56 after the split), and `bs=64`
and `bs=128` were the same run: identical peak memory, identical error,
identical time. With the field cache on, throughput responded to the batch up
to 16 and was flat thereafter.

`peak_vram_bytes` and `seconds_per_epoch` in the metrics JSON are what a sweep
over `batch_size` should be read from. The whole run's cost, stage by stage,
closes the log ([the resource profile](resource-profile)).

### `samples_per_step`

One optimizer step per batch weighs every *batch* alike, and so a structure by
the inverse of its batch's size. On a dataset whose grid shapes are mostly
unique that is far from neutral: a structure alone in its bucket moves the
weights as far as four structures sharing the commonest shape do together, at
the default `batch_size: 4`, and the cells that happen to have odd shapes are
fitted at the expense of the rest.

`samples_per_step: N` accumulates gradients over consecutive shape-bucketed
batches until at least N **structures** have been seen, then takes one step.
Each batch's loss is weighted by its size, so every structure carries the
same weight whatever bucket it fell in. Bucketing is untouched (no batch ever
mixes shapes); it is only the step that spans several of them. The gradient of
a step is the mean over its structures, and the epoch's remainder, fewer than
N, takes a step of its own rescaled to remain a mean. `0`, the default, keeps
the one-step-per-batch behavior. The optimizer then takes about one step per N
structures rather than one per batch, which is a change of schedule as well as
of weighting; compare against a baseline at matched steps, not only at matched
epochs.

### `loss`

Four norms, on two named axes:

| `norm` | normalized per sample | gradients in the objective |
| --- | --- | --- |
| `absolute_l2` | no | no |
| `relative_l2` | yes | no |
| `absolute_h1` | no | yes |
| `relative_h1` | yes | yes |

**Relative** divides each sample's error by its own target norm, so materials
whose fields differ by orders of magnitude contribute equally and the value
reads directly as a fraction. **Absolute** does not, so every *voxel* counts
equally and a denser system contributes proportionally more gradient: right
for a set whose materials are genuinely comparable in magnitude, and usually
wrong for a heterogeneous one, which is why `relative_l2` is the default.

**H1** adds `sobolev_weight` times the same error on the spatial gradients.
Worth using when the derivative matters: $\tau_\mathrm{vW}$ depends on
$\nabla\rho$, so gradient noise is amplified in low-density regions. Both
halves are taken in the *same* norm, so the weight is a pure ratio between
values and derivatives rather than also absorbing a change of scale.

**The validation column follows the objective.** The progress table reports
`val abs L2`, `val rel L2`, `val abs H1` or `val rel H1`, whichever is
running, and the number behind it is the objective's own data term evaluated
on the held-out set:

```text
    train loss: mean PhysicsInformedLoss per batch   |   val rel H1: held-out error, physical units
    val rel H1 = rel L2 + 0.1 x the relative L2 of the gradient, matching the
    objective; the final per-structure table below reports plain rel L2
          epoch     train loss     val rel H1
```

That is not only a label. Early stopping and the checkpoint are decided on this
number, and while it read `val rel L2` regardless a gradient-constrained run
was selecting the model that minimized a functional it was not training on. The
four are not comparable with each other, which is why the **per-structure
table at the end of a run reports relative $L^2$ whatever the objective was**.
That is the number to quote, and the one that lets two runs be put side by
side.

(training-spin)=
### `spin`: the two channels of a spin target

A `CHGCAR` written with `ISPIN = 2` carries $(\rho, m)$, and there are two ways
to score it. The block is read only for a two-channel target.

`joint`, the default, is one norm over both channels: each channel's error
divided by the whole $(\rho, m)$ target's norm and combined in quadrature,
which is identically $\lVert\delta(\rho,m)\rVert/\lVert(\rho,m)\rVert$.
`per_channel` divides each channel by **its own** norm and sums them,

$$\mathcal{L} = \mathcal{L}[\rho] + w_m\,\mathcal{L}[m],$$

with $w_m$ = `magnetization_weight`. Because $m \equiv 0$ exactly on a large
fraction of a real set, the magnetization's denominator is floored:
$\sqrt{\lVert m\rVert^2 + (\eta\lVert\rho\rVert)^2}$ with $\eta$ =
`magnetization_floor`. A single-channel target is identical in both modes, bit
for bit, and under `joint` **`magnetization_weight` does nothing**: there is
one term for it to scale.

**The default is the measured one, and the arithmetic misleads about which
should win.** Under `joint` the magnetization's share of the objective is
$3\times10^{-8}$ in the *median* on the 432-cell CHON set, which is the
observation that makes `per_channel` look obviously correct. Measured head to
head on that cache (two seeds, 60 epochs, magnetic cells scored by predicted
over reference $\lVert m\rVert$ and by $m$'s relative $L^2$):

| training set | `joint` | `per_channel` |
|---|---|---|
| 30 magnetic + 30 noise cells | 1.00 / 0.084 | 1.00 / 0.109 |
| 12 magnetic + 100 noise cells (the real 11 % imbalance) | **0.96 / 0.166** | **0.001 / 1.000** |

What trains the magnetization is the *mean* share, not the median: the magnetic
cells carry real weight under `joint`, and that is enough. `per_channel`
instead hands each of the many noise cells a floored term of order one, and a
hundred cells saying "drive $m$ to zero" bury the dozen saying anything else,
so it collapses the magnetization it was introduced to rescue. Reach for it
only when magnetic cells are not a small minority, and measure rather than
assume.

The progress table's `charge` and `mag` columns are the two parts in whichever
mode is running, and the held-out error early stopping reads carries the same
mode, so the checkpoint is selected on the objective rather than on a third
quantity. Not to be confused with `ext2chg.occupancies.magnetization_weight`,
which weights the magnetization *record set* of the per-atom occupancy target.

### `physics`

Weighted penalties on the predicted fields, each dimensionless, so a single
weight can serve a heterogeneous dataset:

| Key | Constraint it penalizes the violation of | Default |
| --- | --- | --- |
| `electron_count` | $\int\rho\,d\mathbf{r} = N$ (density operator) | `0.0` |
| `positivity` | non-negativity of the predicted field | `0.0` |
| `von_weizsacker` | $\tau \ge \tau_\mathrm{vW}$ (kinetic operator) | `0.0` |

```yaml
training:
  physics:
    electron_count: 0.1
```

All three default to zero, so the objective is the plain supervised baseline
until one is chosen deliberately, and there is no switch beside them: **all
zero is off.** A run with every weight at zero skips the machinery altogether,
which is not merely a change of objective. Every constraint acts on *decoded*
fields, so with one live the loop must copy the reference field to the device
on every batch, invert the target transform over the prediction inside the
autograd graph, invert the input transform too, and in delta-density mode copy
the baseline across and add it back twice. Off, all of that is work whose
result is discarded. Measured: 2.4 % of a step on Apple Metal and **6.6 % on a
V100**, where the per-batch copy crosses PCIe.

(The orbital-free stationarity residual, which used to be a fourth weight, is
not configurable: training never read `xc`, so the term ran without
$v_\mathrm{xc}$. `experiments/euler_lagrange` is where it lives.)

#### Charge conservation

`electron_count` is the first constraint to reach for:

$$
\mathcal{L}_{N} = \left\langle \left(
\frac{\int\hat\rho\,d^3r - \int\rho\,d^3r}{\int\rho\,d^3r}
\right)^{2} \right\rangle ,
$$

the squared relative error between the integral of the **predicted** density
and the integral of the **reference** one, the valence electron count. Two
things make it the first constraint to reach for:

*It needs no labels.* The reference density is in every batch, so $N$ is its
own integral. That is what lets it work on an archive that publishes $\rho$ and
nothing else.

For a density operator with the occupancies on, this constrains the total charge
channel of the grid density. The [VASP CHGCAR
convention](https://vasp.at/wiki/index.php/CHGCAR) defines its integral as
`NELECT`; the PAW occupancy records are separate targets and are not summed
into that integral. Magnetization is not constrained by this term. The loss
uses physical units after the inverse transform and restoration of the atomic
baseline in `delta` mode. Using the reference integral also preserves the
electron count of charged training structures.

This is a **soft penalty**, so a positive weight does not guarantee exact
conservation. The validation score includes the weighted mean squared count
error; the log reports `val |dN/N|` before output normalization. The JSON
history contains `train_electron_count_loss`, `val_electron_count_loss`, and
`val_electron_count_error` (a relative fraction, not a percentage).
Conservation is exact at inference, where the density is renormalized to the
valence count (`--no-normalize` opts out).

*It fixes what the data term cannot.* A relative $L^2$ is indifferent to a
percent of charge spread thinly through the interstitial region, and a total
energy is not. Across a chemical space where $N$ runs from ten to ninety
electrons per cell, nothing else pins the scale per material.

A weight of `0.1` sits an order of magnitude below the data term, which is the
intended balance: the physics guides, the data decides. The configuration
under `experiments/chon_ext2chg/configs/` that used a weight of `1.0` is in the
previous schema and a historical record; its weight has not been selected by a
hyperparameter sweep. The training log reports one number, `train loss`: the
**total** objective the optimizer stepped on, data fidelity plus every weighted
constraint. That total is the only quantity comparable between two runs; the
individual terms are reported by the loss unweighted, so they do not sum to it.

```{warning}
Introduce them one at a time against a measured baseline, and only once the
data term has converged: a physics residual evaluated at random initialization
is noise. A badly scaled constraint degrades accuracy while looking principled.
```

Where a constraint can be imposed structurally, prefer that: `chg2tau.head:
pauli` enforces $\tau\ge\tau_\mathrm{vW}$ exactly, whereas `von_weizsacker` only
discourages violating it.

## `fine_tuning` — start from a trained checkpoint

A block, off unless present. Fine-tuning needs the section in the configuration
(the command-line flags `--pretrained` and `--fine-tune-lr` override its values
and do not switch it on):

| Key | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | `models/poraque_models/poraque_models.poraque` | the base bundle to adapt |
| `learning_rate` | `1.0e-5` | much lower than a fresh fit, which is the point |
| `freeze_lifting` | `false` | hold the input embedding fixed and adapt the rest |
| `lora` | `false` | freeze every weight and learn low-rank adapters — a block |
| `lora.rank` | `8` | adapter rank |
| `lora.alpha` | `16.0` | the update is scaled by alpha / rank |
| `lora.dropout` | `0.0` | adapter dropout |

```yaml
fine_tuning:
  checkpoint: models/mp_general/mp_general.poraque
  learning_rate: 1.0e-5
  lora:
    rank: 8
```

The normalizations come from the checkpoint and are never refitted. It cannot
be combined with `training.validation.kfold`, and it is refused for an operator
with its occupancies on (a pretrained bundle has no occupancy layer to adapt).
See {doc}`fine_tuning/index`.

## `runtime` — the hardware

Settings that change how fast a run is, never what it computes.

| Key | Default | Meaning |
| --- | --- | --- |
| `device` | `auto` | `auto`, `cuda`, `cuda:1`, `mps` or `cpu` |
| `strict_device` | `false` | abort instead of falling back to the CPU — see below |
| `distributed` | `auto` | `auto` or `off`; multi-GPU over NCCL when the launcher describes a group |
| `workers` | `auto` | processes that *build* the cache, one material each — [see below](cache-workers) |
| `loader_workers` | `0` | DataLoader worker processes — read `data.cache.in_memory` first |
| `tf32` | `true` | TensorFloat-32 matmul and convolution; CUDA, Ampere and later |
| `readout_cache_mib` | `1024` | device memory for the `ext2chg` read-out's kernels |

(strict-device)=
### `device` and `strict_device`

`device: auto` prefers CUDA, then Apple Metal, then the CPU. An explicit backend
that is unavailable **warns and falls back to the CPU**, so a configuration
written on a machine with a GPU still runs on a laptop.

That is right on a workstation and expensive in a queue. `strict_device: true`
turns the fallback into an error:

```yaml
runtime:
  device: cuda
  strict_device: true
```

**Every HPC configuration should set it.** Without it, a job that cannot reach
its GPU takes its place in the queue, warns into a log nobody reads until
afterwards, and trains on the CPU *inside* the GPU allocation until the wall
clock ends. The refusal names the probable cause (no GPU visible, an empty
`CUDA_VISIBLE_DEVICES`, a driver older than the wheel's CUDA runtime, or a
build carrying no kernels for this architecture) and prints the full device
report into the run's log.

That last case is worth stating on its own, because `torch.cuda.is_available()`
cannot detect it: it answers "is there a driver and a device", not "can this
binary generate code for this device". A `+cu130` wheel on a V100 reports
available and then aborts at the first kernel launch, since CUDA 13 dropped
Volta. Poraquê checks the compute capability against the build's own
`arch_list` and refuses before any of that. See
[NVIDIA GPUs](nvidia-gpus).

`strict_device` applies to `device: auto` as well, which is the default: it
refuses when `auto` resolves to the CPU, and names `device: cpu` as the way to
ask for CPU training deliberately.

(cache-workers)=
### `workers`

How many processes **build** the cache, one material each. `auto` (the
default), a count, or `0` for the single-process build (`--cache-workers` on
the command line). Not to be confused with
[`loader_workers`](num-workers-and-pin-memory), which loads an already-built
cache during training and is the opposite trade.

The build is not the FFT work it looks like. Measured per material on the
97-cell platinum set at `resolution: 48`:

| stage | share of the clock |
| --- | --- |
| Fortran `E17.11` text reading and writing | **86 %** |
| computing V_ext (POTCAR tables + one FFT) | 8 % |
| **spectral downsampling: the FFT** | **2.2 %** |

So a thread count cannot help: there is no BLAS call anywhere in the path, and
every transform goes through `np.fft`, which is single-threaded. A cache job
that asks for eight cores and sets `OMP_NUM_THREADS=8` uses one of them. What
scales is **materials**: `_build_one` is independent per material, and it
already writes every field to `.partial-<pid>` and moves it into place with
`os.replace`, so a pool needs no lock and the result is byte-identical to the
single-process build.

Measured on eight cores, 16 materials at 108³:

| workers | text storage | HDF5 storage |
| --- | --- | --- |
| 1 | 2.764 s/material | 1.515 s/material |
| 2 | 1.462 s (1.89×) | 0.832 s (1.82×) |
| 4 | 0.868 s (3.18×) | 0.518 s (2.92×) |
| 8 | 0.684 s (4.04×) | 0.440 s (3.44×) |

On the full 97-cell set the `--cache-only` cache stage went from **656.9 s to
246.1 s** at the 4 workers `auto` picks there, 2.67×, not 4×, because
per-material time rises 1.49× under memory-bandwidth contention on the 288³
grids. Against the 961.8 s the build cost before the two serial redundancies
(reading only the header for the grid shape, and splitting the text block in
chunks) were removed, it is 4.0×.

`auto` takes the smallest of three numbers and says which bound it hit:

```text
  workers: 4 (auto: 8 usable core(s); 4 within the 8 GiB budget at ~1995 MiB
  each, sized on the largest grid at 23.89 M voxels; 97 material(s))
```

- the cores this process **may** use (`sched_getaffinity`, which is what
  Slurm's `--cpus-per-task` restricts a task to; the node's core count is the
  wrong number on a shared node);
- a memory budget divided by an estimate taken from the **largest** native grid
  in the set. Measured by building one material per process: a 108³ density
  peaked at 498 MiB and a 288³ one at 1558 MiB, so the model is 400 MiB fixed
  plus 70 B per voxel. The largest rather than the average, because every
  worker may be handed the nanoparticle and sizing on the mean is how a build
  goes into swap on the last eleven materials. The budget is **80 % of the
  job's Slurm allocation** where there is one (`SLURM_MEM_PER_NODE`, or
  `SLURM_MEM_PER_CPU` × `SLURM_CPUS_PER_TASK`) and 8 GiB otherwise; a
  constant would pick four workers on a `--cpus-per-task=8 --mem=32G` node and
  idle half the allocation, which is the exact job this exists for;
- the number of materials.

Below **1 M native voxels in total** `auto` stays single-process whatever the
core count: a `spawn`ed pool costs 0.49–0.60 s for each worker to import the
package, and the build runs at 1.03 s per million voxels, so a pool on a toy
set is pure loss. An explicit count ignores that floor.

Materials are handed out **largest first**: their native grids span a factor of
19 in this set and the cost is linear in voxels, so in discovery order the
biggest cells form a tail with idle cores behind them. Rows therefore print in
completion order rather than alphabetically, and `i/N` counts materials
finished. Raise the job's memory request along with its core count: a 288³
density needs ~1.6 GiB per worker.

(num-workers-and-pin-memory)=
### `loader_workers`

`loader_workers` and [`data.cache.in_memory`](cache-in-memory) are
**alternatives, not complements**, and the measurement is unambiguous. Twenty
epochs on 115 structures at 32³, on a V100:

| variant | training time | speedup |
| --- | --- | --- |
| as shipped | 168.0 s | 1.0× |
| `cache.in_memory: true` | **16.3 s** | **10.3×** |
| `loader_workers: 4`, no cache | 120.2 s | 1.4× |
| both | 18.8 s | 8.9× |

The validation error was identical in all four, as it must be. Each worker is a
*process* with a cache of its own, so it re-parses the whole dataset on its
first epoch; the parse the cache removes is paid N times instead of once.
Raise `loader_workers` only when the data genuinely does not fit in RAM, which
is the case `cache.in_memory: false` exists for. Pinned memory is a CUDA
transfer optimization and nothing else, so it is on there and off elsewhere,
and is not a key.

### `tf32`

`tf32` keeps float32's range and drops its mantissa to ten bits inside the
tensor cores. It is a real speedup on Ampere and later, and **exactly nothing**
before it (a V100 has no TF32 path), so it is on by default and costs nothing
where it does not apply. Ignored off CUDA. An equivariance check under it reads
~1e-3 and is not a bug.

### `readout_cache_mib`

The `ext2chg` read-out's kernels depend only on a cell and its grid, so they
are built once per geometry (on the CPU, with SciPy's Bessel functions) and
kept on the device, least recently used evicted first, within this budget. A
training split whose kernels exceed it rebuilds the evicted ones every epoch,
and that rebuild then dominates the epoch: 64 MiB was far below the ~360 MB the
785 CHONPS geometries need at a 6 Å⁻¹ band, and the 1024 MiB default holds them
all. The setting changes how often a kernel is computed, never what it is, so
it is not part of the checkpoint.

(no-torch-compile)=
### There is no `compile` key

There was one, for a version, and it was removed in 26.9.3 after being
measured. Setting `compile`, `compile_mode` or `compile_dynamic` raises and
says why, rather than being accepted and ignored.

**Inductor does not generate code for complex operators.** Every compiled run
said so:

```text
torch/_inductor/lowering.py:1917: UserWarning: Torchinductor does not support
code generation for complex operators. Performance may be worse than eager.
```

`SpectralConv3d`'s weights are complex, and it *is* the Fourier neural
operator. So the one part of the model compilation was invoked to fuse is the
part Inductor hands back to eager, while the wrapper and the graph breaks are
still paid for. That is the mechanism behind the numbers: 150 epochs, two
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
epoch line** and was canceled. A configuration key whose failure mode is a
silent deadlock inside a queue allocation is the hazard
[`strict_device`](strict-device) exists to remove, pointing the other way.

What was *not* measured, and is worth measuring one day: compiling only the
real-valued submodules (`CellEncoder`, FiLM, the pointwise convolutions, the
projection head) and leaving `SpectralConv3d` in eager. That is where the 44.4 %
of elementwise GPU time lives. It should come back as an internal decision
about which submodules to wrap, not as a switch a config file can throw. The
condition for re-opening whole-model compilation is upstream: Inductor gaining
complex-operator codegen.

(distributed)=
### `distributed`

`auto`, the default, forms a `DistributedDataParallel` group over NCCL **when
the environment already describes one**, and does nothing otherwise. A Slurm
step with more than one task, or a `torchrun` launch, is such an environment; a
workstation is not.

It cannot turn a one-process run into a four-GPU one. The launcher decides the
topology and this key decides only whether to believe it, so on a cluster the
setting that matters is in the submission script:

```bash
#SBATCH --ntasks-per-node=4          # one task per GPU
#SBATCH --gres=gpu:4
srun poraque-train --config configs/train_simple.yaml --distributed auto
```

See [Several GPUs, under Slurm](installation.md) for the full script and for
how `MASTER_ADDR` and `MASTER_PORT` are derived from the allocation.

`off` refuses a group inside a multi-task allocation, which is how four
independent single-GPU jobs are run from one submission and how a scaling
result is bisected against a single-device baseline.

Three consequences, all of them visible in the run log:

- The **effective batch size is `batch_size` × the world size**. A four-rank
  run at `batch_size: 10` steps on 40 samples, so it is not the same optimizer
  as the one-rank run it is compared against: halve the learning rate or
  quarter the batch size if the comparison is meant to be of the same
  optimization.
- **Rank 0 alone writes** the checkpoint, the metrics, the figures and the PDF,
  and alone prints.
- **The batches are split, not the samples.** `ShapeBucketSampler` groups
  materials by grid shape so no padding ever reaches the FFT, and a `DataLoader`
  takes a `sampler` or a `batch_sampler` and never both, so the bucketing runs
  first, identically on every rank, and a real `DistributedSampler` partitions
  the resulting *list of batches*. Each rank gets a unique, non-overlapping
  subset, and every rank gets the same *number* of batches, which is not
  tidiness: DDP all-reduces gradients inside each `backward()`, and a rank that
  runs out of batches first leaves the others in a collective that never
  completes.

The collective timeout is long (30 minutes) on purpose, and is not a key. The
first collective happens after every rank has read its prepared cache, and a
cold parallel-filesystem read of a few hundred densities is minutes; a timeout
that fires there reports itself as a NCCL error and sends the reader looking at
the network.

```{note}
There is no `DataParallel` fallback and no Gloo backend. Both would let a
misconfigured run go quietly slower than a single GPU: the first by
replicating in one process, the second by distributing across CPU cores inside
a GPU allocation. Without CUDA the group is refused with a warning naming the
probable cause, and the run continues on one device.
```

## `report` — what is written

| Key | Default | Meaning |
| --- | --- | --- |
| `figures` | `true` | render `plots/`: loss curves, parity over every structure, field slices |
| `pdf` | `true` | typeset `report/`; needs a LaTeX toolchain |
| `format` | `png` | `png`, `pdf` or `svg` |
| `dpi` | `200` | raster resolution for saved figures |
| `raw_data` | `false` | write the numbers behind each figure beside it |

The checkpoint, the log, the metrics JSON and the resolved configuration are
always written when `run.output` is set; `run.output: null` is the one switch
that silences them all. `--no-report` and `--no-plots` turn the PDF and the
figures off from the shell.

### What the report's performance table says

One row per split (`train`, `validation`, `all`) and five columns:

| Column | What it answers |
| --- | --- |
| rel. $L^2$ | The headline error, and the one every other number in this manual is quoted in. |
| rel. $H^1$ | Values *and* gradients. A prediction can match a density pointwise and still be rough, and $\tau_\mathrm{vW}$ depends on $\nabla\rho$, so a small $L^2$ beside a large $H^1$ is a model whose energetics will disappoint. |
| MAE | The typical voxel, in the field's own units, where a squared error is not readable as a quantity. |
| Max. error | The worst voxel. Means and RMS both hide a single catastrophic site, and near a nucleus is exactly where one occurs. |
| $\lvert\Delta N\rvert$ | Conservation. For the density operator this is the electron-count error in electrons, and it is the metric a pointwise loss controls worst. |

The `train` and `validation` rows read against each other **are** the
generalization gap. That is the single most useful thing the table says, and it
is why the splits are rows.

```{note}
The rel. $H^1$ here is the textbook relative Sobolev norm, which has no free
parameter, **not** the $H^1$ objective `loss.norm: relative_h1` minimizes, whose
value depends on `sobolev_weight` and so cannot be compared between two runs.
Every column is computed without consulting the loss, for exactly that reason.

For a two-channel model every number is the **density** channel; the
magnetization is measured separately, as `magnetization_relative_l2` in the
metrics JSON, and the report says so in its caveats.

The table lists *splits*, not structures: it used to print one row per
material, so a 115-material run made six pages of numbers nobody reads
individually. The per-structure figures, and `mse`, `rmse`, `r2`,
`nrmse_range` and `jsd`, which are not typeset, are all in
`log/<name>.json`, which is machine-readable and has no margins.
```

(resource-profile)=
### The resource profile

The last thing `poraque-train` prints is what the run cost:

```text
[RESOURCES]
  stage                               wall time     peak RSS
  ----------------------------------------------------------
  cache                                   0.0 s    222.6 MiB
  ext2pseudo: setup                          0.0 s    228.4 MiB
  ext2pseudo: training                       0.4 s    319.7 MiB
  ext2pseudo: evaluation and figures         1.6 s    500.5 MiB
  ext2pseudo: PDF report                     0.0 s    500.5 MiB
  pseudo2tau: setup                          0.0 s    501.3 MiB
  pseudo2tau: training                       0.0 s    502.9 MiB
  pseudo2tau: evaluation and figures         0.9 s    552.8 MiB
  pseudo2tau: PDF report                     0.0 s    552.8 MiB
  checkpoint                              0.0 s    552.8 MiB
  ----------------------------------------------------------
  total (wall clock)                      3.1 s    552.8 MiB
  device: cpu   outside the stages above: 0.1 s
  peak RSS: the process's high-water resident set so far (getrusage), so it never falls between rows
```

(That is a synthetic smoke run of the grid-alone operators; the shape is what
matters. Like every block of the log it opens with its `[NAME]` header; see
{doc}`output`.) Each stage is timed after a device synchronization, so a GPU
stage is charged for its compute rather than for the kernels it queued. The
memory columns answer the question an allocation is sized from:

- **peak RSS** is the process's high-water resident set, read from the
  operating system rather than sampled, so a spike between two samples cannot
  be missed. It is a running maximum and never falls; the stage where it jumps
  is the one that needed the memory. DataLoader workers are separate processes
  and are reported on a line of their own.
- **peak VRAM**, on CUDA, is `torch.cuda.max_memory_allocated` *within* each
  stage (the peak statistics are reset at every boundary, so the column does
  fall). The summary adds the most the caching allocator reserved, which is the
  number `nvidia-smi` shows and the one an out-of-memory error is about.
- **MPS at end** is what the Metal driver held when the stage ended. MPS
  exposes no peak, so on a Mac this column is a floor, not a maximum.

The summary prints from a `finally`, so a run that raises (an out-of-memory
error, a bad file, a Ctrl-C) still reports every stage it finished and marks
the one it died in `(interrupted)`: the run whose cost most needs explaining is
the one that did not finish. A job killed by a signal is not so lucky. Slurm
ends a job at its time limit with `SIGTERM`, which Python does not turn into an
exception, so that run leaves no summary; the metrics JSON of a finished run is
the record to size the next allocation from. `--cache-only` prints it too, with
the cache as its only stage. The same record is in the metrics JSON under
`resources`; cross-validation times the whole of each operator's folds as one
stage.

### `raw_data`

A figure is an argument someone will later want to make differently: in a
journal's colors, at a journal's size, with two runs on one axis. Re-running
the model to recover the numbers is the expensive way to do that, so with this
on each figure method writes what it drew, beside what it drew:

```text
plots/ext2pseudo_loss_curves.png       plots/ext2pseudo_loss_curves.csv
plots/ext2pseudo_parity.png            plots/ext2pseudo_parity.csv
                                    plots/ext2pseudo_parity_bin_edges.csv
plots/ext2pseudo_s000_field_slice.png  plots/ext2pseudo_s000_field_slice.npz
```

The sidecar shares the **stem** of its figure, so the pairing is a property of
the directory listing rather than of a convention to remember. Tabular figures
get a CSV; the slice figure gets a compressed `.npz`, because what it draws are
three 2D arrays and a CSV of a 108×108 grid is neither smaller nor easier to
read back.

- **The loss curve** is one row per epoch: `epoch`, `train_loss`, and the
  validation column *named for the norm it holds* (`val_rel_l2`, or
  `val_abs_h1` for an `absolute_h1` run). Epochs on which validation was not
  measured leave that cell **empty** rather than interpolated; filling them
  would put points in the file the run never measured. There is no physics
  column: `train_loss` is the total the optimizer stepped on, and the per-term
  breakdown is not recorded.
- **The parity plot** is one row per *occupied* bin: `split`, `reference`,
  `prediction`, `count`, `density`. Long format, because that pivots straight
  back into a `pcolormesh`; occupied bins only, because a 200-bin grid is
  40 000 cells of which a few thousand carry anything. The bin edges go in
  their own file, since a log axis makes them unrecoverable from the centers.
  `count` is what the figure drew, so on log axes it leaves out every sampled
  voxel at which reference or prediction is not positive; the panel titles give
  the sample itself. A spin-polarized run writes the same pair again for $m$,
  as `<task>_parity_magnetization`.
- **The slice figure** stores the three panels as arrays (`reference`,
  `prediction`, `error`) with the color limits the figure used. The error is
  *stored*, not left to be recomputed: a reader who reconstructs it and gets
  something else then knows the file is inconsistent, rather than trusting
  their own subtraction.

It needs `figures`. The data is written by the figure methods as they draw,
which is what guarantees the file and the image show the same numbers rather
than two independent computations of them. PDF reports are assembled in a
temporary directory that is removed afterwards, so no `.tex`, `.aux` or `.log`
files are left behind.

## Worked examples

The short form of the platinum run, both operators, is
`configs/train_simple.yaml`. The ones below change one thing each.

Train the deployable models: nothing held out, the Pauli head for the kinetic
operator (which, for now, means the grid alone):

```yaml
run:
  name: poraque_models
ext2chg:
  occupancies: false
chg2tau:
  head: pauli
  occupancies: false
training:
  epochs: 200
  validation:
    fraction: 0
```

Measure generalization:

```yaml
run:
  name: poraque_models_cv
ext2chg:
  occupancies: false
training:
  validation:
    kfold: 5
```

Two runs over the same data with different physics, kept side by side. The
baseline leaves the weights at zero, which also skips the per-batch decode the
constraints need:

```yaml
run:
  name: agaupt_baseline
ext2chg:
  occupancies: false
```

```yaml
run:
  name: agaupt_charge_conserving
ext2chg:
  occupancies: false
training:
  physics:
    electron_count: 0.1
```

A Materials Project download, which ships no `POTCAR` and no `TAUCAR`: the
density operator on the grid alone, with the library that generated the data.

```yaml
run:
  name: mp_ag_au_pt
data:
  path: data/MP
  potcar_dir: /opt/vasp/potpaw_PBE
ext2chg:
  occupancies: false
```

A fast smoke test:

```yaml
data:
  grid:
    resolution: 20
ext2chg:
  occupancies: false
  model:
    width: 8
    modes: 4
    layers: 2
    projection_channels: 16
training:
  epochs: 10
```

Higher fidelity, if the hardware allows:

```yaml
data:
  grid:
    resolution: 64
ext2chg:
  model:
    width: 32
    modes: 12
    layers: 4
training:
  epochs: 400
  batch_size: 2
```
