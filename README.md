<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/poraque?style=for-the-badge&logo=pypi&logoColor=white)](https://pypi.org/project/poraque/)
[![Python](https://img.shields.io/pypi/pyversions/poraque?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![ASE](https://img.shields.io/badge/ASE-3.2x-4B8BBE?style=for-the-badge)](https://wiki.fysik.dtu.dk/ase/)
[![h5py](https://img.shields.io/badge/h5py-HDF5-0E7C86?style=for-the-badge)](https://www.h5py.org/)
[![mp-api](https://img.shields.io/badge/mp--api-Materials%20Project-3B3F8C?style=for-the-badge)](https://pypi.org/project/mp-api/)
[![On-line manual](https://img.shields.io/readthedocs/poraque?style=for-the-badge&logo=readthedocs&logoColor=white&label=Manual)](https://poraque.readthedocs.io/)
[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-poraque-181717?style=for-the-badge&logo=github)](https://github.com/seixas-research/poraque)

</div>

# Poraquê

**Poraquê** is a software framework for machine learning operators between the real-space fields of density functional theory. Given only a crystal geometry, it predicts the charge density and the kinetic energy density — no wavefunctions, no self-consistency cycle.

Two **Fourier Neural Operators** are learned: the first maps the external potential to the charge density (`ext2chg`), the second maps the charge density to the kinetic energy density (`chg2tau`). They are not two unrelated regressions — the first is the Hohenberg–Kohn map, whose existence is a theorem, and the second is the kinetic energy density functional, the missing ingredient of orbital-free DFT.

## Install

```bash
pip install poraque
```

Or from source, which is what you want if you intend to change anything:

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

Python 3.11 or newer. Symbolic distillation is an optional extra:
`pip install "poraque[symbolic]"`. Installing registers seven console commands
— `poraque-train`, `poraque-inference`, `poraque-committee`,
`poraque-active-learning`, `poraque-atoms`, `poraque-vasp` and `poraque-mp` —
which run from any directory once the environment is active. All but the last
are the `main()` of the script of the same name under `scripts/`, so
`python scripts/poraque_train.py` is equivalent to `poraque-train` and needs
nothing installed.

### On an NVIDIA GPU

`pyproject.toml` asks for `torch>=2.0` and says nothing about CUDA, because the
right build depends on the machine and pinning one would break Apple Silicon.
PyPI then serves PyTorch's *default* wheel, which is not always the right one —
and **the wrong choice does not raise, it gives a silent CPU run.**

Two things have to line up: the driver has to be new enough for the wheel's
CUDA runtime (`nvidia-smi`, top right), and the GPU's compute capability has to
appear in `torch.cuda.get_arch_list()`. The second is the one that catches
people out, because CUDA 13 dropped Maxwell, Pascal and **Volta**: a `+cu130`
wheel on a V100 reports `is_available() == True` and then aborts at the first
kernel launch. For a V100, `pip install torch --index-url
https://download.pytorch.org/whl/cu126`.

```bash
python -m poraque.ml.device --check      # three seconds, before the queue
```

It prints the torch build, its CUDA runtime, where it was imported from, the
architectures it carries kernels for, and one line per GPU saying whether this
build can use it — and exits non-zero when the answer is no. In a job script,
`|| exit 1`; in a configuration, `training.strict_device: true`, which aborts
rather than spending a GPU allocation on the CPU.

Two settings matter more than the GPU itself. `data.cache_in_memory` (`auto` by
default) keeps decoded fields in RAM between epochs; without it every epoch
re-parses every file, which measured **10.3× on a V100** with the validation
error unchanged to five decimals. And `training.num_workers` is its
*alternative*, not its complement — added to the cache it makes training slower,
since each worker re-parses the set into a cache of its own.

### On several GPUs, under Slurm

Poraquê trains data-parallel over NCCL when the launcher describes a group, and
on one device when it does not. It cannot invent ranks: **the launcher decides
the topology and `training.distributed` decides only whether to believe it.**

```bash
sbatch scripts/slurm/poraque_ddp.sbatch configs/train.yaml
```

The line that matters is `--ntasks-per-node=4` beside `--gres=gpu:4`.
Requesting four GPUs and launching one task is a perfectly good single-GPU run
that looks, from inside the process, exactly like the one you asked for — which
is why the run log prints the Slurm variables it saw. `MASTER_ADDR` comes from
`SLURM_STEP_NODELIST` and `MASTER_PORT` from `SLURM_JOB_ID`, so nothing is
hard-coded and two jobs on one node cannot collide.

The **batches** are split, not the samples. `ShapeBucketSampler` groups
materials by grid shape so no padding ever reaches the FFT, and a `DataLoader`
takes a `sampler` or a `batch_sampler` and never both — so the bucketing runs
first, identically on every rank, and a real `DistributedSampler` partitions the
resulting *list of batches*. Every rank gets a unique, non-overlapping subset
and the same *number* of batches, which is what stops DDP's per-`backward()`
all-reduce from hanging on the rank that ran out first. The effective batch size
is `batch_size` × the world size, and rank 0 alone prints and writes.

### Faster CPU inference (optional)

Poraquê ships a small C kernel for the spectral contraction — the one part of a
Fourier layer PyTorch runs poorly at batch 1, which is the shape every
prediction has. It compiles itself on first use and needs no configuration.

```bash
python -m poraque.ml.backend --benchmark    # compile now and check it
```

**2–3.4× on whole-model CPU inference** at 24–32³ (less at 96³, where the FFTs
dominate). Optional in every sense: without a C compiler everything works and
falls back to `torch.einsum`. Training is unaffected, accuracy is identical to
float32 rounding, and `PORAQUE_C_BACKEND=0` turns it off. See the User Guide
§1.5 for the details.

## Use

```bash
# 1. train one ext2chg and one chg2tau model on all structures
poraque-train --config configs/train.yaml

# 2. measure generalisation
poraque-train --config configs/train.yaml --kfold --k-folds 5

# 3. predict a structure that has never been computed. --models points at
#    the bundle step 1 wrote: models/<task.name>/<task.name>.poraque
poraque-inference new_structure/ --output predictions/new_structure \
    --models models/pt_w16_m8_l3/pt_w16_m8_l3.poraque

# 4. calibrate: does committee disagreement predict error? Run this on
#    LABELLED data first -- it costs nothing and says whether step 5 means
#    anything. Read the Spearman coefficient. --cache is the tagged
#    directory step 1 built (the tag encodes resolution and potentials).
poraque-committee --models "models/committee_*" --task ext2chg \
    --cache data/cache/res32_potcar --against models/<name>/log/<name>.json

# 5. select: which UNLABELLED structures to compute next. This one spends
#    the DFT budget, so it is the one that needs step 4 to have passed.
poraque-active-learning --models "models/committee_*" --task ext2chg \
    --pool data/pool --select 5
```

Steps 4 and 5 are two halves of one loop, not two ways of doing the same
thing. **`poraque-committee` runs on labelled data** and asks whether the
disagreement measure predicts error at all — it costs nothing and produces a
Spearman coefficient. **`poraque-active-learning` runs on an unlabelled pool**
and turns that measure into a spending decision. Run the first one first.

Every predicted field is written in `CHGCAR` format.

### Configuration

**Every key is optional** — a file states only what the run does differently.
That is why `configs/train.yaml` fits on half a page: of the ~80 settings a
full config carries, it changes a handful.

`configs/train_complete_and_commented.yaml` is the reference: it lists the
settings worth knowing about, each with the reasoning behind it. Read it there
and copy only what you need into your own file.

| File | Purpose |
| --- | --- |
| `configs/train.yaml` | the clean starting point — copy this one |
| `configs/train_complete_and_commented.yaml` | the same run with every choice explained; a reference to read, not to copy |
| `configs/train_materialsproject.yaml` | training on a Materials Project download |
| `configs/train_mixed.yaml` | one operator from several datasets at once |

### One directory per model

Everything a run writes lands under `models/<name>/`:

```text
models/pt_w16_m8_l3/
    pt_w16_m8_l3.poraque     the weights
    log/                  training log, metrics JSON, resolved config
    plots/                loss curves, parity, field slices
    report/               the generated PDF
```

A trained model is not one file: it is weights, plus the numbers that say how
good they are, plus the figures behind those numbers, plus the config that
produced them. Naming the run is the only thing you set — `task.name` — and
two runs with different names cannot collide.

Each part can be switched off independently:

```yaml
output:
  root: models              # null disables all output
  write_log: true
  plot_figures: true
  write_pdf_report: true
  checkpoint: true
```

or from the shell with `--output-root DIR`, `--no-plots`, `--no-report`.

### Precision

Two independent settings, because they control different costs:

```yaml
data:  {precision: float64}    # how fields are stored: float16 | float32 | float64
model: {precision: float32}    # what the operator computes in: float32 | float64
```

`data.precision` is memory — a 160³ field is 16 MB in double and 8 MB in
single. `model.precision: float64` roughly doubles time and memory and is for
checking that a physical result is not a single-precision artefact; it needs
`training.device: cpu`, since Metal has no float64 at all.

Or drive it from ASE:

```python
from ase.build import bulk
from poraque.calculator import Poraque

atoms = bulk("Pt", "fcc", a=3.92, cubic=True)
atoms.calc = Poraque("models/poraque_models.poraque", potcar_dir="POTCARs")

rho = atoms.calc.get_charge_density(atoms)   # ChargeDensity, e/Ang^3
print(rho.electron_count())                  # 44.0, the valence count
rho.write("CHGCAR_pred")                     # readable by any DFT tool

charges = atoms.get_charges()                # net charge per atom, +e
print(atoms.calc.charge_analysis)            # populations, valence, method
```

`potcar_dir` is a POTCAR **library** — one subdirectory per element,
`<potcar_dir>/Pt/POTCAR` — not a single POTCAR file. The entries for whatever
elements the `Atoms` happens to contain are assembled on demand, which is what
lets one calculator serve arbitrary compositions.

Forces and stress are not implemented, so this is single points, not
relaxations.

## Training on the Materials Project

`poraque-mp` turns a **chemical space** — a set of elements — into a local
dataset of charge densities. Size it first; over a chemical space the estimate
is exact, because charge densities are objects in a public S3 bucket and their
sizes are read with `HEAD` requests that transfer no payload:

```bash
# a pure dry run: prints to the console and writes nothing at all
poraque-mp --elements Pt Pd Ni --estimate

# download into ./data/MP, skipping anything over 20 MB
poraque-mp --elements Pt Pd Ni --output data/MP --max-size-mb 20
```

`--output` (or `--outdir`) defaults to the **current directory**, so a command
that writes hundreds of megabytes puts them where you ran it. Files stay
gzipped; Poraquê reads compressed volumetric files in place.

The per-object size cap has **two spellings and they are the same cap**:
`--max-size-mb` and `--max-size-gb`, a decimal factor of 1000 apart, matching
every size this tool prints. One `CHGCAR` is an MB-scale object and a whole
download is a GB-scale one, so which unit reads naturally depends on which of
the two you are thinking about. Passing both is refused rather than resolved.

**The whole database is `--all`, stated explicitly.** It selects every material
the index says has a charge density — one server-side query rather than 2ⁿ
chemical systems — and the search filters still apply under it, which is how a
*subset* of everything is taken. Omitting `--elements` is not a second spelling
of it: a command naming neither is refused, since a forgotten flag would
otherwise become a multi-terabyte transfer.

**`--sample N` is what makes `--all` usable.** It draws N materials at random
and *those* become the working set — with `--estimate` it sizes exactly those
N, and without `--estimate` it downloads them. Everything the run writes into
`--output` describes that subset, summary and CIFs included, so a sampled
directory never carries an index of a set it does not hold.

```bash
# what would 20 random materials from the whole database cost?
poraque-mp --all --estimate --sample 20

# fetch exactly those 20
poraque-mp --all --sample 20 --output data/MP

# the trainable subset of it, as compressed HDF5
poraque-mp --all --num-sites 1 12 --max-size-mb 20 \
    --output data/MP --hdf5 --compression gzip
```

A sampled estimate still **projects** to the whole set — the sample mean times
the number advertised — on its own line and labelled as a projection, because
sizing the database is what you usually want to know *before* deciding to fetch
a piece of it. Both numbers are printed; one alone would be read as the other.

`--seed` decides *which* subset, and it is a set rather than a lottery: the
same seed selects the same materials, so an interrupted sampled download
resumes onto the same subset instead of fetching N more beside it. Sizes are
also strongly right-tailed, so two draws of twenty can differ by tens of
percent, and a projection that moved on every run could not be planned against.

A download is **one directory per material, named by its id** — the shape of a
VASP run, and the shape `data.data_paths` already reads:

```text
data/MP/
    summary.csv  manifest.json  manifest.csv
    structures/mp-124.cif
    mp-124/{CHGCAR.gz, INCAR, KPOINTS, POSCAR, mp.json}
    mp-81/ {CHGCAR.gz, INCAR, KPOINTS, POSCAR, mp.json}
    mp-126/{fields.h5, ...}    # with --hdf5
```

The three VASP inputs reconstruct the calculation, so a material directory is
one VASP can be pointed at. `INCAR` and `KPOINTS` come from the task record the
density belongs to — MP's standard static set fills in when a record cannot be
read, and the file says which. The `POSCAR` comes from the **density's own
header**: that is the geometry it was computed at, and taking it from anywhere
else is how a directory ends up with a structure and a density that disagree.

None of them is training data. The loader records the density and nothing else,
and the external potential stays *computed* — `mp.json` is what keeps the
directory from being mistaken for a VASP run now that it looks like one.
`--no-vasp-inputs` skips the three.

### `--hdf5` keeps everything, and VASP still cannot read it

`--hdf5` writes each density as a chunked, compressed field store instead of a
`CHGCAR`. It is a re-encoding rather than a conversion — pymatgen already holds
ρ·Ω, which is what the store keeps — and it carries **everything the text file
does**: the magnetisation block of a spin-polarised density, and the PAW
augmentation records, the one-centre terms `ICHARG = 1` will not restart
without.

What no store can do is be opened by VASP, so a run needs the density written
back out:

```bash
poraque-vasp chgcar data/MP/mp-124/fields.h5 --output CHGCAR

# or straight into a deck, which converts rather than copies
poraque-vasp energy data/MP/mp-124/fields.h5 --like <run> --copy-density
```

Every mode takes a store where it takes a `CHGCAR` — address one field as
`fields.h5::CHGCAR` when the store holds several. A text source is copied byte
for byte rather than parsed and re-emitted: re-rendering a file that was
already correct can only lose digits or shift a column, and the density block
is a column-positional Fortran read.

Then train. **`data_paths` is the only key that names a dataset**, and a
download goes in it exactly as a run tree does — every entry is a directory of
per-material subdirectories, and nothing says which is which:

```yaml
task: ext2chg              # MP publishes no tau, so chg2tau is not trainable
data:
  data_paths:
    - data/MP              # mp-124/, mp-81/, ...
    - data/vasp/structures # optional: your own runs, pooled into the same set
  potcar_dir: /opt/vasp/potpaw_PBE     # see below
  resolution: 32
```

`TAUCAR` is optional per material, so `type: all` on a mixture trains `ext2chg`
on everything and `chg2tau` on whichever materials have a τ.

```bash
poraque-train --config configs/train_materialsproject.yaml
```

**Set `potcar_dir`.** An MP download has a structure and a density and no
pseudopotentials, and the external potential — the model's *input* — cannot be
built exactly without them. Point at the POTCAR library that generated the data
(MP uses the VASP PBE set) and V_ext is VASP's tabulated local potential,
accurate to a relative 2×10⁻⁵. Leave it out and the Gaussian pseudo-ion model
stands in: on the Ag–Pt–Pt set the two differ by **0.38 relative L2** — they
are different fields, not different roundings of one. Missing entries warn and
fall back per element rather than failing the run.

The structure itself needs nothing extra: a `CHGCAR` carries its own `POSCAR`
in its first lines.

## What is in here

| Path | Contents |
| --- | --- |
| `src/poraque/fields/` | Shared-grid scalar fields, VASP and FHI-aims I/O, pluggable ingestion |
| `src/poraque/data/` | Materials Project downloader, format detection, mixed datasets |
| `src/poraque/ml/` | Fourier neural operators, differentiable DFT operators, training |
| `src/poraque/physics/` | Total-energy components integrated from the predicted fields |
| `src/poraque/calculator.py` | ASE calculator wrapping the whole chain |
| `src/poraque/vis/` | Figures and automatic PDF reports |
| `models/` | One folder per trained model: weights, log, plots, report |
| `scripts/` | Validation, training, inference, experiments |
| `configs/` | YAML run definitions — every key is optional |
| `docs/source/` | Sphinx documentation |
| `docs/notes/` | Design and analysis notes — start at `roadmap.md` |
| `latex/user_guide/` | User guide (how to run it) |
| `latex/technical_guide/` | Technical guide (physics and architecture) |

## Design points

- **The external potential is computed natively.** Poraquê reconstructs it from
  the `POTCAR` tables on any standard VASP output, matching a reference
  potential to a relative 5×10⁻⁵. There is no option to import one: the
  training input must be exactly what inference produces. Where the data ships
  no pseudopotentials — a public density archive, or a run whose `POTCAR` was
  stripped — `potcar_dir` supplies them and the same exact construction is
  used; failing that, a Gaussian pseudo-ion model stands in, and the run says
  which of the two it used.
- **Grids may differ between materials.** One model serves all of them: the
  operator's weights live in Fourier-mode space, and batches are bucketed by
  grid shape.
- **Constraints are structural where possible.** For `chg2tau`,
  τ = τ_vW[ρ] + softplus(·) makes the Hoffmann-Ostenhof bound hold by
  construction rather than by penalty.
- **Resampling is spectral.** Fourier truncation is the exact band-limited
  projection for a plane-wave field; interpolation would alias and shift the
  electron count.
- **CUDA, Apple Metal and CPU**, selected automatically — and the run says
  which, with the compute capability, because a request that quietly fell back
  to the CPU is the most expensive silent failure this package has.
- **It is a grid-based operator learner, not an equivariant message-passing
  network.** That is worth stating because it settles a question people ask:
  NVIDIA's cuequivariance accelerates arithmetic on irreducible representations
  of O(3) — tensor products, spherical harmonics, symmetric contractions — and
  Poraquê has no irreps to accelerate. Profiled by CUDA kernel on a V100, its
  share of a training step is 0.0 %. The Fourier layer is a complex `einsum`
  diagonal in the mode index and dense in channels, which cuequivariance cannot
  express; the NVIDIA library that already accelerates this code is cuFFT,
  reached through `torch.fft`.

## Status

Seventeen platinum supercells — ten 27-atom cells and seven 32-atom cells, spanning
four grid shapes. A single 80/20 split at `32³` working resolution, 300 epochs
with early stopping, whole structures held out (`seed=42`):

| Model | held out | training fit |
| --- | --- | --- |
| `ext2chg` | 0.0379 ± 0.0027 | 0.0209 |
| `chg2tau` | 0.0511 ± 0.0031 | 0.0140 |

R² on the held-out structures is 0.9976 (`ext2chg`) and 0.9953 (`chg2tau`). The
`±` is the spread across the three validation structures, not a
cross-validation error bar — see the caveat below.

The learned kinetic functional beats the analytic orbital-free functionals by a
wide margin on this system — on the same held-out fields Thomas-Fermi scores
1.347 and von Weizsäcker 0.738, so `chg2tau` is **26×** and **14×** better
respectively.

> **What this split can and cannot tell you.** With `valid_fraction = 0.2` and
> 17 structures, the held-out set is three structures — and at `seed=42` all
> three (`struct_011`, `struct_012`, `struct_016`) happen to be 32-atom cells.
> So the headline numbers describe the *harder* subset only, there is no
> held-out 27-atom measurement at all, and three structures is too thin a base
> for a meaningful error bar. Earlier 5-fold cross-validation on the
> 12-structure dataset put 32-atom cells at roughly twice the error of 27-atom
> ones; the number above is consistent with that, and is better than the 0.0445
> that subset scored then, which is what four 32-atom cells in training rather
> than one should buy. Use `--kfold` for a figure that covers every structure.

> Still one element. These numbers measure interpolation between geometries of
> platinum and now, weakly, extrapolation across cell size. They say nothing about
> transfer to other chemistry. Growing the dataset remains the main open item —
> see `docs/notes/roadmap.md`.

**Energies are not there yet.** The total energy is a sum of terms of order
10⁴ eV whose physically relevant variation is a fraction of an eV per atom — a
relative ~10⁻⁴ — and a field-level error of 2×10⁻² cannot survive that
cancellation. The last full measurement, on the earlier 12-structure dataset,
put the true spread at 0.27 eV/atom against an error on predicted differences
of 0.29 eV/atom — a ratio of 1.06 with correlation r ≈ −0.1. **Those figures
have not been re-measured on the current 17-structure dataset**, and the
verdict they support is unchanged either way: an error equal to the signal and
no correlation means the predicted energy ordering carries no information. The
energy module itself is validated against
exact Madelung constants and uniform-electron-gas limits; it is the *fields*
that are not yet accurate enough. See `docs/source/energy/index.md`.


## License

Poraquê is released under the [MIT License](LICENSE). Copyright © 2026 Leandro Seixas Rocha.

## Acknowledgements

We thank financial support from INCT Materials Informatics (Grant No. 406447/2022-5).
