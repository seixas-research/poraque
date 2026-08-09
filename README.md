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
[![Python](https://img.shields.io/pypi/pyversions/poraque?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/poraque/)
[![Docs](https://img.shields.io/readthedocs/poraque?style=for-the-badge&logo=readthedocs&logoColor=white)](https://poraque.readthedocs.io/)
[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)

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
`pip install "poraque[symbolic]"`. Installing registers five console commands —
`poraque-train`, `poraque-inference`, `poraque-committee`,
`poraque-active-learning` and `poraque-mp` — which run from any directory once
the environment is active. The first four are the `main()` of the script of the
same name under `scripts/`, so `python scripts/poraque_train.py` is equivalent
to `poraque-train` and needs nothing installed.

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

# 3. predict a structure that has never been computed
poraque-inference new_structure/ --output predictions/new_structure

# 4. calibrate: does committee disagreement predict error? Run this on
#    LABELLED data first -- it costs nothing and says whether step 5 means
#    anything. Read the Spearman coefficient.
poraque-committee --models "models/committee_*" --task ext2chg \
    --cache data/cache --against models/<name>/log/<name>.json

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
That is why `configs/train.yaml` is 19 lines: of the 78 settings a full config
carries, it changes 3.

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
models/au_w16_m8_l3/
    au_w16_m8_l3.pfno     the weights
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

atoms = bulk("Au", "fcc", a=4.08, cubic=True)
atoms.calc = Poraque("models/poraque_models.pfno", potcar_dir="POTCARs")

rho = atoms.calc.get_charge_density(atoms)   # ChargeDensity, e/Ang^3
print(rho.electron_count())                  # 44.0, the valence count
rho.write("CHGCAR_pred")                     # readable by any DFT tool

charges = atoms.get_charges()                # net charge per atom, +e
print(atoms.calc.charge_analysis)            # populations, valence, method
```

`potcar_dir` is a POTCAR **library** — one subdirectory per element,
`<potcar_dir>/Au/POTCAR` — not a single POTCAR file. The entries for whatever
elements the `Atoms` happens to contain are assembled on demand, which is what
lets one calculator serve arbitrary compositions.

Forces and stress are not implemented, so this is single points, not
relaxations.

## Training on the Materials Project

`poraque-mp` turns a **chemical space** — a set of elements — into a local
dataset of charge densities. Size it first; the estimate is exact, because
charge densities are objects in a public S3 bucket and their sizes are read
with `HEAD` requests that transfer no payload:

```bash
# a pure dry run: prints to the console and writes nothing at all
poraque-mp --elements Ag Au Pt --estimate

# download into ./data/MP, skipping anything over 20 MB
poraque-mp --elements Ag Au Pt --output data/MP --max-size-mb 20
```

`--output` (or `--outdir`) defaults to the **current directory**, so a command
that writes hundreds of megabytes puts them where you ran it. Files stay
gzipped; Poraquê reads compressed volumetric files in place.

Then train. `train_paths` is a list, so a download can be trained on alone or
beside your own runs:

```yaml
task: ext2chg              # MP publishes no tau, so chg2tau is not trainable
data:
  train_paths:
    - data/MP              # a bulk archive of standalone CHGCARs
    - data/vasp            # optional: your own calculation directories
  potcar_dir: /opt/vasp/potpaw_PBE     # see below
  resolution: 32
```

```bash
poraque-train --config configs/train_materialsproject.yaml
```

**Set `potcar_dir`.** An MP download has a structure and a density and no
pseudopotentials, and the external potential — the model's *input* — cannot be
built exactly without them. Point at the POTCAR library that generated the data
(MP uses the VASP PBE set) and V_ext is VASP's tabulated local potential,
accurate to a relative 2×10⁻⁵. Leave it out and the Gaussian pseudo-ion model
stands in: on the Ag–Au–Pt set the two differ by **0.38 relative L2** — they
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
- **CUDA, Apple Metal and CPU**, selected automatically.

## Status

Seventeen gold supercells — ten 27-atom cells and seven 32-atom cells, spanning
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
> gold and now, weakly, extrapolation across cell size. They say nothing about
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
