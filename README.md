<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/assets/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)

# Poraquê

**Poraquê learns maps between the three-dimensional scalar fields of
density-functional theory.** Given only a crystal geometry it predicts the
valence charge density and the kinetic energy density — no wavefunctions, no
self-consistency cycle.

```
{POSCAR, INCAR, POTCAR} --analytic--> EXTCAR --Model 1--> CHGCAR --Model 2--> TAUCAR
                                                                                 |
                                                                        integrate v
                                                                              energy
```

The first step is closed-form; only the two field-to-field maps are learned.
They are not unrelated regressions: the first is the **Hohenberg–Kohn map**,
whose existence is a theorem, and the second is the **kinetic energy density
functional**, the missing ingredient of orbital-free DFT.

## Install

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

Python 3.11 or newer. Installing registers five console commands —
`poraque-train`, `poraque-inference`, `poraque-committee`,
`poraque-active-learning` and `poraque-mp` — which run from any directory once
the environment is active. The first four are the `main()` of the script of the
same name under `scripts/`, so `python scripts/poraque_train.py` is equivalent
to `poraque-train` and needs nothing installed.

### Faster CPU inference (optional)

Poraquê ships a small C kernel for the spectral contraction, the one part of a
Fourier layer that PyTorch runs poorly at batch 1. It is **optional**: without
it everything works and simply falls back to `torch.einsum`.

There is nothing to configure. The kernel is compiled on first use — about half
a second, once — and cached in `~/.cache/poraque`. To do that compile now, and
check it:

```bash
python -m poraque.ml.backend --benchmark
```

```text
  compiler  : /usr/bin/cc
  cache dir : /home/you/.cache/poraque
  C spectral backend: loaded from ...poraque_spectral_89feecc6.dylib (pthreads)
  agreement with torch.einsum: 2.76e-07 relative (float32 rounding is ~1e-7)

  one contraction (batch 1, width 32, modes 12^3):
    torch.einsum (4 threads)     4.528 ms
    C, serial                    0.606 ms     7.5x
    C, 4 pthreads                0.200 ms    22.6x
```

All it needs is a C compiler: `xcode-select --install` on macOS, or
`build-essential` on Debian/Ubuntu. Add `--rebuild` after editing the kernel.

| | |
| --- | --- |
| Whole-model inference | 2–3.4× faster at 24–32³, tapering to ~1× at 96³ where the FFTs dominate |
| Threading | pthreads, not OpenMP — a second OpenMP runtime beside PyTorch's raises `OMP: Error #15`. Saturates memory bandwidth at ~4 threads |
| Training | unaffected — the kernel records nothing on the autograd tape, so it is used only under `torch.no_grad()` |
| Accuracy | identical to float32 rounding (~1e-7 relative), checked on every call path by `tests/test_backend.py` |
| To disable | `PORAQUE_C_BACKEND=0` |

`poraque-inference` prints which path it used, so a run that silently fell back
is visible rather than merely slow.

## Use

```bash
# 1. train one ext2chg and one chg2tau model on all structures
poraque-train --write-config configs/train_config.yaml
poraque-train --config configs/train_config.yaml

# 2. measure generalisation
poraque-train --config configs/train_config.yaml --kfold --k-folds 5

# 3. predict a structure that has never been computed
poraque-inference new_structure/ --output predictions/new_structure

# 4. calibrate: does committee disagreement predict error? Run this on
#    LABELLED data first -- it costs nothing and says whether step 5 means
#    anything. Read the Spearman coefficient.
poraque-committee --models "models/committee_*" --task ext2chg \
    --cache data/cache --against logs/kfold.json

# 5. select: which UNLABELLED structures to compute next. This one spends
#    the DFT budget, so it is the one that needs step 4 to have passed.
poraque-active-learning --models "models/committee_*" --task ext2chg \
    --pool data/pool --select 5
```

`poraque-committee` and `poraque-active-learning` are two halves of one loop,
not two ways of doing the same thing. They differ in the data they run on, and
that decides everything else:

| | `poraque-committee` | `poraque-active-learning` |
| --- | --- | --- |
| runs on | a **labelled** dataset — inputs *and* targets | an **unlabelled** pool — inputs only |
| answers | is this measure worth trusting? | which structures do I compute next? |
| produces | a Spearman correlation against the *known* error | a ranking, and a transfer into the training set |
| costs | nothing, the DFT is already done | the DFT runs it selects |

Both share the same committee, the same divergence and the same ranking table,
which is why their output looks alike — they differ in what the number is *for*.

Every predicted field is written in `CHGCAR` format.

### Configuration

**Every key in a config is optional.** A file needs to state only what the run
does differently; everything omitted takes its default. That is why the shipped
examples are short — of the 80 settings a full config carries,
`configs/train_config.yaml` differs in 15.

```bash
poraque-train --write-config /tmp/all.yaml              # every key, with defaults
poraque-train --config mine.yaml --write-config mine.yaml --minimal
```

The second compresses a config you already have down to its differences, and
`--minimal` also works after command-line overrides, so a swept run can be
frozen back into a file:

```bash
poraque-train --config base.yaml --epochs 500 --write-config sweep.yaml --minimal
```

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
atoms.calc = Poraque("models/poraque_models.pfno", potcar="POTCAR")
atoms.get_potential_energy()
print(atoms.calc.components)     # T_s, E_ext, alpha Z, E_H, E_xc, Ewald
```

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
poraque-train --config configs/train_mp_config.yaml
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
| `src/poraque/fields/` | Shared-grid scalar fields, VASP I/O, pluggable ingestion |
| `src/poraque/data/` | Materials Project downloader, format detection, mixed datasets |
| `src/poraque/ml/` | Fourier neural operators, differentiable DFT operators, training |
| `src/poraque/physics/` | Total-energy components integrated from the predicted fields |
| `src/poraque/calculator.py` | ASE calculator wrapping the whole chain |
| `src/poraque/vis/` | Figures and automatic PDF reports |
| `models/` | One folder per trained model: weights, log, plots, report |
| `scripts/` | Validation, training, inference, experiments |
| `configs/` | YAML run definitions — every key is optional; see `--minimal` |
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
