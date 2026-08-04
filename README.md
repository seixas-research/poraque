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

Python 3.11 or newer.

## Use

```bash
# 1. check the data and the external-potential reconstruction
python scripts/validate_vasp_data.py --fit-sigma --form-factor

# 2. train one ext2chg and one chg2tau model on all structures
python scripts/run_train.py --write-config configs/train_config.yaml
python scripts/run_train.py --config configs/train_config.yaml

# 3. measure generalisation
python scripts/run_train.py --config configs/train_config.yaml --kfold --k-folds 5

# 4. predict a structure that has never been computed
python scripts/run_eval.py new_structure/ --output predictions/new_structure
```

Every predicted field is written in `CHGCAR` format and opens in VESTA.

Or drive it from ASE:

```python
from ase.build import bulk
from poraque.calculator import Poraque

atoms = bulk("Au", "fcc", a=4.08, cubic=True)
atoms.calc = Poraque("models/poraque_models.pth", potcar="POTCAR")
atoms.get_potential_energy()
print(atoms.calc.components)     # T_s, E_ext, alpha Z, E_H, E_xc, Ewald
```

Forces and stress are not implemented, so this is single points, not
relaxations.

## What is in here

| Path | Contents |
| --- | --- |
| `src/poraque/fields/` | Shared-grid scalar fields, VASP I/O, pluggable ingestion |
| `src/poraque/ml/` | Fourier neural operators, differentiable DFT operators, training |
| `src/poraque/physics/` | Total-energy components integrated from the predicted fields |
| `src/poraque/calculator.py` | ASE calculator wrapping the whole chain |
| `src/poraque/vis/` | Figures and automatic PDF reports |
| `scripts/` | Validation, training, inference, experiments |
| `configs/` | YAML run definitions |
| `docs/source/` | Sphinx documentation |
| `docs/notes/` | Design and analysis notes — start at `roadmap.md` |
| `latex/user_guide/` | User guide (how to run it) |
| `latex/technical_guide/` | Technical guide (physics and architecture) |

## Design points

- **The external potential is computed natively.** Poraquê reconstructs it from
  the `POTCAR` tables on any standard VASP output, matching a reference
  potential to a relative 5×10⁻⁵. There is no option to import one: the
  training input must be exactly what inference produces.
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

Twelve gold supercells — ten 27-atom cells and two 32-atom cells, spanning four
grid shapes. 5-fold cross-validation, whole structures held out:

| Model | relative L² | R² |
| --- | --- | --- |
| `ext2chg` | 0.0245 ± 0.0130 | 0.9987 |
| `chg2tau` | 0.0444 ± 0.0271 | 0.9951 |

The learned kinetic functional beats the analytic orbital-free functionals by a
wide margin on this system — Thomas-Fermi scores 1.348 and von Weizsäcker
0.738 on the same fields, so `chg2tau` is **30×** and **17×** better
respectively.

### Cell size dominates the error

The aggregate above hides the only interesting thing in it. Split by cell size:

| Subset | `ext2chg` | `chg2tau` |
| --- | --- | --- |
| 27-atom (10 structures) | 0.0205 ± 0.0064 | 0.0355 ± 0.0069 |
| 32-atom (2 structures) | 0.0445 ± 0.0182 | 0.0894 ± 0.0420 |

Held out, a 32-atom cell is **2.2–2.5× harder** than a 27-atom one. That is the
first transfer measurement this project has: with only two examples of that
cell size, holding one out leaves a single sibling, and the operator has to
extrapolate to a grid shape it has barely seen.

Within a familiar cell size, more data helps monotonically — the 27-atom
numbers are the best yet recorded:

| Dataset | `ext2chg` | `chg2tau` |
| --- | --- | --- |
| 5 structures | 0.0295 ± 0.0025 | 0.0525 ± 0.0031 |
| 9 structures | 0.0219 ± 0.0046 | 0.0400 ± 0.0069 |
| 12 structures, 27-atom subset | 0.0205 ± 0.0064 | 0.0355 ± 0.0069 |

> The three rows differ in protocol as well as in data — the 12-structure run
> holds out 3 structures per fold against 1 for the 5-structure run, and uses
> early stopping, which the earlier runs predate. Read the trend, not the
> third decimal.

> Still one element. These numbers measure interpolation between geometries of
> gold and now, weakly, extrapolation across cell size. They say nothing about
> transfer to other chemistry. Growing the dataset remains the main open item —
> see `docs/notes/roadmap.md`.

**Energies are not there yet.** The total energy is a sum of terms of order
10⁴ eV whose physically relevant variation is a fraction of an eV per atom — a
relative ~10⁻⁴ — and a field-level error of 2×10⁻² cannot survive that
cancellation. Across the twelve structures the true spread is 0.27 eV/atom and
the error on predicted differences is 0.29 eV/atom, a ratio of 1.06 with
correlation r ≈ −0.1. That is an improvement on the previous 3× ratio, but an
error equal to the signal and no correlation still means the predicted energy
ordering carries no information. The energy module itself is validated against
exact Madelung constants and uniform-electron-gas limits; it is the *fields*
that are not yet accurate enough. See `docs/source/energy/index.md`.

## License

MIT. See [LICENSE](LICENSE).
