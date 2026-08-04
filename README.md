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
python scripts/train_fno.py --write-config configs/train_config.yaml
python scripts/train_fno.py --config configs/train_config.yaml

# 3. measure generalisation
python scripts/train_fno.py --config configs/train_config.yaml --kfold --k-folds 5

# 4. predict a structure that has never been computed
python scripts/infer_fno.py new_structure/ \
    --ext2chg models/ext2chg.pt --chg2tau models/chg2tau.pt \
    --output predictions/new_structure
```

Every predicted field is written in `CHGCAR` format and opens in VESTA.

## What is in here

| Path | Contents |
| --- | --- |
| `src/poraque/fields/` | Shared-grid scalar fields, VASP I/O, pluggable ingestion |
| `src/poraque/ml/` | Fourier neural operators, differentiable DFT operators, training |
| `src/poraque/vis/` | Figures and automatic PDF reports |
| `scripts/` | Validation, training, inference, experiments |
| `configs/` | YAML run definitions |
| `docs/source/` | Sphinx documentation |
| `docs/notes/` | Design and analysis notes — start at `roadmap.md` |
| `latex/user_guide/` | User guide (how to run it) |
| `latex/technical_guide/` | Technical guide (physics and architecture) |

## Design points

- **No modified VASP required.** The external potential is reconstructed from
  the `POTCAR` tables, matching a reference `EXTCAR` to a relative
  5×10⁻⁵.
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

Measured on five gold supercells, 5-fold cross-validation with whole structures
held out:

| Model | relative L² | R² |
| --- | --- | --- |
| `ext2chg` | 0.0295 ± 0.0025 | 0.9986 |
| `chg2tau` | 0.0525 ± 0.0031 | 0.9950 |

The learned kinetic functional beats Thomas-Fermi and von Weizsäcker by roughly
an order of magnitude on this system.

> These numbers measure interpolation between nearby geometries of a single
> element. They say nothing about transfer to other chemistry. Growing the
> dataset is the main open item — see `docs/notes/roadmap.md`.

## License

MIT. See [LICENSE](LICENSE).
