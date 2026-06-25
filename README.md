<h1 align="center" style="margin-top:20px; margin-bottom:50px;">

<a href="https://github.com/seixas-research/poraque" target="_blank" rel="noopener noreferrer">
  <picture>
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_dark.png" media="(prefers-color-scheme: dark)">
    <source srcset="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" media="(prefers-color-scheme: light)">
    <img src="https://raw.githubusercontent.com/seixas-research/poraque/refs/heads/main/logo/logo_light.png" style="height: auto; width: auto; max-height: 100px; " alt="Poraquê logo">
  </picture>
</a>
</h1>

[![License: MIT](https://img.shields.io/github/license/seixas-research/poraque?color=green&style=for-the-badge)](LICENSE)    [![PyPI](https://img.shields.io/pypi/v/poraque?color=red&style=for-the-badge)](https://pypi.org/project/poraque/)

# Poraquê

**Poraquê** is a compact, readable density-functional theory (DFT) code for
electronic-structure calculations. It implements both **Kohn-Sham (KS-DFT)** and
**orbital-free (OF-DFT)** methods behind a single calculator, and integrates
natively with the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/)
so that structures, workflows, and analysis tools from the wider ecosystem work
out of the box.

## Features

- **Unified calculator.** One ASE calculator, `poraque.ase.Poraque`, selects the
  method dynamically with `mode='ks'` (Kohn-Sham) or `mode='of'` (orbital-free).
- **Plane-wave / real-space basis.** Fields and orbitals live on a uniform
  real-space grid whose discrete Fourier transform spans a **plane-wave basis**.
  The kinetic operator is applied exactly in reciprocal space
  (`½ |G + k|²`); local potentials are applied diagonally in real space. The
  basis completeness is set by the grid density, controllable directly
  (`grid_shape`) or through a plane-wave cutoff (`ecut`).
- **Pseudopotentials.** A modular `poraque.pseudopotentials` package provides a
  transparent core–valence split, built-in analytic local pseudopotentials, and
  a reader for a small standard pseudopotential file format
  (`pseudopotentials='auto'` or a per-element mapping).
- **Periodic systems & k-points.** Full periodic boundary conditions with
  Brillouin-zone sampling via Monkhorst–Pack grids built on
  [`ase.dft.kpoints`](https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html)
  (`kpts=(n1, n2, n3)`), folded by time-reversal symmetry.
- **Energies and forces** reported in ASE units (eV, eV/Å), plus frozen-density
  embedding (FDE) drivers for subsystem calculations.

## Installation

Poraquê targets Python ≥ 3.10. For development, clone the repository and install
in editable mode:

```bash
git clone https://github.com/seixas-research/poraque.git
cd poraque
pip install -e .
```

This pulls in the runtime dependencies (NumPy, SciPy, ASE, pandas, matplotlib).
Run the test suite with:

```bash
pytest
```

## Quick start

The calculator is a drop-in ASE `Calculator`: attach it to an `Atoms` object and
ask for energies or forces.

### 1. Gas-phase molecule — an H₂ molecule (orbital-free DFT)

```python
from ase import Atoms

from poraque.ase import Poraque
from poraque.core import SolverSettings

# H2 molecule in a non-periodic box (Ångström).
h2 = Atoms(
    "H2",
    positions=[[2.0, 2.5, 2.5], [3.0, 2.5, 2.5]],
    cell=[5.0, 5.0, 5.0],
    pbc=False,
)

h2.calc = Poraque(
    mode="of",                       # orbital-free DFT
    grid_shape=(24, 24, 24),         # real-space / plane-wave grid
    external_kwargs={"a": 0.8},      # softening of the nuclear potential
    settings=SolverSettings(max_iter=80, mixing=0.1),
)

print(f"Total energy: {h2.get_potential_energy():.6f} eV")
print("Forces (eV/Å):")
print(h2.get_forces())
```

### 2. Bulk crystal — silicon with k-point sampling (Kohn-Sham DFT)

```python
from ase.build import bulk

from poraque.ase import Poraque
from poraque.core import SolverSettings

# Diamond-structure silicon (2-atom primitive cell).
si = bulk("Si", "diamond", a=5.43)

si.calc = Poraque(
    mode="ks",                       # Kohn-Sham DFT
    grid_shape=(16, 16, 16),
    kpts=(4, 4, 4),                  # Monkhorst-Pack Brillouin-zone sampling
    pseudopotentials="auto",         # 4 valence electrons per Si atom
    settings=SolverSettings(max_iter=40, mixing=0.5, tolerance=1e-5),
)

print(f"Total energy: {si.get_potential_energy():.6f} eV")
print(f"Valence electrons: {si.calc.results['density'].integrate():.4f}")
```

More runnable scripts live in [`examples/`](examples/).

## Method selection at a glance

| Argument            | Meaning                                                        |
| ------------------- | -------------------------------------------------------------- |
| `mode`              | `'ks'` (Kohn-Sham) or `'of'` (orbital-free)                    |
| `grid_shape`        | Real-space grid `(Nx, Ny, Nz)`                                 |
| `ecut`              | Plane-wave cutoff (Hartree); sizes the grid automatically      |
| `kpts`              | `(n1, n2, n3)` Monkhorst-Pack grid, or explicit fractional k-points |
| `pseudopotentials`  | `'auto'`, a `{symbol: spec}` mapping, or a `LocalPseudopotential` |
| `xc`                | Exchange-correlation functional (`'lda'` by default, `None` to disable) |
| `charge`            | Net charge of the system                                       |

## License

This is an open source code under the [MIT License](LICENSE).
