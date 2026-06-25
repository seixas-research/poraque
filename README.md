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
electronic-structure calculations. It implements **Kohn-Sham (KS-DFT)**,
**orbital-free (OF-DFT)**, and **Frozen-Density Embedding (FDE / subsystem DFT)**
behind a single calculator, and integrates natively with the
[Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) so that
structures, workflows, and analysis tools from the wider ecosystem work out of
the box.

Unlike a traditional monolithic DFT package, Poraquê is built around **one shared
real-space / plane-wave numerical core** that is reused by every method. That
design is what makes its three defining capabilities possible: a *native ASE
calculator* as the only user-facing API, the ability to *seamlessly bridge
orbital-free and Kohn-Sham regions in a single system* via freeze-and-thaw
embedding, and a research focus on *machine-learning-derived kinetic energy
density functionals (ML-KEDFs)*.

## What makes Poraquê different

Traditional DFT codes (Quantum ESPRESSO, VASP, ABINIT, GPAW, …) are powerful but
typically commit to a single electronic-structure paradigm (almost always
Kohn-Sham), expose a bespoke input-file language, and treat the kinetic energy
functional as fixed. Poraquê is deliberately different:

- **Native ASE integration, not a wrapper.** The *only* public entry point is a
  standard ASE `Calculator` (`poraque.ase.Poraque`). There is no custom input
  language: you build an `Atoms` object and ask for `get_potential_energy()` /
  `get_forces()`. Every ASE tool — builders, optimizers, equation-of-state,
  databases, NEB — works unchanged. The method is chosen with a single keyword,
  `mode='ks'` or `mode='of'`.
- **A hybrid OF ⇄ KS engine via Freeze-Embedding.** Because OF-DFT and KS-DFT
  share the same grid, functionals, and Hartree/XC machinery, Poraquê can
  partition a system and treat different regions with *different* methods at the
  same time — an accurate Kohn-Sham *active* subsystem embedded in a cheap
  orbital-free *environment* (KS-in-OF), or any combination — coupled through a
  non-additive kinetic + electrostatic + XC embedding potential and relaxed with
  freeze-and-thaw cycles. This OF/KS bridging is awkward or impossible in codes
  built around a single paradigm.
- **Machine-learning-derived KEDFs as a first-class research target.** The
  orbital-free kinetic energy is treated as a *pluggable, learnable* object, not
  a hard-coded term. The functional interface (energy + functional derivative)
  is the same one used by Thomas-Fermi and von Weizsäcker, so a symbolic-
  regression formula or a CNN trained on electron-density slices can drop
  straight into the self-consistent minimizer.
- **One readable core, in plain NumPy/SciPy.** The reference implementation
  favors clarity over micro-optimization, with a clean `Grid` / `System` /
  `Density` / `Result` data model and a pluggable backend layer — a code you can
  actually read, extend, and use to prototype new physics.

| | Traditional KS-DFT codes | **Poraquê** |
| --- | --- | --- |
| Primary interface | Custom input files | **Native ASE `Calculator`** |
| Methods | Kohn-Sham only (usually) | **KS-DFT + OF-DFT + FDE, one API** |
| OF/KS in one system | Not supported | **KS-in-OF / OF-in-KS embedding** |
| Kinetic functional | Fixed | **Pluggable, ML-derived KEDFs** |
| Pseudopotentials | `.upf` (norm-conserving/PAW) | **`.upf` (PseudoDojo/QE) + analytic** |
| Codebase | Large Fortran/C++ | **Compact, readable Python** |

## Features

- **Unified calculator.** One ASE calculator, `poraque.ase.Poraque`, selects the
  method dynamically with `mode='ks'` (Kohn-Sham) or `mode='of'` (orbital-free).
- **Plane-wave basis (mandatory for KS-DFT) with automatic grid generation.**
  Fields and orbitals live on a uniform real-space grid whose discrete Fourier
  transform spans a **plane-wave basis**. The kinetic operator is applied
  exactly in reciprocal space (`½ |G + k|²`); local potentials are applied
  diagonally in real space. Set a plane-wave cutoff with `ecut` and the
  real-space grid is **sized automatically** to resolve it (Nyquist condition);
  a manually supplied `grid_shape` is safely overridden by the optimized grid
  (with a warning).
- **Pseudopotentials, including `.upf`.** A modular `poraque.pseudopotentials`
  package provides a transparent core–valence split, built-in analytic local
  pseudopotentials, *and* a reader for **`.upf` (Unified Pseudopotential Format)**
  files compatible with PseudoDojo and Quantum ESPRESSO. A bundled registry maps
  the chosen exchange-correlation functional to the matching file
  (`pseudopotentials='upf'` picks **LDA or PBE** pseudopotentials to match `xc`).
- **Transparent energy accounting.** Results carry a strictly additive energy
  decomposition — total, kinetic, external/ionic, Hartree, exchange-correlation,
  and nonlocal terms — and an optional `verbose=True` mode prints the generated
  grid, the material structure (cell, positions, PBC), the per-step SCF
  convergence, and the final breakdown to standard output.
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
ask for energies or forces. The full calculation log — generated grid, material
structure, step-by-step SCF convergence, and the final energy decomposition — is
**printed to standard output by default** (pass `verbose=False` to silence it).

You control the plane-wave grid in one of two equivalent ways: pass an explicit
`grid_shape=(Nx, Ny, Nz)`, or pass a plane-wave kinetic-energy cutoff `ecut`
(Hartree) and let Poraquê size the grid automatically. `Grid.from_ecut` exposes
the exact mapping so you can see the dimensions a cutoff produces.

### 1. Gas-phase molecule — an H₂ molecule (orbital-free DFT)

```python
from ase import Atoms

from poraque.ase import Poraque
from poraque.core import Grid, SolverSettings, System

# H2 molecule in a non-periodic box (Ångström).
h2 = Atoms(
    "H2",
    positions=[[2.0, 2.5, 2.5], [3.0, 2.5, 2.5]],
    cell=[5.0, 5.0, 5.0],
    pbc=False,
)

# A user-supplied kinetic-energy cutoff dynamically determines the grid.
ecut = 8.0  # Hartree
system = System.from_ase(h2)
derived = Grid.from_ecut(system.cell, ecut, pbc=system.pbc)
print(f"Ecut = {ecut} Ha  ->  grid_shape = {derived.shape}")  # e.g. (26, 26, 26)

h2.calc = Poraque(
    mode="of",                       # orbital-free DFT
    ecut=ecut,                       # grid generated automatically from the cutoff
    external_kwargs={"a": 0.8},      # softening of the nuclear potential
    settings=SolverSettings(max_iter=80, mixing=0.1),
)
# Equivalent explicit form: Poraque(mode="of", grid_shape=(26, 26, 26), ...)

print(f"Total energy: {h2.get_potential_energy():.6f} eV")  # also prints the log
print("Forces (eV/Å):")
print(h2.get_forces())
```

### 2. Bulk crystal — silicon with k-point sampling (Kohn-Sham DFT)

```python
from ase.build import bulk

from poraque.ase import Poraque
from poraque.core import Grid, SolverSettings, System

# Diamond-structure silicon (2-atom primitive cell).
si = bulk("Si", "diamond", a=5.43)

# Size the plane-wave grid from a cutoff (KS-DFT always uses the plane-wave basis).
ecut = 6.0  # Hartree
derived = Grid.from_ecut(System.from_ase(si).cell, ecut)
print(f"Ecut = {ecut} Ha  ->  grid_shape = {derived.shape}")

si.calc = Poraque(
    mode="ks",                       # Kohn-Sham DFT (plane-wave basis, mandatory)
    ecut=ecut,                       # grid_shape derived automatically; see above
    kpts=(4, 4, 4),                  # Monkhorst-Pack Brillouin-zone sampling
    pseudopotentials="auto",         # 4 valence electrons per Si atom
    settings=SolverSettings(max_iter=40, mixing=0.5, tolerance=1e-5),
)
# Or specify the grid directly: Poraque(mode="ks", grid_shape=(16, 16, 16), ...)

print(f"Total energy: {si.get_potential_energy():.6f} eV")
print(f"Valence electrons: {si.calc.results['density'].integrate():.4f}")
```

> If you pass **both** `ecut` and `grid_shape`, the cutoff wins: the manual grid
> is overridden by the automatically optimized one and a warning is logged.

More runnable scripts live in [`examples/`](examples/).

## Method selection at a glance

| Argument            | Meaning                                                        |
| ------------------- | -------------------------------------------------------------- |
| `mode`              | `'ks'` (Kohn-Sham) or `'of'` (orbital-free)                    |
| `basis`             | Single-particle basis; `'pw'` (plane waves), **mandatory for KS-DFT** |
| `grid_shape`        | Real-space grid `(Nx, Ny, Nz)` (overridden when `ecut` is set) |
| `ecut`              | Plane-wave cutoff (Hartree); sizes the grid automatically      |
| `kpts`              | `(n1, n2, n3)` Monkhorst-Pack grid, or explicit fractional k-points |
| `pseudopotentials`  | `'auto'` (analytic), `'upf'` (bundled UPF), a `{symbol: spec}` mapping, or a `LocalPseudopotential` |
| `pseudo_functional` | Functional for UPF selection (`'LDA'`/`'PBE'`); inferred from `xc` if unset |
| `xc`                | Exchange-correlation functional (`'lda'` by default, `None` to disable) |
| `charge`            | Net charge of the system                                       |
| `verbose`           | Print grid, structure, SCF convergence, and energy decomposition |

## License

This is an open source code under the [MIT License](LICENSE).
