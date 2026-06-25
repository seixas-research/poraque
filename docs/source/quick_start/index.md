# Quick Start

Poraquê exposes a single, unified ASE calculator, `poraque.ase.Poraque`. The
method is chosen dynamically with `mode='ks'` (Kohn-Sham) or `mode='of'`
(orbital-free). Positions and cells are given in Ångström and converted
internally to atomic units (Bohr, Hartree); energies are returned in eV.

## The unified ASE calculator

```python
from ase import Atoms
from poraque.ase import Poraque
from poraque.core import SolverSettings

atoms = Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)

# Orbital-free DFT.
atoms.calc = Poraque(
    mode="of",
    grid_shape=(24, 24, 24),
    settings=SolverSettings(max_iter=60, mixing=0.1),
    external_kwargs={"a": 0.8},
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV / Å (numerical)
```

Switch to Kohn-Sham simply by passing `mode="ks"`. The default OF-DFT stack is
`TFvW + Hartree + LDA (Dirac exchange + PW92 correlation) + External`; KS-DFT
defaults to `LDA` exchange-correlation with a Hartree term.

`Poraque` is a fully compliant `ase.calculators.calculator.Calculator`, so it
plugs into ASE workflows such as geometry optimization.

## Periodic crystals with k-points

For periodic solids, sample the Brillouin zone with a Monkhorst-Pack grid and use
pseudopotentials to keep only the valence electrons:

```python
from ase.build import bulk
from poraque.ase import Poraque
from poraque.core import SolverSettings

si = bulk("Si", "diamond", a=5.43)
si.calc = Poraque(
    mode="ks",
    grid_shape=(16, 16, 16),
    kpts=(4, 4, 4),                 # Monkhorst-Pack sampling (ase.dft.kpoints)
    pseudopotentials="auto",        # 4 valence electrons per Si atom
    settings=SolverSettings(max_iter=40, mixing=0.5),
)
print("Total energy (eV):", si.get_potential_energy())
```

## Low-level drivers

The `System`/`Grid`-level drivers behind the calculator are also available
directly when you need full control over the functional stack:

```python
from poraque import OFDFTCalculator
from poraque.functionals import TFvW, Hartree, LDA, External
from poraque.potentials import build_external_potential
from poraque.core import System, Grid

system = System.from_ase(atoms)
grid = Grid((24, 24, 24), system.cell, system.pbc)
v_ext = build_external_potential(grid, system, kind="soft", a=0.8)
result = OFDFTCalculator(
    system, grid, [TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)]
).calculate()
print("Total energy (Hartree):", result.total_energy)
```

## Frozen-Density Embedding

Partition a system into subsystems, each with its own solver (OF or KS), and run
freeze-and-thaw:

```python
import numpy as np
from poraque.core import Grid, System, SolverSettings
from poraque.backends.numpy import NumpyBackend
from poraque.fde import Subsystem, FDEEngine

cell = np.eye(3) * 12.0
grid = Grid((24, 24, 24), cell, pbc=True)

a = System([[5.0, 6.0, 6.0]], [1], cell, electrons=1)
b = System([[7.0, 6.0, 6.0]], [1], cell, electrons=1)

engine = FDEEngine(
    [Subsystem(a, method="ks"), Subsystem(b, method="of")],   # mixed KS-in-OF
    grid, NumpyBackend(),
    settings=SolverSettings(max_iter=8, tolerance=1e-4),
)
result = engine.freeze_and_thaw()
print("Embedded total energy:", result.total_energy)
```
