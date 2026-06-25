# Quick Start

All Poraquê drivers work directly with ASE `Atoms` objects. Positions and cells
are given in Ångström and converted internally to atomic units (Bohr, Hartree);
energies are returned in eV through the ASE calculator and in Hartree through the
low-level drivers.

## Orbital-Free DFT single point

```python
from ase import Atoms
from poraque import run_ofdft

atoms = Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
result = run_ofdft(atoms, grid_shape=(24, 24, 24))

print("Total energy (Hartree):", result.total_energy)
print("Converged:", result.converged)
print("Components:", result.energy_components)
```

The default OF-DFT stack is `TFvW + Hartree + LDA (Dirac exchange + PW92
correlation) + External`. You can supply your own functional list:

```python
from poraque.functionals import TFvW, Hartree, LDA, External
from poraque.potentials import build_external_potential
from poraque.core import System, Grid

system = System.from_ase(atoms)
grid = Grid((24, 24, 24), system.cell, system.pbc)
v_ext = build_external_potential(grid, system, kind="soft", a=0.8)
result = run_ofdft(atoms, functionals=[TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)])
```

## Kohn-Sham DFT single point

```python
from poraque import run_ksdft

result = run_ksdft(atoms, grid_shape=(32, 32, 32), mixing=0.3)
print("KS total energy:", result.total_energy)
print("Eigenvalues:", result.state.eigenvalues)
```

## The ASE calculator

```python
from poraque.ase import PoraqueASE
from poraque.core import SolverSettings

atoms.calc = PoraqueASE(
    grid_shape=(24, 24, 24),
    settings=SolverSettings(max_iter=60, mixing=0.1),
    external_kwargs={"a": 0.8},
)

energy = atoms.get_potential_energy()   # eV
forces = atoms.get_forces()             # eV / Å (numerical)
```

`PoraqueASE` is a fully compliant `ase.calculators.calculator.Calculator`, so it
plugs into ASE workflows such as geometry optimization.

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
