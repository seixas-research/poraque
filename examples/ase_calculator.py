# -*- coding: utf-8 -*-
"""Use Poraquê as a standard ASE calculator (energy and numerical forces).

The grid is sized from a user-supplied plane-wave kinetic-energy cutoff
(``ecut``); ``Grid.from_ecut`` makes the cutoff -> grid_shape mapping explicit.
The full calculation log is printed to stdout by default (verbose=True).
"""

from ase import Atoms

from poraque.ase import Poraque
from poraque.core import SolverSettings

atoms = Atoms("H2",
              positions=[[2.0, 2.5, 2.5], [3.0, 2.5, 2.5]],
              cell=[5, 5, 5],
              pbc=True)

ecut = 8.0  # Hartree

atoms.calc = Poraque(
    mode="of",
    ecut=ecut,                  # grid generated automatically from the cutoff
    settings=SolverSettings(max_iter=60, mixing=0.1),
    external_kwargs={"a": 0.8},
)

print(f"Potential energy: {atoms.get_potential_energy():.6f} eV")
print("Forces (eV/A):")
print(atoms.get_forces())

