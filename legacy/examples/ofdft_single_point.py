# -*- coding: utf-8 -*-
"""OF-DFT single-point energy for a hydrogen atom in a box."""

from ase import Atoms

from poraque.ase import Poraque
from poraque.core import SolverSettings


atoms = Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
atoms.calc = Poraque(
    mode="of",
    grid_shape=(24, 24, 24),
    settings=SolverSettings(max_iter=80, mixing=0.1),
)

energy = atoms.get_potential_energy()
results = atoms.calc.results
print(f"Converged:      {results['converged']}")
print(f"Total energy:   {energy:.6f} eV")
print(f"Electron count: {results['density'].integrate():.6f}")

