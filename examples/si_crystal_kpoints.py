# -*- coding: utf-8 -*-
"""Periodic KS-DFT for bulk silicon with Monkhorst-Pack k-point sampling.

A diamond-structure silicon crystal is built with ASE, a local pseudopotential
keeps only the four valence electrons per atom, and the Brillouin zone is
sampled with a Monkhorst-Pack grid.
"""

from ase.build import bulk

from poraque.ase import Poraque
from poraque.core import SolverSettings


def main():
    si = bulk("Si", "diamond", a=5.43)  # 2-atom primitive cell

    si.calc = Poraque(
        mode="ks",
        grid_shape=(16, 16, 16),
        kpts=(4, 4, 4),                # Monkhorst-Pack Brillouin-zone sampling
        pseudopotentials="auto",       # 4 valence electrons per Si atom
        settings=SolverSettings(max_iter=40, mixing=0.5, tolerance=1e-5),
    )

    energy = si.get_potential_energy()
    results = si.calc.results
    print(f"Converged:    {results['converged']}")
    print(f"Total energy: {energy:.6f} eV")
    print(f"Electrons:    {results['density'].integrate():.6f}")  # 8 valence e-


if __name__ == "__main__":
    main()
