# -*- coding: utf-8 -*-
"""Use Poraquê as a standard ASE calculator (energy and numerical forces)."""

from ase import Atoms

from poraque.ase import Poraque
from poraque.core import SolverSettings


def main():
    atoms = Atoms("H2", positions=[[2.0, 2.5, 2.5], [3.0, 2.5, 2.5]],
                  cell=[5, 5, 5], pbc=True)
    atoms.calc = Poraque(
        mode="of",
        grid_shape=(24, 24, 24),
        settings=SolverSettings(max_iter=60, mixing=0.1),
        external_kwargs={"a": 0.8},
    )

    print(f"Potential energy: {atoms.get_potential_energy():.6f} eV")
    print("Forces (eV/A):")
    print(atoms.get_forces())


if __name__ == "__main__":
    main()
