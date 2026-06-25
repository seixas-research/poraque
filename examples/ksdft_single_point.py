# -*- coding: utf-8 -*-
"""Kohn-Sham DFT single-point energy for a closed-shell two-electron well."""

from ase import Atoms

from poraque.ase import Poraque
from poraque.core import SolverSettings


def main():
    # A single He-like center (Z=2, 2 electrons).
    atoms = Atoms("He", positions=[[3.0, 3.0, 3.0]], cell=[6, 6, 6], pbc=True)
    atoms.calc = Poraque(
        mode="ks",
        grid_shape=(32, 32, 32),
        settings=SolverSettings(max_iter=80, mixing=0.3),
        external_kwargs={"a": 0.6},
    )

    energy = atoms.get_potential_energy()
    state = atoms.calc.results["density"]
    print(f"Converged:    {atoms.calc.results['converged']}")
    print(f"Total energy: {energy:.6f} eV")
    print(f"Electrons:    {state.integrate():.6f}")


if __name__ == "__main__":
    main()
