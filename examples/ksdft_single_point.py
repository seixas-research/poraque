# -*- coding: utf-8 -*-
"""Kohn-Sham DFT single-point energy for a closed-shell two-electron well."""

from ase import Atoms

from poraque import run_ksdft


def main():
    # A single He-like center (Z=2, 2 electrons).
    atoms = Atoms("He", positions=[[3.0, 3.0, 3.0]], cell=[6, 6, 6], pbc=True)
    result = run_ksdft(atoms, grid_shape=(32, 32, 32), mixing=0.3,
                       max_iter=80, external_kwargs={"a": 0.6})

    print(f"Converged:    {result.converged} ({result.iterations} SCF steps)")
    print(f"Total energy: {result.total_energy:.6f} Hartree")
    print(f"Eigenvalues:  {result.state.eigenvalues[:3]}")
    print(f"Occupations:  {result.state.occupations}")


if __name__ == "__main__":
    main()
