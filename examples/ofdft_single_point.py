# -*- coding: utf-8 -*-
"""OF-DFT single-point energy for a hydrogen atom in a box."""

from ase import Atoms

from poraque import run_ofdft


def main():
    atoms = Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5, 5, 5], pbc=True)
    result = run_ofdft(atoms, grid_shape=(24, 24, 24), max_iter=80)

    print(f"Converged:      {result.converged} ({result.iterations} iters)")
    print(f"Total energy:   {result.total_energy:.6f} Hartree")
    for name, value in result.energy_components.items():
        print(f"  {name:<28s} {value:12.6f}")
    print(f"Electron count: {result.density.integrate():.6f}")


if __name__ == "__main__":
    main()
