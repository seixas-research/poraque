# -*- coding: utf-8 -*-
# file: system.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np

from .units import ANGSTROM, BOHR_TO_ANGSTROM


class System:
    """
    Represents the physical system (ions, electrons, cell).
    """
    def __init__(self, positions, atomic_numbers, cell, pbc=True, electrons=None, spin_polarized=False):
        """
        Initialize the System.

        Parameters
        ----------
        positions : array_like
            Nx3 array of atomic positions.
        atomic_numbers : array_like
            Array of atomic numbers (Z).
        cell : array_like
            3x3 array of lattice vectors.
        pbc : bool or tuple of bool, optional
            Periodic boundary conditions.
        electrons : int, optional
            Total number of electrons. If None, it's inferred from atomic_numbers (neutral).
        spin_polarized : bool, optional
            Whether the system is spin-polarized.
        """
        self.positions = np.array(positions, dtype=float)
        self.atomic_numbers = np.array(atomic_numbers, dtype=int)
        self.cell = np.array(cell, dtype=float)
        
        if isinstance(pbc, bool):
            self.pbc = (pbc, pbc, pbc)
        else:
            self.pbc = tuple(pbc)
            
        if electrons is None:
            self.electrons = int(np.sum(self.atomic_numbers))
        else:
            self.electrons = int(electrons)
            
        self.spin_polarized = spin_polarized
        
    @classmethod
    def from_ase(cls, atoms, charge=0):
        """
        Create a System object from an ASE ``Atoms`` object.

        Positions and cell are converted from Ångström (ASE convention) to
        Bohr (internal atomic units).

        Parameters
        ----------
        atoms : ase.Atoms
            The atomic structure.
        charge : int, optional
            Net charge; the electron count is ``sum(Z) - charge``.
        """
        numbers = atoms.get_atomic_numbers()
        return cls(
            positions=np.asarray(atoms.get_positions()) * ANGSTROM,
            atomic_numbers=numbers,
            cell=np.asarray(atoms.get_cell()) * ANGSTROM,
            pbc=atoms.get_pbc(),
            electrons=int(np.sum(numbers)) - int(charge),
        )

    def to_ase(self):
        """
        Export this system to an ASE ``Atoms`` object.

        Positions and cell are converted from Bohr (internal) back to
        Ångström (ASE convention). The inverse of :meth:`from_ase` for
        neutral systems.

        Returns
        -------
        ase.Atoms
            The atomic structure in ASE units.
        """
        from ase import Atoms

        return Atoms(
            numbers=self.atomic_numbers,
            positions=self.positions * BOHR_TO_ANGSTROM,
            cell=self.cell * BOHR_TO_ANGSTROM,
            pbc=self.pbc,
        )

    def __repr__(self):
        return f"System(atoms={len(self.atomic_numbers)}, electrons={self.electrons}, spin={self.spin_polarized})"
