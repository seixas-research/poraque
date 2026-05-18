# -*- coding: utf-8 -*-
# file: system.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

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
    def from_ase(cls, atoms):
        """
        Create a System object from an ASE Atoms object.
        """
        return cls(
            positions=atoms.get_positions(),
            atomic_numbers=atoms.get_atomic_numbers(),
            cell=atoms.get_cell(),
            pbc=atoms.get_pbc(),
            electrons=int(np.sum(atoms.get_atomic_numbers())) # Default to neutral
        )

    def __repr__(self):
        return f"System(atoms={len(self.atomic_numbers)}, electrons={self.electrons}, spin={self.spin_polarized})"
