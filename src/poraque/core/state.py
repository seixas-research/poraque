# -*- coding: utf-8 -*-
# file: state.py

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

class State:
    """
    Represents the electronic state (orbitals, occupations).
    Mainly used for KS-DFT.
    """
    def __init__(self, grid, orbitals, occupations=None):
        """
        Initialize the State.

        Parameters
        ----------
        grid : Grid
            The grid on which orbitals are defined.
        orbitals : array_like
            (n_orbitals, Nx, Ny, Nz) array of orbitals.
        occupations : array_like, optional
            (n_orbitals,) array of occupation numbers.
        """
        self.grid = grid
        self.orbitals = np.array(orbitals)
        if self.orbitals.ndim == 3: # Single orbital
            self.orbitals = self.orbitals[np.newaxis, ...]
            
        self.n_orbitals = self.orbitals.shape[0]
        
        if occupations is None:
            self.occupations = np.ones(self.n_orbitals) * 2.0 # Default closed shell
        else:
            self.occupations = np.array(occupations)

    def get_density(self):
        """
        Compute the electron density from orbitals and occupations.
        n(r) = sum_i f_i |phi_i(r)|^2
        """
        density_data = np.zeros(self.grid.shape)
        for i in range(self.n_orbitals):
            density_data += self.occupations[i] * np.abs(self.orbitals[i])**2
        return density_data

    def orthonormalize(self):
        """
        Orthonormalize orbitals using Gram-Schmidt or similar.
        Note: For real-space grids, this involves the volume element.
        """
        # Flatten for linear algebra
        flat_orbitals = self.orbitals.reshape(self.n_orbitals, -1)
        
        # Gram-Schmidt (basic implementation)
        q, r = np.linalg.qr(flat_orbitals.T)
        
        # Rescale by sqrt(dV) to maintain normalization on the grid
        self.orbitals = (q.T / np.sqrt(self.grid.volume_element)).reshape(self.orbitals.shape)

    def __repr__(self):
        return f"State(n_orbitals={self.n_orbitals}, shape={self.orbitals.shape})"
