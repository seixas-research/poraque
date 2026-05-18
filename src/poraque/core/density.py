# -*- coding: utf-8 -*-
# file: density.py

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

class Density:
    """
    Represents the electron density on the grid.
    """
    def __init__(self, grid, data=None):
        """
        Initialize the Density.

        Parameters
        ----------
        grid : Grid
            The grid on which the density is defined.
        data : array_like, optional
            The density values on the grid. If None, initialized to zeros.
        """
        self.grid = grid
        if data is None:
            self.data = np.zeros(grid.shape)
        else:
            self.data = np.array(data)
            if self.data.shape != grid.shape:
                raise ValueError(f"Density data shape {self.data.shape} does not match grid shape {grid.shape}")

    def integrate(self):
        """
        Integrate the density over the grid volume.
        """
        return self.grid.integrate(self.data)

    def normalize(self, n_electrons):
        """
        Normalize the density to a given number of electrons.
        """
        current_n = self.integrate()
        if current_n > 0:
            self.data *= (n_electrons / current_n)

    def check_positivity(self):
        """
        Check if the density is positive everywhere.
        Returns the minimum value and its position (indices).
        """
        min_val = np.min(self.data)
        if min_val < 0:
            idx = np.unravel_index(np.argmin(self.data), self.data.shape)
            return False, min_val, idx
        return True, min_val, None

    def __repr__(self):
        return f"Density(integral={self.integrate():.4f}, shape={self.data.shape})"
