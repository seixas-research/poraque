# -*- coding: utf-8 -*-
# file: validation.py

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

def validate_consistency(grid, system=None, density=None, state=None):
    """
    Check consistency between core objects.
    """
    if system is not None:
        # Check if cell in system and grid match
        import numpy as np
        if not np.allclose(grid.cell, system.cell):
            raise ValueError("Grid and System cells are not consistent.")
            
    if density is not None:
        # Check if density grid and grid match
        if density.grid is not grid:
            if density.data.shape != grid.shape:
                 raise ValueError("Density data shape does not match grid shape.")

    if state is not None:
        # Check if state grid and grid match
        if state.grid is not grid:
            if state.orbitals.shape[1:] != grid.shape:
                raise ValueError("State orbitals shape does not match grid shape.")

    return True
