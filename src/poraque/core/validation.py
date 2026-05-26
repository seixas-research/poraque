# -*- coding: utf-8 -*-
# file: validation.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

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
