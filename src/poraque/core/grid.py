# -*- coding: utf-8 -*-
# file: grid.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 


import numpy as np

class Grid:
    """
    Represents the simulation grid and cell.
    """
    def __init__(self, shape, cell, pbc=True):
        """
        Initialize the Grid.

        Parameters
        ----------
        shape : tuple of int
            Number of grid points in each direction (Nx, Ny, Nz).
        cell : array_like
            3x3 array of lattice vectors.
        pbc : bool or tuple of bool, optional
            Periodic boundary conditions in each direction. Default is True (all periodic).
        """
        self.shape = tuple(shape)
        self.Nx, self.Ny, self.Nz = self.shape
        self.N = self.Nx * self.Ny * self.Nz
        
        self.cell = np.array(cell, dtype=float)
        if self.cell.shape != (3, 3):
            raise ValueError("Cell must be a 3x3 array.")
            
        if isinstance(pbc, bool):
            self.pbc = (pbc, pbc, pbc)
        else:
            self.pbc = tuple(pbc)
            
        # Compute spacing and volume
        self.volume = np.abs(np.linalg.det(self.cell))
        self.volume_element = self.volume / self.N
        
        # Grid spacings (for orthogonal cells, these are just cell_lengths / shape)
        # For non-orthogonal cells, it's more subtle, but for now we assume 
        # a simple mapping.
        self.h = np.linalg.norm(self.cell, axis=1) / self.shape

        # Reciprocal lattice vectors
        self.reciprocal_cell = 2 * np.pi * np.linalg.inv(self.cell).T
        
    def get_xyz(self):
        """
        Generate Cartesian coordinates for all grid points.
        """
        # This is a simplified version for orthogonal cells.
        # For non-orthogonal, we'd transform fractional coordinates.
        u = np.linspace(0, 1, self.Nx, endpoint=False)
        v = np.linspace(0, 1, self.Ny, endpoint=False)
        w = np.linspace(0, 1, self.Nz, endpoint=False)
        
        uu, vv, ww = np.meshgrid(u, v, w, indexing='ij')
        coords_fractional = np.stack([uu.flatten(), vv.flatten(), ww.flatten()], axis=1)
        coords_cartesian = coords_fractional @ self.cell
        
        return coords_cartesian.reshape(self.Nx, self.Ny, self.Nz, 3)

    def integrate(self, field):
        """
        Integrate a scalar field over the grid volume.
        """
        return np.sum(field) * self.volume_element

    def __repr__(self):
        return f"Grid(shape={self.shape}, volume={self.volume:.4f}, pbc={self.pbc})"

