# -*- coding: utf-8 -*-
# file: operators.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 

import numpy as np
from ..base import Backend

class NumpyBackend(Backend):
    """
    NumPy-based reference implementation of the Backend.
    """

    def integrate(self, field, grid):
        return np.sum(field) * grid.volume_element

    def gradient(self, field, grid):
        """
        Compute gradient using central finite differences.
        """
        # This is a basic implementation assuming PBC.
        # np.gradient handles spacing and PBC if 'edge_order' is used, 
        # but for DFT we typically want periodic wrap-around.
        grad = []
        for axis in range(3):
            g = (np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2 * grid.h[axis])
            grad.append(g)
        return np.array(grad)

    def laplacian(self, field, grid):
        """
        Compute Laplacian using 2nd order central finite differences.
        """
        lap = np.zeros_like(field)
        for axis in range(3):
            lap += (np.roll(field, -1, axis=axis) - 2 * field + np.roll(field, 1, axis=axis)) / (grid.h[axis]**2)
        return lap

    def laplacian_fft(self, field, grid):
        """
        Compute the Laplacian via reciprocal space: nabla^2 f = IFFT(-G^2 f(G)).

        Spectrally accurate for periodic, band-limited fields.
        """
        g2 = grid.get_g2()
        f_g = np.fft.fftn(field)
        lap_g = -g2 * f_g
        return np.real(np.fft.ifftn(lap_g))

    def poisson(self, charge_density, grid):
        """
        Solve Poisson equation via FFT: V(G) = 4 * pi * n(G) / G^2
        """
        n_g = np.fft.fftn(charge_density)

        # Reuse the grid's |G|^2 (valid for non-orthogonal cells too).
        G2 = grid.get_g2()
        G2[0, 0, 0] = 1.0 # Avoid division by zero

        v_g = 4 * np.pi * n_g / G2
        v_g[0, 0, 0] = 0.0 # Set average potential to zero
        
        return np.real(np.fft.ifftn(v_g))

    def fft(self, field):
        return np.fft.fftn(field)

    def ifft(self, field):
        return np.fft.ifftn(field)

    def dot(self, a, b):
        return np.sum(a * b)

    def norm(self, a):
        return np.sqrt(np.sum(np.abs(a)**2))
