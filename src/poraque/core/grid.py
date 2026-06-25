# -*- coding: utf-8 -*-
# file: grid.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br> 


import numpy as np

class Grid:
    """
    Real-space grid and its dual plane-wave (reciprocal) basis.

    Poraquê represents fields (densities, potentials, Kohn-Sham orbitals) on a
    uniform real-space grid that fills the periodic unit cell. The grid is the
    *primal* representation; its discrete Fourier transform spans a **plane-wave
    basis**

    .. math::

        \\psi(\\mathbf{r}) = \\sum_{\\mathbf{G}} c_{\\mathbf{G}}\\,
        e^{i \\mathbf{G}\\cdot\\mathbf{r}},

    where the wavevectors :math:`\\mathbf{G}` are the reciprocal-lattice points
    sampled by the FFT (see :meth:`get_g_vectors`). This duality is what lets
    the kinetic operator be applied exactly (and diagonally) in reciprocal space
    while local potentials are applied diagonally in real space. The
    completeness of the basis is controlled by the grid density, i.e. by the
    plane-wave kinetic-energy cutoff (see :meth:`from_ecut`).
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

        # Plane-wave cutoff used to build the grid, if any (see from_ecut).
        self.ecut = None

    @classmethod
    def from_ecut(cls, cell, ecut, pbc=True, density_factor=2.0):
        """
        Build a grid dense enough for a given plane-wave kinetic cutoff.

        The wavefunction plane-wave cutoff ``ecut`` (Hartree) corresponds to a
        maximum wavevector ``g_max = sqrt(2 * ecut)``. The electron density,
        being the square of the orbitals, contains components up to
        ``density_factor * g_max`` (``2`` for an exact product), so the grid
        spacing must satisfy the Nyquist condition
        ``h <= pi / (density_factor * g_max)`` along each lattice vector.

        Parameters
        ----------
        cell : array_like
            ``3x3`` lattice vectors (Bohr).
        ecut : float
            Plane-wave kinetic-energy cutoff for the wavefunctions (Hartree).
        pbc : bool or tuple of bool, optional
            Periodic boundary conditions.
        density_factor : float, optional
            Ratio between the density and wavefunction cutoffs (default ``2.0``).

        Returns
        -------
        Grid
            A grid whose shape resolves the requested cutoff. The attribute
            :attr:`ecut` records the cutoff used.
        """
        cell = np.asarray(cell, dtype=float)
        g_max = np.sqrt(2.0 * float(ecut))
        lengths = np.linalg.norm(cell, axis=1)
        # h_i <= pi / (density_factor * g_max)  ->  N_i >= L_i * density_factor * g_max / pi
        n = np.ceil(lengths * density_factor * g_max / np.pi).astype(int)
        # Use even, FFT-friendly sizes (and never fewer than 2 points).
        shape = tuple(int(max(2, ni + (ni % 2))) for ni in n)
        grid = cls(shape, cell, pbc=pbc)
        grid.ecut = float(ecut)
        return grid

    def kinetic_g2(self, kpoint_cartesian=None):
        r"""
        Squared shifted reciprocal vectors :math:`|\mathbf{G} + \mathbf{k}|^2`.

        Parameters
        ----------
        kpoint_cartesian : array_like, optional
            Bloch wavevector in **Cartesian** reciprocal coordinates (1/Bohr).
            Defaults to the :math:`\Gamma` point, recovering :meth:`get_g2`.

        Returns
        -------
        numpy.ndarray
            Array of shape :attr:`shape` used to build the kinetic operator
            ``T = 1/2 |G + k|^2`` in reciprocal space.
        """
        gx, gy, gz = self.get_g_vectors()
        if kpoint_cartesian is None:
            return gx**2 + gy**2 + gz**2
        kx, ky, kz = kpoint_cartesian
        return (gx + kx) ** 2 + (gy + ky) ** 2 + (gz + kz) ** 2

    def kpoint_to_cartesian(self, kpoint_fractional):
        """
        Convert a fractional k-point to Cartesian reciprocal coordinates.

        Parameters
        ----------
        kpoint_fractional : array_like
            ``(3,)`` k-point in fractional reciprocal coordinates (the
            convention used by :func:`ase.dft.kpoints.monkhorst_pack`).

        Returns
        -------
        numpy.ndarray
            ``(3,)`` Cartesian wavevector (1/Bohr).
        """
        return np.asarray(kpoint_fractional, dtype=float) @ self.reciprocal_cell

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

    def get_g_vectors(self):
        """
        Build the reciprocal-space grid vectors (Gx, Gy, Gz).

        These are the angular wavevectors :math:`\\mathbf{G}` sampled by the FFT
        for the current grid, returned as three Cartesian-component arrays each
        shaped like the grid. The construction is valid for **any** Bravais
        lattice: the integer FFT frequencies are projected onto the reciprocal
        lattice vectors (rows of :attr:`reciprocal_cell`), which reduces to the
        orthogonal ``2*pi*fftfreq`` form for a cubic cell.

        Returns
        -------
        tuple of numpy.ndarray
            ``(Gx, Gy, Gz)`` arrays of shape ``self.shape``.
        """
        # Integer FFT frequencies (0, 1, ..., -2, -1) along each axis.
        kx = np.fft.fftfreq(self.Nx) * self.Nx
        ky = np.fft.fftfreq(self.Ny) * self.Ny
        kz = np.fft.fftfreq(self.Nz) * self.Nz
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")

        # Project the integer frequencies onto the reciprocal lattice vectors
        # (rows of self.reciprocal_cell = 2*pi*inv(cell).T).
        b1, b2, b3 = self.reciprocal_cell
        Gx = KX * b1[0] + KY * b2[0] + KZ * b3[0]
        Gy = KX * b1[1] + KY * b2[1] + KZ * b3[1]
        Gz = KX * b1[2] + KY * b2[2] + KZ * b3[2]
        return Gx, Gy, Gz

    def get_g2(self):
        """
        Squared magnitude of the reciprocal grid vectors ``|G|^2``.

        Returns
        -------
        numpy.ndarray
            Array of shape ``self.shape`` with ``Gx^2 + Gy^2 + Gz^2``.
        """
        gx, gy, gz = self.get_g_vectors()
        return gx**2 + gy**2 + gz**2

    def __repr__(self):
        return f"Grid(shape={self.shape}, volume={self.volume:.4f}, pbc={self.pbc})"

