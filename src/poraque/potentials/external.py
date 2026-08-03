# -*- coding: utf-8 -*-
# file: external.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

import warnings

import numpy as np
from scipy.special import erf, erfc

from ..profiling import profiler
from .kernels import ewald_real_energy, pairwise_coulomb_energy


def point_charge_potential(grid, positions, charges, rc=1e-6):
    """
    Compute the electrostatic potential of point charges on a grid.

    To avoid singularities, a regularized ``1/max(r, rc)`` form is used.

    For periodic systems, this would ideally be done via Ewald summation
    (see :func:`ewald_summation`); here we provide a simple real-space
    summation suitable for finite/molecular systems.

    Parameters
    ----------
    grid : Grid
        Grid providing :meth:`get_xyz`.
    positions : array_like
        ``(N, 3)`` array of point-charge coordinates.
    charges : array_like
        ``(N,)`` array of charges (electron-attracting nuclei use positive Z).
    rc : float, optional
        Regularization cutoff distance.

    Returns
    -------
    numpy.ndarray
        External potential ``v_ext(r)`` on the grid (attractive, negative).
    """
    coords = grid.get_xyz()
    v_ext = np.zeros(grid.shape)

    for pos, q in zip(positions, charges):
        dr = coords - pos
        r = np.linalg.norm(dr, axis=-1)
        r_regularized = np.maximum(r, rc)
        v_ext -= q / r_regularized
    return v_ext


def soft_coulomb_potential(grid, positions, charges, a=0.5, mic=True):
    """
    Regularized (soft-Coulomb) ionic potential ``-Z / sqrt(r^2 + a^2)``.

    The softening parameter ``a`` removes the Coulomb singularity at the
    nuclei, which keeps the real-space finite-difference operators stable.
    When ``mic`` is true, distances use the minimum-image convention so the
    potential is consistent with periodic boundary conditions.

    Parameters
    ----------
    grid : Grid
        Grid providing :meth:`get_xyz` and a ``cell`` attribute.
    positions : array_like
        ``(N, 3)`` nuclear coordinates (Bohr).
    charges : array_like
        ``(N,)`` nuclear charges ``Z``.
    a : float, optional
        Softening parameter (Bohr).
    mic : bool, optional
        Use the minimum-image convention for periodic cells.

    Returns
    -------
    numpy.ndarray
        Attractive external potential on the grid.
    """
    coords = grid.get_xyz()
    v_ext = np.zeros(grid.shape)
    cell = np.asarray(grid.cell, dtype=float)
    cell_inv = np.linalg.inv(cell)

    for pos, q in zip(positions, charges):
        dr = coords - pos
        if mic:
            s = dr @ cell_inv
            s -= np.round(s)
            dr = s @ cell
        r2 = np.sum(dr**2, axis=-1)
        v_ext -= q / np.sqrt(r2 + a**2)
    return v_ext


def build_external_potential(grid, system, kind="soft", **kwargs):
    """
    Build the electron-nucleus external potential for a :class:`System`.

    Parameters
    ----------
    grid : Grid
        Real-space grid.
    system : System
        Atomic structure providing ``positions`` and ``atomic_numbers``.
    kind : {"soft", "point", "ewald"}, optional
        Potential model. ``"soft"`` uses :func:`soft_coulomb_potential`
        (default, recommended for finite-difference grids), ``"point"`` uses
        :func:`point_charge_potential`, and ``"ewald"`` uses
        :func:`ewald_summation` for fully periodic cells.
    **kwargs
        Forwarded to the underlying potential builder.

    Returns
    -------
    numpy.ndarray
        External potential on the grid.
    """
    positions = system.positions
    charges = system.atomic_numbers.astype(float)

    if kind == "soft":
        # Wrap with the minimum-image convention only when the cell is actually
        # periodic; finite/molecular systems (pbc=False) must not wrap.
        kwargs.setdefault("mic", any(grid.pbc))
        return soft_coulomb_potential(grid, positions, charges, **kwargs)
    if kind == "point":
        return point_charge_potential(grid, positions, charges, **kwargs)
    if kind == "ewald":
        return ewald_summation(grid, positions, charges, grid.cell, **kwargs)
    raise ValueError(f"Unknown external potential kind: {kind!r}")


def compute_ion_ion_energy(system, grid, charges=None, alpha=None, r_cut=None, k_cut=None):
    """
    Classical electrostatic ion-ion repulsion energy (Hartree).

    Uses Ewald summation when the cell is periodic (``any(grid.pbc)``) and a
    direct pairwise Coulomb sum for finite/molecular systems. This term is
    independent of the electron density but is essential for physically correct
    *absolute* total energies and forces: without it, geometry optimizations
    have no nuclear repulsion and atoms collapse to ``r -> 0``.

    Parameters
    ----------
    system : System
        Provides ``positions`` (Bohr) and ``atomic_numbers``.
    grid : Grid
        Provides ``cell`` (Bohr), ``volume`` and ``pbc``.
    charges : array_like, optional
        Ionic charges ``Z`` per atom. Defaults to ``system.atomic_numbers``
        (correct for all-electron runs). For pseudopotential runs the *valence*
        charges must be supplied so the ion-ion term is consistent with the
        electron-ion potential and electron count.
    alpha, r_cut, k_cut : float, optional
        Ewald splitting parameter and real/reciprocal cutoffs. Sensible
        defaults are derived from the cell when omitted.

    Returns
    -------
    float
        Ion-ion electrostatic energy (Hartree).
    """
    positions = np.asarray(system.positions, dtype=float)
    if charges is None:
        charges = system.atomic_numbers.astype(float)
    else:
        charges = np.asarray(charges, dtype=float)
    n_atoms = len(charges)
    if n_atoms <= 1:
        return 0.0

    # --- Non-periodic: direct pairwise Coulomb repulsion ---
    if not any(grid.pbc):
        with profiler.timer("ion-ion (pairwise)"):
            return pairwise_coulomb_energy(positions, charges)

    # --- Periodic: Ewald summation ---
    cell = np.asarray(grid.cell, dtype=float)
    V_box = grid.volume
    H_inv = np.linalg.inv(cell)
    B = 2 * np.pi * H_inv.T  # rows are reciprocal lattice vectors
    perpendicular_heights = 2 * np.pi / np.linalg.norm(B, axis=1)

    if alpha is None:
        alpha = 5.0 / (V_box ** (1 / 3.0))
    if r_cut is None:
        r_cut = np.min(perpendicular_heights) / 2.0
    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi

    # 1. Real-space term (0.5 factor handles the symmetric double counting).
    #    The triple lattice sum is the dominant scalar cost and is delegated to
    #    the vectorized kernel in poraque.potentials.kernels, which is the seam
    #    a future compiled backend would replace.
    n_max_real = np.ceil(r_cut / perpendicular_heights).astype(int)
    nx_r = np.arange(-n_max_real[0], n_max_real[0] + 1)
    ny_r = np.arange(-n_max_real[1], n_max_real[1] + 1)
    nz_r = np.arange(-n_max_real[2], n_max_real[2] + 1)
    mesh_n_r = np.array(np.meshgrid(nx_r, ny_r, nz_r, indexing='ij')).reshape(3, -1).T
    shift_vectors = np.dot(mesh_n_r, cell)

    with profiler.timer("ion-ion Ewald (real)"):
        e_real = ewald_real_energy(positions, charges, shift_vectors,
                                   alpha, r_cut)

    # 2. Reciprocal-space term.
    e_recip = 0.0
    n_max_recip = np.ceil(k_cut / np.linalg.norm(B, axis=1)).astype(int)
    nx_k = np.arange(-n_max_recip[0], n_max_recip[0] + 1)
    ny_k = np.arange(-n_max_recip[1], n_max_recip[1] + 1)
    nz_k = np.arange(-n_max_recip[2], n_max_recip[2] + 1)
    mesh_n_k = np.array(np.meshgrid(nx_k, ny_k, nz_k, indexing='ij')).reshape(3, -1).T
    k_vectors = np.dot(mesh_n_k, B)

    for k in k_vectors:
        k_sq = np.dot(k, k)
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))
        S_k_sq = np.abs(S_k) ** 2
        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha ** 2)) / k_sq
        e_recip += 0.5 * prefactor * S_k_sq

    # 3. Self-interaction correction.
    e_self = (alpha / np.sqrt(np.pi)) * np.sum(charges ** 2)

    # 4. Neutralizing-background term for charged cells.
    net_charge = np.sum(charges)
    e_bg = -(np.pi / (2.0 * alpha ** 2 * V_box)) * net_charge ** 2

    return e_real + e_recip - e_self + e_bg


def ewald_summation(grid, positions, charges, cell, alpha=None, r_cut=None, k_cut=None, mic=False):
    """
    Compute the electrostatic potential on a grid using Ewald summation
    for an arbitrary periodic unit cell.

    Parameters
    ----------
    grid : Grid
        Object with :meth:`get_xyz` returning coordinates ``(..., 3)``.
    positions : array_like
        ``(N, 3)`` array of point-charge coordinates.
    charges : array_like
        ``(N,)`` array of charge magnitudes.
    cell : array_like
        ``(3, 3)`` array where rows are the lattice vectors ``a1, a2, a3``.
    alpha : float, optional
        Ewald splitting parameter (controls width of Gaussian clouds).
    r_cut : float, optional
        Real-space cutoff distance.
    k_cut : float, optional
        Reciprocal-space cutoff (maximum k-vector magnitude).
    mic : bool, optional
        If True, uses the fast fractional minimum-image convention (assumes
        ``r_cut <= min_height/2`` and a cell that is not severely skewed). If
        False (default), loops over neighboring cells in a bounding box for
        guaranteed accuracy at any cutoff or cell skewness.

    Returns
    -------
    numpy.ndarray
        Electrostatic potential on the grid.
    """
    coords = grid.get_xyz()
    v_total = np.zeros(grid.shape)

    # --- 1. Cell Geometry & Reciprocal Space Setup ---
    H = np.array(cell, dtype=float)
    V_box = np.abs(np.linalg.det(H))
    H_inv = np.linalg.inv(H)

    # B matrix: rows are reciprocal lattice vectors
    B = 2 * np.pi * H_inv.T

    # Perpendicular heights of the real-space cell
    perpendicular_heights = 2 * np.pi / np.linalg.norm(B, axis=1)

    # --- 2. Parameter Heuristics ---
    if alpha is None:
        alpha = 5.0 / (V_box ** (1 / 3.0))

    if r_cut is None:
        r_cut = np.min(perpendicular_heights) / 2.0

    if k_cut is None:
        k_cut = 5.0 * alpha * 2 * np.pi

    # Safety check for the fast MIC method
    if mic and r_cut > np.min(perpendicular_heights) / 2.0 + 1e-6:
        warnings.warn("Warning: r_cut is larger than half the shortest cell height. "
                      "The MIC method (mic=True) may miss interacting periodic images. "
                      "Consider using mic=False.")

    # --- 3. Real-Space Summation ---
    if mic:
        # FAST METHOD: Fractional Rounding Minimum Image Convention
        for pos, q in zip(positions, charges):
            dr = coords - pos
            s = np.dot(dr, H_inv)
            s_mic = s - np.round(s)
            dr_mic = np.dot(s_mic, H)

            r = np.linalg.norm(dr_mic, axis=-1)
            r_safe = np.maximum(r, 1e-12)

            mask = r_safe < r_cut
            v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]

    else:
        # ROBUST METHOD: Bounding Box over neighboring periodic cells
        n_max_real = np.ceil(r_cut / perpendicular_heights).astype(int)

        nx_r = np.arange(-n_max_real[0], n_max_real[0] + 1)
        ny_r = np.arange(-n_max_real[1], n_max_real[1] + 1)
        nz_r = np.arange(-n_max_real[2], n_max_real[2] + 1)
        mesh_n_r = np.array(np.meshgrid(nx_r, ny_r, nz_r, indexing='ij')).reshape(3, -1).T

        shift_vectors = np.dot(mesh_n_r, H)

        for pos, q in zip(positions, charges):
            for shift in shift_vectors:
                dr = coords - (pos + shift)

                r = np.linalg.norm(dr, axis=-1)
                r_safe = np.maximum(r, 1e-12)

                mask = r_safe < r_cut
                v_total[mask] += q * erfc(alpha * r_safe[mask]) / r_safe[mask]

    # --- 4. Reciprocal-Space Summation ---
    n_max_recip = np.ceil(k_cut / np.linalg.norm(B, axis=1)).astype(int)

    nx_k = np.arange(-n_max_recip[0], n_max_recip[0] + 1)
    ny_k = np.arange(-n_max_recip[1], n_max_recip[1] + 1)
    nz_k = np.arange(-n_max_recip[2], n_max_recip[2] + 1)
    mesh_n_k = np.array(np.meshgrid(nx_k, ny_k, nz_k, indexing='ij')).reshape(3, -1).T

    k_vectors = np.dot(mesh_n_k, B)

    for k in k_vectors:
        k_sq = np.dot(k, k)
        if k_sq == 0 or np.sqrt(k_sq) > k_cut:
            continue

        # Structure factor
        S_k = np.sum(charges * np.exp(-1j * np.dot(positions, k)))

        # Grid evaluation
        grid_phase = np.exp(1j * np.dot(coords, k))

        prefactor = (4 * np.pi / V_box) * np.exp(-k_sq / (4 * alpha**2)) / k_sq
        v_total += np.real(prefactor * S_k * grid_phase)

    # --- 5. Constant / Background Terms ---
    net_charge = np.sum(charges)
    if not np.isclose(net_charge, 0.0):
        v_bg = -(np.pi / (alpha**2 * V_box)) * net_charge
        v_total += v_bg

    return v_total
