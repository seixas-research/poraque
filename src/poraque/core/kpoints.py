# -*- coding: utf-8 -*-
# file: kpoints.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Brillouin-zone sampling helpers.

Periodic Kohn-Sham calculations require sampling Bloch states over the
Brillouin zone. Poraquê builds these k-point grids with ASE's
:func:`ase.dft.kpoints.monkhorst_pack`, then optionally folds the grid by
time-reversal symmetry (``k`` and ``-k`` are degenerate for a real
Hamiltonian), returning the irreducible k-points together with their weights.
"""

import numpy as np
from ase.dft.kpoints import monkhorst_pack


def gamma_only():
    """Return the single :math:`\\Gamma`-point grid ``([[0, 0, 0]], [1.0])``."""
    return np.zeros((1, 3)), np.ones(1)


def monkhorst_pack_kpoints(size, reduce_time_reversal=True, tol=1e-8):
    """
    Generate a Monkhorst-Pack k-point grid and its weights.

    Parameters
    ----------
    size : int or sequence of int
        Grid subdivisions ``(n1, n2, n3)``. A scalar is broadcast to all three
        directions. ``(1, 1, 1)`` yields the :math:`\\Gamma` point.
    reduce_time_reversal : bool, optional
        Fold the grid using time-reversal symmetry, combining each ``k`` with
        ``-k`` and summing their weights. This roughly halves the number of
        Hamiltonian diagonalizations for systems without an external magnetic
        field (default ``True``).
    tol : float, optional
        Tolerance for matching ``k`` and ``-k`` modulo a reciprocal-lattice
        vector.

    Returns
    -------
    tuple of numpy.ndarray
        ``(kpoints, weights)`` where ``kpoints`` has shape ``(Nk, 3)`` in
        fractional reciprocal coordinates and ``weights`` sums to one.
    """
    if np.isscalar(size):
        size = (int(size),) * 3
    size = tuple(int(s) for s in size)

    kpoints = monkhorst_pack(size)
    n = len(kpoints)
    weights = np.full(n, 1.0 / n)

    if not reduce_time_reversal:
        return kpoints, weights

    reduced = []
    reduced_weights = []
    for k, w in zip(kpoints, weights):
        found = False
        for idx, kr in enumerate(reduced):
            # k == -kr modulo a reciprocal-lattice vector (integer shift)?
            delta = k + kr
            if np.allclose(delta - np.round(delta), 0.0, atol=tol):
                reduced_weights[idx] += w
                found = True
                break
        if not found:
            reduced.append(k)
            reduced_weights.append(w)

    return np.array(reduced), np.array(reduced_weights)
