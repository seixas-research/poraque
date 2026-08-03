# -*- coding: utf-8 -*-
# file: kernels.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""
Numerical kernels for the classical electrostatic lattice sums.

These are the scalar hot spots of the ion-ion energy: pairwise loops whose cost
grows as ``O(N_atoms^2)`` and, for the Ewald real-space term, as
``O(N_atoms^2 * N_shifts)``. They are written here in **vectorized NumPy** —
no JIT, no compiled extension, no optional dependency — so the package runs on
a bare NumPy/SciPy stack.

.. note::
   This module is deliberately a **narrow seam**. Every function is pure
   (arrays in, float out), free of Poraquê types, and carries no state, so a
   future C/C++ backend can replace the bodies one at a time without touching
   any caller. The previous Numba implementation lived behind exactly this
   interface and was removed; keep new kernels equally self-contained.

Vectorization strategy: the pairwise distance matrix is formed explicitly,
which is ``O(N^2)`` in memory but negligible for the atom counts a plane-wave
DFT cell holds (hundreds). The Ewald sum loops over *lattice shifts* and
vectorizes the atom pairs inside, rather than materializing the full
``(N_shifts, N, N)`` tensor — the shift count is the dimension that explodes
with the real-space cutoff.
"""

import numpy as np
from scipy.special import erfc

__all__ = [
    "pairwise_coulomb_energy",
    "ewald_real_energy",
]

#: Distance floor guarding the ``1/r`` singularity for coincident sites.
_R_MIN = 1e-12


def pairwise_coulomb_energy(positions, charges):
    r"""
    Bare Coulomb energy of a finite set of point charges (atomic units).

    .. math:: E = \sum_{i<j} \frac{q_i q_j}{r_{ij}}

    Each pair is counted once, so no factor of one half is applied.

    Parameters
    ----------
    positions : array_like
        ``(N, 3)`` Cartesian coordinates (Bohr).
    charges : array_like
        ``(N,)`` charges.

    Returns
    -------
    float
        Interaction energy (Hartree).
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    n_atoms = positions.shape[0]
    if n_atoms < 2:
        return 0.0

    delta = positions[:, None, :] - positions[None, :, :]
    distance = np.maximum(np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)), _R_MIN)

    energy = np.outer(charges, charges) / distance
    # Upper triangle (k=1) selects each unordered pair exactly once.
    return float(np.sum(np.triu(energy, k=1)))


def ewald_real_energy(positions, charges, shifts, alpha, r_cut):
    r"""
    Real-space (short-range) term of the Ewald sum.

    .. math::

        E_{\rm real} = \frac{1}{2}\sum_{\mathbf{S}}\sum_{i,j}
        {}^{'}\; q_i q_j \,
        \frac{\mathrm{erfc}(\alpha\, r_{ij\mathbf{S}})}{r_{ij\mathbf{S}}},
        \qquad r_{ij\mathbf{S}} < r_{\rm cut},

    where :math:`\mathbf{S}` runs over lattice translations and the primed sum
    omits the ``i == j`` self term **in the home cell only** — an atom does
    interact with its own periodic images. The explicit one half corrects the
    double counting of the unrestricted ``i, j`` sum.

    Parameters
    ----------
    positions : array_like
        ``(N, 3)`` Cartesian coordinates (Bohr).
    charges : array_like
        ``(N,)`` charges.
    shifts : array_like
        ``(N_shifts, 3)`` lattice translation vectors, including the zero
        vector.
    alpha : float
        Ewald splitting parameter.
    r_cut : float
        Real-space cutoff; pairs beyond it are dropped.

    Returns
    -------
    float
        Real-space energy (Hartree).
    """
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    shifts = np.atleast_2d(np.asarray(shifts, dtype=float))

    charge_products = np.outer(charges, charges)
    identity = np.eye(positions.shape[0], dtype=bool)
    energy = 0.0

    for shift in shifts:
        delta = positions[:, None, :] - (positions[None, :, :] + shift)
        distance = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))

        inside = distance < r_cut
        if np.linalg.norm(shift) < _R_MIN:
            # Home cell: drop the i == j self interaction only.
            inside &= ~identity
        if not inside.any():
            continue

        safe = np.maximum(distance[inside], _R_MIN)
        energy += 0.5 * np.sum(
            charge_products[inside] * erfc(alpha * safe) / safe
        )

    return float(energy)
