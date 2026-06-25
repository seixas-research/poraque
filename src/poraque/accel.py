# -*- coding: utf-8 -*-
# file: accel.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

"""Numba-accelerated, thread-parallel kernels for the hot loops.

The most expensive *scalar* loops in Poraquê are the real-space lattice sums of
the Ewald ion-ion energy (an ``O(N_atoms^2 * N_images)`` triple loop) and the
non-periodic pairwise Coulomb sum. These are pure Python in the reference
implementation and dominate the cost for systems with many atoms or large
real-space cutoffs.

This module JIT-compiles those loops with `Numba <https://numba.pydata.org>`_
and runs them across CPU cores using Numba's ``prange`` (an OpenMP-style
shared-memory thread pool — the appropriate parallelism for this NumPy /
shared-array architecture, as opposed to distributed MPI). The public functions
are thin wrappers that:

* dispatch to the compiled kernel when Numba is available, and
* fall back to an equivalent (slower) pure-Python/NumPy implementation when it
  is not, so Poraquê keeps working without Numba installed.

Because the wrappers and the JIT kernels compute exactly the same quantity, the
fallback also doubles as the *reference* implementation that the JIT test suite
(``tests/test_jit.py``) checks the compiled kernels against.
"""

import math

import numpy as np

try:  # Numba is an optional accelerator, not a hard dependency.
    from numba import njit, prange, get_num_threads, threading_layer
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without Numba
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op stand-in for :func:`numba.njit` when Numba is absent."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    prange = range

    def get_num_threads():
        return 1

    def threading_layer():
        return "none (numba not installed)"


__all__ = [
    "NUMBA_AVAILABLE",
    "parallel_info",
    "pairwise_coulomb_energy",
    "ewald_real_energy",
    "thomas_fermi_energy",
]


# --------------------------------------------------------------------------- #
# Compiled kernels
# --------------------------------------------------------------------------- #
# Each ``prange`` body writes its result into a private slot of a partial-sums
# array and the totals are reduced afterwards. This idiom is more robust across
# Numba versions than relying on automatic scalar-reduction inference inside
# deeply nested loops.
@njit(parallel=True, fastmath=True, cache=True)
def _pairwise_coulomb_kernel(positions, charges):  # pragma: no cover - JIT body
    n_atoms = positions.shape[0]
    partial = np.zeros(n_atoms)
    for i in prange(n_atoms):
        ei = 0.0
        for j in range(i + 1, n_atoms):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dz = positions[i, 2] - positions[j, 2]
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            if r < 1e-12:
                r = 1e-12
            ei += charges[i] * charges[j] / r
        partial[i] = ei
    return partial.sum()


@njit(parallel=True, fastmath=True, cache=True)
def _ewald_real_kernel(positions, charges, shifts, alpha, r_cut):  # pragma: no cover - JIT body
    n_atoms = positions.shape[0]
    n_shift = shifts.shape[0]
    partial = np.zeros(n_atoms)
    for i in prange(n_atoms):
        ei = 0.0
        for j in range(n_atoms):
            qij = charges[i] * charges[j]
            for s in range(n_shift):
                sx = shifts[s, 0]
                sy = shifts[s, 1]
                sz = shifts[s, 2]
                if i == j and abs(sx) < 1e-12 and abs(sy) < 1e-12 and abs(sz) < 1e-12:
                    continue
                dx = positions[i, 0] - (positions[j, 0] + sx)
                dy = positions[i, 1] - (positions[j, 1] + sy)
                dz = positions[i, 2] - (positions[j, 2] + sz)
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                if r < r_cut:
                    rr = r if r > 1e-12 else 1e-12
                    ei += 0.5 * qij * math.erfc(alpha * rr) / rr
        partial[i] = ei
    return partial.sum()


@njit(parallel=True, fastmath=True, cache=True)
def _thomas_fermi_kernel(n_flat, c_tf):  # pragma: no cover - JIT body
    n = n_flat.shape[0]
    partial = np.zeros(n)
    for i in prange(n):
        v = n_flat[i]
        if v > 0.0:
            partial[i] = v ** (5.0 / 3.0)
    return c_tf * partial.sum()


# --------------------------------------------------------------------------- #
# Pure-Python / NumPy reference fallbacks
# --------------------------------------------------------------------------- #
def _pairwise_coulomb_reference(positions, charges):
    n_atoms = positions.shape[0]
    energy = 0.0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            r = np.linalg.norm(positions[i] - positions[j])
            energy += charges[i] * charges[j] / max(r, 1e-12)
    return energy


def _ewald_real_reference(positions, charges, shifts, alpha, r_cut):
    from scipy.special import erfc
    n_atoms = positions.shape[0]
    energy = 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            for s in range(shifts.shape[0]):
                shift = shifts[s]
                if i == j and np.all(np.abs(shift) < 1e-12):
                    continue
                r = np.linalg.norm(positions[i] - (positions[j] + shift))
                if r < r_cut:
                    rr = max(r, 1e-12)
                    energy += 0.5 * charges[i] * charges[j] * erfc(alpha * rr) / rr
    return energy


# --------------------------------------------------------------------------- #
# Public dispatchers
# --------------------------------------------------------------------------- #
def parallel_info():
    """
    Describe the active parallel backend.

    Returns
    -------
    dict
        ``{"numba": bool, "threads": int, "layer": str}`` — whether Numba is
        available, the number of worker threads it will use, and the active
        threading layer (e.g. ``"omp"``, ``"tbb"``, ``"workqueue"``). The
        threading layer is only known after the first compiled call, so it may
        read ``"unknown"`` until a kernel has run.
    """
    if not NUMBA_AVAILABLE:
        return {"numba": False, "threads": 1, "layer": "none (numba not installed)"}
    try:
        layer = threading_layer()
    except Exception:
        layer = "unknown (no parallel kernel run yet)"
    return {"numba": True, "threads": int(get_num_threads()), "layer": layer}


def pairwise_coulomb_energy(positions, charges):
    """
    Sum of ``q_i q_j / r_ij`` over all atom pairs (non-periodic ion-ion energy).

    Parameters
    ----------
    positions : numpy.ndarray
        ``(N, 3)`` Cartesian coordinates (Bohr).
    charges : numpy.ndarray
        ``(N,)`` ionic charges.

    Returns
    -------
    float
        Total pairwise Coulomb energy (Hartree). Uses the Numba kernel when
        available, otherwise an equivalent NumPy reference.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    charges = np.ascontiguousarray(charges, dtype=np.float64)
    if NUMBA_AVAILABLE and positions.shape[0] > 1:
        return float(_pairwise_coulomb_kernel(positions, charges))
    return float(_pairwise_coulomb_reference(positions, charges))


def ewald_real_energy(positions, charges, shifts, alpha, r_cut):
    """
    Real-space part of the Ewald ion-ion energy.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(N, 3)`` Cartesian coordinates (Bohr).
    charges : numpy.ndarray
        ``(N,)`` ionic charges.
    shifts : numpy.ndarray
        ``(M, 3)`` Cartesian lattice-image translations to sum over.
    alpha : float
        Ewald splitting parameter.
    r_cut : float
        Real-space cutoff radius (Bohr).

    Returns
    -------
    float
        Real-space Ewald energy (Hartree), with the conventional ``0.5`` factor
        applied (so it can be added directly to the reciprocal-space term).
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    charges = np.ascontiguousarray(charges, dtype=np.float64)
    shifts = np.ascontiguousarray(shifts, dtype=np.float64)
    if NUMBA_AVAILABLE:
        return float(_ewald_real_kernel(positions, charges, shifts,
                                        float(alpha), float(r_cut)))
    return float(_ewald_real_reference(positions, charges, shifts, alpha, r_cut))


def thomas_fermi_energy(density, c_tf):
    """
    Thomas-Fermi kinetic energy density integral ``c_tf * sum n^(5/3)``.

    This kernel mirrors :class:`poraque.functionals.ThomasFermi` and exists
    mainly as a JIT-vs-NumPy regression target; multiply the result by the grid
    volume element to obtain the energy.

    Parameters
    ----------
    density : numpy.ndarray
        Density values on the grid (any shape; flattened internally).
    c_tf : float
        Thomas-Fermi constant prefactor.

    Returns
    -------
    float
        ``c_tf * sum_i n_i^(5/3)`` over positive density points.
    """
    flat = np.ascontiguousarray(density, dtype=np.float64).ravel()
    if NUMBA_AVAILABLE:
        return float(_thomas_fermi_kernel(flat, float(c_tf)))
    n = np.maximum(flat, 0.0)
    return float(c_tf * np.sum(n ** (5.0 / 3.0)))
