# -*- coding: utf-8 -*-
# file: test_jit.py
"""Correctness and performance tests for the Numba-JIT accelerated kernels.

Scope and a clarification on "JIT"
----------------------------------
These tests cover :mod:`poraque.accel`, whose hot loops are compiled with
**Numba's** ``@njit`` JIT and parallelised with ``prange`` (an OpenMP-style
thread pool). This is *independent* of CPython 3.14's own experimental
interpreter JIT (PEP 744): Numba ships its own LLVM-based compiler and does not
rely on, and is not accelerated by, the CPython JIT. "Python 3.14 compatibility"
here therefore means "Numba's JIT compiles and runs correctly on the 3.14
interpreter", which is exactly what :class:`TestNumbaCompatibility` asserts.

The suite validates three things:

* **Correctness** — each compiled kernel reproduces its pure-Python/NumPy
  reference to machine precision.
* **Compatibility** — the kernels actually JIT-compile and run on this
  interpreter (Python 3.14+).
* **Speedup** — the compiled kernels are faster than the reference once warm
  (skipped when Numba is unavailable, since then the two code paths are
  identical).
"""

import sys
import time

import numpy as np
import pytest

from poraque import accel
from poraque.core import Grid, System
from poraque.potentials.external import compute_ion_ion_energy

numba_required = pytest.mark.skipif(
    not accel.NUMBA_AVAILABLE, reason="Numba is not installed"
)


def _random_atoms(n, seed, box=10.0):
    rng = np.random.default_rng(seed)
    positions = rng.random((n, 3)) * box
    charges = rng.random(n) + 0.5
    return positions, charges


def _best_time(func, repeats=3):
    """Return the fastest wall time of ``func`` over ``repeats`` runs."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - t0)
    return best


class TestKernelCorrectness:
    """Each JIT kernel must equal its reference implementation."""

    def test_pairwise_coulomb_matches_reference(self):
        positions, charges = _random_atoms(40, seed=1)
        jit = accel.pairwise_coulomb_energy(positions, charges)
        ref = accel._pairwise_coulomb_reference(positions, charges)
        assert jit == pytest.approx(ref, rel=1e-10, abs=1e-12)

    def test_ewald_real_matches_reference(self):
        positions, charges = _random_atoms(15, seed=2, box=6.0)
        cell = np.eye(3) * 6.0
        ints = np.arange(-2, 3)
        mesh = np.array(np.meshgrid(ints, ints, ints, indexing="ij")).reshape(3, -1).T
        shifts = mesh @ cell
        jit = accel.ewald_real_energy(positions, charges, shifts, 0.9, 3.0)
        ref = accel._ewald_real_reference(positions, charges, shifts, 0.9, 3.0)
        assert jit == pytest.approx(ref, rel=1e-10, abs=1e-12)

    def test_thomas_fermi_matches_numpy(self):
        rng = np.random.default_rng(3)
        n = rng.random((16, 16, 16))
        c_tf = 0.3 * (3 * np.pi**2) ** (2 / 3)
        jit = accel.thomas_fermi_energy(n, c_tf)
        ref = c_tf * np.sum(np.maximum(n, 0.0) ** (5.0 / 3.0))
        assert jit == pytest.approx(ref, rel=1e-10)

    def test_thomas_fermi_ignores_negative_density(self):
        n = np.array([[-1.0, 0.0], [1.0, 8.0]])
        c_tf = 1.0
        expected = 1.0 ** (5 / 3) + 8.0 ** (5 / 3)
        assert accel.thomas_fermi_energy(n, c_tf) == pytest.approx(expected)


class TestKernelPhysics:
    """The accelerated kernels must preserve the physics of the ion-ion sum."""

    def test_ion_ion_ewald_is_alpha_independent(self):
        # A correct Ewald sum is independent of the splitting parameter for a
        # charge-neutral cell; this exercises the JIT real-space kernel inside
        # the full energy assembly.
        cell = np.eye(3) * 5.0
        grid = Grid((8, 8, 8), cell, pbc=True)
        system = System([[0, 0, 0], [2.5, 2.5, 2.5]], [1, 1], cell, pbc=True)
        charges = np.array([1.0, -1.0])
        energies = [
            compute_ion_ion_energy(system, grid, charges=charges, alpha=a)
            for a in (0.8, 1.0, 1.2)
        ]
        assert np.allclose(energies, energies[0], atol=1e-3)

    def test_pairwise_two_unit_charges(self):
        # Two +1 charges 2 Bohr apart -> 1/2 Hartree.
        positions = np.array([[0.0, 0, 0], [2.0, 0, 0]])
        charges = np.array([1.0, 1.0])
        assert accel.pairwise_coulomb_energy(positions, charges) == pytest.approx(0.5)


class TestNumbaCompatibility:
    """Numba's JIT must compile and run on the active (3.14+) interpreter."""

    def test_running_on_python_314_plus(self):
        assert sys.version_info[:2] >= (3, 14)

    @numba_required
    def test_parallel_info_reports_threads(self):
        # Force at least one compiled call so the threading layer is resolved.
        accel.pairwise_coulomb_energy(*_random_atoms(4, seed=7))
        info = accel.parallel_info()
        assert info["numba"] is True
        assert info["threads"] >= 1
        assert isinstance(info["layer"], str)

    @numba_required
    def test_kernels_are_compiled_dispatchers(self):
        # njit kernels expose Numba's dispatcher API (signatures after a call).
        accel._thomas_fermi_kernel(np.ones(8), 1.0)
        assert hasattr(accel._thomas_fermi_kernel, "signatures")
        assert len(accel._thomas_fermi_kernel.signatures) >= 1


@numba_required
class TestSpeedup:
    """Warm JIT kernels must outrun the pure-Python/NumPy reference."""

    def test_pairwise_speedup(self):
        positions, charges = _random_atoms(350, seed=11)
        # Warm up (exclude one-time compilation from the measurement).
        accel.pairwise_coulomb_energy(positions, charges)

        t_jit = _best_time(lambda: accel.pairwise_coulomb_energy(positions, charges))
        t_ref = _best_time(lambda: accel._pairwise_coulomb_reference(positions, charges))
        assert t_jit < t_ref, f"JIT ({t_jit:.4f}s) not faster than reference ({t_ref:.4f}s)"

    def test_ewald_real_speedup(self):
        positions, charges = _random_atoms(40, seed=12, box=8.0)
        cell = np.eye(3) * 8.0
        ints = np.arange(-1, 2)
        mesh = np.array(np.meshgrid(ints, ints, ints, indexing="ij")).reshape(3, -1).T
        shifts = mesh @ cell
        accel.ewald_real_energy(positions, charges, shifts, 0.8, 4.0)  # warm up

        t_jit = _best_time(
            lambda: accel.ewald_real_energy(positions, charges, shifts, 0.8, 4.0))
        t_ref = _best_time(
            lambda: accel._ewald_real_reference(positions, charges, shifts, 0.8, 4.0))
        assert t_jit < t_ref, f"JIT ({t_jit:.4f}s) not faster than reference ({t_ref:.4f}s)"
