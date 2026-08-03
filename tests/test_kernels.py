# -*- coding: utf-8 -*-
# file: test_kernels.py
"""
Tests for :mod:`poraque.potentials.kernels`.

These replace the former ``test_jit.py``: the Numba backend was removed, so the
tests now check the pure-NumPy kernels against closed-form electrostatics
rather than against a JIT reference. Because they pin *physics* rather than an
implementation, they will keep guarding a future C/C++ backend unchanged.
"""

import numpy as np
import pytest
from scipy.special import erfc

from poraque.potentials.kernels import ewald_real_energy, pairwise_coulomb_energy


class TestPairwiseCoulomb:
    def test_two_unit_charges(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert pairwise_coulomb_energy(positions, [1.0, 1.0]) == pytest.approx(0.5)

    def test_sign_follows_the_charges(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        assert pairwise_coulomb_energy(positions, [1.0, -1.0]) == pytest.approx(-0.5)

    def test_each_pair_counted_once(self):
        """Three charges on an equilateral triangle: 3 pairs, not 6."""
        side = 2.0
        positions = np.array([
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side / 2, side * np.sqrt(3) / 2, 0.0],
        ])
        assert pairwise_coulomb_energy(positions, np.ones(3)) == pytest.approx(3 / side)

    def test_degenerate_inputs(self):
        assert pairwise_coulomb_energy(np.zeros((0, 3)), []) == 0.0
        assert pairwise_coulomb_energy(np.zeros((1, 3)), [1.0]) == 0.0

    def test_translation_and_rotation_invariance(self):
        rng = np.random.default_rng(0)
        positions = rng.normal(size=(12, 3)) * 3.0
        charges = rng.normal(size=12)
        reference = pairwise_coulomb_energy(positions, charges)

        shifted = positions + np.array([10.0, -4.0, 2.5])
        assert pairwise_coulomb_energy(shifted, charges) == pytest.approx(reference)

        angle = 0.7
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert pairwise_coulomb_energy(positions @ rotation.T,
                                       charges) == pytest.approx(reference)

    def test_matches_an_explicit_double_loop(self):
        rng = np.random.default_rng(1)
        positions = rng.normal(size=(9, 3)) * 2.0
        charges = rng.normal(size=9)

        expected = 0.0
        for i in range(9):
            for j in range(i + 1, 9):
                expected += charges[i] * charges[j] / np.linalg.norm(
                    positions[i] - positions[j])
        assert pairwise_coulomb_energy(positions, charges) == pytest.approx(expected)


class TestEwaldRealSpace:
    def test_zero_shift_reduces_to_a_screened_pair_sum(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        charges = np.array([1.0, 1.0])
        alpha, r_cut = 0.5, 10.0

        energy = ewald_real_energy(positions, charges, np.zeros((1, 3)), alpha, r_cut)
        expected = erfc(alpha * 2.0) / 2.0
        assert energy == pytest.approx(expected)

    def test_self_term_excluded_only_in_the_home_cell(self):
        """A lone atom feels its images but not itself."""
        positions = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([1.0])
        alpha, r_cut, length = 0.4, 6.0, 4.0

        assert ewald_real_energy(positions, charges, np.zeros((1, 3)),
                                 alpha, r_cut) == 0.0

        shifts = np.array([[0.0, 0.0, 0.0], [length, 0.0, 0.0], [-length, 0.0, 0.0]])
        energy = ewald_real_energy(positions, charges, shifts, alpha, r_cut)
        expected = 2 * 0.5 * erfc(alpha * length) / length
        assert energy == pytest.approx(expected)

    def test_cutoff_is_respected(self):
        positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
        charges = np.array([1.0, 1.0])
        assert ewald_real_energy(positions, charges, np.zeros((1, 3)),
                                 0.5, 4.0) == 0.0
        assert ewald_real_energy(positions, charges, np.zeros((1, 3)),
                                 0.5, 6.0) > 0.0

    def test_matches_an_explicit_triple_loop(self):
        rng = np.random.default_rng(2)
        positions = rng.uniform(0.0, 5.0, size=(6, 3))
        charges = rng.normal(size=6)
        shifts = np.array([[i * 5.0, j * 5.0, k * 5.0]
                           for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                          dtype=float)
        alpha, r_cut = 0.35, 7.0

        expected = 0.0
        for i in range(6):
            for j in range(6):
                for shift in shifts:
                    if i == j and np.linalg.norm(shift) < 1e-12:
                        continue
                    distance = np.linalg.norm(positions[i] - (positions[j] + shift))
                    if distance < r_cut:
                        expected += (0.5 * charges[i] * charges[j]
                                     * erfc(alpha * distance) / distance)

        assert ewald_real_energy(positions, charges, shifts, alpha,
                                 r_cut) == pytest.approx(expected)

    @staticmethod
    def _rocksalt_madelung(alpha):
        """Madelung constant of rock-salt NaCl from the full Ewald sum."""
        from poraque.core.grid import Grid
        from poraque.core.system import System
        from poraque.potentials.external import compute_ion_ion_energy

        a = 5.64 / 0.529177210903          # conventional cell, Bohr
        cell = np.eye(3) * a
        fractional = np.array([
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5],
        ])
        charges = np.array([1.0] * 4 + [-1.0] * 4)

        system = System(positions=fractional @ cell,
                        atomic_numbers=np.ones(8, dtype=int), cell=cell)
        grid = Grid((16, 16, 16), cell, pbc=True)

        energy = compute_ion_ion_energy(system, grid, charges=charges,
                                        alpha=alpha, r_cut=3 * a,
                                        k_cut=8 * alpha * 2 * np.pi)
        # E = -(N/2) M q^2 / r_nn for N = 8 ions with r_nn = a/2.
        return -energy * (a / 2.0) / 4.0

    def test_reproduces_the_nacl_madelung_constant(self):
        """Rock-salt NaCl must give M = 1.7475645946.

        This drives the real-space kernel inside the complete
        :func:`compute_ion_ion_energy` and is the strongest available check
        that removing the Numba backend preserved the physics.
        """
        assert self._rocksalt_madelung(0.4) == pytest.approx(1.7475645946, rel=1e-8)

    def test_madelung_is_independent_of_the_splitting_parameter(self):
        """alpha only divides work between the real and reciprocal sums.

        Invariance under alpha is the signature of a correctly balanced Ewald
        implementation: it fails immediately if the real-space kernel's cutoff,
        self-term handling or double-counting factor is wrong.
        """
        values = [self._rocksalt_madelung(alpha) for alpha in (0.3, 0.4, 0.5)]
        assert max(values) - min(values) < 1e-8
