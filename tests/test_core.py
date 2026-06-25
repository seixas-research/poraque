# -*- coding: utf-8 -*-
# file: test_core.py
"""Unit tests for the core data model: Grid, Density, System, units."""

import numpy as np
import pytest

from poraque.core import Density, Grid, System
from poraque.core import units


class TestGrid:
    def test_volume_and_element(self, grid):
        # Cubic 10 Bohr cell -> volume 1000, element = 1000 / N
        assert grid.volume == pytest.approx(1000.0)
        assert grid.volume_element == pytest.approx(1000.0 / grid.N)

    def test_spacing(self, grid):
        assert np.allclose(grid.h, 10.0 / 24.0)

    def test_integrate_constant(self, grid):
        # Integral of a constant field equals value * volume.
        field = np.full(grid.shape, 2.5)
        assert grid.integrate(field) == pytest.approx(2.5 * grid.volume)

    def test_get_xyz_shape_and_range(self, grid):
        coords = grid.get_xyz()
        assert coords.shape == (*grid.shape, 3)
        assert coords.min() >= 0.0
        assert coords.max() < 10.0

    def test_rejects_bad_cell(self):
        with pytest.raises(ValueError):
            Grid((4, 4, 4), np.eye(2))

    def test_g2_is_nonnegative_with_single_zero(self, grid):
        g2 = grid.get_g2()
        assert g2.shape == grid.shape
        assert g2.min() == pytest.approx(0.0)
        # Exactly one component (the G=0 term) is zero.
        assert np.count_nonzero(g2 == 0.0) == 1


class TestDensity:
    def test_integrate_constant_gives_electron_number(self, grid):
        # A uniform density of N/V electrons integrates back to N.
        n_electrons = 8.0
        data = np.full(grid.shape, n_electrons / grid.volume)
        rho = Density(grid, data)
        assert rho.integrate() == pytest.approx(n_electrons)

    def test_normalize(self, grid):
        rho = Density(grid, np.random.default_rng(0).random(grid.shape) + 0.1)
        rho.normalize(5.0)
        assert rho.integrate() == pytest.approx(5.0)

    def test_positivity_check(self, grid):
        good = Density(grid, np.ones(grid.shape))
        ok, _, idx = good.check_positivity()
        assert ok and idx is None

        bad_data = np.ones(grid.shape)
        bad_data[0, 0, 0] = -1.0
        bad = Density(grid, bad_data)
        ok, min_val, idx = bad.check_positivity()
        assert not ok
        assert min_val == pytest.approx(-1.0)
        assert idx == (0, 0, 0)

    def test_shape_mismatch_raises(self, grid):
        with pytest.raises(ValueError):
            Density(grid, np.ones((2, 2, 2)))


class TestSystem:
    def test_default_electrons_neutral(self, cubic_cell):
        sys = System([[0, 0, 0], [2, 0, 0]], [1, 8], cubic_cell)
        assert sys.electrons == 9  # neutral H + O

    def test_explicit_electrons(self, cubic_cell):
        sys = System([[0, 0, 0]], [8], cubic_cell, electrons=10)  # O 2-
        assert sys.electrons == 10

    def test_pbc_broadcast(self, cubic_cell):
        sys = System([[0, 0, 0]], [1], cubic_cell, pbc=True)
        assert sys.pbc == (True, True, True)


class TestUnits:
    def test_hartree_ev_roundtrip(self):
        # 1 Hartree expressed in eV then back is consistent.
        assert units.HARTREE_TO_EV * units.EV == pytest.approx(1.0)

    def test_bohr_angstrom_roundtrip(self):
        assert units.BOHR_TO_ANGSTROM * units.ANGSTROM == pytest.approx(1.0)
