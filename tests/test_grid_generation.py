# -*- coding: utf-8 -*-
# file: test_grid_generation.py
"""Tests for automatic plane-wave grid generation and basis enforcement.

These tests verify that the real-space grid is generated automatically from the
plane-wave kinetic-energy cutoff (``ecut``), that the resulting grid dimensions
match the Nyquist condition expected for that cutoff, and that a manually
supplied grid is safely overridden (with a warning) when an ``ecut`` is given.
They also check that the plane-wave basis is mandatory for KS-DFT.
"""

import warnings

import numpy as np
import pytest

from poraque.core import Grid


def _expected_shape(cell, ecut, density_factor=2.0):
    """Reference Nyquist grid dimensions for a plane-wave cutoff.

    The density cutoff wavevector is ``density_factor * sqrt(2 ecut)``; the grid
    spacing must satisfy ``h <= pi / g_max``, i.e.
    ``N_i >= L_i * density_factor * sqrt(2 ecut) / pi``. FFT-friendly even sizes
    are used (never fewer than 2 points).
    """
    cell = np.asarray(cell, dtype=float)
    g_max = np.sqrt(2.0 * ecut)
    lengths = np.linalg.norm(cell, axis=1)
    n = np.ceil(lengths * density_factor * g_max / np.pi).astype(int)
    return tuple(int(max(2, ni + (ni % 2))) for ni in n)


class TestFromEcutDimensions:
    @pytest.mark.parametrize("ecut", [5.0, 10.0, 20.0, 40.0])
    def test_shape_matches_nyquist(self, ecut):
        cell = np.eye(3) * 10.0
        grid = Grid.from_ecut(cell, ecut=ecut)
        assert grid.shape == _expected_shape(cell, ecut)
        assert grid.ecut == ecut

    def test_anisotropic_cell(self):
        cell = np.diag([8.0, 12.0, 6.0])
        grid = Grid.from_ecut(cell, ecut=15.0)
        assert grid.shape == _expected_shape(cell, ecut=15.0)
        # Longer lattice vectors need more points.
        assert grid.Ny > grid.Nx > grid.Nz

    def test_higher_cutoff_is_denser(self):
        cell = np.eye(3) * 10.0
        coarse = Grid.from_ecut(cell, ecut=10.0)
        fine = Grid.from_ecut(cell, ecut=40.0)
        assert all(f >= c for f, c in zip(fine.shape, coarse.shape))
        assert any(f > c for f, c in zip(fine.shape, coarse.shape))

    def test_spacing_resolves_cutoff(self):
        # The grid spacing must satisfy the Nyquist condition h <= pi / g_max.
        cell = np.eye(3) * 9.0
        ecut = 25.0
        grid = Grid.from_ecut(cell, ecut=ecut, density_factor=2.0)
        g_max = 2.0 * np.sqrt(2.0 * ecut)
        assert np.all(grid.h <= np.pi / g_max + 1e-9)

    def test_even_fft_friendly_shape(self):
        grid = Grid.from_ecut(np.eye(3) * 7.0, ecut=18.0)
        assert all(n % 2 == 0 for n in grid.shape)

    def test_density_factor_changes_dimensions(self):
        cell = np.eye(3) * 10.0
        sharp = Grid.from_ecut(cell, ecut=20.0, density_factor=4.0)
        smooth = Grid.from_ecut(cell, ecut=20.0, density_factor=2.0)
        assert all(s >= m for s, m in zip(sharp.shape, smooth.shape))
        assert any(s > m for s, m in zip(sharp.shape, smooth.shape))


class TestCalculatorGridGeneration:
    """The ASE calculator must generate / override grids from ``ecut``."""

    def _atoms(self):
        ase = pytest.importorskip("ase")
        from ase import Atoms
        return Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5.0, 5.0, 5.0], pbc=True)

    def test_ecut_generates_expected_grid(self):
        pytest.importorskip("ase")
        from poraque.ase import Poraque
        from poraque.core import System

        atoms = self._atoms()
        calc = Poraque(mode="ks", ecut=12.0)
        system = System.from_ase(atoms)
        grid = calc._build_grid(system)
        assert grid.shape == _expected_shape(system.cell, ecut=12.0)
        assert grid.ecut == 12.0

    def test_manual_grid_is_overridden_with_warning(self):
        pytest.importorskip("ase")
        from poraque.ase import Poraque
        from poraque.core import System

        atoms = self._atoms()
        # A coarse manual grid together with a cutoff that needs a finer one.
        calc = Poraque(mode="ks", grid_shape=(8, 8, 8), ecut=20.0)
        system = System.from_ase(atoms)
        with pytest.warns(UserWarning, match="ecut"):
            grid = calc._build_grid(system)
        # The automatically optimized grid wins.
        assert grid.shape == _expected_shape(system.cell, ecut=20.0)
        assert grid.shape != (8, 8, 8)

    def test_explicit_grid_used_without_ecut(self):
        pytest.importorskip("ase")
        from poraque.ase import Poraque
        from poraque.core import System

        atoms = self._atoms()
        calc = Poraque(mode="ks", grid_shape=(18, 18, 18))
        system = System.from_ase(atoms)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no override warning expected
            grid = calc._build_grid(system)
        assert grid.shape == (18, 18, 18)


class TestPlaneWaveBasisEnforced:
    def test_ks_rejects_non_pw_basis(self):
        pytest.importorskip("ase")
        from poraque.ase import Poraque

        with pytest.raises(ValueError, match="plane-wave"):
            Poraque(mode="ks", basis="gaussian")

    def test_ks_accepts_pw_aliases(self):
        pytest.importorskip("ase")
        from poraque.ase import Poraque

        for alias in ("pw", "PW", "plane-wave", "planewave"):
            calc = Poraque(mode="ks", basis=alias)
            assert calc.basis == "pw"

    def test_low_level_ks_driver_requires_pw(self):
        from poraque.calculator import KSDFTCalculator
        from poraque.core import Grid, System

        cell = np.eye(3) * 6.0
        grid = Grid((12, 12, 12), cell)
        system = System([[3, 3, 3]], [1], cell, electrons=1)
        v_ext = np.zeros(grid.shape)
        with pytest.raises(ValueError, match="plane-wave"):
            KSDFTCalculator(system, grid, v_ext, basis="lcao")
