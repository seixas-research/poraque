# -*- coding: utf-8 -*-
# file: test_fde.py
"""Tests for the Frozen-Density Embedding (subsystem DFT) engine."""

import numpy as np
import pytest

from poraque.backends.numpy import NumpyBackend
from poraque.core import Density, Grid, SolverSettings, System
from poraque.fde import FDEEngine, Subsystem
from poraque.functionals import TFvW


@pytest.fixture
def dimer_engine():
    L, N = 12.0, 20
    cell = np.eye(3) * L
    grid = Grid((N, N, N), cell, pbc=True)
    sA = System([[5.0, 6.0, 6.0]], [1], cell, electrons=1)
    sB = System([[7.0, 6.0, 6.0]], [1], cell, electrons=1)
    subs = [
        Subsystem(sA, method="of", name="A", external_kwargs={"a": 0.8}),
        Subsystem(sB, method="of", name="B", external_kwargs={"a": 0.8}),
    ]
    return FDEEngine(
        subs, grid, NumpyBackend(),
        settings=SolverSettings(max_iter=8, tolerance=1e-4),
        inner_settings=SolverSettings(max_iter=30, tolerance=1e-6, mixing=0.1),
    )


class TestPartitioning:
    def test_total_density_sums_subsystems(self, dimer_engine):
        n_tot = dimer_engine.total_density()
        expected = sum(s.density.data for s in dimer_engine.subsystems)
        assert np.allclose(n_tot.data, expected)

    def test_total_density_electron_count(self, dimer_engine):
        # Each subsystem has 1 electron -> total integrates to 2.
        assert dimer_engine.total_density().integrate() == pytest.approx(2.0)


class TestEmbeddingPotential:
    def test_vanishes_with_single_subsystem(self):
        L, N = 10.0, 16
        cell = np.eye(3) * L
        grid = Grid((N, N, N), cell, pbc=True)
        s = System([[5, 5, 5]], [1], cell, electrons=1)
        eng = FDEEngine([Subsystem(s, method="of", external_kwargs={"a": 0.8})],
                        grid, NumpyBackend())
        v_emb = eng.embedding_potential(0)
        # No other subsystem -> embedding potential is identically zero.
        assert np.allclose(v_emb, 0.0, atol=1e-12)

    def test_nonadditive_parts_vanish_when_separated(self):
        """The (short-ranged) nonadditive XC/kinetic vanish for disjoint densities."""
        L, N = 16.0, 24
        cell = np.eye(3) * L
        grid = Grid((N, N, N), cell, pbc=True)
        backend = NumpyBackend()
        coords = grid.get_xyz()

        def gaussian(centre, width=0.8):
            r2 = np.sum((coords - centre) ** 2, axis=-1)
            g = np.exp(-r2 / (2 * width**2))
            rho = Density(grid, g)
            rho.normalize(1.0)
            return rho

        n_a = gaussian(np.array([3.0, 8.0, 8.0]))
        n_b = gaussian(np.array([13.0, 8.0, 8.0]))  # far away
        n_tot = Density(grid, n_a.data + n_b.data)

        kedf = TFvW(lambda_vw=1.0)
        v_ts_nad = (kedf.potential(n_tot, None, grid, backend)
                    - kedf.potential(n_a, None, grid, backend))
        # In the region where A lives, B's density is negligible so the
        # nonadditive kinetic potential is essentially zero there.
        near_a = np.sum((coords - np.array([3.0, 8.0, 8.0])) ** 2, axis=-1) < 1.0
        assert np.max(np.abs(v_ts_nad[near_a])) < 1e-3


class TestNonadditiveDerivative:
    def test_nonadditive_kinetic_derivative(self):
        """v_Ts^nad = v_Ts[n_A+n_B] - v_Ts[n_A] matches d/dn_A of T^nad."""
        L, N = 12.0, 20
        cell = np.eye(3) * L
        grid = Grid((N, N, N), cell, pbc=True)
        backend = NumpyBackend()
        coords = grid.get_xyz()

        def gaussian(centre, width=1.0, n=1.0):
            r2 = np.sum((coords - centre) ** 2, axis=-1)
            rho = Density(grid, np.exp(-r2 / (2 * width**2)) + 0.01)
            rho.normalize(n)
            return rho

        n_a = gaussian(np.array([5.0, 6.0, 6.0]))
        n_b = gaussian(np.array([7.0, 6.0, 6.0]))
        kedf = TFvW(lambda_vw=0.2)

        def t_nad(data_a):
            ra = Density(grid, data_a)
            rab = Density(grid, data_a + n_b.data)
            return (kedf.energy(rab, None, grid, backend)
                    - kedf.energy(ra, None, grid, backend)
                    - kedf.energy(n_b, None, grid, backend))

        # Analytic nonadditive kinetic potential.
        n_tot = Density(grid, n_a.data + n_b.data)
        v_nad = (kedf.potential(n_tot, None, grid, backend)
                 - kedf.potential(n_a, None, grid, backend))

        dn = 1e-3 * np.cos(2 * np.pi * coords[..., 0] / L) * np.sin(
            2 * np.pi * coords[..., 1] / L)
        analytic = backend.integrate(v_nad * dn, grid)
        numeric = (t_nad(n_a.data + dn) - t_nad(n_a.data - dn)) / 2.0
        assert numeric == pytest.approx(analytic, rel=1e-4, abs=1e-9)


class TestFreezeAndThaw:
    def test_runs_and_conserves_electrons(self, dimer_engine):
        result = dimer_engine.freeze_and_thaw()
        assert result.density.integrate() == pytest.approx(2.0, rel=1e-5)

    def test_energy_stabilizes(self, dimer_engine):
        result = dimer_engine.freeze_and_thaw()
        energies = np.array(result.history["energy"])
        # The freeze-and-thaw energy converges: later changes shrink.
        diffs = np.abs(np.diff(energies))
        assert diffs[-1] < diffs[0]
        assert diffs[-1] < 1e-3

    def test_density_residual_decreases(self, dimer_engine):
        result = dimer_engine.freeze_and_thaw()
        res = result.history["density_residual"]
        assert res[-1] < res[0]

    def test_energy_components_present(self, dimer_engine):
        result = dimer_engine.freeze_and_thaw()
        keys = result.energy_components.keys()
        assert "Intra" in keys
        assert "Kinetic (nonadditive)" in keys
        assert "XC (nonadditive)" in keys
        assert sum(result.energy_components.values()) == pytest.approx(
            result.total_energy
        )


class TestMixedEmbedding:
    def test_ks_in_of_runs(self):
        """A KS active region embedded in an OF frozen region (mixed FDE)."""
        L, N = 12.0, 18
        cell = np.eye(3) * L
        grid = Grid((N, N, N), cell, pbc=True)
        sA = System([[5.0, 6.0, 6.0]], [2], cell, electrons=2)  # KS region
        sB = System([[8.0, 6.0, 6.0]], [2], cell, electrons=2)  # OF region
        subs = [
            Subsystem(sA, method="ks", name="KS", external_kwargs={"a": 0.7}),
            Subsystem(sB, method="of", name="OF", external_kwargs={"a": 0.7}),
        ]
        eng = FDEEngine(
            subs, grid, NumpyBackend(),
            settings=SolverSettings(max_iter=2, tolerance=1e-3),
            inner_settings=SolverSettings(max_iter=20, tolerance=1e-4, mixing=0.3),
        )
        result = eng.freeze_and_thaw()
        assert np.isfinite(result.total_energy)
        assert result.density.integrate() == pytest.approx(4.0, rel=1e-4)
