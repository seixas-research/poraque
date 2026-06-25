# -*- coding: utf-8 -*-
# file: test_ofdft.py
"""Integration tests for the OF-DFT minimization engine."""

import numpy as np
import pytest

from poraque.calculator import Poraque
from poraque.core import Density, SolverSettings
from poraque.engine import OFDFTEngine
from poraque.functionals import External, Hartree, LDA, TFvW, ThomasFermi
from poraque.potentials import build_external_potential


@pytest.fixture
def ofdft_calc(coarse_grid, hydrogen_system):
    """A TFvW + Hartree + XC + external OF-DFT calculator on a soft H potential."""
    v_ext = build_external_potential(coarse_grid, hydrogen_system, kind="soft", a=0.8)
    functionals = [
        TFvW(lambda_vw=1.0),
        Hartree(),
        LDA(),
        External(v_ext),
    ]
    settings = SolverSettings(max_iter=60, tolerance=1e-6, mixing=0.1)
    return Poraque(hydrogen_system, coarse_grid, functionals, settings=settings)


def test_runs_and_returns_result(ofdft_calc):
    result = ofdft_calc.calculate()
    assert np.isfinite(result.total_energy)
    assert result.iterations >= 1
    assert set(["energy", "residual", "mu"]).issubset(result.history)


def test_electron_number_conserved(ofdft_calc):
    result = ofdft_calc.calculate()
    assert result.density.integrate() == pytest.approx(
        ofdft_calc.system.electrons, rel=1e-6
    )


def test_density_non_negative(ofdft_calc):
    result = ofdft_calc.calculate()
    ok, min_val, _ = result.density.check_positivity()
    assert ok, f"negative density encountered: {min_val}"


def test_energy_monotonically_decreases(ofdft_calc):
    result = ofdft_calc.calculate()
    energies = np.array(result.history["energy"])
    # The backtracking line search guarantees a non-increasing energy.
    diffs = np.diff(energies)
    assert np.all(diffs <= 1e-9)


def test_energy_components_reported(ofdft_calc):
    result = ofdft_calc.calculate()
    names = result.energy_components.keys()
    assert any("von Weizs" in n for n in names)
    assert "Hartree" in names
    assert "External" in names
    # Components sum to the total energy.
    assert sum(result.energy_components.values()) == pytest.approx(
        result.total_energy
    )


def test_cg_converges_to_tolerance(coarse_grid, hydrogen_system):
    """The conjugate-gradient minimizer reaches the requested tolerance."""
    v_ext = build_external_potential(coarse_grid, hydrogen_system, kind="soft", a=0.8)
    functionals = [TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)]
    settings = SolverSettings(max_iter=300, tolerance=1e-6, mixing=0.1)
    result = Poraque(hydrogen_system, coarse_grid, functionals,
                     settings=settings).calculate()
    assert result.converged
    assert result.history["residual"][-1] < 1e-6


def test_cg_beats_steepest_descent(coarse_grid, hydrogen_system):
    """CG needs fewer iterations than steepest descent (cg_restart=1)."""
    v_ext = build_external_potential(coarse_grid, hydrogen_system, kind="soft", a=0.8)
    functionals = [TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)]

    def run(cg_restart):
        settings = SolverSettings(max_iter=500, tolerance=1e-5, mixing=0.1,
                                  cg_restart=cg_restart)
        return Poraque(hydrogen_system, coarse_grid, functionals,
                       settings=settings).calculate()

    cg = run(20)
    sd = run(1)  # restart every step == steepest descent
    assert cg.converged
    assert cg.iterations < sd.iterations
    # Both reach the same minimum when they converge.
    assert cg.total_energy == pytest.approx(sd.total_energy, abs=1e-4)


def test_projected_gradient_is_tangent(coarse_grid, hydrogen_system):
    """The chemical-potential projection keeps the gradient tangent: <g, w> = 0."""
    from poraque.backends.numpy import NumpyBackend
    from poraque.core import Density

    backend = NumpyBackend()
    v_ext = build_external_potential(coarse_grid, hydrogen_system, kind="soft", a=0.8)
    functionals = [TFvW(lambda_vw=1.0), Hartree(), LDA(), External(v_ext)]
    engine = OFDFTEngine(hydrogen_system, coarse_grid, functionals, backend,
                         SolverSettings())

    n0 = hydrogen_system.electrons / coarse_grid.volume
    rho = Density(coarse_grid, np.full(coarse_grid.shape, n0))
    w = np.sqrt(rho.data)
    v_eff = engine.compute_effective_potential(rho)
    mu = backend.integrate(w**2 * v_eff, coarse_grid) / hydrogen_system.electrons
    g = 2 * w * (v_eff - mu)
    assert backend.integrate(g * w, coarse_grid) == pytest.approx(0.0, abs=1e-9)


def test_uniform_tf_reference(coarse_grid):
    """A pure-TF energy on a uniform density matches the analytic value."""
    from poraque.core import System

    system = System([[5, 5, 5]], [1], coarse_grid.cell, electrons=3)
    n0 = system.electrons / coarse_grid.volume
    rho = Density(coarse_grid, np.full(coarse_grid.shape, n0))
    tf = ThomasFermi()
    from poraque.backends.numpy import NumpyBackend

    e = tf.energy(rho, system, coarse_grid, NumpyBackend())
    assert e == pytest.approx(tf.C_TF * n0 ** (5 / 3) * coarse_grid.volume)
