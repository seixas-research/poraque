# -*- coding: utf-8 -*-
# file: test_ksdft.py
"""Tests for the Kohn-Sham DFT real-space SCF engine."""

import numpy as np
import pytest

from poraque.backends.numpy import NumpyBackend
from poraque.core import Grid, SolverSettings, System
from poraque.engine import KSDFTEngine, aufbau_occupations


class _NoXC:
    """A null XC functional, for testing the non-interacting limit."""

    name = "noxc"

    def energy(self, density, system, grid, backend):
        return 0.0

    def potential(self, density, system, grid, backend):
        return np.zeros(grid.shape)


@pytest.fixture
def harmonic():
    """A 3D harmonic well with one electron; exact ground state = 1.5*omega."""
    L, N = 10.0, 32
    cell = np.eye(3) * L
    grid = Grid((N, N, N), cell, pbc=True)
    system = System([[L / 2] * 3], [1], cell, electrons=1)
    coords = grid.get_xyz()
    centre = np.array([L / 2] * 3)
    r2 = np.sum((coords - centre) ** 2, axis=-1)
    omega = 1.0
    v_ext = 0.5 * omega**2 * r2
    return grid, system, v_ext, omega


class TestAufbau:
    def test_closed_shell(self):
        occ = aufbau_occupations(4, 4)
        assert list(occ) == [2.0, 2.0, 0.0, 0.0]

    def test_open_shell_fractional_remainder(self):
        occ = aufbau_occupations(3, 3)
        assert list(occ) == [2.0, 1.0, 0.0]

    def test_total_matches_electrons(self):
        assert aufbau_occupations(5, 7).sum() == pytest.approx(7.0)


class TestKSHarmonic:
    def test_ground_state_eigenvalue(self, harmonic):
        grid, system, v_ext, omega = harmonic
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=20, tolerance=1e-8, mixing=1.0),
                             xc=_NoXC(), hartree=False)
        result = engine.run()
        assert result.state.eigenvalues[0] == pytest.approx(1.5 * omega, abs=1e-4)

    def test_one_electron_energy_equals_eigenvalue(self, harmonic):
        grid, system, v_ext, omega = harmonic
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=20, tolerance=1e-8, mixing=1.0),
                             xc=_NoXC(), hartree=False)
        result = engine.run()
        # For a single non-interacting electron, E_total = epsilon_0.
        assert result.total_energy == pytest.approx(result.state.eigenvalues[0], abs=1e-6)

    def test_excited_states_spacing(self, harmonic):
        grid, system, v_ext, omega = harmonic
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=5, tolerance=1e-8, mixing=1.0),
                             xc=_NoXC(), hartree=False, n_extra_states=3)
        result = engine.run()
        ev = result.state.eigenvalues
        # First excited 3D HO level is at 2.5*omega (spacing omega above ground).
        assert ev[1] == pytest.approx(2.5 * omega, abs=2e-2)

    def test_density_integrates_to_electron_number(self, harmonic):
        grid, system, v_ext, omega = harmonic
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=10, tolerance=1e-8, mixing=1.0),
                             xc=_NoXC(), hartree=False)
        result = engine.run()
        assert result.density.integrate() == pytest.approx(1.0, rel=1e-6)

    def test_orbitals_orthonormal(self, harmonic):
        grid, system, v_ext, omega = harmonic
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=3, tolerance=1e-8, mixing=1.0),
                             xc=_NoXC(), hartree=False, n_extra_states=2)
        result = engine.run()
        orbs = result.state.orbitals
        dV = grid.volume_element
        # <psi_i | psi_j> = delta_ij with the grid volume element.
        n = orbs.shape[0]
        overlap = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                overlap[i, j] = np.sum(orbs[i] * orbs[j]) * dV
        assert np.allclose(overlap, np.eye(n), atol=1e-6)


class TestKSInteracting:
    def test_helium_like_scf_converges(self):
        """A closed-shell 2-electron soft-Coulomb well: SCF should converge."""
        from poraque.potentials import build_external_potential

        L, N = 12.0, 28
        cell = np.eye(3) * L
        grid = Grid((N, N, N), cell, pbc=True)
        system = System([[L / 2] * 3], [2], cell, electrons=2)
        v_ext = build_external_potential(grid, system, kind="soft", a=0.6)
        engine = KSDFTEngine(system, grid, v_ext, NumpyBackend(),
                             SolverSettings(max_iter=60, tolerance=1e-5, mixing=0.3))
        result = engine.run()
        assert result.converged
        assert np.isfinite(result.total_energy)
        assert result.density.integrate() == pytest.approx(2.0, rel=1e-5)
        # Components present and summing consistently is checked elsewhere; here
        # just ensure the SCF lowered/stabilized the energy.
        energies = np.array(result.history["energy"])
        assert np.isfinite(energies).all()
