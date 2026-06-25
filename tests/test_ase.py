# -*- coding: utf-8 -*-
# file: test_ase.py
"""Integration tests for the ASE interoperability layer."""

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms

from poraque.ase import PoraqueASE
from poraque.core import System, SolverSettings
from poraque.core.units import ANGSTROM
from poraque.ofdft import run_ofdft


@pytest.fixture
def h_atoms():
    atoms = Atoms("H", positions=[[2.5, 2.5, 2.5]], cell=[5.0, 5.0, 5.0], pbc=True)
    return atoms


class TestRoundTrip:
    def test_system_roundtrip(self, h_atoms):
        system = System.from_ase(h_atoms)
        back = system.to_ase()
        assert np.allclose(back.get_positions(), h_atoms.get_positions(), atol=1e-9)
        assert np.allclose(back.get_cell(), h_atoms.get_cell(), atol=1e-9)
        assert list(back.get_atomic_numbers()) == list(h_atoms.get_atomic_numbers())
        assert tuple(back.get_pbc()) == tuple(h_atoms.get_pbc())

    def test_units_converted_to_bohr(self, h_atoms):
        system = System.from_ase(h_atoms)
        # 5 Å cell -> ~9.45 Bohr.
        assert system.cell[0, 0] == pytest.approx(5.0 * ANGSTROM)

    def test_charge_changes_electron_count(self, h_atoms):
        neutral = System.from_ase(h_atoms)
        cation = System.from_ase(h_atoms, charge=1)
        assert neutral.electrons == 1
        assert cation.electrons == 0


class TestPBC:
    def test_nonperiodic_pbc_preserved(self):
        atoms = Atoms("H", positions=[[2, 2, 2]], cell=[4, 4, 4], pbc=False)
        system = System.from_ase(atoms)
        assert system.pbc == (False, False, False)
        assert tuple(system.to_ase().get_pbc()) == (False, False, False)

    def test_mixed_pbc_preserved(self):
        atoms = Atoms("H", positions=[[2, 2, 2]], cell=[4, 4, 4],
                      pbc=[True, False, True])
        system = System.from_ase(atoms)
        assert system.pbc == (True, False, True)


class TestCalculator:
    def test_is_ase_calculator(self):
        from ase.calculators.calculator import Calculator
        assert issubclass(PoraqueASE, Calculator)

    def test_single_point_energy_in_eV(self, h_atoms):
        settings = SolverSettings(max_iter=40, mixing=0.1, tolerance=1e-6)
        calc = PoraqueASE(grid_shape=(16, 16, 16), settings=settings,
                          external_kwargs={"a": 0.8})
        h_atoms.calc = calc
        energy = h_atoms.get_potential_energy()
        # Energy is a finite float reported in eV.
        assert np.isfinite(energy)
        assert isinstance(float(energy), float)

    def test_forces_shape_and_symmetry(self, h_atoms):
        settings = SolverSettings(max_iter=30, mixing=0.1, tolerance=1e-5)
        calc = PoraqueASE(grid_shape=(16, 16, 16), settings=settings,
                          external_kwargs={"a": 0.8}, fd_step=0.02)
        h_atoms.calc = calc
        forces = h_atoms.get_forces()
        assert forces.shape == (1, 3)
        # A single atom near the cell centre feels a near-zero net force.
        assert np.linalg.norm(forces) < 0.5  # eV/Å


def test_run_ofdft_convenience(h_atoms):
    result = run_ofdft(h_atoms, grid_shape=(16, 16, 16), max_iter=40,
                       external_kwargs={"a": 0.8})
    assert np.isfinite(result.total_energy)
    assert result.density.integrate() == pytest.approx(1.0, rel=1e-5)
