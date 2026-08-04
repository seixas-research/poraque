# -*- coding: utf-8 -*-
# file: test_calculator.py

"""
Tests for the ASE calculator, :class:`poraque.calculator.Poraque`.

The models are built and saved inside the tests rather than loaded from
``models/``: a test that depends on a checkpoint someone happened to leave in
the working tree passes or fails for reasons that have nothing to do with the
code under test. Untrained weights are fine here — what is being checked is the
plumbing, the ASE protocol and the guard rails, not the physics of the
prediction.
"""

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms                                              # noqa: E402

from poraque.calculator import Poraque, _grid_shape                # noqa: E402
from poraque.ml import FieldOperator                               # noqa: E402


@pytest.fixture
def atoms():
    """Two-atom Au cell, small enough to run in a fraction of a second."""
    return Atoms("Au2", cell=np.eye(3) * 4.08, pbc=True,
                 scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture
def operators(tmp_path):
    """A saved ext2chg/chg2tau pair with untrained weights."""
    paths = {}
    for task in ("ext2chg", "chg2tau"):
        operator = FieldOperator(task, width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu",
                                 training_resolution=16)
        paths[task] = str(tmp_path / f"{task}.pt")
        operator.save(paths[task])
    return paths


@pytest.fixture
def calculator(operators):
    return Poraque(operators["ext2chg"], operators["chg2tau"],
                   charges={"Au": 11.0}, device="cpu")


# ===================================================================== #
# ASE protocol
# ===================================================================== #
class TestAseProtocol:
    def test_implemented_properties(self):
        assert "energy" in Poraque.implemented_properties
        # ASE optimizers ask for 'free_energy' by name; omitting it makes them
        # fail on a calculator that can in fact answer.
        assert "free_energy" in Poraque.implemented_properties

    def test_get_potential_energy_returns_a_finite_scalar(self, atoms,
                                                          calculator):
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning, match="Gaussian"):
            energy = atoms.get_potential_energy()
        assert np.isscalar(energy) or np.ndim(energy) == 0
        assert np.isfinite(energy)

    def test_free_energy_equals_energy(self, atoms, calculator):
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            energy = atoms.get_potential_energy()
        assert calculator.results["free_energy"] == pytest.approx(energy)

    def test_results_carry_the_decomposition(self, atoms, calculator):
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            atoms.get_potential_energy()
        payload = calculator.results["energy_components"]
        assert payload["ewald"] is not None
        assert payload["total"] == pytest.approx(calculator.results["energy"])
        # No POTCAR, so the G=0 remainder is unavailable and must say so.
        assert "alpha_z" in payload["missing"]

    def test_forces_raise_not_implemented(self, atoms, calculator):
        atoms.calc = calculator
        with pytest.raises(NotImplementedError, match="forces"):
            calculator.get_forces(atoms)

    def test_stress_raises_not_implemented(self, atoms, calculator):
        atoms.calc = calculator
        with pytest.raises(NotImplementedError, match="stress"):
            calculator.get_stress(atoms)

    def test_energy_is_recomputed_after_the_geometry_changes(self, atoms,
                                                             calculator):
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            first = atoms.get_potential_energy()
        atoms.set_cell(np.eye(3) * 4.5, scale_atoms=True)
        second = atoms.get_potential_energy()
        assert first != second


# ===================================================================== #
# Pipeline
# ===================================================================== #
class TestPipeline:
    def test_predicts_three_fields_on_one_grid(self, atoms, calculator):
        with pytest.warns(RuntimeWarning):
            fields = calculator.predict_fields(atoms)
        assert set(fields) == {"external", "density", "tau"}
        grids = {tuple(field.grid.shape) for field in fields.values()}
        assert len(grids) == 1, "the three fields must share one mesh"

    def test_fields_are_kept_after_an_energy_call(self, atoms, calculator):
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            atoms.get_potential_energy()
        assert set(calculator.fields) == {"external", "density", "tau"}

    def test_resolution_comes_from_the_checkpoint(self, operators):
        calculator = Poraque(operators["ext2chg"], operators["chg2tau"],
                             charges={"Au": 11.0}, device="cpu")
        assert calculator.resolution == 16

    def test_explicit_resolution_wins(self, operators):
        calculator = Poraque(operators["ext2chg"], operators["chg2tau"],
                             charges={"Au": 11.0}, resolution=20, device="cpu")
        assert calculator.resolution == 20

    def test_external_potential_has_zero_mean(self, atoms, calculator):
        """The G=0 convention that makes V_ext addable to v_H."""
        with pytest.warns(RuntimeWarning):
            potential = calculator.build_external_potential(atoms)
        assert abs(np.asarray(potential).mean()) < 1e-8


# ===================================================================== #
# Guard rails
# ===================================================================== #
class TestGuardRails:
    def test_rejects_a_non_periodic_cell(self, calculator):
        molecule = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]],
                         cell=np.eye(3) * 10.0, pbc=False)
        with pytest.raises(ValueError, match="periodic"):
            calculator.build_external_potential(molecule)

    def test_requires_charges_or_a_potcar(self, operators):
        with pytest.raises(ValueError, match="POTCAR"):
            Poraque(operators["ext2chg"], operators["chg2tau"], device="cpu")

    def test_rejects_swapped_checkpoints(self, operators):
        """Chaining the wrong model produces plausible-looking garbage."""
        with pytest.raises(ValueError, match="task"):
            Poraque(operators["chg2tau"], operators["ext2chg"],
                    charges={"Au": 11.0}, device="cpu")

    def test_rejects_a_live_operator_for_the_wrong_task(self, operators):
        wrong = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                              device="cpu")
        with pytest.raises(ValueError, match="ext2chg"):
            Poraque(wrong, operators["chg2tau"], charges={"Au": 11.0},
                    device="cpu")

    def test_accepts_live_operators(self, atoms):
        pair = [FieldOperator(task, width=4, modes=2, n_layers=1,
                              projection_channels=8, device="cpu")
                for task in ("ext2chg", "chg2tau")]
        calculator = Poraque(*pair, charges={"Au": 11.0}, resolution=12,
                             device="cpu")
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            assert np.isfinite(atoms.get_potential_energy())

    def test_warns_once_about_the_gaussian_fallback(self, atoms, calculator):
        """
        Loud the first time, silent afterwards: an MD-style loop would
        otherwise bury the terminal, and a warning nobody reads is no warning.
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calculator.build_external_potential(atoms)
            calculator.build_external_potential(atoms)
        gaussian = [w for w in caught if "Gaussian" in str(w.message)]
        assert len(gaussian) == 1


# ===================================================================== #
# Grid sizing
# ===================================================================== #
class TestGridShape:
    def test_cubic_cell_gives_a_cubic_grid(self):
        assert _grid_shape(np.eye(3) * 5.0, 32) == (32, 32, 32)

    def test_aspect_ratio_is_preserved(self):
        shape = _grid_shape(np.diag([10.0, 10.0, 20.0]), 32)
        assert shape[2] == 32
        assert shape[0] == shape[1] < shape[2]

    def test_sizes_are_fft_friendly(self):
        for shape in (_grid_shape(np.diag([7.3, 11.9, 5.1]), 30),
                      _grid_shape(np.eye(3) * 3.0, 17)):
            for n in shape:
                remaining = n
                for factor in (2, 3, 5, 7):
                    while remaining % factor == 0:
                        remaining //= factor
                assert remaining == 1, f"{n} has a large prime factor"

    def test_never_degenerates_to_a_tiny_axis(self):
        shape = _grid_shape(np.diag([100.0, 1.0, 1.0]), 32)
        assert min(shape) >= 4
