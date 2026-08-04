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

import gzip
import os

import numpy as np
import pytest

ase = pytest.importorskip("ase")

from ase import Atoms                                              # noqa: E402

from poraque.calculator import Poraque, _grid_shape                # noqa: E402
from poraque.ml import (  # noqa: E402
    BUNDLE_FILENAME,
    FieldOperator,
    save_bundle,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def atoms():
    """Two-atom Au cell, small enough to run in a fraction of a second."""
    return Atoms("Au2", cell=np.eye(3) * 4.08, pbc=True,
                 scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture
def operators(tmp_path):
    """A unified bundle holding both tasks, with untrained weights."""
    built = {
        task: FieldOperator(task, width=4, modes=2, n_layers=1,
                            projection_channels=8, device="cpu",
                            training_resolution=16)
        for task in ("ext2chg", "chg2tau")
    }
    return save_bundle(str(tmp_path / BUNDLE_FILENAME), built)


@pytest.fixture
def calculator(operators):
    return Poraque(operators, charges={"Au": 11.0}, device="cpu")


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
        calculator = Poraque(operators,
                             charges={"Au": 11.0}, device="cpu")
        assert calculator.resolution == 16

    def test_explicit_resolution_wins(self, operators):
        calculator = Poraque(operators,
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
            Poraque(operators, device="cpu")

    def test_rejects_a_bundle_missing_a_task(self, tmp_path):
        """
        A half-populated bundle must fail loudly. Falling back to whatever the
        other key holds would chain the wrong model and produce
        plausible-looking garbage.
        """
        partial = save_bundle(str(tmp_path / BUNDLE_FILENAME), {
            "ext2chg": FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                     device="cpu")})
        with pytest.raises(KeyError, match="chg2tau"):
            Poraque(partial, charges={"Au": 11.0}, device="cpu")

    def test_rejects_a_single_operator_checkpoint(self, tmp_path):
        """A bare FieldOperator file is not a bundle, and says so."""
        path = str(tmp_path / "ext2chg.pfno")
        FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                      device="cpu").save(path)
        with pytest.raises(ValueError, match="not a Poraque model bundle"):
            Poraque(path, charges={"Au": 11.0}, device="cpu")

    def test_rejects_a_live_operator_for_the_wrong_task(self, operators):
        wrong = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                              device="cpu")
        with pytest.raises(ValueError, match="ext2chg"):
            Poraque(operators, ext2chg=wrong, charges={"Au": 11.0},
                    device="cpu")

    def test_accepts_live_operators(self, atoms):
        pair = {task: FieldOperator(task, width=4, modes=2, n_layers=1,
                                    projection_channels=8, device="cpu")
                for task in ("ext2chg", "chg2tau")}
        calculator = Poraque(ext2chg=pair["ext2chg"], chg2tau=pair["chg2tau"],
                             charges={"Au": 11.0}, resolution=12, device="cpu")
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
# POTCAR library lookup
# ===================================================================== #
@pytest.fixture
def potcar_text():
    """A real POTCAR from the reference dataset, or skip."""
    path = os.path.join(_ROOT, "data", "vasp", "struct_000", "POTCAR")
    if not os.path.exists(path):
        pytest.skip("reference POTCAR not available")
    with open(path, "r", errors="replace") as handle:
        return handle.read()


def _library(tmp_path, layout, text, element="Au"):
    """Write a one-element POTCAR library in the requested layout."""
    root = tmp_path / f"lib_{layout}"
    if layout == "nested":
        (root / element).mkdir(parents=True)
        (root / element / "POTCAR").write_text(text)
    elif layout == "gzip":
        (root / element).mkdir(parents=True)
        with gzip.open(root / element / "POTCAR.gz", "wt") as handle:
            handle.write(text)
    elif layout == "flat":
        root.mkdir(parents=True)
        (root / f"POTCAR.{element}").write_text(text)
    elif layout == "flat-suffix":
        root.mkdir(parents=True)
        (root / f"{element}.POTCAR").write_text(text)
    elif layout == "variant":
        (root / f"{element}_pv").mkdir(parents=True)
        (root / f"{element}_pv" / "POTCAR").write_text(text)
    elif layout == "ambiguous":
        for suffix in ("_pv", "_sv"):
            (root / f"{element}{suffix}").mkdir(parents=True)
            (root / f"{element}{suffix}" / "POTCAR").write_text(text)
    else:                                                  # pragma: no cover
        raise ValueError(layout)
    return str(root)


class TestPotcarDir:
    @pytest.fixture
    def gold(self):
        return Atoms("Au", cell=np.eye(3) * 4.08, pbc=True,
                     scaled_positions=[[0.0, 0.0, 0.0]])

    @pytest.mark.parametrize("layout", ["nested", "gzip", "flat",
                                        "flat-suffix"])
    def test_every_layout_is_found(self, tmp_path, potcar_text, operators,
                                   gold, layout):
        calculator = Poraque(operators,
                             potcar_dir=_library(tmp_path, layout, potcar_text),
                             resolution=12, device="cpu")
        gold.calc = calculator
        assert np.isfinite(gold.get_potential_energy())
        # A real POTCAR carries the tables, so the G=0 term is available.
        assert calculator.components.alpha_z is not None
        assert calculator.components.missing == ()

    def test_all_layouts_agree(self, tmp_path, potcar_text, operators, gold):
        energies = []
        for layout in ("nested", "gzip", "flat"):
            calculator = Poraque(
                operators, resolution=12,
                potcar_dir=_library(tmp_path, layout, potcar_text),
                device="cpu")
            atoms = gold.copy()
            atoms.calc = calculator
            energies.append(atoms.get_potential_energy())
        assert energies[0] == pytest.approx(energies[1])
        assert energies[0] == pytest.approx(energies[2])

    def test_entries_are_cached_per_composition(self, tmp_path, potcar_text,
                                                operators, gold):
        """Parsing the tables is the expensive part of a scan over geometries."""
        calculator = Poraque(operators,
                             potcar_dir=_library(tmp_path, "nested",
                                                 potcar_text),
                             resolution=12, device="cpu")
        gold.calc = calculator
        gold.get_potential_energy()
        first = calculator._potcar_cache[("Au",)]
        gold.set_cell(np.eye(3) * 4.2, scale_atoms=True)
        gold.get_potential_energy()
        assert calculator._potcar_cache[("Au",)] is first

    def test_single_variant_is_used_with_a_warning(self, tmp_path, potcar_text,
                                                   operators, gold):
        calculator = Poraque(operators,
                             potcar_dir=_library(tmp_path, "variant",
                                                 potcar_text),
                             resolution=12, device="cpu")
        gold.calc = calculator
        with pytest.warns(RuntimeWarning, match="variant"):
            assert np.isfinite(gold.get_potential_energy())

    def test_several_variants_raise_rather_than_guess(self, tmp_path,
                                                      potcar_text, operators,
                                                      gold):
        """
        Fe vs Fe_pv differ in ZVAL and therefore in every energy. Choosing
        one by sort order would be a silent physics decision.
        """
        calculator = Poraque(operators,
                             potcar_dir=_library(tmp_path, "ambiguous",
                                                 potcar_text),
                             resolution=12, device="cpu")
        gold.calc = calculator
        with pytest.raises(ValueError, match="variants"):
            gold.get_potential_energy()

    def test_missing_element_names_what_it_looked_for(self, tmp_path,
                                                      potcar_text, operators):
        calculator = Poraque(operators,
                             potcar_dir=_library(tmp_path, "nested",
                                                 potcar_text),
                             resolution=12, device="cpu")
        silicon = Atoms("Si", cell=np.eye(3) * 5.43, pbc=True,
                        scaled_positions=[[0.0, 0.0, 0.0]])
        silicon.calc = calculator
        with pytest.raises(FileNotFoundError, match="Si"):
            silicon.get_potential_energy()

    def test_rejects_a_directory_that_does_not_exist(self, operators, tmp_path):
        with pytest.raises(ValueError, match="not a directory"):
            Poraque(operators,
                    potcar_dir=str(tmp_path / "absent"), device="cpu")

    def test_explicit_potcar_that_misses_a_species_is_reported(
            self, tmp_path, potcar_text, operators):
        """A POTCAR for Au cannot silently be used for a cell containing Si."""
        path = tmp_path / "POTCAR"
        path.write_text(potcar_text)
        calculator = Poraque(operators,
                             potcar=str(path), resolution=12, device="cpu")
        silicon = Atoms("Si", cell=np.eye(3) * 5.43, pbc=True,
                        scaled_positions=[[0.0, 0.0, 0.0]])
        silicon.calc = calculator
        with pytest.raises(ValueError, match="potcar_dir"):
            silicon.get_potential_energy()

    def test_repr_names_the_library(self, tmp_path, potcar_text, operators):
        directory = _library(tmp_path, "nested", potcar_text)
        calculator = Poraque(operators,
                             potcar_dir=directory, device="cpu")
        assert "potcar_dir" in repr(calculator)


# ===================================================================== #
# Exchange-correlation selection
# ===================================================================== #
class TestFunctionalSelection:
    def test_default_is_pbe(self, operators):
        """The reference data is PAW_PBE with LEXCH = PE."""
        calculator = Poraque(operators,
                             charges={"Au": 11.0}, device="cpu")
        assert calculator.functional == "pbe"

    def test_choice_reaches_the_energy(self, atoms, operators):
        """
        The calculator's E_xc must equal what the energy module returns for
        that functional on the very field it predicted. Comparing the two
        functionals against each other would instead test the physics, which
        untrained weights cannot exercise — they predict a non-positive
        density, and every functional clips that to exactly zero.
        """
        from poraque.physics import xc_energy

        for name in ("pbe", "lda", "none"):
            calculator = Poraque(operators, charges={"Au": 11.0},
                                 functional=name, resolution=12, device="cpu")
            copy = atoms.copy()
            copy.calc = calculator
            with pytest.warns(RuntimeWarning):
                copy.get_potential_energy()

            density = calculator.fields["density"]
            assert calculator.components.xc == pytest.approx(
                xc_energy(density, density.grid, functional=name))
            assert calculator.components.functional == name

    def test_pbe_and_lda_differ_on_a_physical_density(self, operators):
        """The selection is not cosmetic — checked where rho is positive."""
        from poraque.fields import FieldGrid
        from poraque.physics import xc_energy

        grid = FieldGrid((16, 16, 16), np.eye(3) * 8.0)
        x = grid.scaled_coordinates()[..., 0]
        density = 0.3 * (1.0 + 0.5 * np.cos(2.0 * np.pi * x))
        pbe = xc_energy(density, grid, "pbe")
        lda = xc_energy(density, grid, "lda")
        assert pbe < lda < 0.0

    def test_functional_is_recorded_in_the_components(self, atoms, operators):
        calculator = Poraque(operators,
                             charges={"Au": 11.0}, functional="lda",
                             resolution=12, device="cpu")
        atoms.calc = calculator
        with pytest.warns(RuntimeWarning):
            atoms.get_potential_energy()
        assert calculator.components.functional == "lda"
        assert calculator.results["energy_components"]["functional"] == "lda"


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
