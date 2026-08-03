# -*- coding: utf-8 -*-
# file: test_io.py
"""
Tests for the code-agnostic ingestion layer and spectral resampling.

The point of :mod:`poraque.fields.io` is that adding Quantum ESPRESSO or GPAW
must not require touching anything downstream. These tests therefore pin the
*contract* — the neutral types, the registry, the units — rather than VASP's
file format, which is covered by ``test_fields.py``.
"""

import numpy as np
import pytest

from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid, Structure
from poraque.fields.io import (
    CalculationParameters,
    CalculationReader,
    EspressoReader,
    GpawReader,
    PseudopotentialInfo,
    VaspReader,
    available_codes,
    detect_reader,
    get_reader,
    register_reader,
    resolve_reader,
)
from poraque.fields.resample import (
    downsample_shape,
    resample_field,
    spectral_resample,
)
from poraque.fields.vasp import Poscar

POSCAR_TEXT = """Si2 diamond
   5.43000000000000
     0.0000000000000000    0.5000000000000000    0.5000000000000000
     0.5000000000000000    0.0000000000000000    0.5000000000000000
     0.5000000000000000    0.5000000000000000    0.0000000000000000
   Si
     2
Direct
  0.0000000000000000  0.0000000000000000  0.0000000000000000
  0.2500000000000000  0.2500000000000000  0.2500000000000000
"""

INCAR_TEXT = "ENCUT = 245.345\nPREC = Accurate\n"

POTCAR_TEXT = """ PAW_PBE Si 05Jan2001
   4.00000000000000
   LEXCH  = PE
   TITEL  = PAW_PBE Si 05Jan2001
   POMASS =   28.085; ZVAL   =    4.000    mass and valenz
   RCORE  =    1.900    outmost cutoff radius
   ENMAX  =  245.345; ENMIN  = 184.009 eV
 End of Dataset
"""


@pytest.fixture
def vasp_dir(tmp_path):
    (tmp_path / "POSCAR").write_text(POSCAR_TEXT)
    (tmp_path / "INCAR").write_text(INCAR_TEXT)
    (tmp_path / "POTCAR").write_text(POTCAR_TEXT)
    return tmp_path


# --------------------------------------------------------------------- #
# Neutral structure
# --------------------------------------------------------------------- #
class TestStructure:
    def test_poscar_is_a_structure(self):
        structure = Poscar.from_string(POSCAR_TEXT)
        assert isinstance(structure, Structure)
        assert structure.formula == "Si2"
        assert structure.elements == ["Si"]

    def test_decorated_symbols_resolve_to_elements(self):
        structure = Structure(np.eye(3) * 5, ["Fe_pv", "O.pbe-n-kjpaw"], [1, 2],
                              np.zeros((3, 3)))
        assert structure.elements == ["Fe", "O"]
        assert list(structure.atomic_numbers) == [26, 8, 8]

    def test_species_slices_are_contiguous(self):
        structure = Structure(np.eye(3) * 5, ["Si", "O"], [2, 3], np.zeros((5, 3)))
        slices = dict(structure.species_slices())
        assert slices["Si"] == slice(0, 2)
        assert slices["O"] == slice(2, 5)

    def test_from_structure_wraps_without_copying_geometry(self):
        base = Structure(np.eye(3) * 5, ["Si"], [1], [[0.1, 0.2, 0.3]])
        wrapped = Poscar.from_structure(base)
        assert isinstance(wrapped, Poscar)
        assert np.allclose(wrapped.scaled_positions, base.scaled_positions)


# --------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------- #
class TestRegistry:
    def test_builtin_codes_are_registered(self):
        assert set(available_codes()) >= {"vasp", "espresso", "gpaw"}

    def test_get_reader_by_name_class_and_instance(self):
        assert isinstance(get_reader("vasp"), VaspReader)
        assert isinstance(get_reader(VaspReader), VaspReader)
        instance = VaspReader()
        assert get_reader(instance) is instance

    def test_unknown_code(self):
        with pytest.raises(KeyError, match="Unknown code"):
            get_reader("siesta")

    def test_detect_vasp(self, vasp_dir):
        assert isinstance(detect_reader(vasp_dir), VaspReader)

    def test_detect_fails_on_an_unknown_layout(self, tmp_path):
        (tmp_path / "something.txt").write_text("hello")
        with pytest.raises(ValueError, match="Could not identify"):
            detect_reader(tmp_path)

    def test_resolve_reader_auto_and_explicit(self, vasp_dir):
        assert isinstance(resolve_reader(vasp_dir, "auto"), VaspReader)
        assert isinstance(resolve_reader(None, "vasp"), VaspReader)
        with pytest.raises(ValueError, match="needs a directory"):
            resolve_reader(None, "auto")

    def test_registering_a_new_code(self):
        class DummyReader(CalculationReader):
            code = "dummy_for_test"
            structure_files = ("dummy.in",)
            field_files = {"density": "dummy.rho"}

            def read_structure(self, directory):
                return None

            def read_parameters(self, directory):
                return CalculationParameters()

            def read_pseudopotentials(self, directory):
                return {}

            def read_field(self, path, field_class, grid=None):
                return None

            def write_field(self, field, path, comment=None):
                return path

        register_reader(DummyReader)
        try:
            assert "dummy_for_test" in available_codes()
            assert isinstance(get_reader("dummy_for_test"), DummyReader)
        finally:
            from poraque.fields.io import _READERS

            _READERS.pop("dummy_for_test")

    def test_register_rejects_non_readers(self):
        with pytest.raises(TypeError):
            register_reader(dict)


# --------------------------------------------------------------------- #
# VASP reader against the neutral contract
# --------------------------------------------------------------------- #
class TestVaspReader:
    def test_read_structure(self, vasp_dir):
        structure = VaspReader().read_structure(vasp_dir)
        assert isinstance(structure, Structure)
        assert structure.natoms == 2

    def test_read_parameters_are_in_ev(self, vasp_dir):
        parameters = VaspReader().read_parameters(vasp_dir)
        assert isinstance(parameters, CalculationParameters)
        assert parameters.cutoff == pytest.approx(245.345)
        assert parameters.precision == "accurate"

    def test_read_pseudopotentials_returns_neutral_type(self, vasp_dir):
        pseudos = VaspReader().read_pseudopotentials(vasp_dir)
        assert set(pseudos) == {"Si"}
        info = pseudos["Si"]
        assert isinstance(info, PseudopotentialInfo)
        assert info.valence_charge == 4.0
        # RCORE is written in Bohr by VASP; the contract requires Angstrom.
        assert info.core_radius == pytest.approx(1.9 * 0.529177210903)
        assert info.recommended_cutoff == pytest.approx(245.345)

    def test_field_path_and_unknown_kind(self, vasp_dir):
        reader = VaspReader()
        assert reader.field_path(vasp_dir, "density").endswith("CHGCAR")
        assert reader.field_path(vasp_dir, "kinetic").endswith("TAUCAR")
        with pytest.raises(KeyError, match="does not define a file"):
            reader.field_path(vasp_dir, "spin")

    def test_valence_charges_with_overrides(self, vasp_dir):
        reader = VaspReader()
        assert reader.valence_charges(vasp_dir) == {"Si": 4.0}
        assert reader.valence_charges(vasp_dir, {"Si": 12.0}) == {"Si": 12.0}

    def test_missing_optional_files_degrade_gracefully(self, tmp_path):
        (tmp_path / "POSCAR").write_text(POSCAR_TEXT)
        reader = VaspReader()
        assert reader.read_parameters(tmp_path).cutoff is None
        assert reader.read_pseudopotentials(tmp_path) == {}


# --------------------------------------------------------------------- #
# The generic entry point
# --------------------------------------------------------------------- #
class TestFromCalculation:
    def test_matches_the_vasp_specific_path(self, vasp_dir):
        generic = ExternalPotential.from_calculation(vasp_dir)
        specific = ExternalPotential.from_vasp(vasp_dir)
        assert generic.shape == specific.shape
        assert np.abs(generic.data - specific.data).max() < 1e-12

    def test_records_the_code_in_metadata(self, vasp_dir):
        assert ExternalPotential.from_calculation(vasp_dir).metadata["code"] == "vasp"

    def test_honours_a_shared_grid(self, vasp_dir):
        grid = FieldGrid((20, 20, 20), Poscar.from_string(POSCAR_TEXT).cell)
        assert ExternalPotential.from_calculation(vasp_dir,
                                                  grid=grid).shape == (20, 20, 20)

    def test_requires_valence_charges(self, vasp_dir):
        (vasp_dir / "POTCAR").unlink()
        with pytest.raises(ValueError, match="No valence charge"):
            ExternalPotential.from_calculation(vasp_dir, encut=245.0)
        assert ExternalPotential.from_calculation(vasp_dir, encut=245.0,
                                                  zval={"Si": 4.0}) is not None


class TestGridFromParameters:
    def test_explicit_shape_wins(self):
        structure = Poscar.from_string(POSCAR_TEXT)
        parameters = CalculationParameters(cutoff=400.0, grid_shape=(12, 12, 12))
        assert FieldGrid.from_parameters(structure, parameters).shape == (12, 12, 12)
        assert FieldGrid.from_parameters(structure, parameters,
                                         shape=(8, 8, 8)).shape == (8, 8, 8)

    def test_falls_back_to_the_recommended_cutoff(self):
        structure = Poscar.from_string(POSCAR_TEXT)
        pseudos = {"Si": PseudopotentialInfo("Si", "Si", 4.0,
                                             recommended_cutoff=245.345)}
        grid = FieldGrid.from_parameters(structure, CalculationParameters(),
                                         pseudos)
        assert grid.encut == pytest.approx(245.345)

    def test_raises_without_any_cutoff(self):
        structure = Poscar.from_string(POSCAR_TEXT)
        with pytest.raises(ValueError, match="Cannot size the grid"):
            FieldGrid.from_parameters(structure, CalculationParameters())


# --------------------------------------------------------------------- #
# Skeletons must fail loudly, not silently
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("reader_class", [EspressoReader, GpawReader])
class TestSkeletons:
    def test_every_method_raises_not_implemented(self, reader_class, tmp_path):
        reader = reader_class()
        for call in (
            lambda: reader.read_structure(tmp_path),
            lambda: reader.read_parameters(tmp_path),
            lambda: reader.read_pseudopotentials(tmp_path),
            lambda: reader.read_field("x", ChargeDensity),
            lambda: reader.write_field(None, "x"),
        ):
            with pytest.raises(NotImplementedError):
                call()

    def test_declares_all_three_field_kinds(self, reader_class, tmp_path):
        assert set(reader_class.field_files) == {"external", "density", "kinetic"}


def test_espresso_reader_exposes_the_rydberg_conversion():
    """The unit trap QE support must handle; pinned so it cannot drift."""
    from poraque.fields.io.espresso import RY_TO_EV

    assert RY_TO_EV == pytest.approx(13.605693122994)


# --------------------------------------------------------------------- #
# Spectral resampling
# --------------------------------------------------------------------- #
class TestSpectralResample:
    def test_exact_for_a_band_limited_field(self):
        """Downsampling a field the coarse grid can represent is lossless."""
        grid = FieldGrid((64, 64, 64), np.eye(3) * 10.0)
        xyz = grid.cartesian_coordinates()
        k = 2 * np.pi / 10.0
        field = (np.sin(k * xyz[..., 0]) + 0.5 * np.cos(2 * k * xyz[..., 1])
                 + 0.3 * np.sin(k * (xyz[..., 2] + xyz[..., 0])) + 2.0)
        coarse = spectral_resample(field, (32, 32, 32))
        assert np.abs(coarse - field[::2, ::2, ::2]).max() < 1e-10

    def test_output_is_real(self):
        rng = np.random.default_rng(0)
        assert np.isrealobj(spectral_resample(rng.random((30, 24, 36)), (16, 16, 16)))

    @pytest.mark.parametrize("target", [(30, 24, 36), (20, 16, 24), (90, 60, 80)])
    def test_cell_average_is_preserved_exactly(self, target):
        """The G=0 coefficient is copied unchanged, so integrals survive."""
        rng = np.random.default_rng(1)
        data = rng.random((60, 48, 72))
        assert spectral_resample(data, target).mean() == pytest.approx(data.mean(),
                                                                      abs=1e-12)

    def test_downsampling_is_a_projection(self):
        rng = np.random.default_rng(2)
        data = rng.random((60, 48, 72))
        once = spectral_resample(data, (30, 24, 36))
        twice = spectral_resample(spectral_resample(once, (60, 48, 72)),
                                  (30, 24, 36))
        assert np.abs(once - twice).max() < 1e-10

    def test_identity(self):
        rng = np.random.default_rng(3)
        data = rng.random((16, 16, 16))
        assert np.abs(spectral_resample(data, (16, 16, 16)) - data).max() == 0.0

    def test_rejects_degenerate_targets(self):
        rng = np.random.default_rng(4)
        with pytest.raises(ValueError):
            spectral_resample(rng.random((16, 16, 16)), (1, 16, 16))

    def test_resample_field_preserves_type_and_integral(self):
        cell = np.eye(3) * 10.0
        structure = Poscar(cell, ["Si"], [1], [[0.1, 0.2, 0.3]])
        rng = np.random.default_rng(5)
        field = ChargeDensity(rng.random((32, 32, 32)) + 0.1,
                              FieldGrid((32, 32, 32), cell), structure)
        target_grid = FieldGrid((16, 16, 16), cell)
        coarse = resample_field(field, (16, 16, 16), grid=target_grid)

        assert isinstance(coarse, ChargeDensity)
        assert coarse.grid is target_grid
        assert coarse.integrate() == pytest.approx(field.integrate(), rel=1e-10)
        assert coarse.metadata["resampled_from"] == (32, 32, 32)

    def test_resample_field_rejects_an_inconsistent_grid(self):
        cell = np.eye(3) * 10.0
        structure = Poscar(cell, ["Si"], [1], [[0.0, 0.0, 0.0]])
        field = ChargeDensity(np.ones((16, 16, 16)), FieldGrid((16, 16, 16), cell),
                              structure)
        with pytest.raises(ValueError, match="expected"):
            resample_field(field, (8, 8, 8), grid=FieldGrid((12, 12, 12), cell))

    def test_downsample_shape_keeps_the_aspect_ratio(self):
        assert downsample_shape((128, 128, 128), target_max=32) == (32, 32, 32)
        # A ragged source must stay ragged, or materials silently become alike.
        assert downsample_shape((120, 128, 128), target_max=32) == (30, 32, 32)

    def test_downsample_shape_by_factor(self):
        assert downsample_shape((64, 64, 64), factor=2) == (32, 32, 32)

    def test_downsample_shape_never_upsamples(self):
        assert downsample_shape((16, 16, 16), target_max=64) == (16, 16, 16)

    def test_downsample_shape_requires_an_argument(self):
        with pytest.raises(ValueError, match="factor.*target_max"):
            downsample_shape((32, 32, 32))
