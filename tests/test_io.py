# -*- coding: utf-8 -*-
# file: test_io.py
"""
Tests for the code-agnostic ingestion layer and spectral resampling.

The point of :mod:`poraque.fields.io` is that adding Quantum ESPRESSO or GPAW
must not require touching anything downstream. These tests therefore pin the
*contract* — the neutral types, the registry, the units — rather than VASP's
file format, which is covered by ``test_fields.py``.
"""

import importlib.util
import os
import sys

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


# ===================================================================== #
# PAW augmentation records
# ===================================================================== #
class TestAugmentation:
    """
    The one-centre PAW terms live inside the augmentation spheres and are not
    representable on the plane-wave grid, so no grid-based model produces
    them. VASP's ``ICHARG=1`` needs them, which is why they are borrowed from
    a reference calculation rather than predicted.
    """

    def _chgcar(self, path, shape=(2, 2, 2), columns=5, augmentation=(),
                spin=False):
        """A minimal CHGCAR-format file, optionally with a spin channel."""
        import numpy as np

        n = int(np.prod(shape))
        values = [f"{i:.5E}" for i in range(n)]
        lines = ["Au", "   1.0", "  2.0 0.0 0.0", "  0.0 2.0 0.0",
                 "  0.0 0.0 2.0", "  Au", "   1", "Direct",
                 "  0.0 0.0 0.0", ""]
        lines.append("  {}  {}  {}".format(*shape))
        for start in range(0, n, columns):
            lines.append(" ".join(values[start:start + columns]))
        lines.extend(augmentation)
        if spin:
            lines.append("  {}  {}  {}".format(*shape))
            for start in range(0, n, columns):
                lines.append(" ".join(values[start:start + columns]))
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    def test_extracts_the_records(self, tmp_path):
        from poraque.fields.vasp.volumetric import read_augmentation

        block = ["augmentation occupancies   1  4",
                 "  0.1 0.2 0.3 0.4"]
        path = self._chgcar(tmp_path / "CHGCAR", augmentation=block)
        shape, extracted = read_augmentation(path)
        assert shape == (2, 2, 2)
        assert extracted == block

    def test_a_file_without_records_yields_nothing(self, tmp_path):
        """A CHG has the same layout and never carries them."""
        from poraque.fields.vasp.volumetric import read_augmentation

        path = self._chgcar(tmp_path / "CHG")
        assert read_augmentation(path)[1] == []

    def test_a_spin_channel_is_not_dragged_along(self, tmp_path):
        """
        The second grid block is the reference's magnetisation, which has
        nothing to do with a prediction; extraction stops before it.
        """
        from poraque.fields.vasp.volumetric import read_augmentation

        block = ["augmentation occupancies   1  2", "  0.5 0.5"]
        path = self._chgcar(tmp_path / "CHGCAR", augmentation=block, spin=True)
        assert read_augmentation(path)[1] == block

    def test_column_width_does_not_shift_the_boundary(self, tmp_path):
        """
        CHGCAR writes 5 values a line and CHG 10. Counting lines instead of
        values would land in the middle of the grid on one of them.
        """
        from poraque.fields.vasp.volumetric import read_augmentation

        block = ["augmentation occupancies   1  1", "  0.25"]
        for columns in (5, 10, 3):
            path = self._chgcar(tmp_path / f"CHGCAR{columns}",
                                columns=columns, augmentation=block)
            assert read_augmentation(path)[1] == block

    def test_counts_the_records(self, tmp_path):
        from poraque.fields.vasp.volumetric import count_augmentation_records

        block = ["augmentation occupancies   1  1", " 0.1",
                 "augmentation occupancies   2  1", " 0.2"]
        assert count_augmentation_records(block) == 2

    def test_written_records_round_trip(self, tmp_path):
        """Appended verbatim, and readable back out unchanged."""
        import numpy as np

        from poraque.fields import ChargeDensity, FieldGrid
        from poraque.fields.vasp.poscar import Poscar
        from poraque.fields.vasp.volumetric import read_augmentation

        grid = FieldGrid((4, 4, 4), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Au"], [1], np.zeros((1, 3)))
        density = ChargeDensity(np.ones(grid.shape) * 0.5, grid, structure)

        block = ["augmentation occupancies   1  3", "  0.1 0.2 0.3"]
        path = str(tmp_path / "CHGCAR")
        density.write(path, augmentation=block)
        assert read_augmentation(path)[1] == block

    def test_writing_without_records_is_unchanged(self, tmp_path):
        import numpy as np

        from poraque.fields import ChargeDensity, FieldGrid
        from poraque.fields.vasp.poscar import Poscar

        grid = FieldGrid((4, 4, 4), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Au"], [1], np.zeros((1, 3)))
        density = ChargeDensity(np.ones(grid.shape) * 0.5, grid, structure)

        plain = density.write(str(tmp_path / "a"))
        with_none = density.write(str(tmp_path / "b"), augmentation=None)
        assert open(plain).read() == open(with_none).read()

    def test_the_density_still_reads_back(self, tmp_path):
        """Appending must not disturb the grid block above it."""
        import numpy as np

        from poraque.fields import ChargeDensity, FieldGrid
        from poraque.fields.vasp.poscar import Poscar

        grid = FieldGrid((4, 4, 4), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Au"], [1], np.zeros((1, 3)))
        values = np.random.default_rng(0).random(grid.shape)
        density = ChargeDensity(values, grid, structure)

        path = str(tmp_path / "CHGCAR")
        density.write(path, augmentation=["augmentation occupancies   1  1",
                                          "  0.5"])
        restored = ChargeDensity.read(path, grid=FieldGrid.from_file(path))
        assert np.allclose(restored.data, values, rtol=1e-8)


class TestAugmentationReference:
    """
    The transferable per-element table: read off the training calculations,
    averaged, carried by the model, written out for a structure that has no
    reference of its own.
    """

    def test_fortran_exponent_convention(self):
        """
        Fortran normalises the mantissa to [0.1, 1) and Python to [1, 10), so
        6.424378 is 0.6424378E+01 in one and 6.4243780E+00 in the other. VASP
        reads these with a fixed-format read; the difference is not cosmetic.
        """
        from poraque.fields.vasp.augmentation import fortran_exponential

        assert fortran_exponential(6.424378) == "  0.6424378E+01"
        assert fortran_exponential(-2.173598e-16) == " -0.2173598E-15"
        assert fortran_exponential(0.0) == "  0.0000000E+00"
        assert len(fortran_exponential(1.0)) == 15

    def test_rounding_never_produces_an_illegal_mantissa(self):
        """0.99999995 rounds to 1.0000000, which Fortran would not write."""
        from poraque.fields.vasp.augmentation import fortran_exponential

        for value in (0.99999999, 9.9999999, -0.999999999):
            rendered = fortran_exponential(value).strip()
            mantissa = float(rendered.split("E")[0])
            assert abs(mantissa) < 1.0

    def test_round_trips_through_the_formatter(self):
        from poraque.fields.vasp.augmentation import (
            format_augmentation,
            parse_augmentation,
        )

        import numpy as np

        records = [np.array([1.5, -2.25e-8, 0.0, 3.0]),
                   np.array([0.125, 7.5e12, -1.0, 2.0])]
        parsed = parse_augmentation(format_augmentation(records))
        assert len(parsed) == 2
        for original, restored in zip(records, parsed):
            assert np.allclose(original, restored, rtol=1e-6, atol=1e-30)

    def test_header_matches_the_fortran_format(self):
        """``("augmentation occupancies",2I4)`` -- four columns each."""
        from poraque.fields.vasp.augmentation import format_augmentation

        import numpy as np

        lines = format_augmentation([np.zeros(3)])
        assert lines[0] == "augmentation occupancies   1   3"

    def test_five_values_per_line(self):
        from poraque.fields.vasp.augmentation import format_augmentation

        import numpy as np

        lines = format_augmentation([np.arange(12, dtype=float)])
        data = lines[1:]
        assert [len(line) // 15 for line in data] == [5, 5, 2]

    def test_one_record_per_atom_in_species_order(self):
        from poraque.fields.vasp.augmentation import records_for_structure
        from poraque.fields.vasp.poscar import Poscar

        import numpy as np

        structure = Poscar(np.eye(3) * 5.0, ["Si", "O"], [1, 2],
                           np.zeros((3, 3)))
        reference = {"Si": {"values": [1.0, 2.0], "atoms": 1, "structures": 1},
                     "O": {"values": [3.0, 4.0], "atoms": 2, "structures": 1}}
        lines, missing = records_for_structure(structure, reference)
        assert missing == []
        headers = [line for line in lines if "augmentation" in line]
        assert len(headers) == 3
        assert headers[0].endswith("   1   2")
        # Si first, then the two O -- the order the POSCAR lists them.
        assert "0.1000000E+01" in lines[1]
        assert "0.3000000E+01" in lines[3]

    def test_a_missing_element_yields_nothing_rather_than_a_partial_block(self):
        """
        A file with records for some atoms and not others is worse than one
        with none: VASP would read it and silently mis-assign them.
        """
        from poraque.fields.vasp.augmentation import records_for_structure
        from poraque.fields.vasp.poscar import Poscar

        import numpy as np

        structure = Poscar(np.eye(3) * 5.0, ["Si", "O"], [1, 1],
                           np.zeros((2, 3)))
        reference = {"Si": {"values": [1.0], "atoms": 1, "structures": 1}}
        lines, missing = records_for_structure(structure, reference)
        assert lines == []
        assert missing == ["O"]

    def test_building_a_reference_averages_over_atoms(self, tmp_path):
        from poraque.fields.vasp.augmentation import build_reference

        import numpy as np

        # Two "calculations" of one element, values 2 and 4 -> mean 3.
        for name, value in (("a", 2.0), ("b", 4.0)):
            directory = tmp_path / name
            directory.mkdir()
            self._write_chgcar(directory / "CHGCAR", value)
        reference = build_reference([str(tmp_path / "a"), str(tmp_path / "b")])
        assert set(reference) == {"Au"}
        assert np.allclose(reference["Au"]["values"], 3.0)
        assert reference["Au"]["atoms"] == 2
        assert reference["Au"]["structures"] == 2

    def test_a_directory_without_records_contributes_nothing(self, tmp_path):
        from poraque.fields.vasp.augmentation import build_reference

        directory = tmp_path / "plain"
        directory.mkdir()
        self._write_chgcar(directory / "CHGCAR", 1.0, augmentation=False)
        assert build_reference([str(directory)]) == {}

    @staticmethod
    def _write_chgcar(path, value, augmentation=True):
        lines = ["Au", "   1.0", "  2.0 0.0 0.0", "  0.0 2.0 0.0",
                 "  0.0 0.0 2.0", "  Au", "   1", "Direct",
                 "  0.0 0.0 0.0", "", "  2  2  2",
                 "0.0 0.0 0.0 0.0 0.0", "0.0 0.0 0.0"]
        if augmentation:
            lines += ["augmentation occupancies   1   2",
                      f"  {value:.7E}  {value:.7E}"]
        path.write_text("\n".join(lines) + "\n")


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_inference():
    """Import ``scripts/poraque_inference.py`` as a module."""
    path = os.path.join(_ROOT, "scripts", "poraque_inference.py")
    spec = importlib.util.spec_from_file_location("_poraque_inference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_poraque_inference"] = module
    spec.loader.exec_module(module)
    return module


poraque_inference = _load_inference()


# ===================================================================== #
# Grid: ENCUT / PREC resolution
# ===================================================================== #
class TestCutoffSettings:
    """
    `--from-incar` versus the manual flags. The failure this guards against is
    silent: a grid that disagrees with the input file it was told to follow.
    """

    class _Args:
        def __init__(self, **kwargs):
            self.encut = 200.0
            self.prec_accurate = False
            self.from_incar = None
            self.__dict__.update(kwargs)

    def _resolve(self, args):
        lines = []
        settings = poraque_inference.resolve_cutoff_settings(args, lines.append)
        return settings, lines

    def test_defaults_to_normal_precision(self):
        (cutoff, prec, source), _ = self._resolve(self._Args())
        assert cutoff == 200.0
        assert prec == "normal"
        assert source == "--encut"

    def test_prec_accurate_switches_the_rule(self):
        (_, prec, _), _ = self._resolve(self._Args(prec_accurate=True))
        assert prec == "accurate"

    def test_incar_supplies_both(self, tmp_path):
        path = tmp_path / "INCAR"
        path.write_text("ENCUT = 450\nPREC = Accurate\n")
        (cutoff, prec, source), _ = self._resolve(
            self._Args(from_incar=str(path)))
        assert cutoff == 450.0
        assert prec == "accurate"
        assert str(path) in source

    def test_incar_overrides_both_flags(self, tmp_path):
        """The whole point: the file wins, and the log says what it displaced."""
        path = tmp_path / "INCAR"
        path.write_text("ENCUT = 300\nPREC = Normal\n")
        (cutoff, prec, _), lines = self._resolve(
            self._Args(encut=800.0, prec_accurate=True, from_incar=str(path)))
        assert cutoff == 300.0
        assert prec == "normal"
        message = " ".join(lines)
        assert "--encut 800" in message
        assert "--prec-accurate" in message

    def test_no_override_notice_when_nothing_was_overridden(self, tmp_path):
        path = tmp_path / "INCAR"
        path.write_text("ENCUT = 450\nPREC = Accurate\n")
        _, lines = self._resolve(self._Args(from_incar=str(path)))
        assert not any("override" in line for line in lines)

    def test_high_counts_as_accurate(self, tmp_path):
        path = tmp_path / "INCAR"
        path.write_text("ENCUT = 400\nPREC = High\n")
        (_, prec, _), _ = self._resolve(self._Args(from_incar=str(path)))
        assert prec in poraque_inference.ACCURATE_PRECISIONS

    def test_a_missing_incar_is_an_error(self):
        import pytest

        with pytest.raises(SystemExit, match="does not exist"):
            self._resolve(self._Args(from_incar="/nope/INCAR"))

    def test_an_incar_without_encut_is_an_error(self, tmp_path):
        """It cannot size a grid, and guessing one would be worse."""
        import pytest

        path = tmp_path / "INCAR"
        path.write_text("PREC = Accurate\n")
        with pytest.raises(SystemExit, match="no ENCUT"):
            self._resolve(self._Args(from_incar=str(path)))


class TestPrecGridDensity:
    def test_accurate_gives_a_denser_grid_than_normal(self):
        """
        PREC=Accurate uses a factor of 2 against Normal's 3/2, so the grid is
        about 4/3 longer on each axis.
        """
        import numpy as np

        from poraque.fields import FieldGrid

        cell = np.eye(3) * 12.0
        normal = FieldGrid.from_encut(cell, 200.0, prec="normal")
        accurate = FieldGrid.from_encut(cell, 200.0, prec="accurate")
        assert max(accurate.shape) > max(normal.shape)
        ratio = max(accurate.shape) / max(normal.shape)
        assert 1.2 < ratio < 1.45


# ===================================================================== #
# VASP's own FFT grid rule
# ===================================================================== #
class TestValidFFTGridSize:
    """
    ``FFTCH1`` accepts a length only when dividing out 2, 3, 5 and 7 leaves 1
    *and* the factor 2 occurs at least once — 7-smooth **and** even.
    """

    def test_known_roundings(self):
        from poraque.fields.vasp.fftgrid import get_valid_fft_grid_size

        assert get_valid_fft_grid_size(61) == 64
        assert get_valid_fft_grid_size(64) == 64
        assert get_valid_fft_grid_size(109) == 112
        assert get_valid_fft_grid_size(1) == 2

    def test_results_are_seven_smooth_and_even(self):
        from poraque.fields.vasp.fftgrid import get_valid_fft_grid_size

        for request in range(2, 300):
            size = get_valid_fft_grid_size(request)
            assert size >= request
            assert size % 2 == 0
            residue = size
            for factor in (2, 3, 5, 7):
                while residue % factor == 0:
                    residue //= factor
            assert residue == 1, f"{size} has a prime factor above 7"

    def test_it_is_the_smallest_such_integer(self):
        from poraque.fields.vasp.fftgrid import get_valid_fft_grid_size

        def admissible(value):
            if value % 2:
                return False
            for factor in (2, 3, 5, 7):
                while value % factor == 0:
                    value //= factor
            return value == 1

        for request in range(2, 200):
            size = get_valid_fft_grid_size(request)
            assert not any(admissible(v) for v in range(request, size))

    def test_a_fractional_request_is_raised_first(self):
        from poraque.fields.vasp.fftgrid import get_valid_fft_grid_size

        assert get_valid_fft_grid_size(60.1) == 64
        assert get_valid_fft_grid_size(63.999) == 64


class TestVaspGridRule:
    """
    Pinned against real VASP output. The failure mode is a factor of two:
    rounding the density size in one step instead of rounding the coarse grid
    and doubling it gives 64 where VASP gives 128.
    """

    #: 27-atom gold cell, ENCUT 450, PREC=Accurate -> VASP writes 128^3.
    GOLD_CELL = [[0.0, 6.233932, 6.233932],
                 [6.233932, 0.0, 6.233932],
                 [6.233932, 6.233932, 0.0]]

    def test_reproduces_a_real_vasp_grid(self):
        from poraque.fields.vasp.fftgrid import vasp_grid_shapes

        coarse, fine = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Accurate")
        assert coarse == (64, 64, 64)
        assert fine == (128, 128, 128)

    def test_the_coarse_grid_is_rounded_before_doubling(self):
        """
        The ordering *is* the algorithm. 4 x 15.25 = 61 rounds to 64; doubling
        gives 128. Rounding 61 x 2 = 122 in one step would give 128 too, but
        rounding the density target 61 alone gives 64 -- half of VASP's.
        """
        from poraque.fields.vasp.fftgrid import (
            cutoff_indices,
            get_valid_fft_grid_size,
            vasp_grid_shapes,
        )

        index = cutoff_indices(self.GOLD_CELL, 450.0)[0]
        assert 15.0 < index < 15.5
        naive = get_valid_fft_grid_size(4 * index + 1)
        coarse, fine = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Accurate")
        assert naive == 64                       # what one-shot rounding gives
        assert fine[0] == 2 * coarse[0] == 128   # what VASP actually builds

    def test_normal_uses_the_smaller_coarse_multiplier(self):
        from poraque.fields.vasp.fftgrid import vasp_grid_shapes

        accurate, _ = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Accurate")
        normal, _ = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Normal")
        assert max(normal) < max(accurate)       # WFACT 3 against 4

    def test_normal_also_doubles_for_the_density(self):
        from poraque.fields.vasp.fftgrid import vasp_grid_shapes

        coarse, fine = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Normal")
        assert fine == tuple(2 * n for n in coarse)

    def test_single_uses_one_grid(self):
        """PREC=Single switches off the double-grid technique entirely."""
        from poraque.fields.vasp.fftgrid import vasp_grid_shapes

        coarse, fine = vasp_grid_shapes(self.GOLD_CELL, 450.0, prec="Single")
        assert coarse == fine

    def test_only_the_first_letter_of_prec_matters(self):
        from poraque.fields.vasp.fftgrid import vasp_density_grid

        for spelling in ("Accurate", "accurate", "A", "aCcUrAtE"):
            assert vasp_density_grid(self.GOLD_CELL, 450.0,
                                     prec=spelling) == (128, 128, 128)

    def test_an_anisotropic_cell_gives_an_anisotropic_grid(self):
        """The complaint that started this: grids must follow the cell."""
        from poraque.fields.vasp.fftgrid import vasp_density_grid

        cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]]
        shape = vasp_density_grid(cell, 400.0, prec="Accurate")
        assert shape[0] == shape[1]
        assert shape[2] > shape[0]

    def test_matches_every_reference_calculation(self):
        """Data-driven, and skipped when the raw calculations are absent."""
        import glob

        from poraque.fields import FieldGrid
        from poraque.fields.vasp.fftgrid import vasp_density_grid
        from poraque.fields.vasp.incar import Incar

        root = os.path.join(_ROOT, "data", "vasp")
        checked = 0
        for directory in sorted(glob.glob(os.path.join(root, "struct_*"))):
            chgcar = os.path.join(directory, "CHGCAR")
            incar = os.path.join(directory, "INCAR")
            if not (os.path.exists(chgcar) and os.path.exists(incar)):
                continue
            reference = FieldGrid.from_file(chgcar)
            settings = Incar.from_file(incar)
            computed = vasp_density_grid(
                reference.cell, settings.get_float("ENCUT"),
                prec=str(settings.get("PREC", "normal")))
            assert tuple(reference.shape) == computed, directory
            checked += 1
        if not checked:
            pytest.skip("no reference VASP calculations available")
