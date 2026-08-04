# -*- coding: utf-8 -*-
# file: test_fields.py
"""Tests for :mod:`poraque.fields` — VASP I/O, the shared grid, and EXTCAR."""

import numpy as np
import pytest

from poraque.fields import (
    COULOMB_CONSTANT_EV_ANGSTROM,
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
    fft_friendly_size,
    thomas_fermi_tau,
    von_weizsacker_tau,
)
from poraque.fields.vasp import Incar, Poscar, Potcar

# --------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------- #
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

INCAR_TEXT = """SYSTEM = Si bulk
 ENCUT = 245.345   ! plane-wave cutoff
 PREC = Accurate
 ISMEAR = 0; SIGMA = 0.05
# a full-line comment
"""

POTCAR_TEXT = """ PAW_PBE Si 05Jan2001
   4.00000000000000
 parameters from PSCTR are:
   VRHFIN =Si: s2p2
   LEXCH  = PE
   TITEL  = PAW_PBE Si 05Jan2001
   POMASS =   28.085; ZVAL   =    4.000    mass and valenz
   RCORE  =    1.900    outmost cutoff radius
   ENMAX  =  245.345; ENMIN  = 184.009 eV
 END of PSCTR-controll parameters
  local part
             4.00000000000000
  0.4899969775558059E+03  0.4897893015154668E+03  0.4891667097127825E+03
 gradient corrections used for XC
    1
 End of Dataset
"""


@pytest.fixture
def vasp_dir(tmp_path):
    """A directory holding a minimal but realistic POSCAR/INCAR/POTCAR set."""
    (tmp_path / "POSCAR").write_text(POSCAR_TEXT)
    (tmp_path / "INCAR").write_text(INCAR_TEXT)
    (tmp_path / "POTCAR").write_text(POTCAR_TEXT)
    return tmp_path


@pytest.fixture
def poscar():
    return Poscar.from_string(POSCAR_TEXT)


# --------------------------------------------------------------------- #
# VASP parsers
# --------------------------------------------------------------------- #
class TestPoscar:
    def test_geometry(self, poscar):
        assert poscar.natoms == 2
        assert poscar.symbols == ["Si"]
        assert poscar.counts == [2]
        assert poscar.volume == pytest.approx(5.43 ** 3 / 4.0)
        assert list(poscar.atomic_numbers) == [14, 14]

    def test_cartesian_positions(self, poscar):
        expected = np.array([[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]])
        assert np.allclose(poscar.positions, expected)

    def test_negative_scale_is_target_volume(self):
        text = POSCAR_TEXT.replace("   5.43000000000000", "  -40.0257")
        assert Poscar.from_string(text).volume == pytest.approx(40.0257)

    def test_cartesian_input_round_trips(self, poscar):
        text = poscar.to_string(direct=False)
        assert np.allclose(Poscar.from_string(text).scaled_positions,
                           poscar.scaled_positions)

    def test_string_round_trip(self, poscar):
        again = Poscar.from_string(poscar.to_string())
        assert np.allclose(again.cell, poscar.cell)
        assert np.allclose(again.scaled_positions, poscar.scaled_positions)

    def test_vasp4_needs_symbols(self):
        text = POSCAR_TEXT.replace("   Si\n", "")
        with pytest.raises(ValueError, match="species-symbol line"):
            Poscar.from_string(text)
        assert Poscar.from_string(text, symbols=["Si"]).natoms == 2


class TestIncar:
    def test_parsing(self):
        incar = Incar.from_string(INCAR_TEXT)
        assert incar.encut == pytest.approx(245.345)
        assert incar.prec == "accurate"
        assert incar.get_float("SIGMA") == pytest.approx(0.05)
        assert incar.get_int("ISMEAR") == 0
        assert "SYSTEM" in incar

    def test_comments_are_stripped(self):
        assert Incar.from_string(INCAR_TEXT)["ENCUT"] == "245.345"

    def test_explicit_grid_tags(self):
        incar = Incar.from_string("NGXF = 48\nNGYF = 48\nNGZF = 60\n")
        assert incar.fine_shape == (48, 48, 60)
        assert incar.coarse_shape is None


class TestPotcar:
    def test_header_quantities(self):
        potcar = Potcar.from_string(POTCAR_TEXT, parse_tables=True)
        assert potcar.symbols == ["Si"]
        assert potcar.zval_map == {"Si": 4.0}
        assert potcar.enmax == pytest.approx(245.345)
        assert potcar[0].rcore == pytest.approx(1.9)
        assert potcar[0].rcore_angstrom == pytest.approx(1.9 * 0.529177210903)
        assert potcar[0].functional == "PE"
        assert potcar[0].local_part is not None

    def test_decorated_symbol(self):
        text = POTCAR_TEXT.replace("PAW_PBE Si ", "PAW_PBE Fe_pv ")
        entry = Potcar.from_string(text)[0]
        assert entry.symbol == "Fe_pv"
        assert entry.element == "Fe"
        assert entry.atomic_number == 26

    def test_species_order_check(self, poscar):
        assert Potcar.from_string(POTCAR_TEXT).matches(poscar)


# --------------------------------------------------------------------- #
# Shared grid
# --------------------------------------------------------------------- #
class TestFieldGrid:
    def test_fft_friendly_sizes(self):
        assert fft_friendly_size(17) == 18
        assert fft_friendly_size(18) == 18
        assert fft_friendly_size(23) == 24
        # never returns an odd number or a size with a large prime factor
        for n in range(2, 200):
            size = fft_friendly_size(n)
            assert size >= n and size % 2 == 0
            residue = size
            for factor in (2, 3, 5, 7):
                while residue % factor == 0:
                    residue //= factor
            assert residue == 1

    def test_from_encut_scales_with_cutoff(self, poscar):
        coarse = FieldGrid.from_encut(poscar.cell, 100.0, prec="normal")
        fine = FieldGrid.from_encut(poscar.cell, 400.0, prec="normal")
        assert all(f > c for f, c in zip(fine.shape, coarse.shape))

    def test_prec_accurate_is_denser(self, poscar):
        normal = FieldGrid.from_encut(poscar.cell, 245.345, prec="normal")
        accurate = FieldGrid.from_encut(poscar.cell, 245.345, prec="accurate")
        assert all(a >= n for a, n in zip(accurate.shape, normal.shape))
        assert accurate.shape != normal.shape

    def test_precedence_explicit_shape_wins(self, poscar):
        incar = Incar.from_string("ENCUT = 400\nNGXF = 12\nNGYF = 12\nNGZF = 12\n")
        assert FieldGrid.from_vasp_inputs(poscar, incar=incar).shape == (12, 12, 12)
        assert FieldGrid.from_vasp_inputs(poscar, incar=incar,
                                          shape=(8, 8, 8)).shape == (8, 8, 8)

    def test_falls_back_to_potcar_enmax(self, poscar):
        potcar = Potcar.from_string(POTCAR_TEXT)
        grid = FieldGrid.from_vasp_inputs(poscar, incar=Incar(), potcar=potcar)
        assert grid.encut == pytest.approx(245.345)

    def test_raises_without_any_cutoff(self, poscar):
        with pytest.raises(ValueError, match="Cannot size the grid"):
            FieldGrid.from_vasp_inputs(poscar, incar=Incar())

    def test_reciprocal_lattice_is_dual(self, poscar):
        grid = FieldGrid((12, 12, 12), poscar.cell)
        assert np.allclose(grid.cell @ grid.reciprocal_cell.T,
                           2 * np.pi * np.eye(3))

    def test_integrate_constant(self, poscar):
        grid = FieldGrid((10, 10, 10), poscar.cell)
        assert grid.integrate(np.ones(grid.shape)) == pytest.approx(grid.volume)

    def test_matches(self, poscar):
        a = FieldGrid((10, 10, 10), poscar.cell)
        assert a.matches(FieldGrid((10, 10, 10), poscar.cell))
        assert not a.matches(FieldGrid((10, 10, 12), poscar.cell))
        assert not a.matches(FieldGrid((10, 10, 10), poscar.cell * 1.1))


# --------------------------------------------------------------------- #
# External potential
# --------------------------------------------------------------------- #
class TestExternalPotential:
    def test_from_vasp_directory(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        assert potential.shape == potential.grid.shape
        assert potential.metadata["charges"] == {"Si": 4.0}
        assert np.isfinite(potential.data).all()

    def test_cell_average_vanishes(self, vasp_dir):
        # V(G=0) = 0 is the neutralizing-background convention.
        assert ExternalPotential.from_vasp(vasp_dir).mean() == pytest.approx(0.0,
                                                                            abs=1e-10)

    def test_attractive_minimum_sits_on_an_atom(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        index = np.unravel_index(np.argmin(potential.data), potential.shape)
        fractional = np.asarray(index) / np.asarray(potential.shape)
        delta = potential.structure.scaled_positions - fractional
        delta -= np.round(delta)
        distance = np.linalg.norm(delta @ potential.grid.cell, axis=1).min()
        assert distance < 1.5 * potential.grid.spacing.max()

    def test_satisfies_poisson_equation(self, poscar):
        """laplacian(V) = 4 pi K (n_ion - <n_ion>), with n_ion built in real space.

        This validates the reciprocal-space construction against a completely
        independent evaluation of the ionic charge distribution.
        """
        length, sigma, charge = 12.0, 0.6, 3.0
        cell = np.eye(3) * length
        grid = FieldGrid((48, 48, 48), cell)
        structure = Poscar(cell, ["H"], [1], [[0.35, 0.5, 0.65]])
        potential = ExternalPotential.compute(structure, grid, {"H": charge},
                                              widths={"H": sigma})

        coordinates = grid.cartesian_coordinates()
        centre = structure.positions[0]
        normalization = charge / (2 * np.pi * sigma ** 2) ** 1.5
        n_ion = np.zeros(grid.shape)
        for image in np.ndindex(3, 3, 3):
            offset = (np.asarray(image) - 1) * length
            delta = coordinates - (centre + offset)
            n_ion += normalization * np.exp(-np.sum(delta ** 2, axis=-1)
                                            / (2 * sigma ** 2))
        assert grid.integrate(n_ion) == pytest.approx(charge, abs=1e-6)

        laplacian = np.real(np.fft.ifftn(-grid.get_g2()
                                         * np.fft.fftn(potential.data)))
        rhs = 4 * np.pi * COULOMB_CONSTANT_EV_ANGSTROM * (n_ion - n_ion.mean())
        assert np.abs(laplacian - rhs).max() < 1e-6 * np.abs(rhs).max()

    def test_translation_covariance(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        grid, structure = potential.grid, potential.structure
        shift = 6
        moved = Poscar(
            structure.cell, structure.symbols, structure.counts,
            (structure.scaled_positions
             + np.array([shift / grid.shape[0], 0.0, 0.0])) % 1.0,
        )
        shifted = ExternalPotential.compute(
            moved, grid, potential.metadata["charges"],
            widths=potential.metadata["widths"],
        )
        assert np.abs(shifted.data
                      - np.roll(potential.data, shift, axis=0)).max() < 1e-10

    def test_linear_in_ionic_charge(self, poscar):
        grid = FieldGrid((16, 16, 16), poscar.cell)
        full = ExternalPotential.compute(poscar, grid, {"Si": 4.0},
                                         widths={"Si": 0.5})
        half = ExternalPotential.compute(poscar, grid, {"Si": 2.0},
                                         widths={"Si": 0.5})
        assert np.abs(2.0 * half.data - full.data).max() < 1e-10

    def test_gaussian_model_is_grid_converged(self, poscar):
        cell = np.eye(3) * 12.0
        structure = Poscar(cell, ["H"], [1], [[0.35, 0.5, 0.65]])
        coarse = ExternalPotential.compute(structure, FieldGrid((30, 30, 30), cell),
                                           {"H": 3.0}, widths={"H": 0.6})
        fine = ExternalPotential.compute(structure, FieldGrid((60, 60, 60), cell),
                                         {"H": 3.0}, widths={"H": 0.6})
        assert np.abs(fine.data[::2, ::2, ::2] - coarse.data).max() < 1e-3

    def test_shared_grid_is_honoured(self, vasp_dir):
        grid = FieldGrid((20, 20, 20), Poscar.from_string(POSCAR_TEXT).cell)
        assert ExternalPotential.from_vasp(vasp_dir, grid=grid).shape == (20, 20, 20)

    def test_rejects_mismatched_potcar(self, vasp_dir):
        (vasp_dir / "POTCAR").write_text(POTCAR_TEXT.replace("Si", "Ge"))
        with pytest.raises(ValueError, match="do not match POSCAR"):
            ExternalPotential.from_vasp(vasp_dir)

    def test_requires_valence_charges(self, vasp_dir):
        (vasp_dir / "POTCAR").unlink()
        with pytest.raises(ValueError, match="No valence charge"):
            ExternalPotential.from_vasp(vasp_dir)
        assert ExternalPotential.from_vasp(vasp_dir, zval={"Si": 4.0}) is not None

    def test_unknown_model(self, poscar):
        grid = FieldGrid((8, 8, 8), poscar.cell)
        with pytest.raises(ValueError, match="Unknown pseudo-ion model"):
            ExternalPotential.compute(poscar, grid, {"Si": 4.0}, model="jellium")


# --------------------------------------------------------------------- #
# CHGCAR-format I/O shared by every field
# --------------------------------------------------------------------- #
class TestVolumetricIO:
    def test_extcar_round_trip(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        path = vasp_dir / "EXTCAR"
        potential.write(path)

        again = ExternalPotential.read(path)
        assert again.shape == potential.shape
        assert np.abs(again.data - potential.data).max() < 1e-9
        assert np.allclose(again.structure.cell, potential.structure.cell)

    def test_extcar_layout_matches_chgcar(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        path = vasp_dir / "EXTCAR"
        potential.write(path)
        lines = path.read_text().splitlines()

        assert lines[0].startswith("Si2")          # comment line
        assert lines[5].split() == ["Si"]          # species
        assert lines[6].split() == ["2"]           # counts
        assert lines[7].strip() == "Direct"
        assert lines[10].strip() == ""             # blank separator
        assert [int(t) for t in lines[11].split()] == list(potential.shape)
        assert len(lines[12].split()) == 5         # CHGCAR uses 5 columns

    def test_chgcar_volume_scaling(self, vasp_dir):
        grid = FieldGrid((12, 12, 12), Poscar.from_string(POSCAR_TEXT).cell)
        structure = Poscar.from_string(POSCAR_TEXT)
        values = np.abs(np.random.default_rng(0).normal(size=grid.shape)) * 0.1
        density = ChargeDensity(values, grid, structure)

        path = vasp_dir / "CHGCAR"
        density.write(path)
        # On disk VASP stores rho * Omega, not rho.
        raw = np.array(path.read_text().splitlines()[12].split(), dtype=float)
        assert raw[0] == pytest.approx(values.ravel(order="F")[0] * grid.volume,
                                       rel=1e-9)

        again = ChargeDensity.read(path, grid=grid)
        assert np.abs(again.data - values).max() < 1e-9
        assert again.electron_count() == pytest.approx(density.electron_count())

    def test_read_rejects_foreign_grid(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        path = vasp_dir / "EXTCAR"
        potential.write(path)
        wrong = FieldGrid((4, 4, 4), potential.structure.cell)
        with pytest.raises(ValueError, match="same mesh"):
            ExternalPotential.read(path, grid=wrong)

    def test_grid_from_file(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        potential.write(vasp_dir / "EXTCAR")
        assert FieldGrid.from_file(vasp_dir / "EXTCAR").matches(potential.grid)

    def test_fields_compose_only_on_a_shared_grid(self, vasp_dir):
        potential = ExternalPotential.from_vasp(vasp_dir)
        other = ExternalPotential.compute(
            potential.structure, FieldGrid((8, 8, 8), potential.structure.cell),
            {"Si": 4.0},
        )
        assert (potential + potential).data.max() == pytest.approx(
            2 * potential.data.max())
        with pytest.raises(ValueError, match="different grids"):
            potential + other


# --------------------------------------------------------------------- #
# Orbital-free kinetic energy densities
# --------------------------------------------------------------------- #
class TestKineticEnergyDensities:
    def test_thomas_fermi_uniform_gas(self):
        """tau_TF = C_TF rho^(5/3); check against the closed form in a.u."""
        bohr = 0.529177210903
        rho_atomic = 0.05                                  # e/Bohr^3
        tau = thomas_fermi_tau(np.full((4, 4, 4), rho_atomic / bohr ** 3))
        expected = 2.871234000188191 * rho_atomic ** (5.0 / 3.0) \
            * 27.211386245988 / bohr ** 3
        assert np.allclose(tau, expected)

    def test_von_weizsacker_vanishes_for_uniform_density(self, poscar):
        grid = FieldGrid((12, 12, 12), poscar.cell)
        tau = von_weizsacker_tau(np.full(grid.shape, 0.3), grid)
        assert np.abs(tau).max() < 1e-12

    def test_von_weizsacker_exact_for_a_single_orbital(self):
        r"""For rho = psi^2 with psi = cos(kx), tau_vW must equal |grad psi|^2 / 2."""
        length = 10.0
        cell = np.eye(3) * length
        grid = FieldGrid((64, 4, 4), cell)
        x = grid.cartesian_coordinates()[..., 0]
        k = 2 * np.pi / length
        psi = np.cos(k * x) + 1.5                      # nodeless -> tau_vW exact
        rho = psi ** 2

        bohr = 0.529177210903
        tau = von_weizsacker_tau(rho / bohr ** 3, grid)          # eV/Ang^3
        expected = 0.5 * (k * bohr * np.sin(k * x)) ** 2 \
            * 27.211386245988 / bohr ** 3
        assert np.abs(tau - expected).max() < 1e-8 * np.abs(expected).max() + 1e-9

    def test_taucar_round_trip(self, vasp_dir):
        structure = Poscar.from_string(POSCAR_TEXT)
        grid = FieldGrid((10, 10, 10), structure.cell)
        values = np.abs(np.random.default_rng(1).normal(size=grid.shape))
        tau = KineticEnergyDensity(values, grid, structure)
        tau.write(vasp_dir / "TAUCAR")
        again = KineticEnergyDensity.read(vasp_dir / "TAUCAR", grid=grid)
        assert np.abs(again.data - values).max() < 1e-9

    def test_compute_is_deferred(self):
        with pytest.raises(NotImplementedError):
            ChargeDensity.compute()
        with pytest.raises(NotImplementedError):
            KineticEnergyDensity.compute()


# --------------------------------------------------------------------- #
# Exact local pseudopotential from the POTCAR tables
#
# Layout recovered from the VASP source (pseudo.F:294-305, pot.F POTION);
# see docs/vasp_analysis_report.md.
# --------------------------------------------------------------------- #
class TestPotcarLocalTables:
    def test_first_value_is_psgmax_not_zval(self):
        """The number after 'local part' is PSGMAX, a wavevector, not ZVAL.

        Misreading it was the root cause of the EXTCAR discrepancy, so it is
        pinned here explicitly.
        """
        text = POTCAR_TEXT.replace(
            "  local part\n             4.00000000000000\n",
            "  local part\n   75.5890395431569\n",
        )
        entry = Potcar.from_string(text, parse_tables=True)[0]
        assert entry.psgmax == pytest.approx(75.5890395431569)
        assert entry.zval == 4.0                      # unchanged, separate quantity

    def test_q_grid_is_uniform_from_zero(self):
        text = POTCAR_TEXT.replace(
            "  local part\n             4.00000000000000\n",
            "  local part\n   100.0\n",
        )
        entry = Potcar.from_string(text, parse_tables=True)[0]
        q = entry.local_q_grid
        # The spacing is always PSGMAX/NPSPTS; the *length* tracks how many
        # samples were actually present, so q and the values always pair up.
        assert q[0] == 0.0
        assert len(q) == len(entry.local_potential)
        assert q[1] == pytest.approx(100.0 / entry.NPSPTS)
        assert np.allclose(np.diff(q), 100.0 / entry.NPSPTS)

    def test_tables_absent_without_parse_tables(self):
        entry = Potcar.from_string(POTCAR_TEXT)[0]
        assert entry.psgmax is None
        assert entry.local_potential is None
        assert entry.pscore is None


# --------------------------------------------------------------------- #
# Gaussian smoothing
# --------------------------------------------------------------------- #
class TestSmoothing:
    @staticmethod
    def _field(cell, shape=(24, 24, 24)):
        grid = FieldGrid(shape, cell)
        structure = Poscar(cell, ["Si"], [2], [[0.25, 0.25, 0.25],
                                               [0.6, 0.4, 0.7]])
        return ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                         widths={"Si": 0.5})

    def test_zero_sigma_is_a_copy(self):
        field = self._field(np.eye(3) * 8.0)
        for sigma in (0, 0.0, None):
            out = field.smooth(sigma)
            assert np.array_equal(out.data, field.data)
            assert out is not field

    def test_preserves_the_cell_average(self):
        """The G=0 coefficient is untouched, so integrals survive exactly."""
        field = self._field(np.eye(3) * 8.0)
        for sigma in (0.2, 0.5, 1.0):
            assert field.smooth(sigma).mean() == pytest.approx(field.mean(),
                                                               abs=1e-12)

    def test_monotonically_reduces_variance(self):
        field = self._field(np.eye(3) * 8.0)
        deviations = [field.smooth(s).data.std() for s in (0.0, 0.2, 0.4, 0.8)]
        assert all(a > b for a, b in zip(deviations, deviations[1:]))

    def test_methods_agree_on_an_orthogonal_cell(self):
        """On a cubic cell the two routes are the same convolution.

        The residual is the discrete kernel's truncation, so it shrinks as the
        field is better resolved; a coarse grid with a sharp potential leaves a
        few tenths of a percent.
        """
        field = self._field(np.eye(3) * 8.0, shape=(32, 32, 32))
        for sigma in (0.3, 0.5):
            spectral = field.smooth(sigma, "spectral").data
            ndimage = field.smooth(sigma, "ndimage").data
            assert np.abs(spectral - ndimage).max() < 1e-2 * np.abs(spectral).max()

    def test_methods_disagree_on_a_skewed_cell(self):
        """ndimage blurs along lattice axes, which is anisotropic in Cartesian
        space unless the cell is orthogonal. This is why 'spectral' is default;
        the gold data set is fcc, so the distinction is not academic.
        """
        cell = 6.0 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        field = self._field(cell)
        spectral = field.smooth(0.4, "spectral").data
        ndimage = field.smooth(0.4, "ndimage").data
        assert np.abs(spectral - ndimage).max() > 1e-2 * np.abs(spectral).max()

    def test_rejects_bad_arguments(self):
        field = self._field(np.eye(3) * 8.0)
        with pytest.raises(ValueError, match="non-negative"):
            field.smooth(-0.5)
        with pytest.raises(ValueError, match="Unknown smoothing method"):
            field.smooth(0.3, "box")

    def test_records_provenance(self):
        out = self._field(np.eye(3) * 8.0).smooth(0.3, "spectral")
        assert out.metadata["gaussian_blur"] == 0.3
        assert out.metadata["gaussian_blur_method"] == "spectral"

    def test_preserves_type_and_grid(self):
        field = self._field(np.eye(3) * 8.0)
        out = field.smooth(0.3)
        assert isinstance(out, ExternalPotential)
        assert out.grid is field.grid

    def test_from_calculation_applies_the_blur(self, vasp_dir):
        sharp = ExternalPotential.from_calculation(vasp_dir, encut=245.0)
        blurred = ExternalPotential.from_calculation(vasp_dir, encut=245.0,
                                                     gaussian_blur=0.3)
        assert blurred.data.std() < sharp.data.std()
        assert blurred.metadata["gaussian_blur"] == 0.3
        # smoothing must not shift the neutralising-background convention
        assert blurred.mean() == pytest.approx(0.0, abs=1e-10)

    def test_blur_is_off_by_default(self, vasp_dir):
        field = ExternalPotential.from_calculation(vasp_dir, encut=245.0)
        assert "gaussian_blur" not in field.metadata

    def test_truncated_local_table_is_rejected(self):
        """A short 'local part' block must not reach the spline.

        It parses without error but cannot be interpolated onto the PSGMAX
        mesh; gating on `has_local_table` makes it fall back to the analytic
        model instead of raising from inside SciPy.
        """
        entry = Potcar.from_string(POTCAR_TEXT, parse_tables=True)[0]
        assert entry.local_potential is not None
        assert len(entry.local_potential) < entry.NPSPTS
        assert entry.has_local_table is False
        # the q grid must still pair with whatever was read
        assert len(entry.local_q_grid) == len(entry.local_potential)
