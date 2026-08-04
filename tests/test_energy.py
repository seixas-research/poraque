# -*- coding: utf-8 -*-
# file: test_energy.py

"""
Tests for :mod:`poraque.physics.energy`.

Every term here is checked against something *external* to the implementation —
an analytic integral, a tabulated Madelung constant, a uniform-electron-gas
limit — rather than against a previously recorded output. A regression test
that only compares to yesterday's number cannot tell a correct implementation
from a consistently wrong one.
"""

import numpy as np
import pytest

from poraque.fields import FieldGrid
from poraque.fields.constants import (
    BOHR_TO_ANGSTROM,
    COULOMB_CONSTANT_EV_ANGSTROM,
    HARTREE_TO_EV,
)
from poraque.fields.structure import Structure
from poraque.physics import (
    EnergyCalculator,
    EnergyComponents,
    ewald_energy,
    hartree_energy,
    hartree_potential,
    lda_exchange_energy,
    pw92_correlation_energy,
    xc_energy,
)
from poraque.physics.energy import alpha_z_energy


@pytest.fixture
def cubic_grid():
    return FieldGrid((32, 32, 32), np.eye(3) * 10.0)


# ===================================================================== #
# Hartree
# ===================================================================== #
class TestHartree:
    def test_uniform_density_gives_zero(self, cubic_grid):
        """With G=0 removed the electrons are exactly neutralized."""
        uniform = np.full(cubic_grid.shape, 0.3)
        assert abs(hartree_energy(uniform, cubic_grid)) < 1e-10

    def test_single_cosine_matches_the_analytic_integral(self, cubic_grid):
        r"""
        For :math:`\rho = \rho_0 + A\cos(Gx)` the energy is
        :math:`\pi e^2 A^2\Omega/G^2` exactly — one occupied mode, so the
        FFT result must reproduce it to round-off, not merely approximately.
        """
        length = 10.0
        x = cubic_grid.scaled_coordinates()[..., 0]
        amplitude = 0.05
        density = 0.3 + amplitude * np.cos(2.0 * np.pi * x)

        g = 2.0 * np.pi / length
        expected = (np.pi * COULOMB_CONSTANT_EV_ANGSTROM * amplitude ** 2
                    * cubic_grid.volume / g ** 2)
        assert hartree_energy(density, cubic_grid) == pytest.approx(expected,
                                                                   rel=1e-12)

    def test_energy_is_non_negative(self, cubic_grid):
        rng = np.random.default_rng(0)
        density = 0.5 + 0.1 * rng.standard_normal(cubic_grid.shape)
        assert hartree_energy(density, cubic_grid) > 0.0

    def test_potential_has_zero_mean(self, cubic_grid):
        """The G=0 convention is what makes v_H addable to V_ext."""
        rng = np.random.default_rng(1)
        density = 0.5 + 0.1 * rng.standard_normal(cubic_grid.shape)
        assert abs(hartree_potential(density, cubic_grid).mean()) < 1e-10

    def test_scales_quadratically_with_the_density(self, cubic_grid):
        x = cubic_grid.scaled_coordinates()[..., 0]
        density = 0.3 + 0.05 * np.cos(2.0 * np.pi * x)
        single = hartree_energy(density, cubic_grid)
        double = hartree_energy(2.0 * density, cubic_grid)
        assert double == pytest.approx(4.0 * single, rel=1e-12)

    def test_rejects_a_mismatched_shape(self, cubic_grid):
        with pytest.raises(ValueError, match="shape"):
            hartree_energy(np.zeros((8, 8, 8)), cubic_grid)


# ===================================================================== #
# Exchange and correlation
# ===================================================================== #
class TestExchangeCorrelation:
    @pytest.mark.parametrize("r_s", [1.0, 2.0, 5.0])
    def test_dirac_exchange_matches_the_uniform_gas(self, cubic_grid, r_s):
        r"""
        :math:`\varepsilon_x = -0.458165293/r_s` Hartree per electron, exactly.
        """
        density = self._uniform_at(cubic_grid, r_s)
        n_electrons = cubic_grid.integrate(density)
        per_electron = (lda_exchange_energy(density, cubic_grid)
                        / n_electrons / HARTREE_TO_EV)
        assert per_electron == pytest.approx(-0.458165293 / r_s, rel=1e-9)

    @pytest.mark.parametrize("r_s, expected", [
        (1.0, -0.059774), (2.0, -0.044760), (5.0, -0.028216),
    ])
    def test_pw92_correlation_matches_published_values(self, cubic_grid,
                                                       r_s, expected):
        """Perdew and Wang, PRB 45, 13244 (1992), Table I, unpolarized."""
        density = self._uniform_at(cubic_grid, r_s)
        n_electrons = cubic_grid.integrate(density)
        per_electron = (pw92_correlation_energy(density, cubic_grid)
                        / n_electrons / HARTREE_TO_EV)
        assert per_electron == pytest.approx(expected, abs=1e-6)

    def test_correlation_is_smaller_than_exchange(self, cubic_grid):
        density = self._uniform_at(cubic_grid, 2.0)
        assert abs(pw92_correlation_energy(density, cubic_grid)) < \
               abs(lda_exchange_energy(density, cubic_grid))

    def test_negative_density_is_clipped_not_propagated(self, cubic_grid):
        """
        Band-limiting rings, so a resampled field can dip below zero.
        rho**(4/3) of a negative number is not real; those points carry no
        electrons and must contribute nothing.
        """
        density = np.full(cubic_grid.shape, 0.1)
        density[0, 0, 0] = -1e-3
        assert np.isfinite(lda_exchange_energy(density, cubic_grid))
        assert np.isfinite(pw92_correlation_energy(density, cubic_grid))

    def test_zero_density_is_finite(self, cubic_grid):
        """r_s diverges as rho -> 0; the integrand must still vanish."""
        assert pw92_correlation_energy(np.zeros(cubic_grid.shape),
                                       cubic_grid) == pytest.approx(0.0)

    def test_functional_selection(self, cubic_grid):
        density = self._uniform_at(cubic_grid, 2.0)
        assert xc_energy(density, cubic_grid, "none") == 0.0
        assert xc_energy(density, cubic_grid, "x-only") == pytest.approx(
            lda_exchange_energy(density, cubic_grid))
        assert xc_energy(density, cubic_grid, "lda") == pytest.approx(
            lda_exchange_energy(density, cubic_grid)
            + pw92_correlation_energy(density, cubic_grid))

    def test_rejects_an_unknown_functional(self, cubic_grid):
        with pytest.raises(ValueError, match="Unknown functional"):
            xc_energy(np.zeros(cubic_grid.shape), cubic_grid, "pbe")

    @staticmethod
    def _uniform_at(grid, r_s):
        """Uniform density with the given Wigner-Seitz radius, in e/Å³."""
        n_bohr = 3.0 / (4.0 * np.pi * r_s ** 3)
        return np.full(grid.shape, n_bohr / BOHR_TO_ANGSTROM ** 3)


# ===================================================================== #
# Ewald
# ===================================================================== #
class TestEwald:
    """
    Madelung constants are the sharpest available test: they are known to many
    digits and depend on every part of the sum — real space, reciprocal space,
    the self term and the background — being individually right.
    """

    @pytest.mark.parametrize("name, madelung, tolerance", [
        ("rocksalt", 1.747564594633, 1e-9),
        ("cesium_chloride", 1.762674773, 1e-8),
        ("zincblende", 1.6380550, 1e-6),
    ])
    def test_madelung_constant(self, name, madelung, tolerance):
        structure, r_nn = _ionic_crystal(name)
        energy = ewald_energy(structure, {"A": 1.0, "B": -1.0})
        computed = -energy * r_nn / (COULOMB_CONSTANT_EV_ANGSTROM
                                     * structure.natoms / 2)
        assert computed == pytest.approx(madelung, abs=tolerance)

    def test_independent_of_the_requested_accuracy(self):
        """
        The splitting parameter eta is derived from `accuracy`, and the exact
        result cannot depend on it. This is the cheapest check that the
        background term is present and correctly signed.
        """
        structure, _ = _ionic_crystal("rocksalt")
        charges = {"A": 1.0, "B": -1.0}
        loose = ewald_energy(structure, charges, accuracy=1e-8)
        tight = ewald_energy(structure, charges, accuracy=1e-16)
        assert loose == pytest.approx(tight, abs=1e-6)

    def test_charged_cell_uses_the_neutralizing_background(self):
        r"""
        A single +q ion in a cube is the classic jellium Madelung problem,
        :math:`E = -\xi q^2 e^2/2L` with :math:`\xi = 2.8372974795`.
        """
        length, charge = 4.0, 11.0
        structure = Structure(np.eye(3) * length, ["Au"], [1],
                              np.zeros((1, 3)))
        expected = (-2.8372974795 * charge ** 2
                    * COULOMB_CONSTANT_EV_ANGSTROM / (2.0 * length))
        assert ewald_energy(structure, {"Au": charge}) == pytest.approx(
            expected, rel=1e-7)

    def test_translation_invariance(self):
        structure, _ = _ionic_crystal("rocksalt")
        shifted = Structure(structure.cell, structure.symbols, structure.counts,
                            (structure.scaled_positions + 0.137) % 1.0)
        charges = {"A": 1.0, "B": -1.0}
        assert ewald_energy(shifted, charges) == pytest.approx(
            ewald_energy(structure, charges), abs=1e-6)

    def test_accepts_per_atom_charges(self):
        structure, _ = _ionic_crystal("rocksalt")
        per_atom = [1.0] * 4 + [-1.0] * 4
        assert ewald_energy(structure, per_atom) == pytest.approx(
            ewald_energy(structure, {"A": 1.0, "B": -1.0}))

    def test_rejects_a_wrong_number_of_charges(self):
        structure, _ = _ionic_crystal("rocksalt")
        with pytest.raises(ValueError, match="charges"):
            ewald_energy(structure, [1.0, -1.0])

    def test_rejects_a_missing_species(self):
        structure, _ = _ionic_crystal("rocksalt")
        with pytest.raises(KeyError):
            ewald_energy(structure, {"A": 1.0})


# ===================================================================== #
# alpha Z
# ===================================================================== #
class TestAlphaZ:
    def test_scales_with_the_electron_count_and_the_atom_count(self):
        structure = Structure(np.eye(3) * 4.0, ["Au"], [2],
                              np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))
        energy = alpha_z_energy(structure, {"Au": 100.0}, n_electrons=22.0)
        assert energy == pytest.approx(22.0 * 2 * 100.0 / 64.0)

    def test_raises_rather_than_defaulting_to_zero(self):
        """
        Silently dropping a term of order eV per atom would look like a
        working calculation while corrupting every comparison.
        """
        structure = Structure(np.eye(3) * 4.0, ["Au"], [1], np.zeros((1, 3)))
        with pytest.raises(KeyError, match="PSCORE"):
            alpha_z_energy(structure, {"Ag": 1.0}, n_electrons=11.0)

    def test_matches_a_decorated_potcar_symbol(self):
        structure = Structure(np.eye(3) * 4.0, ["Au_pv"], [1], np.zeros((1, 3)))
        assert alpha_z_energy(structure, {"Au": 10.0}, 1.0) == pytest.approx(
            10.0 / 64.0)


# ===================================================================== #
# Assembly
# ===================================================================== #
class TestEnergyComponents:
    def test_total_is_kinetic_plus_potential(self):
        components = EnergyComponents(kinetic=1.0, external=-2.0, hartree=3.0,
                                      xc=-4.0, alpha_z=5.0, ewald=-6.0)
        assert components.potential == pytest.approx(-2.0 + 3.0 - 4.0 + 5.0 - 6.0)
        assert components.total == pytest.approx(1.0 + components.potential)

    def test_absent_terms_are_reported_not_silently_zero(self):
        components = EnergyComponents(kinetic=1.0, external=-2.0, hartree=3.0,
                                      xc=-4.0)
        assert components.missing == ("alpha_z", "ewald")
        assert "incomplete" in str(components)

    def test_complete_result_reports_nothing_missing(self):
        components = EnergyComponents(kinetic=1.0, external=-2.0, hartree=3.0,
                                      xc=-4.0, alpha_z=0.5, ewald=-6.0)
        assert components.missing == ()
        assert "incomplete" not in str(components)

    def test_as_dict_round_trips_the_derived_values(self):
        components = EnergyComponents(kinetic=1.0, external=-2.0, hartree=3.0,
                                      xc=-4.0, alpha_z=5.0, ewald=-6.0,
                                      n_electrons=7.0)
        payload = components.as_dict()
        assert payload["total"] == pytest.approx(components.total)
        assert payload["n_electrons"] == 7.0


class TestEnergyCalculator:
    def test_kinetic_energy_is_the_integral_of_tau(self, cubic_grid):
        tau = np.full(cubic_grid.shape, 2.0)
        calculator = EnergyCalculator(cubic_grid)
        assert calculator.kinetic_energy(tau) == pytest.approx(
            2.0 * cubic_grid.volume)

    def test_external_energy_uses_the_electron_convention(self, cubic_grid):
        """
        V_ext is already the potential energy *of an electron*, so a negative
        potential and a positive density give a negative (bound) energy.
        """
        calculator = EnergyCalculator(cubic_grid)
        energy = calculator.external_energy(np.full(cubic_grid.shape, 0.5),
                                            np.full(cubic_grid.shape, -3.0))
        assert energy == pytest.approx(-1.5 * cubic_grid.volume)

    def test_ewald_is_none_without_charges(self, cubic_grid):
        assert EnergyCalculator(cubic_grid).ewald_energy() is None

    def test_compute_reports_the_electron_count(self, cubic_grid):
        calculator = EnergyCalculator(cubic_grid)
        density = np.full(cubic_grid.shape, 0.25)
        components = calculator.compute(density, np.zeros(cubic_grid.shape),
                                        np.zeros(cubic_grid.shape))
        assert components.n_electrons == pytest.approx(0.25 * cubic_grid.volume)
        assert components.missing == ("alpha_z", "ewald")

    def test_potential_and_total_agree_with_compute(self, cubic_grid):
        rng = np.random.default_rng(2)
        density = np.abs(rng.standard_normal(cubic_grid.shape)) * 0.1
        tau = np.abs(rng.standard_normal(cubic_grid.shape))
        potential = rng.standard_normal(cubic_grid.shape)

        calculator = EnergyCalculator(cubic_grid)
        components = calculator.compute(density, tau, potential)
        assert calculator.potential_energy(density, tau, potential) == \
            pytest.approx(components.potential)
        assert calculator.total_energy(density, tau, potential) == \
            pytest.approx(components.total)

    def test_from_potential_picks_up_grid_structure_and_charges(self):
        from poraque.fields import ExternalPotential

        structure = Structure(np.eye(3) * 6.0, ["Au"], [1], np.zeros((1, 3)))
        grid = FieldGrid((16, 16, 16), structure.cell)
        potential = ExternalPotential.compute(structure, grid, {"Au": 11.0})

        calculator = EnergyCalculator.from_potential(potential)
        assert calculator.grid is grid
        assert calculator.charges == {"Au": 11.0}
        assert calculator.ewald_energy() is not None
        # No POTCAR tables were involved, so alpha_z must be absent, not zero.
        components = calculator.compute(np.zeros(grid.shape),
                                        np.zeros(grid.shape), potential)
        assert components.alpha_z is None
        assert "alpha_z" in components.missing


# ===================================================================== #
# Helpers
# ===================================================================== #
def _ionic_crystal(name):
    """``(Structure, nearest-neighbour distance)`` for a textbook lattice."""
    if name == "rocksalt":
        a = 5.64
        positions = np.array([
            [0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],       # cation
            [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5],       # anion
        ], dtype=float)
        return Structure(np.eye(3) * a, ["A", "B"], [4, 4], positions), a / 2

    if name == "cesium_chloride":
        a = 4.11
        positions = np.array([[0, 0, 0], [.5, .5, .5]], dtype=float)
        return (Structure(np.eye(3) * a, ["A", "B"], [1, 1], positions),
                a * np.sqrt(3) / 2)

    if name == "zincblende":
        a = 5.41
        positions = np.array([
            [0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
            [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25],
        ], dtype=float)
        return (Structure(np.eye(3) * a, ["A", "B"], [4, 4], positions),
                a * np.sqrt(3) / 4)

    raise ValueError(name)
