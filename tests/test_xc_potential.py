# -*- coding: utf-8 -*-
# file: test_xc_potential.py
"""
Tests for the exchange-correlation potential.

``euler_lagrange_residual`` has always accepted a ``v_xc`` argument and there
was nothing in the package able to produce one, so every residual ever
computed here silently dropped the term. It is not a small term: in a valence
region v_xc is of order -15 eV, the same order as the kinetic potential it is
weighed against in the Euler-Lagrange equation.

A potential is the functional derivative of an energy, so that is what these
check: against the energy in :mod:`poraque.physics.energy` by finite
differences, and against the two closed-form identities that hold exactly.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.fields.grid import FieldGrid  # noqa: E402
from poraque.ml.physics import (  # noqa: E402
    lda_exchange_potential,
    pw92_correlation_potential,
    xc_potential,
)
from poraque.physics.energy import (  # noqa: E402
    lda_exchange_energy,
    pw92_correlation_energy,
)


@pytest.fixture
def grid():
    return FieldGrid((8, 8, 8), np.eye(3) * 4.0)


@pytest.fixture
def density():
    rng = np.random.default_rng(0)
    return rng.uniform(0.05, 2.0, (8, 8, 8))


def volume_element(grid):
    return grid.volume / np.prod(grid.shape)


class TestItIsTheDerivativeOfTheEnergy:
    """
    The defining property, checked the only way that cannot be circular:
    against the independently written numpy energy, by finite differences.
    """

    @pytest.mark.parametrize("voxel", [(1, 2, 3), (4, 4, 4), (7, 0, 5),
                                       (0, 0, 0)])
    def test_matches_finite_differences(self, density, grid, voxel):
        def energy(values):
            return (lda_exchange_energy(values, grid)
                    + pw92_correlation_energy(values, grid))

        analytic = xc_potential(torch.tensor(density), "lda").numpy()[voxel]

        step = 1e-6
        up, down = density.copy(), density.copy()
        up[voxel] += step
        down[voxel] -= step
        # dE/drho_i = v_i * dv, by the definition of a functional derivative.
        numerical = ((energy(up) - energy(down))
                     / (2 * step) / volume_element(grid))

        assert numerical == pytest.approx(analytic, rel=1e-5)

    def test_exchange_alone_matches_its_own_energy(self, density, grid):
        """Exchange separately, so a correlation error cannot mask one here."""
        analytic = lda_exchange_potential(torch.tensor(density)).numpy()

        step = 1e-6
        voxel = (2, 5, 1)
        up, down = density.copy(), density.copy()
        up[voxel] += step
        down[voxel] -= step
        numerical = ((lda_exchange_energy(up, grid)
                      - lda_exchange_energy(down, grid))
                     / (2 * step) / volume_element(grid))

        assert numerical == pytest.approx(analytic[voxel], rel=1e-5)


class TestClosedFormIdentities:
    def test_exchange_virial_relation(self, density, grid):
        r"""
        Dirac exchange is homogeneous of degree 4/3 in :math:`\rho`, so
        Euler's theorem gives :math:`E_x = \tfrac34\int\rho\,v_x`. This is
        exact and independent of the finite-difference check.
        """
        v_x = lda_exchange_potential(torch.tensor(density)).numpy()
        integrated = 0.75 * (density * v_x).sum() * volume_element(grid)
        assert integrated == pytest.approx(lda_exchange_energy(density, grid),
                                           rel=1e-12)

    def test_exchange_scales_as_the_cube_root(self):
        r""":math:`v_x \propto \rho^{1/3}`, so doubling gives :math:`2^{1/3}`."""
        rho = torch.full((4, 4, 4), 0.5, dtype=torch.float64)
        one = lda_exchange_potential(rho)
        two = lda_exchange_potential(2 * rho)
        assert torch.allclose(two / one,
                              torch.full_like(one, 2.0 ** (1.0 / 3.0)))

    def test_both_terms_are_negative(self, density):
        """Exchange and PW92 correlation are negative at every density."""
        rho = torch.tensor(density)
        assert (lda_exchange_potential(rho) < 0).all()
        assert (pw92_correlation_potential(rho) < 0).all()

    def test_exchange_dominates_correlation(self, density):
        """At valence densities, by roughly an order of magnitude."""
        rho = torch.tensor(density)
        x = lda_exchange_potential(rho).abs().mean()
        c = pw92_correlation_potential(rho).abs().mean()
        assert x > 5 * c


class TestNumericalBehaviour:
    def test_vacuum_does_not_produce_nan(self):
        """
        Vacuum is where this breaks if it breaks: r_s diverges as rho goes to
        zero, and a plane-wave density reaches zero, and below it.
        """
        rho = torch.tensor([0.0, -1e-12, 1e-30, 1e-8, 1.0],
                           dtype=torch.float64)
        for name in ("lda", "x"):
            values = xc_potential(rho, name)
            assert torch.isfinite(values).all(), name

    def test_it_is_differentiable(self, density):
        """It has to survive being placed in a loss that is backpropagated."""
        rho = torch.tensor(density, requires_grad=True)
        xc_potential(rho, "lda").sum().backward()
        assert rho.grad is not None
        assert torch.isfinite(rho.grad).all()

    def test_shape_and_dtype_are_preserved(self, density):
        rho = torch.tensor(density, dtype=torch.float32).reshape(1, 1, 8, 8, 8)
        out = xc_potential(rho, "lda")
        assert out.shape == rho.shape
        assert out.dtype == rho.dtype


class TestSelection:
    def test_none_returns_zeros(self, density):
        out = xc_potential(torch.tensor(density), "none")
        assert torch.count_nonzero(out) == 0

    def test_exchange_only_omits_correlation(self, density):
        rho = torch.tensor(density)
        assert torch.allclose(xc_potential(rho, "x"),
                              lda_exchange_potential(rho))

    def test_lda_is_the_sum_of_its_parts(self, density):
        rho = torch.tensor(density)
        assert torch.allclose(
            xc_potential(rho, "lda"),
            lda_exchange_potential(rho) + pw92_correlation_potential(rho))

    def test_an_unknown_functional_raises(self, density):
        with pytest.raises(ValueError, match="Unknown xc functional"):
            xc_potential(torch.tensor(density), "b3lyp")


class TestItReachesTheEulerLagrangeResidual:
    def test_the_residual_changes_when_xc_is_supplied(self, density):
        """
        The regression this module exists to prevent: for as long as nothing
        could build a v_xc, every residual was computed without one.
        """
        from poraque.ml.physics import euler_lagrange_residual

        rho = torch.tensor(density).reshape(1, 1, 8, 8, 8)
        cell = (torch.eye(3).unsqueeze(0) * 4.0).double()
        v_ext = torch.zeros_like(rho)

        without = euler_lagrange_residual(rho, v_ext, cell)
        with_xc = euler_lagrange_residual(rho, v_ext, cell,
                                          v_xc=xc_potential(rho, "lda"))
        # Not merely different: different by a physically significant amount.
        assert (with_xc - without).std() > 0.1 * without.std()


class TestEulerLagrangeInversion:
    """
    The Euler-Lagrange equation read backwards, which is the only route to a
    pointwise label for delta T_s / delta rho.
    """

    @pytest.fixture
    def fields(self):
        torch.manual_seed(0)
        rho = torch.rand(1, 1, 8, 8, 8, dtype=torch.float64) * 0.4 + 0.1
        v_ext = torch.randn(1, 1, 8, 8, 8, dtype=torch.float64) * 5.0
        cell = (torch.eye(3).unsqueeze(0) * 6.0).double()
        return rho, v_ext, cell

    def test_it_closes_the_euler_lagrange_equation_exactly(self, fields):
        """
        The defining property: substituting the inverted potential back into
        the residual must annihilate it. If this fails the inversion and the
        residual disagree about what the equation is.
        """
        from poraque.ml.physics import (euler_lagrange_residual,
                                        exact_kinetic_potential)

        rho, v_ext, cell = fields
        kinetic = exact_kinetic_potential(rho, v_ext, cell, xc="lda")
        residual = euler_lagrange_residual(
            rho, v_ext, cell, v_xc=xc_potential(rho, "lda"), kinetic=kinetic)
        assert residual.abs().max() < 1e-12

    def test_it_is_zero_mean_without_a_chemical_potential(self, fields):
        from poraque.ml.physics import exact_kinetic_potential

        rho, v_ext, cell = fields
        assert exact_kinetic_potential(rho, v_ext, cell).mean().abs() < 1e-12

    def test_mu_shifts_it_rigidly(self, fields):
        """mu is an additive constant, so it must not change any difference."""
        from poraque.ml.physics import exact_kinetic_potential

        rho, v_ext, cell = fields
        zero = exact_kinetic_potential(rho, v_ext, cell)
        shifted = exact_kinetic_potential(rho, v_ext, cell, mu=7.5)
        spread = shifted - shifted.mean()
        assert torch.allclose(zero, spread, atol=1e-12)

    def test_omitting_xc_changes_the_answer_materially(self, fields):
        from poraque.ml.physics import exact_kinetic_potential

        rho, v_ext, cell = fields
        with_xc = exact_kinetic_potential(rho, v_ext, cell, xc="lda")
        without = exact_kinetic_potential(rho, v_ext, cell, xc="none")
        assert (with_xc - without).std() > 0.1


class TestLevyPerdewSahni:
    """
    LPS is the Euler-Lagrange equation in a different variable, not an
    independent condition. These pin that down so the claim cannot rot.
    """

    @pytest.fixture
    def fields(self):
        torch.manual_seed(1)
        rho = torch.rand(1, 1, 8, 8, 8, dtype=torch.float64) * 0.4 + 0.1
        v_ext = torch.randn(1, 1, 8, 8, 8, dtype=torch.float64) * 5.0
        cell = (torch.eye(3).unsqueeze(0) * 6.0).double()
        return rho, v_ext, cell

    def test_kinetic_potential_splits_into_bosonic_and_pauli(self, fields):
        r"""
        :math:`\delta T_s/\delta\rho = \delta T_{\rm vW}/\delta\rho +
        v_{\rm P}` identically. This IS the equivalence of LPS and
        Euler-Lagrange, and it holds to machine precision or not at all.
        """
        from poraque.ml.physics import (exact_kinetic_potential,
                                        exact_pauli_potential,
                                        von_weizsacker_potential)

        rho, v_ext, cell = fields
        kinetic = exact_kinetic_potential(rho, v_ext, cell)
        pauli = exact_pauli_potential(rho, v_ext, cell)
        vw = von_weizsacker_potential(rho, cell)
        vw = vw - vw.mean()
        assert torch.allclose(kinetic, vw + pauli, atol=1e-12)

    def test_a_one_orbital_density_has_no_pauli_term(self):
        r"""
        For a density that IS a single orbital, :math:`T_s = T_{\rm vW}` and
        the Pauli potential vanishes. Constructed by choosing the external
        potential that makes a chosen density stationary, which is the
        inversion this module performs.
        """
        from poraque.ml.physics import (exact_pauli_potential,
                                        hartree_potential,
                                        von_weizsacker_potential)

        torch.manual_seed(2)
        axis = torch.arange(12, dtype=torch.float64) / 12 * 2 * np.pi
        x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
        rho = (1.0 + 0.3 * torch.sin(x) + 0.2 * torch.cos(y))[None, None]
        cell = (torch.eye(3).unsqueeze(0) * 6.0).double()

        # Pick v_ext so that dT_vW/drho + v_ext + v_H + v_xc is constant.
        vw = von_weizsacker_potential(rho, cell)
        v_ext = -(vw + hartree_potential(rho, cell)
                  + xc_potential(rho, "pbe", cell=cell))

        pauli = exact_pauli_potential(rho, v_ext, cell, xc="pbe")
        assert pauli.abs().max() < 1e-10

    def test_mu_can_always_make_the_pauli_potential_non_negative(self, fields):
        """
        Which is why "v_P >= 0" is not by itself a test: the constraint is
        only meaningful once mu is fixed independently.
        """
        from poraque.ml.physics import exact_pauli_potential

        rho, v_ext, cell = fields
        zero_mean = exact_pauli_potential(rho, v_ext, cell)
        lifted = exact_pauli_potential(rho, v_ext, cell,
                                       mu=0.0) + zero_mean.max() + 1.0
        assert (lifted - lifted.min() >= 0).all()


class TestPBE:
    """
    PBE is the default because the reference data is PBE (``PAW_PBE``
    potentials, ``LEXCH = PE``). Its potential carries a divergence term,
    ``-div(de/d grad rho)``, which is taken by autograd rather than derived by
    hand; these check that the autograd route is actually the derivative.
    """

    @pytest.fixture
    def smooth(self):
        """Band-limited, so the spectral gradient is meaningful."""
        axis = np.arange(12) / 12 * 2 * np.pi
        x, y, _ = np.meshgrid(axis, axis, axis, indexing="ij")
        return 0.8 + 0.3 * np.sin(x) + 0.2 * np.cos(y)

    @pytest.fixture
    def cell12(self):
        return (torch.eye(3).unsqueeze(0) * 6.0).double()

    @pytest.fixture
    def grid12(self):
        return FieldGrid((12, 12, 12), np.eye(3) * 6.0)

    def test_energy_matches_the_numpy_implementation(self, smooth, cell12,
                                                     grid12):
        from poraque.ml.physics import integrate, xc_energy_density
        from poraque.physics.energy import (pbe_correlation_energy,
                                            pbe_exchange_energy)

        field = torch.tensor(smooth, dtype=torch.float64)[None, None]
        mine = integrate(xc_energy_density(field, cell12, "pbe"), cell12).item()
        reference = (pbe_exchange_energy(smooth, grid12)
                     + pbe_correlation_energy(smooth, grid12))
        assert mine == pytest.approx(reference, rel=1e-8)

    @pytest.mark.parametrize("voxel", [(1, 2, 3), (6, 6, 6), (0, 0, 0)])
    def test_potential_matches_finite_differences(self, smooth, cell12,
                                                  grid12, voxel):
        """
        The check that the divergence term is right. A GGA potential is NOT
        d(e)/d(rho) at a point, and omitting the gradient piece would still
        produce a plausible field.
        """
        from poraque.physics.energy import (pbe_correlation_energy,
                                            pbe_exchange_energy)

        def energy(values):
            return (pbe_exchange_energy(values, grid12)
                    + pbe_correlation_energy(values, grid12))

        field = torch.tensor(smooth, dtype=torch.float64)[None, None]
        analytic = xc_potential(field, "pbe", cell=cell12).squeeze().numpy()

        step = 1e-6
        up, down = smooth.copy(), smooth.copy()
        up[voxel] += step
        down[voxel] -= step
        dv = grid12.volume / smooth.size
        numerical = (energy(up) - energy(down)) / (2 * step) / dv

        assert numerical == pytest.approx(analytic[voxel], rel=1e-6)

    def test_it_reduces_to_lda_on_a_uniform_density(self, cell12):
        """
        Both PBE enhancement factors vanish when the gradient does, so the
        functional must collapse exactly onto Dirac plus PW92.
        """
        flat = torch.full((1, 1, 8, 8, 8), 0.5, dtype=torch.float64)
        cell = (torch.eye(3).unsqueeze(0) * 5.0).double()
        difference = (xc_potential(flat, "pbe", cell=cell)
                      - xc_potential(flat, "lda"))
        assert difference.abs().max() < 1e-12

    def test_it_differs_from_lda_where_the_reduced_gradient_is_large(self):
        """
        Not everywhere: on a dense, slowly varying density the enhancement
        factors are near 1 and the two agree to a milli-eV. They separate in
        the dilute, sharply varying region, which is exactly the valence tail
        that decides a kinetic functional.
        """
        axis = np.arange(16) / 16 * 2 * np.pi
        x, y, _ = np.meshgrid(axis, axis, axis, indexing="ij")
        cell = (torch.eye(3).unsqueeze(0) * 8.0).double()

        dense = torch.tensor(0.8 + 0.3 * np.sin(x))[None, None]
        dilute = torch.tensor(
            0.05 + 0.045 * np.sin(3 * x) + 0.02 * np.cos(2 * y))[None, None]

        near = (xc_potential(dense, "pbe", cell=cell)
                - xc_potential(dense, "lda")).abs().max()
        far = (xc_potential(dilute, "pbe", cell=cell)
               - xc_potential(dilute, "lda")).abs().max()

        assert near < 0.01                          # eV
        assert far > 0.1                            # eV

    def test_a_gradient_functional_without_a_cell_raises(self, smooth):
        field = torch.tensor(smooth, dtype=torch.float64)[None, None]
        with pytest.raises(ValueError, match="needs the density gradient"):
            xc_potential(field, "pbe")

    def test_it_is_differentiable_with_respect_to_the_density(self, smooth,
                                                             cell12):
        """
        The density is normally an operator's OUTPUT, so an Euler-Lagrange
        residual built from a predicted density only trains anything if the
        gradient reaches back through v_xc. This is a second derivative of the
        energy, and it is why this path does not go through
        ``functional_derivative``, which detaches.
        """
        field = torch.tensor(smooth, dtype=torch.float64).reshape(
            1, 1, 12, 12, 12).requires_grad_(True)
        potential = xc_potential(field, "pbe", cell=cell12, create_graph=True)
        potential.sum().backward()
        assert field.grad is not None and torch.isfinite(field.grad).all()


class TestFunctionalIsDeclarable:
    """
    The user must be able to say which functional produced the data, and the
    package must be able to work it out when they do not.
    """

    def test_an_explicit_declaration_wins(self):
        from poraque.fields.io import resolve_xc

        assert resolve_xc(declared="lda") == "lda"
        assert resolve_xc(declared="pbe") == "pbe"

    def test_auto_needs_a_directory(self):
        from poraque.fields.io import resolve_xc

        with pytest.raises(ValueError, match="needs a directory"):
            resolve_xc(declared="auto")

    def test_paw_pbe_resolves_to_pbe(self, tmp_path):
        """
        The case that matters here: a PAW_PBE POTCAR carries LEXCH = PE, so a
        dataset built from one needs nothing declared.
        """
        from poraque.fields.io import resolve_xc

        (tmp_path / "POSCAR").write_text(
            "Au\n1.0\n4 0 0\n0 4 0\n0 0 4\nAu\n1\nDirect\n0 0 0\n")
        (tmp_path / "POTCAR").write_text(
            "  PAW_PBE Au 04Oct2007\n   ZVAL   =   11.000\n"
            "   LEXCH  = PE\n   TITEL  = PAW_PBE Au 04Oct2007\n")
        assert resolve_xc(str(tmp_path), declared="auto") == "pbe"

    def test_an_lda_potcar_resolves_to_lda(self, tmp_path):
        from poraque.fields.io import resolve_xc

        (tmp_path / "POSCAR").write_text(
            "Au\n1.0\n4 0 0\n0 4 0\n0 0 4\nAu\n1\nDirect\n0 0 0\n")
        (tmp_path / "POTCAR").write_text(
            "  PAW Au 04Oct2007\n   ZVAL   =   11.000\n"
            "   LEXCH  = CA\n   TITEL  = PAW Au 04Oct2007\n")
        assert resolve_xc(str(tmp_path), declared="auto") == "lda"

    def test_an_unimplemented_functional_warns_rather_than_lying(self,
                                                                tmp_path):
        """
        PW91 is recognised and not implemented. Substituting PBE silently
        would put a mislabelled potential into the Euler-Lagrange residual.
        """
        from poraque.fields.io import resolve_xc

        (tmp_path / "POSCAR").write_text(
            "Au\n1.0\n4 0 0\n0 4 0\n0 0 4\nAu\n1\nDirect\n0 0 0\n")
        (tmp_path / "POTCAR").write_text(
            "  PAW Au\n   ZVAL   =   11.000\n   LEXCH  = 91\n"
            "   TITEL  = PAW Au\n")
        with pytest.warns(RuntimeWarning, match="not implemented"):
            assert resolve_xc(str(tmp_path), declared="auto") == "pbe"

    def test_an_undeterminable_directory_warns(self, tmp_path):
        from poraque.fields.io import resolve_xc

        with pytest.warns(RuntimeWarning, match="Could not determine"):
            assert resolve_xc(str(tmp_path), declared="auto") == "pbe"

    def test_the_config_carries_it(self):
        """``data.xc`` must reach a run, and default to detection."""
        from poraque.ml.config import TrainingConfig

        assert TrainingConfig.from_dict({}).data.xc == "auto"
        assert TrainingConfig.from_dict(
            {"data": {"xc": "pbe"}}).data.xc == "pbe"
