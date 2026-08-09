# -*- coding: utf-8 -*-
# file: test_functional_derivative.py
"""
Tests for the autograd functional derivative delta T_s / delta rho.

This is the mechanism that turns a learned tau into a usable kinetic energy
functional, so it is validated against closed forms rather than against itself.
The discretisation factor 1/dv is the failure mode worth guarding: omitting it
rescales the whole potential by the number of grid points, silently.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.ml.physics import (  # noqa: E402
    euler_lagrange_residual,
    functional_derivative,
    integrate,
    kinetic_potential,
    operator_kinetic_potential,
    thomas_fermi_potential,
    thomas_fermi_tau,
    volume_element,
    von_weizsacker_potential,
    von_weizsacker_tau,
)


@pytest.fixture
def cell():
    return (torch.eye(3).unsqueeze(0) * 8.0).double()


@pytest.fixture
def density():
    torch.manual_seed(0)
    return torch.rand(1, 1, 16, 16, 16, dtype=torch.float64) * 0.4 + 0.08


def smooth_density(n=32, modes=1):
    """A band-limited density, for comparisons against spectral derivatives."""
    axis = torch.arange(n, dtype=torch.float64) / n * 2 * np.pi
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    field = torch.ones_like(x)
    for k in range(1, modes + 1):
        field = field + (0.25 / k) * torch.sin(k * x) + (0.2 / k) * torch.cos(k * y)
    return field.unsqueeze(0).unsqueeze(0)


class TestAgainstClosedForms:
    def test_thomas_fermi_is_exact(self, density, cell):
        """TF is pointwise, so autograd must reproduce it to machine precision."""
        computed = kinetic_potential(thomas_fermi_tau, density, cell)
        expected = thomas_fermi_potential(density)
        assert (computed - expected).abs().max() < 1e-12 * expected.abs().max()

    def test_von_weizsacker_on_a_smooth_density(self, cell):
        r"""vW involves derivatives, so the two routes are different discretisations.

        Autograd differentiates the discrete :math:`\tau_{\rm vW}` exactly, while
        the analytic form applies a spectral Laplacian to :math:`\sqrt\rho` --
        which is not band-limited even when :math:`\rho` is. They agree to the
        aliasing error, not to machine precision.
        """
        rho = smooth_density(modes=1)
        computed = kinetic_potential(lambda r: von_weizsacker_tau(r, cell), rho, cell)
        expected = von_weizsacker_potential(rho, cell)
        assert (computed - expected).abs().max() < 1e-3 * expected.abs().max()

    def test_von_weizsacker_agreement_degrades_with_roughness(self, cell):
        """The discrepancy is aliasing: it grows as the density gains modes."""
        errors = []
        for modes in (1, 4):
            rho = smooth_density(modes=modes)
            computed = kinetic_potential(lambda r: von_weizsacker_tau(r, cell),
                                         rho, cell)
            expected = von_weizsacker_potential(rho, cell)
            errors.append(float((computed - expected).abs().max()
                                / expected.abs().max()))
        assert errors[1] > errors[0]

    def test_uniform_density_gives_a_constant_potential(self, cell):
        uniform = torch.full((1, 1, 12, 12, 12), 0.25, dtype=torch.float64)
        potential = kinetic_potential(thomas_fermi_tau, uniform, cell)
        assert float(potential.std()) < 1e-12
        assert float(potential.mean()) == pytest.approx(
            float(thomas_fermi_potential(uniform).mean()), rel=1e-10)


class TestDiscretisationFactor:
    def test_matches_finite_differences(self, density, cell):
        """The definitive check on the 1/dv factor."""
        dv = float(volume_element(density, cell).flatten()[0])
        potential = kinetic_potential(thomas_fermi_tau, density, cell)
        step = 1e-6

        for index in [(0, 0, 3, 7, 11), (0, 0, 9, 2, 14), (0, 0, 5, 5, 5)]:
            plus, minus = density.clone(), density.clone()
            plus[index] += step
            minus[index] -= step
            derivative = (float(integrate(thomas_fermi_tau(plus), cell))
                          - float(integrate(thomas_fermi_tau(minus), cell))) / (2 * step)
            assert derivative / dv == pytest.approx(float(potential[index]), rel=1e-5)

    def test_volume_element_is_omega_over_points(self, density, cell):
        dv = volume_element(density, cell)
        expected = 8.0 ** 3 / 16 ** 3
        assert float(dv.flatten()[0]) == pytest.approx(expected, rel=1e-10)

    def test_scaling_with_grid_size(self, cell):
        r"""A physically identical field on two grids gives the same potential.

        Without the 1/dv factor the two would differ by the ratio of grid-point
        counts -- a factor of eight here.
        """
        coarse = torch.full((1, 1, 8, 8, 8), 0.3, dtype=torch.float64)
        fine = torch.full((1, 1, 16, 16, 16), 0.3, dtype=torch.float64)
        a = float(kinetic_potential(thomas_fermi_tau, coarse, cell).mean())
        b = float(kinetic_potential(thomas_fermi_tau, fine, cell).mean())
        assert a == pytest.approx(b, rel=1e-10)


class TestGraphSemantics:
    def test_does_not_mutate_the_input(self, density, cell):
        before = density.clone()
        kinetic_potential(thomas_fermi_tau, density, cell)
        assert torch.equal(density, before)
        assert not density.requires_grad

    def test_create_graph_allows_second_order(self, density, cell):
        """Required when the derivative itself appears in a backpropagated loss."""
        potential = kinetic_potential(thomas_fermi_tau, density, cell,
                                      create_graph=True)
        assert potential.requires_grad
        assert potential.grad_fn is not None

    def test_without_create_graph_the_result_is_detached(self, density, cell):
        potential = kinetic_potential(thomas_fermi_tau, density, cell)
        assert not potential.requires_grad

    def test_generic_functional_derivative(self, density, cell):
        r"""For :math:`F=\int\rho^2`, :math:`\delta F/\delta\rho = 2\rho`."""
        computed = functional_derivative(
            lambda r: integrate(r ** 2, cell), density, cell)
        assert (computed - 2 * density).abs().max() < 1e-10

    def test_rejects_non_tensor(self, cell):
        with pytest.raises(TypeError):
            functional_derivative(lambda r: r.sum(), [1.0, 2.0], cell)


class TestEulerLagrangeIntegration:
    def test_accepts_a_callable_kinetic_functional(self, density, cell):
        potential = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64) * 5.0
        residual = euler_lagrange_residual(
            density, potential, cell, kinetic=thomas_fermi_tau)
        assert residual.shape == density.shape
        assert float(residual.mean().abs().detach()) < 1e-8      # mu removed

    def test_callable_matches_a_precomputed_tensor(self, density, cell):
        potential = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64) * 5.0
        precomputed = kinetic_potential(thomas_fermi_tau, density, cell)
        a = euler_lagrange_residual(density, potential, cell, kinetic=thomas_fermi_tau)
        b = euler_lagrange_residual(density, potential, cell, kinetic=precomputed)
        assert (a - b).abs().max() < 1e-8

    def test_analytic_surrogate_still_the_default(self, density, cell):
        potential = torch.randn(1, 1, 16, 16, 16, dtype=torch.float64) * 5.0
        residual = euler_lagrange_residual(density, potential, cell)
        assert residual.shape == density.shape
        assert float(residual.mean().abs()) < 1e-8

    def test_learned_kinetic_term_is_differentiable(self, cell):
        """The residual must backpropagate into the density that produced it."""
        rho = (torch.rand(1, 1, 12, 12, 12, dtype=torch.float64) * 0.3 + 0.1)
        rho.requires_grad_(True)
        potential = torch.randn(1, 1, 12, 12, 12, dtype=torch.float64) * 5.0
        residual = euler_lagrange_residual(rho, potential, cell,
                                           kinetic=thomas_fermi_tau)
        residual.pow(2).mean().backward()
        assert rho.grad is not None and torch.isfinite(rho.grad).all()


class TestOperatorInterface:
    def test_requires_a_chg2tau_operator(self):
        from poraque.ml import FieldOperator

        operator = FieldOperator("ext2chg", width=8, modes=4, n_layers=1,
                                 projection_channels=16, device="cpu")
        with pytest.raises(ValueError, match="chg2tau"):
            operator_kinetic_potential(operator, torch.rand(1, 1, 8, 8, 8),
                                       torch.eye(3).unsqueeze(0) * 6.0)

    def test_returns_a_field_shaped_potential(self):
        from poraque.ml import FieldOperator

        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16, device="cpu")
        rho = torch.rand(1, 1, 12, 12, 12) * 0.3 + 0.1
        cell = torch.eye(3).unsqueeze(0) * 6.0
        potential = operator_kinetic_potential(operator, rho, cell)
        assert potential.shape == rho.shape
        assert torch.isfinite(potential).all()


class TestHomogeneitySumRule:
    r"""
    Euler's theorem, as a check that needs no reference potential.

    A functional homogeneous of degree :math:`n` in :math:`\rho` satisfies

    .. math::

        \int \frac{\delta F}{\delta\rho}\,\rho\,d^3r = n\,F ,

    exactly. Thomas-Fermi has :math:`n = 5/3` and von Weizsacker
    :math:`n = 1`. The ratio is computable from a *learned* functional too,
    without knowing its derivative — which makes it the one diagnostic
    available when there is no closed form to compare against.

    It is also the sharpest one, because the ratio is precisely the response
    of :math:`T_s` to a **uniform** rescaling of the density. A model trained
    only on densities of one mean never sees that direction; measured, such a
    model fit :math:`\tau` to 5.6e-03 and returned a sum rule of **0.057**.
    Trained on densities spanning a range of means, the same architecture and
    the same :math:`\tau` error gave **0.973**.
    """

    @staticmethod
    def _ratio(potential, rho, cell, energy, degree):
        return (integrate(potential * rho, cell)
                / (degree * integrate(energy, cell))).mean().item()

    def test_thomas_fermi_autograd_is_exact(self, density, cell):
        potential = kinetic_potential(thomas_fermi_tau, density, cell)
        ratio = self._ratio(potential, density, cell,
                            thomas_fermi_tau(density), 5.0 / 3.0)
        assert abs(ratio - 1.0) < 1e-10, ratio

    def test_thomas_fermi_analytic_is_exact(self, density, cell):
        ratio = self._ratio(thomas_fermi_potential(density), density, cell,
                            thomas_fermi_tau(density), 5.0 / 3.0)
        assert abs(ratio - 1.0) < 1e-10, ratio

    @pytest.mark.parametrize("modes", [1, 3, 6])
    def test_von_weizsacker_autograd_is_exact_at_every_roughness(
            self, modes, cell):
        """
        The autograd derivative obeys the sum rule whatever the density does.
        """
        rho = smooth_density(n=24, modes=modes) * 0.3
        ratio = self._ratio(
            kinetic_potential(lambda r: von_weizsacker_tau(r, cell), rho, cell),
            rho, cell, von_weizsacker_tau(rho, cell), 1.0)
        # ~1e-7 rather than machine epsilon: the density floor in
        # `von_weizsacker_tau` does not scale with rho, so it breaks exact
        # homogeneity by a hair. Measured 1.1e-07 at one mode, falling to
        # 1.0e-08 on white noise.
        assert abs(ratio - 1.0) < 1e-6, (modes, ratio)

    def test_autograd_beats_the_analytic_potential_on_a_rough_density(
            self, density, cell):
        """
        On white noise the *analytic* von Weizsacker potential drifts from the
        sum rule by several percent while autograd stays exact. Differentiating
        the energy is more reliable than a hand-derived potential where the
        derivation assumed smoothness.
        """
        energy = von_weizsacker_tau(density, cell)
        automatic = self._ratio(
            kinetic_potential(lambda r: von_weizsacker_tau(r, cell),
                              density, cell), density, cell, energy, 1.0)
        analytic = self._ratio(von_weizsacker_potential(density, cell),
                               density, cell, energy, 1.0)
        # Measured on this fixture: autograd 1.0e-08, analytic 6.8e-02 --
        # six orders of magnitude, so the margin demanded here is modest.
        assert abs(automatic - 1.0) < 1e-6
        assert abs(analytic - 1.0) > 100 * abs(automatic - 1.0)

    def test_a_learned_functional_can_be_checked_without_a_reference(self):
        """
        The point of the rule: it applies to an operator whose derivative has
        no closed form. An untrained model has no reason to satisfy it, which
        is what makes a *trained* model's ratio informative.
        """
        from poraque.ml import FieldOperator

        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16, device="cpu")
        rho = torch.rand(1, 1, 12, 12, 12) * 0.3 + 0.1
        cell = torch.eye(3).unsqueeze(0) * 6.0

        potential = operator_kinetic_potential(operator, rho, cell)
        ratio = (integrate(potential * rho, cell)
                 / integrate(operator.target_transform.inverse(
                     operator.model(operator.input_transform(rho), cell)),
                     cell)).mean()
        assert torch.isfinite(ratio), "the diagnostic must at least be computable"
