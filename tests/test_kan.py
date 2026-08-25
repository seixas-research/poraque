# -*- coding: utf-8 -*-
# file: test_kan.py
"""Tests for :mod:`poraque.ml.kan` -- the pluggable Fourier-block activations,
including the two Kolmogorov-Arnold Network (KAN) style learnable ones."""

import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")
import torch.nn.functional as F  # noqa: E402

import numpy as np  # noqa: E402
import sympy  # noqa: E402 -- same optional dep poraque.ml.symbolic already uses

from poraque.ml import FNO3d, FieldOperator, load_bundle, save_bundle  # noqa: E402
from poraque.ml.kan import (  # noqa: E402
    ACTIVATIONS,
    KAN_ACTIVATIONS,
    BSplineKANActivation,
    ChebyKANActivation,
    RationalKANActivation,
    RBFKANActivation,
    build_activation,
    symbolic_expression,
)


def _grad_map(module):
    """``{name: grad}`` for every parameter of ``module``, after a backward pass."""
    return {name: parameter.grad for name, parameter in module.named_parameters()}


def _forward_at(activation, channel, points):
    """``activation``'s actual output at ``channel``, for the scalar values
    in ``points`` -- every channel fed the same points, so channel 0's row
    of the output is directly comparable to a SymPy expression lambdified
    over those same points."""
    x = torch.tensor(points, dtype=torch.float32).view(1, 1, -1) \
            .expand(1, activation.channels, -1).contiguous()
    with torch.no_grad():
        return activation(x)[0, channel].numpy()


def _symbolic_at(activation, channel, points, decimals=None):
    """``symbolic_expression(activation, channel)``, lambdified and
    evaluated at the same ``points`` -- ``modules=["scipy", "numpy"]`` for
    parity with plain ``numpy`` lambdification of any special function a
    future variant's base or residual might introduce (today's four need
    only ``exp``/``tanh``, which plain ``numpy`` already covers)."""
    x = sympy.Symbol("x")
    expr = symbolic_expression(activation, channel, decimals=decimals)
    fn = sympy.lambdify(x, expr, modules=["scipy", "numpy"])

    return np.asarray(fn(np.asarray(points, dtype=float)), dtype=float)


# --------------------------------------------------------------------- #
# ChebyKANActivation
# --------------------------------------------------------------------- #
class TestChebyKANActivation:
    def test_shape_and_dtype_are_preserved(self):
        activation = ChebyKANActivation(channels=5, degree=4)
        x = torch.randn(3, 5, 6, 7, 8)
        out = activation(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_accepts_tensors_with_any_number_of_spatial_axes(self):
        """Only axis 0 (batch) and axis 1 (channel) are assumed -- the unit
        tests exercise small 2D/3D tensors, not only the 5D (B,C,Nx,Ny,Nz)
        shape the Fourier block actually calls this with."""
        activation = ChebyKANActivation(channels=4, degree=3)
        for shape in [(2, 4, 9), (2, 4, 3, 3)]:
            out = activation(torch.randn(*shape))
            assert out.shape == shape

    def test_gradients_flow_to_every_parameter(self):
        activation = ChebyKANActivation(channels=6, degree=5)
        x = torch.randn(2, 6, 4, 4, 4)
        activation(x).pow(2).mean().backward()
        grads = _grad_map(activation)
        assert set(grads) == {"base_weight", "cheby_coeff"}
        assert all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_close_to_silu_at_init(self):
        """At init the residual is small, so the KAN activation is close to
        the base nonlinearity it can learn to deviate from -- see the class
        docstring. Not exact (the residual is a small random draw, not
        zero), so the tolerance is generous, not tight."""
        torch.manual_seed(0)
        activation = ChebyKANActivation(channels=8, degree=6)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.silu(x)).abs()
        assert deviation.max().item() < 0.5
        assert deviation.mean().item() < 0.3

    def test_rejects_a_channel_axis_mismatch(self):
        activation = ChebyKANActivation(channels=4, degree=2)
        with pytest.raises(ValueError, match="4 channels"):
            activation(torch.randn(2, 3, 4, 4, 4))

    def test_rejects_a_negative_degree(self):
        with pytest.raises(ValueError, match="degree"):
            ChebyKANActivation(channels=4, degree=-1)

    def test_parameter_count_matches_the_documented_formula(self):
        """channels * (degree + 2): one base weight and degree+1 coefficients."""
        channels, degree = 10, 7
        activation = ChebyKANActivation(channels=channels, degree=degree)
        n_params = sum(p.numel() for p in activation.parameters())
        assert n_params == channels * (degree + 2)


# --------------------------------------------------------------------- #
# BSplineKANActivation
# --------------------------------------------------------------------- #
class TestBSplineKANActivation:
    def test_shape_and_dtype_are_preserved(self):
        activation = BSplineKANActivation(channels=5, grid_size=6, spline_order=2)
        x = torch.randn(3, 5, 6, 7, 8)
        out = activation(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_gradients_flow_to_every_parameter(self):
        activation = BSplineKANActivation(channels=6, grid_size=8, spline_order=3)
        x = torch.randn(2, 6, 4, 4, 4)
        activation(x).pow(2).mean().backward()
        grads = _grad_map(activation)
        assert set(grads) == {"base_weight", "spline_coeff"}
        assert all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_close_to_silu_at_init(self):
        torch.manual_seed(0)
        activation = BSplineKANActivation(channels=8, grid_size=8, spline_order=3)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.silu(x)).abs()
        assert deviation.max().item() < 0.5
        assert deviation.mean().item() < 0.3

    def test_handles_inputs_far_outside_grid_range_without_exploding(self):
        """The whole point of clamping: a wide pre-activation tail (routine
        for an untrained network) must not blow the *residual* up.

        The overall output does NOT saturate -- by design, the SiLU base
        term keeps tracking the true unclamped x (SiLU(x) ~ x for large
        positive x), which is exactly the "never actually saturates" property
        documented on the class. What must saturate is the *spline residual*
        alone: clamping means it is evaluated at a fixed boundary point no
        matter how far x is beyond grid_range.
        """
        activation = BSplineKANActivation(channels=4, grid_size=8, spline_order=3,
                                          grid_range=(-2.0, 2.0))
        x = torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0]).view(1, 1, 5) \
                .expand(1, 4, 5).contiguous()
        out = activation(x)
        assert torch.isfinite(out).all()

        residual = out - F.silu(x) * activation.base_weight.view(1, -1, 1)
        # The two positive-side tails (50 and 1000) both clamp to the same
        # boundary point, so their residuals must agree; same on the
        # negative side.
        assert torch.allclose(residual[:, :, 3], residual[:, :, 4], atol=1e-4)
        assert torch.allclose(residual[:, :, 0], residual[:, :, 1], atol=1e-4)

    def test_basis_is_a_partition_of_unity_inside_the_grid(self):
        """A defining property of a B-spline basis: the basis functions sum
        to 1 everywhere inside the knot support."""
        activation = BSplineKANActivation(channels=1, grid_size=8, spline_order=3,
                                          grid_range=(-2.0, 2.0))
        x = torch.linspace(-1.99, 1.99, 101)
        basis = activation._basis(x)
        assert torch.allclose(basis.sum(-1), torch.ones_like(x), atol=1e-4)

    def test_knot_buffer_moves_with_the_module(self):
        activation = BSplineKANActivation(channels=3, grid_size=4, spline_order=2)
        activation = activation.to(torch.float64)
        assert activation.knots.dtype == torch.float64
        out = activation(torch.randn(2, 3, 4, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_rejects_a_channel_axis_mismatch(self):
        activation = BSplineKANActivation(channels=4)
        with pytest.raises(ValueError, match="4 channels"):
            activation(torch.randn(2, 3, 4, 4, 4))

    def test_rejects_a_non_increasing_grid_range(self):
        with pytest.raises(ValueError, match="grid_range"):
            BSplineKANActivation(channels=4, grid_range=(2.0, -2.0))

    def test_parameter_count_matches_the_documented_formula(self):
        """channels * (grid_size + spline_order + 1)."""
        channels, grid_size, spline_order = 10, 6, 3
        activation = BSplineKANActivation(channels=channels, grid_size=grid_size,
                                          spline_order=spline_order)
        n_params = sum(p.numel() for p in activation.parameters())
        assert n_params == channels * (grid_size + spline_order + 1)


# --------------------------------------------------------------------- #
# RBFKANActivation
# --------------------------------------------------------------------- #
class TestRBFKANActivation:
    def test_shape_and_dtype_are_preserved(self):
        activation = RBFKANActivation(channels=5, grid_size=6)
        x = torch.randn(3, 5, 6, 7, 8)
        out = activation(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_gradients_flow_to_every_parameter(self):
        activation = RBFKANActivation(channels=6, grid_size=8)
        x = torch.randn(2, 6, 4, 4, 4)
        activation(x).pow(2).mean().backward()
        grads = _grad_map(activation)
        assert set(grads) == {"base_weight", "rbf_coeff"}
        assert all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_close_to_silu_at_init(self):
        torch.manual_seed(0)
        activation = RBFKANActivation(channels=8, grid_size=8)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.silu(x)).abs()
        assert deviation.max().item() < 0.5
        assert deviation.mean().item() < 0.3

    def test_residual_decays_far_outside_the_grid_range_without_clamping(self):
        """Unlike the B-spline basis, nothing here needs clamping: a Gaussian
        decays to zero on its own, so the residual should vanish (not
        saturate at a boundary value) for a pre-activation far outside
        grid_range, leaving only the SiLU base term."""
        activation = RBFKANActivation(channels=4, grid_size=8, grid_range=(-2.0, 2.0))
        x = torch.tensor([1000.0, -1000.0]).view(1, 1, 2).expand(1, 4, 2).contiguous()
        out = activation(x)
        assert torch.isfinite(out).all()
        residual = out - F.silu(x) * activation.base_weight.view(1, -1, 1)
        assert residual.abs().max().item() < 1e-6

    def test_centers_buffer_moves_with_the_module(self):
        activation = RBFKANActivation(channels=3, grid_size=4)
        activation = activation.to(torch.float64)
        assert activation.centers.dtype == torch.float64
        out = activation(torch.randn(2, 3, 4, dtype=torch.float64))
        assert out.dtype == torch.float64

    def test_rejects_a_channel_axis_mismatch(self):
        activation = RBFKANActivation(channels=4)
        with pytest.raises(ValueError, match="4 channels"):
            activation(torch.randn(2, 3, 4, 4, 4))

    def test_rejects_a_non_increasing_grid_range(self):
        with pytest.raises(ValueError, match="grid_range"):
            RBFKANActivation(channels=4, grid_range=(2.0, -2.0))

    def test_parameter_count_matches_the_documented_formula(self):
        """channels * (grid_size + 2): one base weight and grid_size+1 RBF
        coefficients."""
        channels, grid_size = 10, 6
        activation = RBFKANActivation(channels=channels, grid_size=grid_size)
        n_params = sum(p.numel() for p in activation.parameters())
        assert n_params == channels * (grid_size + 2)


# --------------------------------------------------------------------- #
# RationalKANActivation
# --------------------------------------------------------------------- #
class TestRationalKANActivation:
    def test_shape_and_dtype_are_preserved(self):
        activation = RationalKANActivation(channels=5, num_degree=3, den_degree=3)
        x = torch.randn(3, 5, 6, 7, 8)
        out = activation(x)
        assert out.shape == x.shape
        assert out.dtype == x.dtype

    def test_gradients_flow_to_every_parameter(self):
        activation = RationalKANActivation(channels=6, num_degree=4, den_degree=4)
        x = torch.randn(2, 6, 4, 4, 4)
        activation(x).pow(2).mean().backward()
        grads = _grad_map(activation)
        assert set(grads) == {"base_weight", "num_coeff", "den_coeff"}
        assert all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_close_to_silu_at_init(self):
        torch.manual_seed(0)
        activation = RationalKANActivation(channels=8, num_degree=4, den_degree=4)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.silu(x)).abs()
        assert deviation.max().item() < 0.5
        assert deviation.mean().item() < 0.3

    def test_denominator_never_drops_below_one_so_never_divides_by_zero(self):
        """The whole point of the |b| guard: Q_c(x) = 1 + sum |b_k| x^(2k)
        sums 1 with non-negative terms, so it can never approach zero -- even
        after an adversarial weight update pushes a coefficient to a large
        magnitude of either sign."""
        activation = RationalKANActivation(channels=3, num_degree=2, den_degree=2)
        with torch.no_grad():
            activation.den_coeff.copy_(torch.tensor([[-50.0, 30.0]] * 3))
        x = torch.linspace(-10, 10, 21).view(1, 1, 21).expand(1, 3, 21).contiguous()
        out = activation(x)
        assert torch.isfinite(out).all()

    def test_residual_decays_for_a_wide_pre_activation_tail(self):
        """With the default degrees the denominator's highest power (x^8)
        outgrows the numerator's (x^4), so the residual -> 0 as |x| -> inf --
        no clamping needed, unlike BSplineKANActivation."""
        activation = RationalKANActivation(channels=4, num_degree=4, den_degree=4)
        x = torch.tensor([1000.0, -1000.0]).view(1, 1, 2).expand(1, 4, 2).contiguous()
        out = activation(x)
        assert torch.isfinite(out).all()
        residual = out - F.silu(x) * activation.base_weight.view(1, -1, 1)
        assert residual.abs().max().item() < 1e-3

    def test_rejects_a_channel_axis_mismatch(self):
        activation = RationalKANActivation(channels=4)
        with pytest.raises(ValueError, match="4 channels"):
            activation(torch.randn(2, 3, 4, 4, 4))

    def test_rejects_a_negative_degree(self):
        with pytest.raises(ValueError, match="num_degree"):
            RationalKANActivation(channels=4, num_degree=-1)
        with pytest.raises(ValueError, match="den_degree"):
            RationalKANActivation(channels=4, den_degree=-1)

    def test_zero_den_degree_is_a_pure_polynomial_numerator(self):
        """den_degree=0 drops den_coeff entirely (Q_c = 1 identically), which
        must not raise or silently divide by a stray tensor."""
        activation = RationalKANActivation(channels=3, num_degree=2, den_degree=0)
        assert activation.den_coeff is None
        out = activation(torch.randn(2, 3, 4, 4, 4))
        assert torch.isfinite(out).all()

    def test_parameter_count_matches_the_documented_formula(self):
        """channels * (num_degree + den_degree + 2): one base weight,
        num_degree+1 numerator coefficients, den_degree denominator
        coefficients."""
        channels, num_degree, den_degree = 10, 5, 3
        activation = RationalKANActivation(channels=channels, num_degree=num_degree,
                                           den_degree=den_degree)
        n_params = sum(p.numel() for p in activation.parameters())
        assert n_params == channels * (num_degree + den_degree + 2)


# --------------------------------------------------------------------- #
# The factory and the registry
# --------------------------------------------------------------------- #
class TestBuildActivation:
    @pytest.mark.parametrize("name, reference", [
        ("gelu", F.gelu), ("relu", F.relu), ("silu", F.silu),
        ("tanh", torch.tanh),
    ])
    def test_stateless_variants_match_the_functional_form_exactly(self, name, reference):
        activation = build_activation(name, channels=5)
        x = torch.randn(2, 5, 4, 4, 4)
        assert torch.equal(activation(x), reference(x))
        assert list(activation.parameters()) == []

    def test_kan_variants_are_channel_wise_and_learnable(self):
        for name in KAN_ACTIVATIONS:
            activation = build_activation(name, channels=6)
            assert any(p.requires_grad for p in activation.parameters())

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            build_activation("swish_glu_v3", channels=4)

    def test_every_declared_activation_actually_builds(self):
        for name in ACTIVATIONS:
            build_activation(name, channels=4)

    def test_kan_kwargs_reach_the_right_variant(self):
        bspline = build_activation("kan_bspline", channels=4, kan_grid_size=5,
                                   kan_spline_order=2, kan_grid_range=(-1.0, 1.0))
        assert bspline.grid_size == 5 and bspline.spline_order == 2
        assert bspline.grid_range == (-1.0, 1.0)

        cheby = build_activation("kan_cheby", channels=4, kan_degree=9)
        assert cheby.degree == 9

        rbf = build_activation("kan_rbf", channels=4, kan_grid_size=5,
                               kan_grid_range=(-1.0, 1.0))
        assert rbf.grid_size == 5 and rbf.grid_range == (-1.0, 1.0)

        rational = build_activation("kan_rational", channels=4,
                                    kan_rational_num_degree=2,
                                    kan_rational_den_degree=3)
        assert rational.num_degree == 2 and rational.den_degree == 3


# --------------------------------------------------------------------- #
# Backward compatibility: every activation, default or explicit, is
# self-consistent and round-trips through a checkpoint unchanged
#
# `silu` became the default 2026-08-17 (was `gelu` before that -- see
# FUTURE.md for the measurements that motivated the switch), so "backward
# compatible" here means: (a) a checkpoint always records its own literal
# `activation`, read back on load rather than re-inferred from whatever the
# *current* default happens to be, so old `gelu` checkpoints are unaffected
# by the default changing under them, and (b) `gelu` itself still works
# exactly as before when asked for explicitly.
# --------------------------------------------------------------------- #
class TestBackwardCompatibility:
    def test_default_activation_is_now_silu(self):
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8)
        assert model.activation == "silu"
        assert isinstance(model.blocks[0].activation, torch.nn.SiLU)

    def test_gelu_is_still_available_and_unchanged_when_asked_for_explicitly(self):
        """The refactor from a bare function to an nn.Module wrapper must not
        change what a `gelu` model contains: torch.nn.GELU is stateless.
        `gelu` is no longer the default, but remains a fully supported,
        explicit choice."""
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8,
                      activation="gelu")
        assert model.blocks[0].activation.state_dict() == {}
        assert model.n_parameters() == 17859     # pinned: see test_ml.py's
        # equivalents for width=6/modes=3/n_layers=2/projection_channels=8;
        # a change here means the gelu model's parameter count moved. SiLU
        # is also stateless, so an equivalent silu model has the same count.

    def test_a_pre_kan_checkpoint_still_loads(self, tmp_path):
        """Simulates a bundle saved before the `architecture` dict carried
        the kan_* keys: strip them out and confirm the model still rebuilds,
        falling back to the (harmless, since neither silu nor gelu consumes
        them) constructor defaults for the missing keys."""
        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        state = operator.state()
        for key in ("kan_grid_size", "kan_spline_order", "kan_grid_range",
                    "kan_degree", "kan_rational_num_degree",
                    "kan_rational_den_degree", "kan_use_base"):
            state["architecture"].pop(key, None)

        restored = FieldOperator.from_state(state, device="cpu")
        assert restored.model.activation == "silu"     # the default this
        # operator was actually built with, since `activation=` was not
        # passed to FieldOperator(...) above.
        assert restored.model.kan_grid_size == 8      # constructor default
        assert restored.model.kan_use_base is True     # constructor default
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_default_bundle_round_trip_is_unaffected(self, tmp_path):
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        path = save_bundle(str(tmp_path / "m.pfno"), {"chg2tau": operator})
        restored = load_bundle(path, "chg2tau", device="cpu")
        assert restored.model.activation == "silu"
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_an_explicit_gelu_bundle_round_trip_is_unaffected(self, tmp_path):
        """The same round trip, but for a checkpoint that explicitly asked
        for the no-longer-default `gelu` -- its own recorded architecture
        must win over the current default on load."""
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, activation="gelu",
                                 device="cpu")
        path = save_bundle(str(tmp_path / "m.pfno"), {"chg2tau": operator})
        restored = load_bundle(path, "chg2tau", device="cpu")
        assert restored.model.activation == "gelu"
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)


# --------------------------------------------------------------------- #
# The FNO with a KAN activation: shapes, gradients, checkpoints
# --------------------------------------------------------------------- #
class TestFNOWithKANActivations:
    @pytest.mark.parametrize("activation", sorted(KAN_ACTIVATIONS))
    def test_forward_and_backward_on_a_small_grid(self, activation):
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8,
                      activation=activation)
        cell = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 5.0
        x = torch.randn(2, 1, 8, 8, 8, requires_grad=True)

        out = model(x, cell)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

        out.pow(2).mean().backward()
        activation_params = [p for block in model.blocks
                             for p in block.activation.parameters()]
        assert activation_params, "the KAN activation should own parameters"
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in activation_params)

    def test_kan_bspline_checkpoint_round_trip_preserves_hyperparameters(self, tmp_path):
        operator = FieldOperator(
            "ext2chg", width=4, modes=2, n_layers=1, projection_channels=8,
            activation="kan_bspline", kan_grid_size=5, kan_spline_order=2,
            kan_grid_range=(-3.0, 3.0), device="cpu",
        )
        path = save_bundle(str(tmp_path / "m.pfno"), {"ext2chg": operator})
        restored = load_bundle(path, "ext2chg", device="cpu")

        assert restored.model.activation == "kan_bspline"
        assert restored.model.kan_grid_size == 5
        assert restored.model.kan_spline_order == 2
        assert restored.model.kan_grid_range == (-3.0, 3.0)
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_kan_cheby_checkpoint_round_trip_preserves_hyperparameters(self, tmp_path):
        operator = FieldOperator(
            "chg2tau", width=4, modes=2, n_layers=1, projection_channels=8,
            activation="kan_cheby", kan_degree=9, device="cpu",
        )
        path = save_bundle(str(tmp_path / "m.pfno"), {"chg2tau": operator})
        restored = load_bundle(path, "chg2tau", device="cpu")

        assert restored.model.activation == "kan_cheby"
        assert restored.model.kan_degree == 9
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_rbf_checkpoint_round_trip_preserves_hyperparameters(self, tmp_path):
        operator = FieldOperator(
            "ext2chg", width=4, modes=2, n_layers=1, projection_channels=8,
            activation="kan_rbf", kan_grid_size=5, kan_grid_range=(-3.0, 3.0),
            device="cpu",
        )
        path = save_bundle(str(tmp_path / "m.pfno"), {"ext2chg": operator})
        restored = load_bundle(path, "ext2chg", device="cpu")

        assert restored.model.activation == "kan_rbf"
        assert restored.model.kan_grid_size == 5
        assert restored.model.kan_grid_range == (-3.0, 3.0)
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_rational_checkpoint_round_trip_preserves_hyperparameters(self, tmp_path):
        operator = FieldOperator(
            "chg2tau", width=4, modes=2, n_layers=1, projection_channels=8,
            activation="kan_rational", kan_rational_num_degree=3,
            kan_rational_den_degree=2, device="cpu",
        )
        path = save_bundle(str(tmp_path / "m.pfno"), {"chg2tau": operator})
        restored = load_bundle(path, "chg2tau", device="cpu")

        assert restored.model.activation == "kan_rational"
        assert restored.model.kan_rational_num_degree == 3
        assert restored.model.kan_rational_den_degree == 2
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_mismatched_kan_hyperparameters_fail_loudly_not_silently(self, tmp_path):
        """Without the architecture record this would load a kan_cheby model
        with degree=6 (the constructor default) instead of the 9 it was
        trained with -- a shape-mismatched state_dict, caught at load time,
        rather than a silently wrong model."""
        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=8, activation="kan_cheby",
                                 kan_degree=9, device="cpu")
        state = operator.state()
        state["architecture"].pop("kan_degree", None)
        with pytest.raises(RuntimeError, match="size mismatch|Missing key"):
            FieldOperator.from_state(state)


# --------------------------------------------------------------------- #
# Integration smoke test: both tasks, every activation
# --------------------------------------------------------------------- #
class TestIntegrationBothTasks:
    @pytest.mark.parametrize("task", ["ext2chg", "chg2tau"])
    @pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
    def test_forward_backward_on_tiny_dummy_data(self, task, activation):
        operator = FieldOperator(task, width=4, modes=2, n_layers=1,
                                 projection_channels=8, activation=activation,
                                 device="cpu")
        cell = torch.eye(3).unsqueeze(0) * 6.0
        x = torch.randn(1, operator.in_channels, 6, 6, 6, requires_grad=True)

        out = operator.model(x, cell)
        assert out.shape == (1, operator.out_channels, 6, 6, 6)
        assert torch.isfinite(out).all()

        out.pow(2).mean().backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                  for p in operator.model.parameters())


# --------------------------------------------------------------------- #
# symbolic_expression: reading a channel's learned function straight out
# of its stored coefficients, no fitting
# --------------------------------------------------------------------- #
class TestSymbolicExpression:
    POINTS = [-6.0, -3.0, -2.5, -1.0, -0.3, 0.0, 0.3, 1.0, 2.5, 3.0, 6.0]

    @pytest.mark.parametrize("build", [
        lambda: ChebyKANActivation(channels=3, degree=4),
        lambda: RBFKANActivation(channels=3, grid_size=6, grid_range=(-2.0, 2.0)),
        lambda: RationalKANActivation(channels=3, num_degree=3, den_degree=3),
        lambda: BSplineKANActivation(channels=3, grid_size=6, spline_order=3,
                                     grid_range=(-2.0, 2.0)),
    ], ids=["kan_cheby", "kan_rbf", "kan_rational", "kan_bspline"])
    def test_matches_the_forward_pass_numerically(self, build):
        """The whole point of a *readout* rather than a fit: the symbolic
        expression must reproduce forward() exactly (up to float rounding),
        including points chosen well outside every variant's clamp/decay
        range (-6, 6), not just the well-behaved interior."""
        torch.manual_seed(0)
        activation = build()
        for channel in range(activation.channels):
            forward_out = _forward_at(activation, channel, self.POINTS)
            symbolic_out = _symbolic_at(activation, channel, self.POINTS,
                                        decimals=None)
            assert np.allclose(forward_out, symbolic_out, atol=1e-4), (
                f"channel {channel}: forward={forward_out} vs. "
                f"symbolic={symbolic_out}")

    def test_matches_at_random_points_too(self):
        """Not just the hand-picked grid above -- a Chebyshev/RBF/rational
        residual should agree with its symbolic readout everywhere, not
        only at points that happen to have been checked by hand."""
        torch.manual_seed(1)
        activation = ChebyKANActivation(channels=2, degree=5)
        points = np.random.default_rng(0).uniform(-4, 4, size=25).tolist()
        forward_out = _forward_at(activation, 1, points)
        symbolic_out = _symbolic_at(activation, 1, points, decimals=None)
        assert np.allclose(forward_out, symbolic_out, atol=1e-4)

    def test_expression_is_a_function_of_x_alone(self):
        activation = RationalKANActivation(channels=2, num_degree=2, den_degree=2)
        expr = symbolic_expression(activation, channel=0)
        assert expr.free_symbols == {sympy.Symbol("x")}

    def test_decimals_rounds_every_coefficient(self):
        activation = ChebyKANActivation(channels=2, degree=3)
        expr = symbolic_expression(activation, channel=0, decimals=2)
        for atom in expr.atoms(sympy.Float):
            rounded = round(float(atom), 2)
            assert float(atom) == pytest.approx(rounded, abs=1e-9)

    def test_rejects_an_out_of_range_channel(self):
        activation = ChebyKANActivation(channels=3, degree=2)
        with pytest.raises(ValueError, match="channel"):
            symbolic_expression(activation, channel=3)

    def test_rejects_a_stateless_activation(self):
        """gelu/relu/silu/tanh have nothing learned to read out -- they
        already are their own closed form."""
        gelu = build_activation("gelu", channels=4)
        with pytest.raises(TypeError, match="stateless activation"):
            symbolic_expression(gelu, channel=0)

    def test_reads_out_a_channel_from_a_trained_operator(self, tmp_path):
        """The realistic path: pull the activation module straight off a
        loaded FieldOperator's Fourier blocks, exactly as
        operator.model.blocks[layer].activation would be used in practice."""
        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=2,
                                 projection_channels=8, activation="kan_cheby",
                                 kan_degree=3, device="cpu")
        activation = operator.model.blocks[0].activation
        expr = symbolic_expression(activation, channel=0)
        assert isinstance(expr, sympy.Expr)
        # round-trips through a checkpoint just like every other tensor
        path = save_bundle(str(tmp_path / "m.pfno"), {"ext2chg": operator})
        restored = load_bundle(path, "ext2chg", device="cpu")
        restored_expr = symbolic_expression(
            restored.model.blocks[0].activation, channel=0)
        forward_out = _forward_at(activation, 0, self.POINTS)
        restored_out = _symbolic_at(restored.model.blocks[0].activation, 0,
                                    self.POINTS, decimals=None)
        assert np.allclose(forward_out, restored_out, atol=1e-4)
        assert restored_expr is not None


# --------------------------------------------------------------------- #
# use_base=False: "pure" KAN -- no fixed nonlinearity mixed in at all
# --------------------------------------------------------------------- #
class TestPureKANMode:
    """``use_base=False`` drops the ``w_c * silu(x)`` term entirely, so a
    channel is nothing but its learned residual. Every check here compares
    a ``use_base=True`` module against a ``use_base=False`` one built from
    an *identical* random draw (re-seeding right before each construction):
    ``torch.ones(...)`` (the base weight init) and ``torch.arange``/
    ``torch.linspace`` (the fixed knots/centers) never touch the RNG stream,
    so the residual coefficients -- drawn by ``torch.randn`` immediately
    after -- are bit-identical between the two regardless of ``use_base``.
    That makes "with minus without" an exact, direct measurement of the base
    term alone, not an approximation."""

    CASES = [
        (ChebyKANActivation, {"channels": 4, "degree": 3}, "kan_cheby"),
        (BSplineKANActivation,
         {"channels": 4, "grid_size": 6, "spline_order": 3}, "kan_bspline"),
        (RBFKANActivation, {"channels": 4, "grid_size": 6}, "kan_rbf"),
        (RationalKANActivation,
         {"channels": 4, "num_degree": 3, "den_degree": 3}, "kan_rational"),
    ]

    @pytest.mark.parametrize("cls, kwargs, name", CASES,
                             ids=[c[2] for c in CASES])
    def test_output_differs_from_the_base_variant_by_exactly_the_base_term(
            self, cls, kwargs, name):
        torch.manual_seed(0)
        with_base = cls(**kwargs, use_base=True)
        torch.manual_seed(0)
        without_base = cls(**kwargs, use_base=False)

        x = torch.randn(2, 4, 5, 5, 5) * 2.0
        with torch.no_grad():
            out_with = with_base(x)
            out_without = without_base(x)
        expected_base = F.silu(x) * with_base.base_weight.view(1, -1, 1, 1, 1)
        assert torch.allclose(out_with - out_without, expected_base, atol=1e-5)

    @pytest.mark.parametrize("cls, kwargs, name", CASES,
                             ids=[c[2] for c in CASES])
    def test_no_base_weight_parameter_exists(self, cls, kwargs, name):
        activation = cls(**kwargs, use_base=False)
        assert activation.base_weight is None
        assert activation.use_base is False
        assert "base_weight" not in dict(activation.named_parameters())

    @pytest.mark.parametrize("cls, kwargs, name", CASES,
                             ids=[c[2] for c in CASES])
    def test_gradients_still_flow_to_every_remaining_parameter(self, cls, kwargs, name):
        activation = cls(**kwargs, use_base=False)
        x = torch.randn(2, 4, 4, 4, 4)
        activation(x).pow(2).mean().backward()
        grads = _grad_map(activation)
        assert "base_weight" not in grads
        assert grads   # at least one learned coefficient tensor remains
        assert all(g is not None and g.abs().sum() > 0 for g in grads.values())

    def test_build_activation_forwards_kan_use_base(self):
        for name in ["kan_cheby", "kan_bspline", "kan_rbf", "kan_rational"]:
            pure = build_activation(name, channels=4, kan_use_base=False)
            assert pure.use_base is False and pure.base_weight is None
            default = build_activation(name, channels=4)
            assert default.use_base is True

    def test_symbolic_expression_omits_the_base_term_entirely(self):
        """A pure channel's expression must differ from the base-carrying
        one by exactly the SiLU base term, symbolically -- not merely
        "look shorter"."""
        torch.manual_seed(0)
        with_base = ChebyKANActivation(channels=2, degree=3, use_base=True)
        torch.manual_seed(0)
        without_base = ChebyKANActivation(channels=2, degree=3, use_base=False)

        expr_with = symbolic_expression(with_base, channel=0, decimals=None)
        expr_without = symbolic_expression(without_base, channel=0, decimals=None)

        points = [-4.0, -1.0, 0.0, 1.0, 4.0]
        out_with = _symbolic_at(with_base, 0, points, decimals=None)
        out_without = _symbolic_at(without_base, 0, points, decimals=None)
        expected_base = F.silu(torch.tensor(points)).numpy() \
            * with_base.base_weight[0].item()
        assert np.allclose(out_with - out_without, expected_base, atol=1e-4)
        # Direct structural check, not only numeric: no exp() term from a
        # base function survives -- ChebyKANActivation's own residual has
        # no exp of any kind, so this is unambiguous for this variant.
        assert expr_without.has(sympy.exp) is False
        assert expr_with.has(sympy.exp) is True   # silu = x/(1+exp(-x))

    def test_pure_kan_checkpoint_round_trip_preserves_use_base(self, tmp_path):
        operator = FieldOperator(
            "ext2chg", width=4, modes=2, n_layers=1, projection_channels=8,
            activation="kan_cheby", kan_degree=3, kan_use_base=False,
            device="cpu",
        )
        assert operator.model.blocks[0].activation.base_weight is None

        path = save_bundle(str(tmp_path / "m.pfno"), {"ext2chg": operator})
        restored = load_bundle(path, "ext2chg", device="cpu")

        assert restored.model.kan_use_base is False
        assert restored.model.blocks[0].activation.base_weight is None
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    @pytest.mark.parametrize("activation", sorted(KAN_ACTIVATIONS))
    def test_fno_forward_and_backward_with_pure_kan_activations(self, activation):
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8,
                      activation=activation, kan_use_base=False)
        for block in model.blocks:
            assert block.activation.base_weight is None
        cell = torch.eye(3).unsqueeze(0).repeat(2, 1, 1) * 5.0
        x = torch.randn(2, 1, 8, 8, 8, requires_grad=True)

        out = model(x, cell)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

        out.pow(2).mean().backward()
        activation_params = [p for block in model.blocks
                             for p in block.activation.parameters()]
        assert activation_params
        assert all(p.grad is not None and p.grad.abs().sum() > 0
                  for p in activation_params)
