# -*- coding: utf-8 -*-
# file: test_kan.py
"""Tests for :mod:`poraque.ml.kan` -- the pluggable Fourier-block activations,
including the two Kolmogorov-Arnold Network (KAN) style learnable ones."""

import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")
import torch.nn.functional as F  # noqa: E402

from poraque.ml import FNO3d, FieldOperator, load_bundle, save_bundle  # noqa: E402
from poraque.ml.kan import (  # noqa: E402
    ACTIVATIONS,
    KAN_ACTIVATIONS,
    BSplineKANActivation,
    ChebyKANActivation,
    build_activation,
)


def _grad_map(module):
    """``{name: grad}`` for every parameter of ``module``, after a backward pass."""
    return {name: parameter.grad for name, parameter in module.named_parameters()}


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

    def test_close_to_gelu_at_init(self):
        """At init the residual is small, so the KAN activation is close to
        the base nonlinearity it can learn to deviate from -- see the class
        docstring. Not exact (the residual is a small random draw, not
        zero), so the tolerance is generous, not tight."""
        torch.manual_seed(0)
        activation = ChebyKANActivation(channels=8, degree=6)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.gelu(x)).abs()
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

    def test_close_to_gelu_at_init(self):
        torch.manual_seed(0)
        activation = BSplineKANActivation(channels=8, grid_size=8, spline_order=3)
        x = torch.randn(4, 8, 5, 5, 5) * 1.5
        with torch.no_grad():
            deviation = (activation(x) - F.gelu(x)).abs()
        assert deviation.max().item() < 0.5
        assert deviation.mean().item() < 0.3

    def test_handles_inputs_far_outside_grid_range_without_exploding(self):
        """The whole point of clamping: a wide pre-activation tail (routine
        for an untrained network) must not blow the *residual* up.

        The overall output does NOT saturate -- by design, the GELU base
        term keeps tracking the true unclamped x (GELU(x) ~ x for large
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

        residual = out - F.gelu(x) * activation.base_weight.view(1, -1, 1)
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


# --------------------------------------------------------------------- #
# Backward compatibility: the default (GELU) model is untouched
# --------------------------------------------------------------------- #
class TestBackwardCompatibility:
    def test_default_activation_is_still_gelu(self):
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8)
        assert model.activation == "gelu"
        assert isinstance(model.blocks[0].activation, torch.nn.GELU)

    def test_gelu_model_has_no_extra_parameters(self):
        """The refactor from a bare function to an nn.Module wrapper must not
        change what a `gelu` model contains: torch.nn.GELU is stateless."""
        model = FNO3d(width=6, modes=3, n_layers=2, projection_channels=8,
                      activation="gelu")
        assert model.blocks[0].activation.state_dict() == {}
        assert model.n_parameters() == 17859     # pinned: see test_ml.py's
        # equivalents for width=6/modes=3/n_layers=2/projection_channels=8;
        # a change here means the default model's parameter count moved.

    def test_a_pre_kan_checkpoint_still_loads(self, tmp_path):
        """Simulates a bundle saved before the `architecture` dict carried
        the kan_* keys: strip them out and confirm the model still rebuilds,
        falling back to the (harmless, since activation=gelu) defaults."""
        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        state = operator.state()
        for key in ("kan_grid_size", "kan_spline_order", "kan_grid_range",
                    "kan_degree"):
            state["architecture"].pop(key, None)

        restored = FieldOperator.from_state(state, device="cpu")
        assert restored.model.activation == "gelu"
        assert restored.model.kan_grid_size == 8      # constructor default
        for a, b in zip(operator.model.state_dict().values(),
                        restored.model.state_dict().values()):
            assert torch.equal(a, b)

    def test_gelu_bundle_round_trip_is_unaffected(self, tmp_path):
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
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
