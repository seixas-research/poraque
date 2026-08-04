# -*- coding: utf-8 -*-
# file: test_heads.py
"""
Tests for :mod:`poraque.ml.heads`.

The point of a constraint-enforcing head is that the constraint holds for
*every* weight configuration, not merely after successful training. These
tests therefore hammer the invariant at random initialization, after
deliberately destructive optimizer steps, and with pathological inputs — the
situations where a soft penalty would quietly fail.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.fields import (  # noqa: E402
    ChargeDensity,
    FieldGrid,
    KineticEnergyDensity,
)
from poraque.fields import von_weizsacker_tau as vw_numpy  # noqa: E402
from poraque.fields.vasp import Poscar  # noqa: E402
from poraque.ml import (  # noqa: E402
    FNO3d,
    FieldOperator,
    FieldPairDataset,
    PauliResidualOperator,
    fit_pauli_scale,
    pauli_bound_violation,
    train,
)
from poraque.ml.physics import von_weizsacker_tau  # noqa: E402
from poraque.ml.transforms import Asinh, Identity  # noqa: E402


@pytest.fixture
def dataset_root(tmp_path):
    """Three materials with tau = tau_vW + a positive Pauli term, by construction."""
    rng = np.random.default_rng(0)
    layout = [
        (np.eye(3) * 6.0, (16, 16, 16)),
        (np.diag([5.0, 6.0, 7.0]), (12, 16, 20)),
        (np.eye(3) * 7.0, (20, 20, 20)),
    ]
    for index, (cell, shape) in enumerate(layout):
        directory = tmp_path / f"mat_{index:02d}"
        directory.mkdir()
        grid = FieldGrid(shape, cell)
        structure = Poscar(cell, ["Si"], [2], rng.random((2, 3)))

        from poraque.fields import ExternalPotential

        potential = ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                              widths={"Si": 0.6})
        potential.write(directory / "EXTCAR")

        density = np.exp(-(potential.data - potential.data.min()) / 25.0) * 0.4 + 0.02
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")

        # A physically-shaped target that respects the bound everywhere.
        tau = vw_numpy(density, grid) + 3.0 * density ** (5.0 / 3.0) + 0.5
        KineticEnergyDensity(tau, grid, structure).write(directory / "TAUCAR")

    return tmp_path


@pytest.fixture
def chg2tau(dataset_root):
    dataset = FieldPairDataset(dataset_root, task="chg2tau")
    dataset.fit_transforms()
    return dataset


def build_head(dataset, scale=1.0, learn_scale=True, **kwargs):
    backbone = FNO3d(width=8, modes=4, n_layers=2, projection_channels=16, **kwargs)
    return PauliResidualOperator(backbone, dataset.input_transform,
                                 dataset.target_transform, scale=scale,
                                 learn_scale=learn_scale)


def physical(head, batch):
    """Decode the head's normalized output back to eV/Å³."""
    return head.target_transform.inverse(head(batch["input"], batch["cell"]))


def as_batch(dataset, index):
    sample = dataset[index]
    return {"input": sample["input"].unsqueeze(0),
            "cell": sample["cell"].unsqueeze(0)}


# --------------------------------------------------------------------- #
# The invariant
# --------------------------------------------------------------------- #
class TestHardConstraint:
    def test_holds_at_random_initialization(self, chg2tau):
        """No training required: the bound is algebra, not a learned property."""
        torch.manual_seed(0)
        head = build_head(chg2tau, scale=2.0)
        for index in range(len(chg2tau)):
            batch = as_batch(chg2tau, index)
            tau = physical(head, batch)
            bound = head.von_weizsacker_term(batch["input"], batch["cell"])
            assert (tau - bound).min() > 0.0

    def test_saturates_to_the_single_orbital_limit(self, chg2tau):
        """Driving the backbone to -inf gives tau == tau_vW exactly.

        This is the correct physical floor, not a degenerate failure: a
        one-orbital (nodeless) region has zero Pauli term, and the head can
        represent that limit exactly rather than only approaching it.
        """
        torch.manual_seed(0)
        head = build_head(chg2tau, scale=1.0)
        with torch.no_grad():
            for parameter in head.backbone.parameters():
                parameter.zero_()
            # Bias the final projection hard negative so softplus underflows.
            head.backbone.project[-1].bias.fill_(-100.0)

        batch = as_batch(chg2tau, 0)
        pauli = head.pauli_term(batch["input"], batch["cell"])
        # softplus(-100) underflows to a float32 subnormal (~4e-44): zero to
        # every precision that matters, and never negative.
        assert 0.0 <= float(pauli.min().detach())
        assert float(pauli.max().detach()) < 1e-40

        tau = physical(head, batch)
        bound = head.von_weizsacker_term(batch["input"], batch["cell"])
        assert torch.allclose(tau, bound, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
    def test_holds_for_arbitrary_weights(self, chg2tau, seed):
        """Randomised weights of large magnitude must not break it.

        The exact invariant lives on the Pauli term: ``softplus`` is
        non-negative for every real argument, so ``tau >= tau_vW`` identically.
        The bound is non-strict, and saturation to ``tau == tau_vW`` is the
        physically correct single-orbital limit, not a failure.

        The decoded ``tau`` additionally passes through the normalization and
        back in float32, so it is compared with a relative tolerance — blowing
        the backbone up by 50x drives the field to ~1e4 eV/A^3, where float32
        has only ~1e-3 absolute resolution.
        """
        torch.manual_seed(seed)
        head = build_head(chg2tau, scale=1.0)
        with torch.no_grad():
            for parameter in head.backbone.parameters():
                parameter.mul_(50.0)          # blow the backbone up
        batch = as_batch(chg2tau, 0)

        # Exact, no round trip through the transform.
        assert float(head.pauli_term(batch["input"], batch["cell"]).min().detach()) >= 0.0

        tau = physical(head, batch)
        margin = tau - head.von_weizsacker_term(batch["input"], batch["cell"])
        assert float(margin.min().detach()) > -1e-6 * float(tau.abs().max().detach())

    def test_survives_destructive_optimizer_steps(self, chg2tau):
        """An absurd learning rate may ruin accuracy but cannot break the bound."""
        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16,
                                 input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu", pauli_residual=True,
                                 pauli_scale=fit_pauli_scale(chg2tau))
        train(operator, chg2tau, epochs=15, batch_size=1, learning_rate=0.5,
              verbose=False)

        for index in range(len(chg2tau)):
            density, _ = chg2tau.load_fields(index)
            prediction = operator.predict(density)
            bound = vw_numpy(density.data, density.grid)
            assert (prediction.data - bound).min() > -1e-6

    def test_baseline_backbone_can_violate_the_bound(self, chg2tau):
        """Control: without the head the bound is not guaranteed.

        If an untrained bare model never violated the bound, the head would be
        solving a non-problem and these tests would prove nothing.
        """
        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16,
                                 input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu")
        worst = min(
            float((operator.predict(chg2tau.load_fields(i)[0]).data
                   - vw_numpy(chg2tau.load_fields(i)[0].data,
                              chg2tau.load_fields(i)[0].grid)).min())
            for i in range(len(chg2tau))
        )
        assert worst < 0.0

    def test_soft_penalty_is_exactly_zero_with_the_head(self, chg2tau):
        """The head makes von_weizsacker_bound_loss redundant, by construction."""
        from poraque.ml.physics import von_weizsacker_bound_loss

        torch.manual_seed(0)
        head = build_head(chg2tau, scale=2.0)
        batch = as_batch(chg2tau, 0)
        density = head.decode_density(batch["input"])
        loss = von_weizsacker_bound_loss(physical(head, batch), density,
                                         batch["cell"])
        assert float(loss.detach()) == 0.0


# --------------------------------------------------------------------- #
# Decomposition
# --------------------------------------------------------------------- #
class TestDecomposition:
    def test_tau_equals_vw_plus_pauli(self, chg2tau):
        torch.manual_seed(0)
        head = build_head(chg2tau, scale=3.0)
        batch = as_batch(chg2tau, 0)
        total = physical(head, batch)
        parts = (head.von_weizsacker_term(batch["input"], batch["cell"])
                 + head.pauli_term(batch["input"], batch["cell"]))
        assert torch.allclose(total, parts, atol=1e-4)

    def test_pauli_term_is_non_negative(self, chg2tau):
        torch.manual_seed(3)
        head = build_head(chg2tau, scale=1.0)
        batch = as_batch(chg2tau, 0)
        assert float(head.pauli_term(batch["input"], batch["cell"]).min().detach()) >= 0.0

    def test_von_weizsacker_term_matches_the_numpy_reference(self, chg2tau):
        """The analytic part must be the same function the physics module uses."""
        head = build_head(chg2tau, scale=1.0)
        density, _ = chg2tau.load_fields(0)
        batch = as_batch(chg2tau, 0)

        computed = head.von_weizsacker_term(batch["input"],
                                            batch["cell"]).numpy()[0, 0]
        reference = vw_numpy(density.data, density.grid)
        assert np.abs(computed - reference).max() < 1e-3 * reference.max()

    def test_decoded_density_round_trips(self, chg2tau):
        head = build_head(chg2tau, scale=1.0)
        density, _ = chg2tau.load_fields(0)
        decoded = head.decode_density(as_batch(chg2tau, 0)["input"]).numpy()[0, 0]
        assert np.abs(decoded - density.data).max() < 1e-4

    def test_scale_is_positive_and_learnable(self, chg2tau):
        head = build_head(chg2tau, scale=4.0, learn_scale=True)
        assert float(head.scale.detach()) == pytest.approx(4.0)
        assert any(p is head.log_scale for p in head.parameters())

        fixed = build_head(chg2tau, scale=4.0, learn_scale=False)
        assert all(p is not fixed.log_scale for p in fixed.parameters())

    def test_rejects_a_non_positive_scale(self, chg2tau):
        with pytest.raises(ValueError, match="must be positive"):
            build_head(chg2tau, scale=0.0)

    def test_requires_a_cell(self, chg2tau):
        head = build_head(chg2tau, scale=1.0)
        with pytest.raises(ValueError, match="requires `cell`"):
            head(as_batch(chg2tau, 0)["input"], None)

    def test_handles_ragged_grid_shapes(self, chg2tau):
        """One head, several materials, several grid shapes."""
        torch.manual_seed(0)
        head = build_head(chg2tau, scale=2.0)
        shapes = set()
        for index in range(len(chg2tau)):
            batch = as_batch(chg2tau, index)
            output = head(batch["input"], batch["cell"])
            assert output.shape == batch["input"].shape
            shapes.add(tuple(output.shape[-3:]))
        assert len(shapes) > 1


# --------------------------------------------------------------------- #
# Fitting helpers and diagnostics
# --------------------------------------------------------------------- #
class TestFitAndDiagnostics:
    def test_fit_pauli_scale_is_positive_and_sane(self, chg2tau):
        scale = fit_pauli_scale(chg2tau)
        assert scale > 0.0
        density, tau = chg2tau.load_fields(0)
        pauli = tau.data - vw_numpy(density.data, density.grid)
        assert scale <= pauli.max()

    def test_bound_violation_reports_a_clean_dataset(self, chg2tau):
        for entry in pauli_bound_violation(chg2tau):
            assert entry["violations"] == 0
            assert entry["worst_deficit"] >= 0.0
            assert 0.0 < entry["vw_fraction"] < 1.0

    def test_bound_violation_detects_a_dirty_dataset(self, dataset_root):
        """A target below tau_vW must be reported, not silently accepted."""
        directory = sorted(p for p in os.listdir(dataset_root))[0]
        grid = FieldGrid.from_file(os.path.join(dataset_root, directory, "CHGCAR"))
        density = ChargeDensity.read(
            os.path.join(dataset_root, directory, "CHGCAR"), grid=grid)
        broken = vw_numpy(density.data, grid) * 0.5      # deliberately too small
        KineticEnergyDensity(broken, grid, density.structure).write(
            os.path.join(dataset_root, directory, "TAUCAR"))

        dataset = FieldPairDataset(dataset_root, task="chg2tau")
        entry = next(e for e in pauli_bound_violation(dataset)
                     if e["material"] == directory)
        assert entry["violations"] > 0
        assert entry["worst_deficit"] < 0.0


# --------------------------------------------------------------------- #
# Integration with FieldOperator
# --------------------------------------------------------------------- #
class TestFieldOperatorIntegration:
    def test_operator_builds_and_trains_with_the_head(self, chg2tau):
        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16,
                                 input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu", pauli_residual=True,
                                 pauli_scale=fit_pauli_scale(chg2tau))
        assert isinstance(operator.model, PauliResidualOperator)
        history = train(operator, chg2tau, epochs=8, batch_size=1,
                        learning_rate=3e-3, verbose=False)
        assert np.isfinite(history["train_loss"]).all()
        assert history["train_loss"][-1] < history["train_loss"][0]

    def test_head_is_rejected_for_ext2chg(self, dataset_root):
        """tau_vW is a functional of the density; ext2chg's input is not one."""
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        dataset.fit_transforms()
        with pytest.raises(ValueError, match="only defined for the chg2tau"):
            FieldOperator("ext2chg", width=8, modes=4, n_layers=1,
                          projection_channels=16, device="cpu",
                          pauli_residual=True)

    def test_checkpoint_restores_the_head_without_being_told(self, chg2tau, tmp_path):
        torch.manual_seed(0)
        kwargs = dict(width=8, modes=4, n_layers=2, projection_channels=16)
        operator = FieldOperator("chg2tau", input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu", pauli_residual=True,
                                 pauli_scale=2.5, **kwargs)
        density, _ = chg2tau.load_fields(0)
        expected = operator.predict(density).data

        path = tmp_path / "pauli.pt"
        operator.save(path)
        restored = FieldOperator.load(path, device="cpu", **kwargs)

        assert restored.pauli_residual is True
        assert isinstance(restored.model, PauliResidualOperator)
        assert restored.pauli_scale == pytest.approx(2.5)
        assert np.abs(restored.predict(density).data - expected).max() < 1e-6

    def test_plain_checkpoint_still_loads_without_a_head(self, chg2tau, tmp_path):
        kwargs = dict(width=8, modes=4, n_layers=1, projection_channels=16)
        operator = FieldOperator("chg2tau", input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu", **kwargs)
        path = tmp_path / "plain.pt"
        operator.save(path)
        restored = FieldOperator.load(path, device="cpu", **kwargs)
        assert restored.pauli_residual is False
        assert not isinstance(restored.model, PauliResidualOperator)

    def test_prediction_is_a_writable_taucar(self, chg2tau, tmp_path):
        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=1,
                                 projection_channels=16,
                                 input_transform=chg2tau.input_transform,
                                 target_transform=chg2tau.target_transform,
                                 device="cpu", pauli_residual=True,
                                 pauli_scale=2.0)
        density, _ = chg2tau.load_fields(0)
        prediction = operator.predict(density)
        assert isinstance(prediction, KineticEnergyDensity)

        path = tmp_path / "TAUCAR_pred"
        prediction.write(path)
        assert KineticEnergyDensity.read(path).shape == density.shape
