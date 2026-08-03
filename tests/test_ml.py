# -*- coding: utf-8 -*-
# file: test_ml.py
"""Tests for :mod:`poraque.ml` — the FNO, its grid-shape invariance, and physics."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

from poraque.fields import (  # noqa: E402
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
)
from poraque.fields import thomas_fermi_tau as tf_numpy  # noqa: E402
from poraque.fields import von_weizsacker_tau as vw_numpy  # noqa: E402
from poraque.fields.vasp import Poscar  # noqa: E402
from poraque.ml import (  # noqa: E402
    FNO3d,
    FieldOperator,
    FieldPairDataset,
    PhysicsInformedLoss,
    RelativeL2Loss,
    SpectralConv3d,
    collate_fields,
    make_dataloader,
    train,
)
from poraque.ml.physics import (  # noqa: E402
    electron_count_loss,
    euler_lagrange_residual,
    hartree_potential,
    integrate,
    spectral_laplacian,
    thomas_fermi_tau,
    von_weizsacker_bound_loss,
    von_weizsacker_tau,
)
from poraque.ml.transforms import Asinh, FieldTransform, Log, Standardize  # noqa: E402

COULOMB = 14.399645478425668


@pytest.fixture
def dataset_root(tmp_path):
    """Six synthetic materials spanning four different grid shapes."""
    rng = np.random.default_rng(0)
    layout = [
        (np.eye(3) * 5.0, (16, 16, 16)),
        (np.eye(3) * 5.0, (16, 16, 16)),
        (np.eye(3) * 6.5, (20, 20, 20)),
        (np.eye(3) * 6.5, (20, 20, 20)),
        (np.diag([4.0, 6.0, 8.0]), (12, 18, 24)),
        (np.eye(3) * 7.5, (24, 24, 24)),
    ]
    for index, (cell, shape) in enumerate(layout):
        directory = tmp_path / f"mat_{index:02d}"
        directory.mkdir()
        grid = FieldGrid(shape, cell)
        structure = Poscar(cell, ["Si"], [2], rng.random((2, 3)),
                           comment=f"mat_{index:02d}")

        potential = ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                              widths={"Si": 0.5})
        potential.write(directory / "EXTCAR")

        density = np.exp(-(potential.data - potential.data.min()) / 20.0) * 0.2 + 0.01
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")
        tau = 2.871234 * density ** (5.0 / 3.0) * 51.42 + 0.01
        KineticEnergyDensity(tau, grid, structure).write(directory / "TAUCAR")

    return tmp_path


# --------------------------------------------------------------------- #
# The grid-shape invariance requirement
# --------------------------------------------------------------------- #
class TestGridShapeInvariance:
    @pytest.mark.parametrize("shape", [(8, 8, 8), (16, 24, 32), (33, 12, 20)])
    def test_spectral_conv_accepts_any_shape(self, shape):
        conv = SpectralConv3d(3, 5, modes=(16, 16, 16))
        assert conv(torch.randn(2, 3, *shape)).shape == (2, 5, *shape)

    def test_modes_are_clamped_to_what_the_grid_offers(self):
        conv = SpectralConv3d(1, 1, modes=(16, 16, 16))
        assert conv.effective_modes((8, 8, 8)) == (4, 4, 5)
        assert conv.effective_modes((64, 64, 64)) == (16, 16, 16)
        # never zero, even on a degenerate axis
        assert all(m >= 1 for m in conv.effective_modes((2, 2, 2)))

    def test_parameter_count_is_grid_independent(self):
        conv = SpectralConv3d(4, 4, modes=(8, 8, 8))
        before = conv.weight.numel()
        conv(torch.randn(1, 4, 12, 12, 12))
        conv(torch.randn(1, 4, 48, 48, 48))
        assert conv.weight.numel() == before

    @pytest.mark.parametrize("shape,cell_diag", [
        ((16, 16, 16), (5.0, 5.0, 5.0)),
        ((24, 24, 24), (8.0, 8.0, 8.0)),
        ((18, 30, 12), (4.0, 7.0, 3.0)),
        ((40, 20, 28), (9.0, 5.0, 6.5)),
        ((10, 10, 10), (3.0, 3.0, 3.0)),
    ])
    def test_one_model_serves_every_shape(self, shape, cell_diag):
        model = FNO3d(width=16, modes=8, n_layers=2, projection_channels=32)
        cell = torch.diag(torch.tensor(cell_diag)).unsqueeze(0)
        assert model(torch.randn(1, 1, *shape), cell).shape == (1, 1, *shape)

    def test_weights_are_untouched_across_shapes(self):
        model = FNO3d(width=12, modes=6, n_layers=2, projection_channels=16)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        cell = torch.eye(3).unsqueeze(0) * 5.0
        for shape in [(12, 12, 12), (20, 16, 24), (8, 8, 8)]:
            model(torch.randn(1, 1, *shape), cell)
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    def test_gradients_flow_for_any_shape(self):
        model = FNO3d(width=12, modes=4, n_layers=2, projection_channels=16)
        cell = torch.eye(3).unsqueeze(0) * 6.0
        for shape in [(12, 12, 12), (20, 16, 24)]:
            model.zero_grad()
            model(torch.randn(1, 1, *shape), cell).pow(2).mean().backward()
            assert any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())

    def test_discretization_invariance(self):
        """A band-limited field sampled at 2 resolutions gives the same operator output.

        This is what the ``norm="forward"`` FFT convention buys: coefficients
        approximate continuous Fourier-series coefficients, so shared weights
        mean the same thing on a 16^3 and on a 32^3 grid.
        """
        model = FNO3d(width=8, modes=4, n_layers=2, projection_channels=16,
                      use_coordinates=False, cell_conditioning=False).eval()
        cell = torch.eye(3).unsqueeze(0) * 6.0

        def sample(n):
            axis = torch.arange(n) / n * 2 * np.pi
            x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
            field = torch.sin(x) + 0.5 * torch.cos(2 * y) + 0.3 * torch.sin(z + x)
            return field.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            coarse = model(sample(16), cell)
            fine = model(sample(32), cell)
        error = (fine[..., ::2, ::2, ::2] - coarse).norm() / coarse.norm()
        assert error < 0.05

    def test_physical_mode_selection_tracks_the_cell(self):
        model = FNO3d(width=8, modes=16, n_layers=1, projection_channels=16,
                      mode_selection="physical", g_max=3.0)
        small = model.physical_modes(torch.eye(3).unsqueeze(0) * 5.0, (16, 16, 16))
        large = model.physical_modes(torch.eye(3).unsqueeze(0) * 15.0, (32, 32, 32))
        # A larger cell has a denser reciprocal lattice -> more modes below g_max.
        assert all(l > s for l, s in zip(large, small))

    def test_physical_mode_selection_requires_g_max(self):
        with pytest.raises(ValueError, match="requires g_max"):
            FNO3d(mode_selection="physical")

    def test_cell_conditioning_requires_a_cell(self):
        model = FNO3d(width=8, modes=4, n_layers=1, projection_channels=16)
        with pytest.raises(ValueError, match="requires `cell`"):
            model(torch.randn(1, 1, 8, 8, 8))


# --------------------------------------------------------------------- #
# Differentiable physics
# --------------------------------------------------------------------- #
class TestPhysics:
    @pytest.fixture
    def cell(self):
        return torch.eye(3).unsqueeze(0) * 10.0

    def test_hartree_potential_solves_poisson(self, cell):
        density = torch.rand(1, 1, 24, 24, 24) * 0.1 + 0.05
        laplacian = spectral_laplacian(hartree_potential(density, cell), cell)
        rhs = -4 * np.pi * COULOMB * (density - density.mean())
        assert (laplacian - rhs).abs().max() < 1e-4 * rhs.abs().max()

    def test_hartree_potential_has_zero_average(self, cell):
        density = torch.rand(1, 1, 16, 16, 16) + 0.1
        assert hartree_potential(density, cell).mean().abs() < 1e-4

    def test_integrate_recovers_the_volume(self, cell):
        ones = torch.ones(1, 1, 20, 20, 20)
        assert float(integrate(ones, cell)) == pytest.approx(1000.0, rel=1e-5)

    def test_thomas_fermi_matches_the_numpy_implementation(self, cell):
        density = torch.rand(1, 1, 12, 12, 12) * 0.3 + 0.05
        reference = tf_numpy(density.numpy()[0, 0])
        assert np.abs(thomas_fermi_tau(density).numpy()[0, 0]
                      - reference).max() < 1e-4 * reference.max()

    def test_von_weizsacker_matches_the_numpy_implementation(self, cell):
        grid = FieldGrid((16, 16, 16), np.eye(3) * 10.0)
        density = torch.rand(1, 1, 16, 16, 16) * 0.3 + 0.05
        reference = vw_numpy(density.numpy()[0, 0], grid)
        assert np.abs(von_weizsacker_tau(density, cell).numpy()[0, 0]
                      - reference).max() < 1e-4 * reference.max()

    def test_von_weizsacker_vanishes_for_a_uniform_density(self, cell):
        uniform = torch.full((1, 1, 16, 16, 16), 0.3)
        assert von_weizsacker_tau(uniform, cell).abs().max() < 1e-6

    def test_von_weizsacker_is_non_negative(self, cell):
        density = torch.rand(1, 1, 16, 16, 16) + 0.05
        assert (von_weizsacker_tau(density, cell) >= -1e-9).all()

    def test_von_weizsacker_bound_loss_is_one_sided(self, cell):
        density = torch.rand(1, 1, 16, 16, 16) * 0.2 + 0.05
        bound = von_weizsacker_tau(density, cell)
        assert float(von_weizsacker_bound_loss(bound * 1.5, density, cell)) == 0.0
        assert float(von_weizsacker_bound_loss(bound * 0.5, density, cell)) > 0.0

    def test_von_weizsacker_bound_loss_is_scale_free(self, cell):
        """Scaling rho must not change the loss by orders of magnitude."""
        density = torch.rand(1, 1, 16, 16, 16) * 0.2 + 0.05
        small = von_weizsacker_bound_loss(torch.zeros_like(density), density, cell)
        large = von_weizsacker_bound_loss(torch.zeros_like(density), density * 100,
                                          cell)
        assert float(small) == pytest.approx(float(large), rel=1e-3)

    def test_electron_count_loss_vanishes_at_the_truth(self, cell):
        density = torch.rand(1, 1, 16, 16, 16) + 0.1
        count = integrate(density, cell)
        assert float(electron_count_loss(density, cell, count)) < 1e-10
        assert float(electron_count_loss(density, cell, count * 1.5)) > 1e-3

    def test_euler_lagrange_residual_is_zero_mean(self, cell):
        density = torch.rand(1, 1, 12, 12, 12) * 0.2 + 0.05
        potential = torch.randn(1, 1, 12, 12, 12) * 5.0
        residual = euler_lagrange_residual(density, potential, cell)
        assert residual.mean().abs() < 1e-3
        assert residual.shape == density.shape

    def test_physics_terms_are_differentiable(self, cell):
        density = (torch.rand(1, 1, 12, 12, 12) * 0.2 + 0.05).requires_grad_(True)
        loss = (von_weizsacker_tau(density, cell).mean()
                + hartree_potential(density, cell).pow(2).mean()
                + thomas_fermi_tau(density).mean())
        loss.backward()
        assert density.grad is not None and torch.isfinite(density.grad).all()


# --------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------- #
class TestTransforms:
    @pytest.mark.parametrize("transform", [
        Standardize(2.0, 3.0), Asinh(0.1), Log(1e-6),
    ])
    def test_round_trip(self, transform):
        values = torch.rand(500) * 5.0 + 1e-3
        assert torch.allclose(transform.inverse(transform(values)), values,
                              atol=1e-4)

    def test_numpy_and_torch_agree(self):
        transform = Asinh(0.05)
        values = np.linspace(0.01, 5.0, 100)
        assert np.allclose(transform(values),
                           transform(torch.as_tensor(values)).numpy(), atol=1e-9)

    def test_log_transform_guarantees_positivity(self):
        transform = Log(1e-6)
        assert (transform.inverse(torch.randn(1000) * 20) >= -1e-6).all()

    def test_state_dict_round_trip(self):
        for transform in (Standardize(1.0, 2.0), Asinh(0.3), Log(1e-5)):
            restored = FieldTransform.from_state_dict(transform.state_dict())
            assert type(restored) is type(transform)
            values = torch.rand(50) + 0.1
            assert torch.allclose(restored(values), transform(values))


# --------------------------------------------------------------------- #
# Dataset and ragged batching
# --------------------------------------------------------------------- #
class TestDataset:
    def test_discovery_and_shapes(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        assert len(dataset) == 6
        assert dataset.shapes() == [(16, 16, 16), (16, 16, 16), (20, 20, 20),
                                    (20, 20, 20), (12, 18, 24), (24, 24, 24)]

    def test_sample_layout(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        sample = dataset[0]
        assert sample["input"].shape == (1, 16, 16, 16)
        assert sample["target"].shape == (1, 16, 16, 16)
        assert sample["cell"].shape == (3, 3)
        assert sample["material"] == "mat_00"

    def test_shape_buckets_never_mix_shapes(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        loader = make_dataloader(dataset, batch_size=3)
        seen = 0
        for batch in loader:
            assert batch["input"].shape[-3:] == torch.Size(batch["shape"])
            seen += len(batch["material"])
        assert seen == len(dataset)

    def test_mixed_shape_collate_is_rejected(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        with pytest.raises(ValueError, match="Cannot batch mixed grid shapes"):
            collate_fields([dataset[0], dataset[2]])

    def test_fit_transforms_picks_sensible_types(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        source, target = dataset.fit_transforms()
        assert isinstance(source, Standardize)   # signed potential
        assert isinstance(target, Asinh)         # positive, many decades

    def test_split_is_by_material(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        first, second = dataset.split(0.7, seed=1)
        assert len(first) + len(second) == len(dataset)
        names = {m.identifier for m in first.materials}
        assert names.isdisjoint({m.identifier for m in second.materials})

    def test_chg2tau_task(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="chg2tau")
        assert dataset.task.input_field == "CHGCAR"
        assert dataset.task.target_field == "TAUCAR"
        assert dataset[0]["input"].shape == (1, 16, 16, 16)

    def test_unknown_task(self, dataset_root):
        with pytest.raises(KeyError, match="Unknown task"):
            FieldPairDataset(dataset_root, task="chg2nothing")

    def test_empty_root(self, tmp_path):
        with pytest.raises(ValueError, match="No material directories"):
            FieldPairDataset(tmp_path, task="ext2chg")


# --------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------- #
class TestTraining:
    def test_loss_decreases_and_predicts_on_an_unseen_shape(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        source, target = dataset.fit_transforms()
        train_set, val_set = dataset.split(0.7, seed=1)

        operator = FieldOperator("ext2chg", width=12, modes=6, n_layers=2,
                                 projection_channels=24, input_transform=source,
                                 target_transform=target, device="cpu")
        history = train(operator, train_set, validation=val_set, epochs=6,
                        batch_size=2, learning_rate=3e-3, verbose=False)
        assert history["train_loss"][-1] < history["train_loss"][0]

        # A grid shape that appears nowhere in the training set.
        grid = FieldGrid((14, 22, 18), np.diag([4.5, 7.0, 5.5]))
        structure = Poscar(grid.cell, ["Si"], [2],
                           np.random.default_rng(3).random((2, 3)))
        potential = ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                              widths={"Si": 0.5})
        prediction = operator.predict(potential)

        assert isinstance(prediction, ChargeDensity)
        assert prediction.shape == (14, 22, 18)
        assert prediction.grid.matches(grid)
        assert np.isfinite(prediction.data).all()

    def test_prediction_is_a_writable_chgcar(self, dataset_root, tmp_path):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        source, target = dataset.fit_transforms()
        operator = FieldOperator("ext2chg", width=8, modes=4, n_layers=1,
                                 projection_channels=16, input_transform=source,
                                 target_transform=target, device="cpu")
        potential = ExternalPotential.read(
            os.path.join(dataset_root, "mat_00", "EXTCAR"))
        path = tmp_path / "CHGCAR_pred"
        operator.predict(potential).write(path)
        assert ChargeDensity.read(path).shape == potential.shape

    def test_checkpoint_round_trip(self, dataset_root, tmp_path):
        dataset = FieldPairDataset(dataset_root, task="ext2chg")
        source, target = dataset.fit_transforms()
        kwargs = dict(width=8, modes=4, n_layers=1, projection_channels=16)
        operator = FieldOperator("ext2chg", input_transform=source,
                                 target_transform=target, device="cpu", **kwargs)

        potential = ExternalPotential.read(
            os.path.join(dataset_root, "mat_00", "EXTCAR"))
        expected = operator.predict(potential).data

        path = tmp_path / "operator.pt"
        operator.save(path)
        restored = FieldOperator.load(path, device="cpu", **kwargs)
        assert np.abs(restored.predict(potential).data - expected).max() < 1e-6

    def test_physics_informed_loss_trains(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="chg2tau")
        source, target = dataset.fit_transforms()
        loss = PhysicsInformedLoss(task="chg2tau", von_weizsacker_weight=1.0,
                                   positivity_weight=1.0, sobolev_weight=0.1)
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16, input_transform=source,
                                 target_transform=target, device="cpu")
        history = train(operator, dataset, epochs=4, batch_size=2, loss=loss,
                        learning_rate=3e-3, verbose=False)
        assert np.isfinite(history["train_loss"]).all()

    def test_default_loss_is_purely_supervised(self):
        """Physics weights default to zero, so the baseline is unchanged."""
        criterion = PhysicsInformedLoss(task="ext2chg")
        prediction = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        cell = torch.eye(3).repeat(2, 1, 1) * 5.0
        terms = criterion(prediction, target, cell=cell,
                          physical_prediction=prediction.abs(),
                          physical_input=target)
        assert set(terms) == {"total", "data"}
        assert float(terms["total"]) == pytest.approx(
            float(RelativeL2Loss()(prediction, target)))
