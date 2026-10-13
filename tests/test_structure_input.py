# -*- coding: utf-8 -*-
# file: test_structure_input.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Structure channels beside the potential, memoized samples, and the two loss
weightings added for the CHONPS ``ext2chg`` study (2026-10-06).

``model.structure_input`` hands the operator :math:`\rho_{\rm sup}` and one
superposition per element next to :math:`v_{\rm ext}`. On 785 CHONPS cells the
same backbone went from 4.4 % to 3.7 % held-out rel L2 at epoch 10 with them;
what is pinned here is that they are what they claim to be, that they survive a
checkpoint, and that inference rebuilds them from the structure alone.
"""

import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from poraque.fields import ChargeDensity, ExternalPotential, FieldGrid  # noqa: E402
from poraque.fields.atomic import (  # noqa: E402
    AtomicReferenceLibrary,
    atomic_superposition,
    structure_channels,
)
from poraque.fields.vasp import Poscar  # noqa: E402
from poraque.ml import FieldOperator, FieldPairDataset, train  # noqa: E402
from poraque.ml.paw import OccupancyTransform  # noqa: E402
from test_atomic import synthetic_reference  # noqa: E402

SPEC = {"superposition": True, "elements": ["C", "Si"]}


@pytest.fixture
def library():
    references = [synthetic_reference("Si", 4.0, 0.7),
                  synthetic_reference("C", 4.0, 0.5)]
    return AtomicReferenceLibrary({r.key: r for r in references})


def _structure(side=6.0, symbols=("Si", "C"), counts=(2, 1), seed=0):
    rng = np.random.default_rng(seed)
    return Poscar(np.eye(3) * side, list(symbols), list(counts),
                  rng.random((sum(counts), 3)))


@pytest.fixture
def dataset_root(tmp_path, library):
    """Four SiC-like cells over two grid shapes, written as a cache would be."""
    for index, (side, shape) in enumerate([(6.0, (16, 16, 16)),
                                           (6.0, (16, 16, 16)),
                                           (7.0, (20, 20, 20)),
                                           (7.0, (20, 20, 20))]):
        directory = tmp_path / f"mat_{index:02d}"
        directory.mkdir()
        grid = FieldGrid(shape, np.eye(3) * side)
        structure = _structure(side, seed=index)
        potential = ExternalPotential.compute(
            structure, grid, {"Si": 4.0, "C": 4.0},
            widths={"Si": 0.5, "C": 0.4})
        potential.write(directory / "EXTCAR")
        density = atomic_superposition(structure, grid, library).data
        density = density * (1.0 + 0.05 * np.cos(potential.data / 10.0))
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")
    return tmp_path


class TestTheStructureChannelsAreTheSuperpositionTakenApart:
    """The channels sum to the baseline, and each holds its element alone."""

    def test_the_element_channels_sum_to_the_superposition(self, library):
        grid = FieldGrid((20, 20, 20), np.eye(3) * 6.0)
        structure = _structure()
        channels = structure_channels(structure, grid, library, **SPEC)
        total = atomic_superposition(structure, grid, library).data
        assert channels.shape == (3, 20, 20, 20)
        np.testing.assert_allclose(channels[0], total, atol=1e-12)
        np.testing.assert_allclose(channels[1:].sum(axis=0), total,
                                   atol=1e-12)

    def test_each_channel_integrates_to_its_own_atoms(self, library):
        grid = FieldGrid((20, 20, 20), np.eye(3) * 6.0)
        channels = structure_channels(_structure(), grid, library, **SPEC)
        volume_element = grid.volume / np.prod(grid.shape)
        counts = channels.sum(axis=(1, 2, 3)) * volume_element
        np.testing.assert_allclose(counts, [12.0, 4.0, 8.0], rtol=1e-3)

    def test_an_absent_element_is_a_channel_of_zeros(self, library):
        grid = FieldGrid((16, 16, 16), np.eye(3) * 6.0)
        structure = _structure(symbols=("Si",), counts=(2,))
        channels = structure_channels(structure, grid, library, **SPEC)
        assert not channels[1].any()

    def test_an_element_without_a_channel_is_refused(self, library):
        grid = FieldGrid((16, 16, 16), np.eye(3) * 6.0)
        with pytest.raises(KeyError, match="never trained"):
            structure_channels(_structure(), grid, library,
                               superposition=True, elements=["Si"])


class TestTheDatasetAppendsTheChannelsAndMemoizesItsSamples:
    """The input gains the channels; a new transform invalidates the memo."""

    def test_the_input_carries_the_structure_channels(self, dataset_root,
                                                      library):
        dataset = FieldPairDataset(dataset_root, task="ext2pseudo", cache=True,
                                   baseline=library, structure_input=SPEC)
        assert dataset.channels == (4, 1)
        dataset.fit_transforms()
        sample = dataset[0]
        assert sample["input"].shape[0] == 4
        assert type(dataset.input_transform).__name__ == "Channelwise"

    def test_assigning_a_transform_drops_the_memoized_samples(
            self, dataset_root, library):
        from poraque.ml.transforms import Standardize

        dataset = FieldPairDataset(dataset_root, task="ext2pseudo", cache=True,
                                   baseline=library)
        dataset.fit_transforms()
        first = dataset[0]["input"].clone()
        assert dataset[0]["input"] is dataset[0]["input"]
        dataset.input_transform = Standardize(0.0, 1.0)
        np.testing.assert_allclose(dataset[0]["input"].numpy(),
                                   dataset.sample_tensors(0)[0].numpy())
        assert not torch.equal(dataset[0]["input"], first)

    def test_without_a_library_the_request_is_refused(self, dataset_root):
        with pytest.raises(ValueError, match="isolated-atom library"):
            FieldPairDataset(dataset_root, task="ext2pseudo", baseline=None,
                             structure_input=SPEC)


class TestTheOperatorRebuildsItsChannelsFromTheStructure:
    """A checkpoint records the channels; ``predict`` builds them itself."""

    def test_predict_matches_the_training_path_after_a_round_trip(
            self, dataset_root, library):
        dataset = FieldPairDataset(dataset_root, task="ext2pseudo", cache=True,
                                   baseline=library, structure_input=SPEC,
                                   dtype=torch.float64)
        source_transform, target_transform = dataset.fit_transforms()
        operator = FieldOperator(
            "ext2pseudo", input_transform=source_transform,
            target_transform=target_transform, device="cpu", width=8,
            modes=4, n_layers=1, use_coordinates=False, in_channels=4,
            out_channels=1, baseline=library, structure_input=SPEC,
            init_seed=0)
        operator.set_precision("float64")
        restored = FieldOperator.from_state(operator.state(), device="cpu")
        restored.set_precision("float64")
        assert restored.structure_input == SPEC
        assert restored.in_channels == 4

        source, _ = dataset.load_fields(0)
        predicted = restored.predict(source).data
        sample = dataset[0]
        with torch.no_grad():
            direct = restored.target_transform.inverse(restored.model(
                sample["input"][None], sample["cell"][None]))[0, 0].numpy()
        direct = direct + dataset.baseline_for(0)
        np.testing.assert_allclose(predicted, direct, atol=1e-10)


class TestAccumulationWeighsEveryStructureAlike:
    """``samples_per_step`` steps on the mean over structures, not batches."""

    def test_one_step_over_the_whole_set_is_the_full_batch_gradient(
            self, dataset_root, library):
        dataset = FieldPairDataset(dataset_root, task="ext2pseudo", cache=True,
                                   baseline=library, dtype=torch.float64)
        dataset.fit_transforms()

        def trained(**kwargs):
            operator = FieldOperator(
                "ext2pseudo", input_transform=dataset.input_transform,
                target_transform=dataset.target_transform, device="cpu",
                width=4, modes=3, n_layers=1, use_coordinates=False,
                init_seed=0, baseline=library)
            operator.set_precision("float64")
            train(operator, dataset, epochs=1, learning_rate=1e-2,
                  optimizer="sgd", scheduler=None, grad_clip=0,
                  verbose=False, seed=0, **kwargs)
            return torch.cat([p.detach().flatten() for p in
                              operator.model.parameters()])

        # Four structures, one step: the gradient of the mean loss over all
        # four, whatever the buckets -- the same as one batch of four would
        # give if the shapes allowed it, and independent of the batch size.
        two = trained(batch_size=2, samples_per_step=4)
        one = trained(batch_size=1, samples_per_step=4)
        torch.testing.assert_close(two, one, rtol=1e-9, atol=1e-12)


class TestTheMagnetizationSetCanBeWeightedDown:
    """``set_weights`` of ones is the old loss to the bit."""

    def _transform(self, weights):
        layouts = {14: (0, 1)}
        from poraque.fields.vasp.augmentation import irreps_blocks

        length = sum(index.size for index in irreps_blocks((0, 1)).values())
        return OccupancyTransform(layouts, 2, {14: np.zeros((2, length))},
                                  {14: np.ones((2, length))},
                                  set_weights=weights), length

    def test_unit_weights_leave_the_loss_unchanged(self):
        rng = torch.Generator().manual_seed(0)
        plain, length = self._transform(None)
        ones, _ = self._transform([1.0, 1.0])
        predicted = torch.randn(1, 3, 2, length, generator=rng,
                                dtype=torch.float64)
        target = torch.randn(1, 3, 2, length, generator=rng,
                             dtype=torch.float64)
        species = torch.full((1, 3), 14)
        mask = torch.ones(1, 3, dtype=torch.float64)
        assert plain.loss(predicted, target, species, mask) == \
            ones.loss(predicted, target, species, mask)

    def test_a_zero_weight_ignores_the_magnetization_set(self):
        rng = torch.Generator().manual_seed(1)
        weighted, length = self._transform([1.0, 0.0])
        predicted = torch.randn(1, 3, 2, length, generator=rng,
                                dtype=torch.float64)
        target = predicted.clone()
        target[:, :, 1] += 5.0
        species = torch.full((1, 3), 14)
        mask = torch.ones(1, 3, dtype=torch.float64)
        assert float(weighted.loss(predicted, target, species, mask)) == 0.0
        restored = OccupancyTransform.from_state_dict(weighted.state_dict())
        assert restored.set_weights == [1.0, 0.0]
