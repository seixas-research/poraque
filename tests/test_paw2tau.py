# -*- coding: utf-8 -*-
# file: test_paw2tau.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
``paw2tau``: the kinetic energy density from the pseudo-density **and** every
atom's PAW occupancies.

The occupancies have no grid representation, so they enter through
:class:`~poraque.ml.paw.SiteWriter`, which writes each atom's record as
atom-centered solid-harmonic Gaussians in reciprocal space. What is pinned here
is that the written field is the real-space sum it claims to be, that it moves
and inverts with the atoms (the condition for the operator reading it to stay
equivariant), and that the task trains, survives a checkpoint and predicts
through the same path it trained on.
"""

import itertools
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="poraque.ml requires PyTorch")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "tests"))

from poraque.fields import ChargeDensity, FieldGrid, KineticEnergyDensity  # noqa: E402
from poraque.fields.vasp import Poscar  # noqa: E402
from poraque.fields.vasp.augmentation import (  # noqa: E402
    format_augmentation,
    irreps_blocks,
    record_layout,
)
from poraque.ml import FieldOperator, FieldPairDataset, train  # noqa: E402
from poraque.ml.paw import OccupancyTransform, SiteWriter, solid_harmonics  # noqa: E402
from poraque.ml.tasks import resolve_task  # noqa: E402

#: Silicon-like s and p projector channels: L = 0, 1, 2 blocks.
LAYOUT = {14: (0, 1)}
LENGTH = len(record_layout(LAYOUT[14]))


def _writer(channels=2):
    torch.manual_seed(0)
    return SiteWriter(LAYOUT, sets=1, channels=channels).double()


def _inputs(positions, seed=0):
    rng = np.random.default_rng(seed)
    count = len(positions)
    records = torch.as_tensor(rng.standard_normal((1, count, 1, LENGTH)))
    species = torch.full((1, count), 14)
    frac = torch.as_tensor(np.asarray(positions, dtype=float))[None]
    mask = torch.ones(1, count, dtype=torch.bool)
    return records, species, frac, mask


class TestTheWriterIsTheGaussianItClaims:
    """
    The reciprocal-space construction equals the real-space sum of solid
    harmonics times Gaussians over the periodic images, by Hobson's theorem.
    """

    def test_the_written_field_matches_the_direct_sum(self):
        writer = _writer()
        records, species, frac, mask = _inputs([[0.1, 0.2, 0.3],
                                                [0.6, 0.55, 0.8]])
        cell = torch.tensor([[7.0, 0, 0], [0.5, 6.5, 0], [0.3, 0.2, 8.0]],
                            dtype=torch.float64)[None]
        # Fine enough for the narrowest Gaussian (0.25 A) to be band-limited
        # on it: the construction is exact, the grid is what truncates it.
        shape = (56, 52, 64)
        field = writer(records, species, frac, mask, cell, shape)[0]

        coefficients = writer.coefficients(records[0], species[0], 2)
        lattice = cell[0].numpy()
        mesh = np.stack(np.meshgrid(*[np.arange(n) / n for n in shape],
                                    indexing="ij"), -1) @ lattice
        direct = np.zeros((2,) + shape)
        for atom in range(2):
            center = frac[0, atom].numpy() @ lattice
            for image in itertools.product(range(-1, 2), repeat=3):
                d = mesh - (center + np.asarray(image) @ lattice)
                r2 = (d ** 2).sum(-1)
                harmonics = solid_harmonics(torch.from_numpy(d), 2)
                for L in range(3):
                    for w, width in enumerate(writer.widths):
                        gauss = np.exp(-r2 / (2 * width ** 2)) \
                            * float(writer.norms[L, w])
                        direct += np.einsum(
                            "cm,xyzm->cxyz",
                            coefficients[L][atom, :, w].detach().numpy(),
                            harmonics[L].numpy()) * gauss[None]
        error = np.abs(field.detach().numpy() - direct).max()
        assert error < 1e-6 * np.abs(direct).max()


class TestTheWrittenFieldMovesWithTheAtoms:
    """Translation and inversion covariance: what keeps the operator equivariant."""

    def test_a_translation_by_grid_points_rolls_the_field(self):
        writer = _writer()
        shape = (24, 24, 24)
        cell = torch.eye(3, dtype=torch.float64)[None] * 6.0
        positions = np.array([[0.1, 0.2, 0.3], [0.45, 0.7, 0.55]])
        shift = np.array([3, 5, 7])
        records, species, frac, mask = _inputs(positions)
        moved = torch.as_tensor((positions + shift / 24.0) % 1.0)[None]
        field = writer(records, species, frac, mask, cell, shape)[0]
        shifted = writer(records, species, moved, mask, cell, shape)[0]
        expected = torch.roll(field, shifts=tuple(int(s) for s in shift),
                              dims=(1, 2, 3))
        torch.testing.assert_close(shifted, expected, atol=1e-9, rtol=0)

    def test_an_inversion_flips_the_odd_blocks(self):
        """
        Inverting the atoms through the origin and negating every odd-L block
        of their records writes the inverted field: the records transform as
        their L says, the kernel follows.
        """
        writer = _writer()
        shape = (24, 24, 24)
        cell = torch.eye(3, dtype=torch.float64)[None] * 6.0
        positions = np.array([[0.1, 0.2, 0.3], [0.45, 0.7, 0.55]])
        records, species, frac, mask = _inputs(positions)
        inverted_records = records.clone()
        for L, index in irreps_blocks(LAYOUT[14]).items():
            if L % 2:
                inverted_records[..., torch.as_tensor(index.reshape(-1))] *= -1
        inverted = torch.as_tensor((-positions) % 1.0)[None]
        field = writer(records, species, frac, mask, cell, shape)[0]
        flipped = writer(inverted_records, species, inverted, mask, cell,
                         shape)[0]
        index = [(-np.arange(n)) % n for n in shape]
        expected = field[:, index[0]][:, :, index[1]][:, :, :, index[2]]
        torch.testing.assert_close(flipped, expected, atol=1e-9, rtol=0)


@pytest.fixture
def dataset_root(tmp_path):
    """Four Si cells with occupancy records in their CHGCAR and a TAUCAR."""
    rng = np.random.default_rng(3)
    for index, (side, shape) in enumerate([(5.0, (16, 16, 16)),
                                           (5.0, (16, 16, 16)),
                                           (6.0, (20, 20, 20)),
                                           (6.0, (20, 20, 20))]):
        directory = tmp_path / f"mat_{index:02d}"
        directory.mkdir()
        grid = FieldGrid(shape, np.eye(3) * side)
        structure = Poscar(np.eye(3) * side, ["Si"], [2], rng.random((2, 3)))
        density = 0.05 + 0.02 * rng.random(shape)
        records = [rng.standard_normal(LENGTH) for _ in range(2)]
        ChargeDensity(density, grid, structure).write(
            directory / "CHGCAR", augmentation=format_augmentation(records))
        tau = 2.87 * density ** (5.0 / 3.0) * 51.4
        KineticEnergyDensity(tau, grid, structure).write(directory / "TAUCAR")
    return tmp_path


class TestThePaw2tauOperatorTrainsAndReloads:
    """The task end to end: data, transform, training, checkpoint, predict."""

    def _operator(self, dataset):
        transform = OccupancyTransform.fit(dataset, LAYOUT, sets=1)
        operator = FieldOperator(
            "paw2tau", input_transform=dataset.input_transform,
            target_transform=dataset.target_transform, device="cpu",
            width=4, modes=3, n_layers=1, use_coordinates=False,
            in_channels=1, out_channels=1, site_layouts=LAYOUT,
            site_transform=transform, occupancy_channels=2, init_seed=0)
        operator.set_precision("float64")
        return operator

    def test_the_task_reads_the_records_from_its_input(self):
        task = resolve_task("paw2tau")
        assert task.site_input == "augmentation"
        assert task.site_field == "CHGCAR" == task.input_field

    def test_one_epoch_trains_and_the_checkpoint_predicts_the_same(
            self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="paw2tau", cache=True,
                                   dtype=torch.float64)
        dataset.fit_transforms()
        operator = self._operator(dataset)
        history = train(operator, dataset, epochs=1, verbose=False, seed=0,
                        validation=dataset)
        assert np.isfinite(history["train_loss"][0])

        restored = FieldOperator.from_state(operator.state(), device="cpu")
        restored.set_precision("float64")
        restored.model.load_state_dict(operator.model.state_dict())
        assert type(restored.model).__name__ == "OccupancyInputModel"
        assert restored.model.writer.channels == 2

        source, _ = dataset.load_fields(0)
        sites = dataset.site_records(0)
        np.testing.assert_allclose(restored.predict(source, sites=sites).data,
                                   operator.predict(source, sites=sites).data,
                                   atol=1e-12)

    def test_predict_refuses_a_field_without_its_records(self, dataset_root):
        dataset = FieldPairDataset(dataset_root, task="paw2tau", cache=True,
                                   dtype=torch.float64)
        dataset.fit_transforms()
        operator = self._operator(dataset)
        source, _ = dataset.load_fields(0)
        with pytest.raises(ValueError, match="occupancies"):
            operator.predict(source)
