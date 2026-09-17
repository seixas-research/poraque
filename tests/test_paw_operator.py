# -*- coding: utf-8 -*-
# file: test_paw_operator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The ``ext2paw`` operator: FNO backbone, grid-to-atom readout, equivariant head.

What each part promises, pinned one at a time:

* the record layout is read from the POTCAR's ``Non local Part``, and on the
  platinum data it puts a bulk site's non-zero values exactly where cubic
  symmetry allows;
* the readout is a property of the **field**, not of the grid it came on --- a
  grid-dependent readout failed at 184 % on held-out nanoparticles --- and it
  is the integral it claims to be;
* readout, neighbour expansion, head and the whole operator (with an
  equivariant backbone) rotate their output with the crystal;
* the target transform is exactly invertible and weights every L block alike;
* it trains in the one loop, reloads bit-identically, and ``poraque-train`` and
  ``poraque-inference`` both drive it.
"""

import math
import os
import sys
import warnings

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from poraque.fields.vasp.augmentation import (  # noqa: E402
    irreps_blocks,
    record_layout,
)
from poraque.ml.paw import (  # noqa: E402
    EquivariantHead,
    NeighbourExpansion,
    OccupancyTransform,
    SiteReadout,
    solid_harmonics,
)

LIBRARY = os.environ.get("PORAQUE_TEST_POTCAR_DIR") or None
PLATINUM = os.path.join(_ROOT, "data", "vasp", "structures", "structure_0000",
                        "CHGCAR")


def _wigner(rotation, lmax=4, seed=0):
    """Real Wigner matrices by least squares: Y_L(R u) = D_L Y_L(u)."""
    rng = np.random.default_rng(seed)
    u = torch.tensor(rng.normal(size=(500, 3)))
    u = u / torch.linalg.vector_norm(u, dim=1, keepdim=True)
    before = solid_harmonics(u, lmax)
    after = solid_harmonics(u @ rotation.T, lmax)
    return {L: torch.linalg.lstsq(before[L], after[L]).solution.T
            for L in range(lmax + 1)}


def _rotation():
    from scipy.spatial.transform import Rotation

    return torch.tensor(Rotation.from_euler("zyx", [0.7, -0.4, 1.1])
                        .as_matrix())


# ===================================================================== #
# The record layout
# ===================================================================== #
class TestTheLayoutComesFromTheNonLocalPart:
    """
    ``l  n_projectors  r_max`` per block, in file order --- not the Description,
    whose last line is the local potential.
    """

    def test_a_synthetic_dataset(self):
        from poraque.fields.vasp.potcar import Potcar
        from test_paw_profiles import potcar_text

        entry = Potcar.from_string(potcar_text(), parse_tables=True)[0]
        assert entry.projector_channels == (1,)
        assert entry.core_profile["projector_channels"] == [1]

    @pytest.mark.skipif(LIBRARY is None,
                        reason="set PORAQUE_TEST_POTCAR_DIR to a POTCAR library")
    @pytest.mark.parametrize("symbol, channels", [
        ("Pt", (2, 2, 0, 0, 1, 1)), ("Fe_pv", (1, 1, 2, 2, 0, 0)),
        ("Si", (0, 0, 1, 1))])
    def test_real_datasets(self, symbol, channels):
        from poraque.fields.vasp.potcar import Potcar

        path = os.path.join(LIBRARY, symbol, "POTCAR")
        if not os.path.exists(path):
            pytest.skip(f"no {symbol} in the library")
        assert Potcar.from_file(path, parse_tables=True)[0] \
            .projector_channels == channels

    def test_the_blocks_are_a_permutation_of_the_record(self):
        channels = (2, 2, 0, 0, 1, 1)
        blocks = irreps_blocks(channels)
        assert len(record_layout(channels)) == 138
        assert {L: index.shape for L, index in blocks.items()} == {
            0: (9, 1), 1: (8, 3), 2: (10, 5), 3: (4, 7), 4: (3, 9)}
        flat = np.concatenate([index.reshape(-1) for index in blocks.values()])
        assert sorted(flat.tolist()) == list(range(138))

    @pytest.mark.skipif(not os.path.exists(PLATINUM),
                        reason="the platinum dataset is not in this checkout")
    def test_a_bulk_site_is_non_zero_only_where_cubic_symmetry_allows(self):
        """L = 0 and L = 4, and L = 4 as Y_40 + sqrt(5/7) Y_44."""
        from poraque.fields.vasp.augmentation import occupancy_arrays
        from poraque.fields.vasp.volumetric import read_augmentation_blocks

        _, sets = read_augmentation_blocks(PLATINUM)
        records = occupancy_arrays(sets)[0][:, 0, :]
        blocks = irreps_blocks((2, 2, 0, 0, 1, 1))
        scale = np.abs(records).max()
        for L, index in blocks.items():
            size = np.abs(records[:, index]).max()
            assert (size > 1e-6 * scale) == (L in (0, 4)), L
        cubic = records[0, blocks[4][0]]
        assert cubic[8] / cubic[4] == pytest.approx(math.sqrt(5 / 7), rel=1e-5)


# ===================================================================== #
# The readout
# ===================================================================== #
class _Field:
    """
    An analytic field on a skewed cell, stored on any grid the way the cache
    stores one: as its **Fourier truncation** to the modes that grid holds.

    ``kmax`` sets how far up it reaches. With a large one, two grids of
    different size hold genuinely different bands of the same field, which is
    the situation the readout exists to be indifferent to.
    """

    def __init__(self, g_below=None, kmax=3, count=14, seed=1):
        self.cell = torch.tensor([[6.1, 0.0, 0.0], [0.9, 7.3, 0.0],
                                  [0.4, 0.6, 8.2]], dtype=torch.float64)
        reciprocal = 2 * math.pi * torch.linalg.inv(self.cell).T
        rng = np.random.default_rng(seed)
        k = torch.tensor(rng.integers(-kmax, kmax + 1, size=(count, 3)),
                         dtype=torch.float64)
        size = torch.linalg.vector_norm(k @ reciprocal, dim=1)
        keep = size > 0 if g_below is None else (size > 0) & (size < g_below)
        self.k, self.G = k[keep], (k @ reciprocal)[keep]
        self.amplitude = torch.tensor(rng.normal(size=int(keep.sum())))
        self.phase = torch.tensor(rng.uniform(0, 2 * math.pi,
                                              int(keep.sum())))

    def at(self, cartesian, include=None):
        terms = torch.cos(cartesian @ self.G.T + self.phase) * self.amplitude
        if include is not None:
            terms = terms * include
        return terms.sum(-1)

    def grid(self, shape):
        axes = [torch.arange(n, dtype=torch.float64) / n for n in shape]
        frac = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
        held = torch.all(self.k.abs() < torch.tensor(shape) / 2.0, dim=1)
        return self.at(frac @ self.cell, held.to(torch.float64))[None]


class TestTheReadoutIsAPropertyOfTheFieldNotTheGrid:
    """
    The measurement behind it: a readout integrated on the stored grid scored
    3 % on nanoparticles it had seen and 184 % on ones it had not, whose grid
    spacing is 0.31 Å against 0.12 Å in bulk. Restricted to one band and
    integrated on one spacing, the same features are 4.9 %.
    """

    def test_two_grids_give_the_same_features(self):
        """
        The coarse grid holds modes up to |k| = 7 and the fine one up to 19,
        so the two arrays differ above the band --- and must agree on it.
        """
        field = _Field(kmax=12, count=60)
        coarse_grid, fine_grid = (16, 18, 20), (40, 44, 52)
        held = torch.all(field.k.abs() < torch.tensor(coarse_grid) / 2.0,
                         dim=1)
        assert not held.all()
        assert float(torch.linalg.vector_norm(field.G[~held], dim=1).min()) \
            > 6.5

        readout = SiteReadout(g_max=6.5).double()
        frac = torch.tensor([[0.13, 0.41, 0.77], [0.5, 0.52, 0.1]],
                            dtype=torch.float64)
        coarse = readout(field.grid(coarse_grid), field.cell, frac)
        fine = readout(field.grid(fine_grid), field.cell, frac)
        for L in range(5):
            assert torch.allclose(coarse[L], fine[L], rtol=1e-10, atol=1e-12)

    def test_it_is_the_integral_it_claims_to_be(self):
        from poraque.ml.paw import READOUT_RADIUS

        field = _Field(g_below=6.0)
        readout = SiteReadout(g_max=6.5).double()
        frac = torch.tensor([[0.13, 0.41, 0.77]], dtype=torch.float64)
        got = readout(field.grid((30, 34, 40)), field.cell, frac)

        h = 0.05
        t = torch.arange(-READOUT_RADIUS, READOUT_RADIUS + h / 2, h,
                         dtype=torch.float64)
        d = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), -1)
        d = d.reshape(-1, 3)
        r = torch.linalg.vector_norm(d, dim=-1)
        d, r = d[r < READOUT_RADIUS], r[r < READOUT_RADIUS]
        values = field.at(d + frac @ field.cell)
        window = 0.5 * (1 + torch.cos(math.pi * r / READOUT_RADIUS))
        radial = torch.exp(-r[:, None] ** 2 / (2 * readout.widths ** 2)) \
            * window[:, None]
        harmonics = solid_harmonics(d, 4)
        for L, tolerance in ((0, 1e-5), (1, 1e-4), (2, 1e-3), (4, 5e-3)):
            reference = torch.einsum("p,pn,pm->nm", values, radial,
                                     harmonics[L]) * h ** 3
            reference = reference * readout.norms[L][:, None]
            error = torch.linalg.norm(got[L][0] - reference.reshape(-1, 2 * L + 1))
            assert float(error / torch.linalg.norm(reference)) < tolerance, L

    def test_a_grid_too_coarse_for_the_band_is_refused(self):
        field = _Field(g_below=6.0)
        readout = SiteReadout(g_max=6.5).double()
        with pytest.raises(ValueError, match="narrower band"):
            readout(field.grid((12, 12, 12)), field.cell,
                    torch.zeros(1, 3, dtype=torch.float64))

    def test_the_harmonics_are_the_convention_vasp_writes_in(self):
        """scipy's real harmonics, m = -L..L, which the records were pinned to."""
        from scipy.special import sph_harm_y

        rng = np.random.default_rng(2)
        u = rng.normal(size=(100, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        ours = solid_harmonics(torch.tensor(u), 4)
        theta, phi = np.arccos(u[:, 2]), np.arctan2(u[:, 1], u[:, 0])
        for L in range(5):
            for m in range(-L, L + 1):
                c = sph_harm_y(L, abs(m), theta, phi)
                expected = (c.real if m == 0 else np.sqrt(2) * (-1) ** m
                            * (c.real if m > 0 else c.imag))
                np.testing.assert_allclose(ours[L][:, L + m].numpy(), expected,
                                           atol=1e-12)


# ===================================================================== #
# Equivariance
# ===================================================================== #
class TestEveryPartRotatesWithTheCrystal:
    """``D^L`` in, ``D^L`` out, for every L block."""

    def test_the_readout_and_the_neighbours(self):
        field = _Field(g_below=6.0)
        rotation = _rotation()
        D = _wigner(rotation)
        rotated_cell = field.cell @ rotation.T
        frac = torch.tensor([[0.13, 0.41, 0.77], [0.6, 0.2, 0.3],
                             [0.9, 0.8, 0.5]], dtype=torch.float64)
        values = field.grid((30, 34, 40))
        readout = SiteReadout(g_max=6.5).double()
        neighbours = NeighbourExpansion().double()
        a, b = readout(values, field.cell, frac), readout(values,
                                                          rotated_cell, frac)
        c, d = neighbours(field.cell, frac), neighbours(rotated_cell, frac)
        for L in range(5):
            assert torch.allclose(b[L], a[L] @ D[L].T, atol=1e-10)
            assert torch.allclose(d[L], c[L] @ D[L].T, atol=1e-10)

    def test_the_head(self):
        torch.manual_seed(0)
        pairs = {0: 3, 1: 2, 2: 4, 4: 1}
        head = EquivariantHead({L: 5 for L in range(5)}, pairs, sets=2).double()
        features = {L: torch.randn(7, 5, 2 * L + 1, dtype=torch.float64)
                    for L in range(5)}
        D = _wigner(_rotation())
        plain = head(features)
        rotated = head({L: f @ D[L].T for L, f in features.items()})
        for L in pairs:
            assert torch.allclose(rotated[L], plain[L] @ D[L].T, atol=1e-10)

    def test_the_whole_operator_with_an_equivariant_backbone(self):
        """
        A quarter turn about z of a cubic cell maps the grid onto itself, so the
        rotated crystal is an index permutation of the same arrays --- and the
        predicted occupancies of every atom must come out rotated by D^L.
        """
        from poraque.ml.fno import FNO3d, set_precision
        from poraque.ml.paw import PAWOperatorModel

        torch.manual_seed(3)
        backbone = FNO3d(width=6, modes=4, n_layers=2, projection_channels=8,
                         use_coordinates=False, equivariant=True, n_radial=6)
        channels = (2, 1)
        model = PAWOperatorModel(backbone, {78: channels}, g_max=4.0)
        set_precision(model, "float64")
        model.eval()

        n, length = 16, 6.0
        cell = (torch.eye(3, dtype=torch.float64) * length)[None]
        x = torch.randn(1, 1, n, n, n, dtype=torch.float64)
        frac = torch.tensor([[[0.10, 0.20, 0.30], [0.55, 0.35, 0.80]]],
                            dtype=torch.float64)
        species = torch.tensor([[78, 78]])
        mask = torch.ones_like(species, dtype=torch.bool)

        # r -> R r with R = (x, y, z) -> (-y, x, z): f'(r) = f(R^T r), so
        # f'[i, j, k] = f[j, -i, k]; and frac' = R frac.
        index = torch.arange(n)
        rotated_x = x[:, :, :, :, :][:, :, index[:, None], (-index[None, :]) % n, :]
        rotated_x = rotated_x.permute(0, 1, 3, 2, 4)
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0]], dtype=torch.float64)
        rotated_frac = (frac @ rotation.T) % 1.0

        _, before = model(x, cell, {"species": species, "positions": frac,
                                    "atom_mask": mask})
        _, after = model(rotated_x, cell, {"species": species,
                                           "positions": rotated_frac,
                                           "atom_mask": mask})
        D = _wigner(rotation)
        for L, block in irreps_blocks(channels).items():
            flat = torch.as_tensor(block.reshape(-1))
            b = before[0][:, :, flat].reshape(2, 1, *block.shape)
            a = after[0][:, :, flat].reshape(2, 1, *block.shape)
            assert torch.allclose(a, b @ D[L].T, atol=1e-8), L


# ===================================================================== #
# The target transform
# ===================================================================== #
class _Sites:
    """A dataset of site targets alone, for fitting the transform."""

    def __init__(self, channels=(2, 1), atoms=5, structures=6, seed=0):
        rng = np.random.default_rng(seed)
        length = len(record_layout(channels))
        self.items = []
        for _ in range(structures):
            values = rng.normal(size=(atoms, 2, length)) * 0.3
            values[:, :, irreps_blocks(channels)[0][:, 0]] += 5.0
            self.items.append({"occupancies": values,
                               "lengths": np.full(atoms, length),
                               "species": np.full(atoms, 78)})

    def __len__(self):
        return len(self.items)

    def site_targets(self, index):
        return self.items[index]


class TestTheTargetTransform:
    """Centre L = 0, scale each (set, L, pair); exactly invertible."""

    def test_it_inverts_exactly_and_centres_only_l0(self):
        channels = (2, 1)
        data = _Sites(channels)
        transform = OccupancyTransform.fit(data, {78: channels}, sets=2)
        records = torch.as_tensor(np.stack([item["occupancies"]
                                            for item in data.items]))
        species = torch.full(records.shape[:2], 78)
        z = transform.normalize(records, species)
        assert torch.allclose(transform.inverse(z, species), records,
                              atol=1e-12)
        blocks = irreps_blocks(channels)
        l0 = torch.as_tensor(blocks[0][:, 0])
        assert np.abs(transform.mean[78][:, np.concatenate(
            [b.reshape(-1) for L, b in blocks.items() if L])]).max() == 0.0
        assert abs(float(z[..., l0].mean())) < 1e-10

    def test_the_scale_is_shared_across_m(self):
        channels = (2, 1)
        transform = OccupancyTransform.fit(_Sites(channels), {78: channels}, 2)
        for L, index in irreps_blocks(channels).items():
            block = transform.scale[78][:, index]
            assert np.allclose(block, block[:, :, :1])

    def test_every_l_block_weighs_the_same_in_the_loss(self):
        channels = (2, 1)
        transform = OccupancyTransform.fit(_Sites(channels), {78: channels}, 2)
        _, _, weight = transform._lookup(torch.tensor([[78]]), "cpu",
                                         torch.float64)
        blocks = irreps_blocks(channels)
        shares = [float(weight[0, 0, 0, torch.as_tensor(index.reshape(-1))]
                        .sum()) for index in blocks.values()]
        assert shares == pytest.approx([1.0 / len(blocks)] * len(blocks))

    def test_a_symmetry_zero_block_is_not_divided_by_its_round_off(self):
        channels = (2, 1)
        data = _Sites(channels)
        four = irreps_blocks(channels)[4].reshape(-1)
        for item in data.items:
            item["occupancies"][:, :, four] = 1e-17
        transform = OccupancyTransform.fit(data, {78: channels}, 2)
        assert transform.scale[78][:, four].min() > 1e-8

    def test_the_state_round_trips(self):
        channels = (2, 1)
        transform = OccupancyTransform.fit(_Sites(channels), {78: channels}, 2)
        back = OccupancyTransform.from_state_dict(transform.state_dict())
        assert back.layouts == transform.layouts
        np.testing.assert_array_equal(back.scale[78], transform.scale[78])

    def test_records_of_the_wrong_length_are_refused(self):
        with pytest.raises(ValueError, match="not the one the data"):
            OccupancyTransform.fit(_Sites((2, 1)), {78: (2, 2, 1)}, 2)
        with pytest.raises(ValueError, match="no projector channels"):
            OccupancyTransform.fit(_Sites((2, 1)), {14: (2, 1)}, 2)


# ===================================================================== #
# Training, persistence, the scripts
# ===================================================================== #
def _synthetic_runs(root, count=4):
    from test_data_sources import write_calculation
    from test_ext2paw import _with_records
    from test_paw_profiles import potcar_text

    potcar = os.path.join(root, "Si.POTCAR")
    with open(potcar, "w") as handle:
        handle.write(potcar_text())
    for index in range(count):
        _with_records(write_calculation(
            os.path.join(root, "runs", f"structure_{index:04d}"),
            shape=(16, 16, 16), seed=index, potcar=potcar), spin=True)
    return os.path.join(root, "runs")


class TestTheOperatorTrainsInTheOneLoop:
    """Field objective plus occupancy term, one optimiser, one checkpoint."""

    @pytest.fixture
    def dataset(self, tmp_path):
        from poraque.data import build_field_cache
        from poraque.ml.data import FieldPairDataset

        runs = _synthetic_runs(str(tmp_path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            build_field_cache([runs], str(tmp_path / "cache"), resolution=16,
                              spin=True, charges={"Si": 4.0},
                              log=lambda *_: None)
        return FieldPairDataset(str(tmp_path / "cache"), "ext2paw")

    @staticmethod
    def _operator(dataset):
        from poraque.ml import FieldOperator

        layouts = {14: (1,)}
        return FieldOperator(
            "ext2paw", width=6, modes=4, n_layers=1, projection_channels=8,
            device="cpu", in_channels=1, out_channels=2,
            site_layouts=layouts, readout_g_max=3.0,
            site_transform=OccupancyTransform.fit(dataset, layouts, sets=2),
            init_seed=0)

    def test_it_learns_and_reports_the_occupancy_error(self, dataset):
        from poraque.ml import train

        operator = self._operator(dataset)
        history = train(operator, dataset, epochs=25, batch_size=2,
                        learning_rate=5e-3, validation=dataset, eval_every=5,
                        verbose=False)
        assert history["train_loss"][-1] < history["train_loss"][0]
        assert len(history["val_occupancy_error"]) == 5
        assert history["val_occupancy_error"][-1] \
            < history["val_occupancy_error"][0]
        assert "occ" in history["val_metric"]

    def test_the_checkpoint_rebuilds_the_head_and_its_transform(self, dataset,
                                                                tmp_path):
        from poraque.fields import ExternalPotential
        from poraque.ml import load_bundle, save_bundle

        operator = self._operator(dataset)
        path = save_bundle(str(tmp_path / "m.poraque"), {"ext2paw": operator})
        back = load_bundle(path, "ext2paw", device="cpu")
        potential = ExternalPotential.read(
            os.path.join(dataset.root, "structure_0000", "EXTCAR"))
        a = operator.predict_occupancies(potential)
        b = back.predict_occupancies(potential)
        assert len(a) == 2 and len(a[0]) == 2 and a[0][0].shape == (6,)
        for first, second in zip(a, b):
            for x, y in zip(first, second):
                np.testing.assert_array_equal(x, y)

    def test_without_its_layout_the_task_is_refused(self):
        from poraque.ml import FieldOperator

        with pytest.raises(ValueError, match="site_layouts"):
            FieldOperator("ext2paw", width=4, modes=2, n_layers=1,
                          device="cpu")

    def test_a_field_operator_has_no_occupancies_to_give(self, dataset):
        from poraque.fields import ExternalPotential
        from poraque.ml import FieldOperator

        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 device="cpu")
        potential = ExternalPotential.read(
            os.path.join(dataset.root, "structure_0000", "EXTCAR"))
        with pytest.raises(ValueError, match="no occupancy head"):
            operator.predict_occupancies(potential)


class TestTheScriptsDriveIt:
    """``poraque-train`` trains it; ``poraque-inference`` writes its records."""

    @pytest.fixture
    def trained(self, tmp_path):
        import yaml

        import poraque_train

        runs = _synthetic_runs(str(tmp_path), count=5)
        config = tmp_path / "train.yaml"
        config.write_text(yaml.safe_dump({
            "task": {"type": "all", "name": "paw_run"},
            "data": {"data_paths": [runs], "cache": str(tmp_path / "c"),
                     "resolution": 16, "delta_density": False,
                     "paw_source": "material", "spin": "auto"},
            "model": {"width": 4, "modes": 2, "n_layers": 1,
                      "projection_channels": 4, "use_coordinates": False,
                      "paw": {"enable": True, "width": 6, "modes": 4,
                              "n_layers": 1}},
            "training": {"epochs": 3, "batch_size": 2, "valid_fraction": 0.4,
                         "eval_epoch": 1, "device": "cpu"},
            "output": {"root": str(tmp_path / "models"),
                       "plot_figures": False, "write_pdf_report": False},
        }))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poraque_train.run(["--config", str(config)])
        return tmp_path, runs

    def test_an_all_run_with_the_block_trains_all_three(self, trained):
        import json

        from poraque.ml import bundle_tasks, read_bundle

        tmp_path, _ = trained
        bundle = tmp_path / "models" / "paw_run" / "paw_run.poraque"
        assert bundle_tasks(str(bundle)) == ["chg2tau", "ext2chg", "ext2paw"]
        site = read_bundle(str(bundle))["model_state_dict"]["ext2paw"]["site"]
        assert site["layouts"] == {"14": [1]}
        assert site["readout_g_max"] > 0 and site["transform"]

        log = (tmp_path / "models" / "paw_run" / "log"
               / "paw_run.log").read_text()
        assert "occupancy layout: Z=14  channels p  -> 6 values" in log
        assert "val occ" in log
        assert "augmentation occupancies (relative RMS" in log
        metrics = json.loads((tmp_path / "models" / "paw_run" / "log"
                              / "paw_run.json").read_text())
        assert "occupancy_relative_rms" in json.dumps(metrics)

    def test_inference_writes_the_operators_own_records(self, trained):
        import shutil

        import poraque_inference

        from poraque.fields import ExternalPotential
        from poraque.fields.vasp.augmentation import occupancy_arrays
        from poraque.fields.vasp.volumetric import read_augmentation_blocks
        from poraque.ml import load_bundle

        tmp_path, runs = trained
        bundle = str(tmp_path / "models" / "paw_run" / "paw_run.poraque")
        new = tmp_path / "new"
        new.mkdir()
        for name in ("POSCAR", "POTCAR", "INCAR"):
            shutil.copy(os.path.join(runs, "structure_0000", name), new / name)
        out = tmp_path / "predicted"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = poraque_inference.predict([
                str(new), "--models", bundle, "--output", str(out),
                "--grid", "16", "16", "16", "--add-paw",
                "--functional", "skip", "--device", "cpu"])
        assert results["paw_augmentation"]["records"] == 4       # 2 atoms x 2 sets
        assert results["paw_augmentation"]["sets"] == 2

        _, blocks = read_augmentation_blocks(str(out / "CHGCAR"))
        written, _ = occupancy_arrays(blocks)
        operator = load_bundle(bundle, "ext2paw", device="cpu")
        predicted = operator.predict_occupancies(
            ExternalPotential.read(str(out / "EXTCAR")))
        np.testing.assert_allclose(written[:, 0, :], np.stack(predicted[0]),
                                   atol=1e-6 * np.abs(written).max())

    def test_naming_the_model_source_without_one_is_an_error(self, trained,
                                                            tmp_path):
        import shutil

        import poraque_inference

        from poraque.ml import load_bundle, save_bundle

        base, runs = trained
        bundle = str(base / "models" / "paw_run" / "paw_run.poraque")
        chain = {task: load_bundle(bundle, task, device="cpu")
                 for task in ("ext2chg", "chg2tau")}
        without = save_bundle(str(tmp_path / "chain.poraque"), chain)
        new = tmp_path / "bare"
        new.mkdir()
        for name in ("POSCAR", "POTCAR", "INCAR"):
            shutil.copy(os.path.join(runs, "structure_0000", name), new / name)
        with pytest.raises(SystemExit, match="holds no ext2paw"):
            poraque_inference.predict([
                str(new), "--models", without, "--output",
                str(tmp_path / "o"), "--grid", "16", "16", "16", "--add-paw",
                "--paw-source", "model", "--functional", "skip",
                "--device", "cpu"])
