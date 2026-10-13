# -*- coding: utf-8 -*-
# file: test_paw_operator.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The ``ext2chg`` operator: FNO backbone, grid-to-atom readout, equivariant head.

What each part promises, pinned one at a time:

* the record layout is read from the POTCAR's ``Non local Part``, and on the
  platinum data it puts a bulk site's non-zero values exactly where cubic
  symmetry allows;
* the readout is a property of the **field**, not of the grid it came on --- a
  grid-dependent readout failed at 184 % on held-out nanoparticles --- and it
  is the integral it claims to be;
* readout, neighbor expansion, head and the whole operator (with an
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
    NeighborExpansion,
    OccupancyTransform,
    SiteReadout,
    solid_harmonics,
)

LIBRARY = os.environ.get("PORAQUE_TEST_POTCAR_DIR") or None
PLATINUM = os.path.join(_ROOT, "data", "vasp", "structures", "structure_0000",
                        "CHGCAR")


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "mps", marks=[pytest.mark.gpu, pytest.mark.skipif(
        not torch.backends.mps.is_available(), reason="requires MPS")])])
def test_hydrogen_only_batch_in_a_multielement_operator(device):
    """CHON output pads to 33; a hydrogen-only batch has 15 targets."""
    from poraque.ml import FieldOperator
    from poraque.ml.paw import occupancy_error
    from poraque.ml.training import validate_occupancies

    layouts = {1: (0, 0, 1), 6: (0, 0, 1, 1)}
    lengths = {z: len(record_layout(channels)) for z, channels in layouts.items()}
    transform = OccupancyTransform(
        layouts, 1, {z: np.zeros((1, n)) for z, n in lengths.items()},
        {z: np.ones((1, n)) for z, n in lengths.items()})
    operator = FieldOperator(
        "ext2chg", device=device, width=4, modes=2, n_layers=1,
        use_coordinates=False, site_layouts=layouts, readout_g_max=2,
        site_transform=transform)
    batch = {"input": torch.randn(1, 1, 8, 8, 8),
             "cell": torch.eye(3)[None] * 4,
             "species": torch.ones(1, 1, dtype=torch.long),
             "positions": torch.zeros(1, 1, 3),
             "atom_mask": torch.ones(1, 1, dtype=torch.bool),
             "paw": torch.ones(1, 1, 1, 15),
             "paw_mask": torch.ones(1, 1, 15, dtype=torch.bool)}
    batch = {k: v.to(device) for k, v in batch.items()}
    sites = {k: batch[k] for k in ("species", "positions", "atom_mask")}
    _, predicted = operator.model(batch["input"], batch["cell"], sites=sites)
    assert predicted.shape[-1] == 33
    wanted = transform.normalize(batch["paw"], batch["species"])
    loss = transform.loss(predicted, wanted, batch["species"], batch["atom_mask"])
    reference = transform.loss(predicted[..., :15], wanted,
                               batch["species"], batch["atom_mask"])
    torch.testing.assert_close(loss, reference)
    loss.backward()
    assert operator.model.heads["1"].scalar_out.weight.grad is not None
    physical = transform.inverse(predicted.detach(), batch["species"])
    squared, norm = occupancy_error(physical, batch["paw"], batch["paw_mask"])
    error, measured_loss = validate_occupancies(operator, [batch])
    assert error == pytest.approx(float(torch.sqrt(squared / norm)), rel=1e-5)
    assert measured_loss == pytest.approx(float(loss), rel=1e-5)


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

    @pytest.mark.parametrize("g_max", [6.5, 2.239581])
    def test_two_grids_give_the_same_features(self, g_max):
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

        readout = SiteReadout(g_max=g_max).double()
        frac = torch.tensor([[0.13, 0.41, 0.77], [0.5, 0.52, 0.1]],
                            dtype=torch.float64)
        coarse = readout(field.grid(coarse_grid), field.cell, frac)
        fine = readout(field.grid(fine_grid), field.cell, frac)
        for L in range(5):
            assert torch.allclose(coarse[L], fine[L], rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("g_max", [6.5, 2.239581])
    def test_it_is_the_integral_it_claims_to_be(self, g_max):
        from poraque.ml.paw import READOUT_RADIUS

        field = _Field(g_below=g_max - 0.1, kmax=2, count=40)
        readout = SiteReadout(g_max=g_max).double()
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

    def test_low_band_constant_has_no_position_error_or_multipoles(self):
        from scipy.integrate import quad
        from poraque.ml.paw import READOUT_RADIUS

        readout = SiteReadout(g_max=2.239581).double()
        cell = torch.eye(3, dtype=torch.float64) * 6
        frac = torch.tensor([[0, 0, 0], [0.137, 0.416, 0.771]], dtype=torch.float64)
        got = readout(torch.ones(2, 12, 12, 12, dtype=torch.float64), cell, frac)
        reference = []
        for width, norm in zip(readout.widths, readout.norms[0]):
            integral = quad(lambda r: r ** 2 * math.exp(-r ** 2 / (2 * float(width) ** 2))
                            * 0.5 * (1 + math.cos(math.pi * r / READOUT_RADIUS)),
                            0, READOUT_RADIUS, epsabs=1e-12)[0]
            reference.append(math.sqrt(4 * math.pi) * float(norm) * integral)
        expected = torch.tensor(reference * 2, dtype=torch.float64)
        torch.testing.assert_close(got[0][..., 0], expected.expand(2, -1),
                                   atol=1e-12, rtol=1e-12)
        for L in range(1, 5):
            assert float(got[L].abs().max()) < 1e-12

    def test_low_band_field_and_position_gradients(self):
        torch.manual_seed(1)
        readout = SiteReadout(2, lmax=2).double()
        fields = torch.randn(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
        frac = torch.tensor([[0.13, 0.41, 0.77]], dtype=torch.float64,
                            requires_grad=True)
        cell = torch.eye(3, dtype=torch.float64) * 4
        def project(f, positions):
            return tuple(readout(f, cell, positions).values())
        assert torch.autograd.gradcheck(project, (fields, frac), fast_mode=True)

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

    @pytest.mark.parametrize("g_max", [6.5, 2.239581])
    def test_the_readout_and_the_neighbors(self, g_max):
        field = _Field(g_below=6.0)
        rotation = _rotation()
        D = _wigner(rotation)
        rotated_cell = field.cell @ rotation.T
        frac = torch.tensor([[0.13, 0.41, 0.77], [0.6, 0.2, 0.3],
                             [0.9, 0.8, 0.5]], dtype=torch.float64)
        values = field.grid((30, 34, 40))
        readout = SiteReadout(g_max=g_max).double()
        neighbors = NeighborExpansion().double()
        a, b = readout(values, field.cell, frac), readout(values,
                                                          rotated_cell, frac)
        c, d = neighbors(field.cell, frac), neighbors(rotated_cell, frac)
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

    def site_records(self, index):
        return self.items[index]


class TestTheTargetTransform:
    """Center L = 0, scale each (set, L, pair); exactly invertible."""

    def test_it_inverts_exactly_and_centers_only_l0(self):
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
    from test_ext2chg import _with_records
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
    """Field objective plus occupancy term, one optimizer, one checkpoint."""

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
        return FieldPairDataset(str(tmp_path / "cache"), "ext2chg")

    @staticmethod
    def _operator(dataset):
        from poraque.ml import FieldOperator

        layouts = {14: (1,)}
        return FieldOperator(
            "ext2chg", width=6, modes=4, n_layers=1, projection_channels=8,
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

    def test_the_best_epoch_is_chosen_on_the_objective_it_trained(
            self, dataset):
        """
        The score was the field error plus w x the *physical* relative RMS of
        the records. 99 % of the records' squared norm is the charge set's
        L = 0 block, so the score could not see the L > 0 blocks the loss
        weights equally --- a held-out slab read 3 % while its L = 2 block was
        35 % wrong --- nor the magnetization set at all. The score is now the
        held-out objective: the field error plus w x the training term.
        """
        from poraque.ml import train

        operator = self._operator(dataset)
        history = train(operator, dataset, epochs=4, batch_size=2,
                        learning_rate=5e-3, validation=dataset, eval_every=2,
                        occupancy_weight=2.5, verbose=False)
        for score, field, loss in zip(history["val_error"],
                                      history["val_field_error"],
                                      history["val_occupancy_loss"]):
            assert score == pytest.approx(field + 2.5 * loss)
        assert "occ loss" in history["val_metric"]

    def test_charge_constraint_uses_absolute_density_and_selects_on_charge(
            self, dataset):
        from poraque.ml import train
        from poraque.ml.data import FieldPairDataset
        from poraque.ml.losses import PhysicsInformedLoss
        from poraque.ml.physics import integrate
        from test_pipeline_defaults import synthetic_library

        residuals = FieldPairDataset(
            dataset.root, "ext2chg",
            baseline=synthetic_library(element="Si", n_electrons=4.))
        source, target = residuals.fit_transforms()
        operator = self._operator(residuals)
        operator.input_transform, operator.target_transform = source, target
        reference_counts = sorted(float(field.grid.integrate(field.total))
                                  for _, field in (residuals.load_fields(i)
                                                   for i in range(len(residuals))))
        seen_counts = []

        class RecordingLoss(PhysicsInformedLoss):
            def forward(self, prediction, target, **kwargs):
                count = integrate(kwargs["physical_target"][:, :1], kwargs["cell"])
                seen_counts.extend(count.detach().cpu().tolist())
                return super().forward(prediction, target, **kwargs)

        history = train(operator, residuals, epochs=2, batch_size=2,
                        validation=residuals, eval_every=1, early_stopping=2,
                        occupancy_weight=2.5, learning_rate=3e-3, verbose=False,
                        loss=RecordingLoss(task="ext2chg", electron_count_weight=1.5))
        np.testing.assert_allclose(sorted(seen_counts), sorted(reference_counts * 2),
                                   rtol=2e-6)
        assert np.isfinite(history["train_electron_count_loss"]).all()
        assert max(history["train_electron_count_loss"]) > 0
        assert len(history["val_electron_count_error"]) == 2
        for score, field, paw, count in zip(history["val_error"],
                                           history["val_field_error"],
                                           history["val_occupancy_loss"],
                                           history["val_electron_count_loss"]):
            assert score == pytest.approx(field + 2.5 * paw + 1.5 * count)
        assert history["best_error"] == min(history["val_error"])
        assert "N loss" in history["val_metric"]

    def test_the_held_out_term_is_the_training_term(self, dataset):
        """One batch, so the pooled mean must equal the loss itself."""
        from poraque.ml.data import collate_fields
        from poraque.ml.training import validate_occupancies

        operator = self._operator(dataset)
        samples = [dataset[index] for index in range(len(dataset))]
        batch = collate_fields(samples)
        loader = [batch]
        _, measured = validate_occupancies(operator, loader)

        transform = operator.site_transform
        with torch.no_grad():
            _, occupancy = operator.model(batch["input"], batch["cell"],
                                          sites=batch)
            expected = transform.loss(
                occupancy, transform.normalize(batch["paw"], batch["species"]),
                batch["species"], batch["atom_mask"].to(occupancy.dtype))
        assert measured == pytest.approx(float(expected), rel=1e-6)

    @pytest.mark.parametrize("neighbor_min", [None, 0.6])
    @pytest.mark.parametrize("legacy_metadata", [False, True])
    def test_the_checkpoint_rebuilds_the_head_and_its_transform(self, dataset,
                                                                tmp_path, neighbor_min,
                                                                legacy_metadata):
        from poraque.fields import ExternalPotential
        from poraque.ml import load_bundle, save_bundle

        operator = self._operator(dataset)
        if neighbor_min is not None:
            centers = operator.model.neighbors.centers
            centers.copy_(torch.linspace(neighbor_min, float(centers[-1]), len(centers),
                                         dtype=centers.dtype, device=centers.device))
        path = save_bundle(str(tmp_path / "m.poraque"), {"ext2chg": operator})
        if legacy_metadata:
            from poraque.ml import read_bundle
            state = read_bundle(path)
            state["model_state_dict"]["ext2chg"]["site"].pop("neighbor_min")
            torch.save(state, path)
        back = load_bundle(path, "ext2chg", device="cpu")
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
            FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                          device="cpu")

    def test_a_field_operator_has_no_occupancies_to_give(self, dataset):
        from poraque.fields import ExternalPotential
        from poraque.ml import FieldOperator

        operator = FieldOperator("ext2pseudo", width=4, modes=2, n_layers=1,
                                 device="cpu")
        potential = ExternalPotential.read(
            os.path.join(dataset.root, "structure_0000", "EXTCAR"))
        with pytest.raises(ValueError, match="no occupancy head"):
            operator.predict_occupancies(potential)


class TestTheScriptsDriveIt:
    """``poraque-train`` trains it; ``poraque-inference`` writes its records."""

    @staticmethod
    def _train(tmp_path, runs, name, occupancies=True):
        """
        One synthetic ``poraque-train`` run: the density operator (with its
        per-atom head unless ``occupancies`` is false) and the grid-only
        kinetic operator.
        """
        import yaml

        import poraque_train

        head = ({"neighbor_min": 0.6} if occupancies else False)
        config = tmp_path / f"{name}.yaml"
        config.write_text(yaml.safe_dump({
            "run": {"name": name, "output": str(tmp_path / "models")},
            "data": {"path": runs, "atoms": None, "spin": "auto",
                     "grid": {"resolution": 16},
                     "cache": {"dir": str(tmp_path / "c")}},
            "ext2chg": {"target": "full",
                        "model": {"width": 6, "modes": 4, "layers": 1,
                                  "projection_channels": 4},
                        "occupancies": head},
            "chg2tau": {"occupancies": False,
                        "model": {"width": 4, "modes": 2, "layers": 1,
                                  "projection_channels": 4}},
            "training": {"epochs": 3, "batch_size": 2,
                         "validation": {"fraction": 0.4, "every": 1}},
            "runtime": {"device": "cpu"},
            "report": {"figures": False, "pdf": False},
        }))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poraque_train.run(["--config", str(config)])
        return str(tmp_path / "models" / name / f"{name}{'.poraque'}")

    @pytest.fixture
    def trained(self, tmp_path):
        runs = _synthetic_runs(str(tmp_path), count=5)
        self._train(tmp_path, runs, "paw_run")
        return tmp_path, runs

    def test_a_run_with_both_sections_trains_both_operators(self, trained):
        """
        The density operator with its per-atom head beside the grid-only
        kinetic operator, in one bundle. (Before 2026-10-13 an ``all`` run
        with ``model.paw`` on trained three operators; a run now trains one
        per section.)
        """
        import json

        from poraque.ml import bundle_tasks, read_bundle

        tmp_path, _ = trained
        bundle = tmp_path / "models" / "paw_run" / "paw_run.poraque"
        assert bundle_tasks(str(bundle)) == ["ext2chg", "pseudo2tau"]
        site = read_bundle(str(bundle))["model_state_dict"]["ext2chg"]["site"]
        assert site["layouts"] == {"14": [1]}
        assert site["readout_g_max"] > 0 and site["transform"]
        assert site["neighbor_min"] == pytest.approx(0.6)

        log = (tmp_path / "models" / "paw_run" / "log"
               / "paw_run.log").read_text()
        assert "occupancy layout: Z=14  channels p  -> 6 values" in log
        assert "val occ" in log
        assert "[OCCUPANCIES]" in log
        metrics = json.loads((tmp_path / "models" / "paw_run" / "log"
                              / "paw_run.json").read_text())
        assert "occupancy_relative_rms" in json.dumps(metrics)

    def test_the_error_is_reported_per_set_and_per_L(self, trained):
        """
        The pooled number is 99 % the charge set's L = 0 block, so alone it
        read 3 % on a held-out slab whose L = 2 block was 35 % wrong.
        """
        import json

        tmp_path, _ = trained
        folder = tmp_path / "models" / "paw_run" / "log"
        log = (folder / "paw_run.log").read_text()
        assert "magnetization L = 2" in log

        def find(node):
            if isinstance(node, dict):
                # The resolved config carries an `ext2chg.occupancies` block
                # too; the summary is the one with per-split entries.
                found = node.get("occupancies")
                if isinstance(found, dict) and "train" in found:
                    return found
                for value in node.values():
                    found = find(value)
                    if found:
                        return found
            if isinstance(node, list):
                for value in node:
                    found = find(value)
                    if found:
                        return found
            return None

        summary = find(json.loads((folder / "paw_run.json").read_text()))
        # p x p couples to L = 0 and 2 (|l - l'| to l + l' in steps of 2), in
        # both sets of a spin run.
        for split in ("train", "validation"):
            blocks = summary[split]["blocks"]
            assert sorted(blocks) == ["set 0", "set 1"]
            assert sorted(blocks["set 0"]) == ["L0", "L2"]
            assert 0.0 <= blocks["set 0"]["L0"]["relative_rms"]

    def test_inference_accepts_a_standalone_paw_bundle(self, trained):
        import shutil
        import poraque_inference
        from poraque.ml import read_bundle
        from poraque.fields.vasp.volumetric import read_augmentation_blocks
        from poraque.fields.vasp.augmentation import occupancy_arrays

        tmp_path, runs = trained
        original = tmp_path / "models" / "paw_run" / "paw_run.poraque"
        state = read_bundle(str(original))
        state["model_state_dict"] = {"ext2chg": state["model_state_dict"]["ext2chg"]}
        bundle = tmp_path / "paw_only.poraque"
        torch.save(state, bundle)
        new = tmp_path / "new_paw_only"
        new.mkdir()
        # Keep a tau reference to check that compare/plot also skip a field
        # the bundle cannot predict, even if a reference file is present.
        for name in ("POSCAR", "POTCAR", "INCAR", "CHGCAR", "TAUCAR"):
            shutil.copy(os.path.join(runs, "structure_0000", name), new / name)
        out = tmp_path / "predicted_paw_only"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = poraque_inference.predict([
                str(new), "--models", str(bundle), "--output", str(out),
                "--grid", "16", "16", "16", "--add-paw", "--paw-source", "model",
                "--compare", "--plot-dir", str(tmp_path / "plots_paw_only"), "--device", "cpu"])
        assert set(result["outputs"]) == {"EXTCAR", "CHGCAR"}
        assert "energy" not in result and "kinetic_energy" not in result
        assert "TAUCAR" not in result["comparison"]
        assert result["paw_augmentation"]["source"] == "model"
        _, blocks = read_augmentation_blocks(str(out / "CHGCAR"))
        records, lengths = occupancy_arrays(blocks)
        assert records.shape == (2, 2, 6)
        assert lengths.tolist() == [6, 6]

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
        operator = load_bundle(bundle, "ext2chg", device="cpu")
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
        bundle = self._train(base, runs, "grid_run", occupancies=False)
        chain = {task: load_bundle(bundle, task, device="cpu")
                 for task in ("ext2pseudo", "pseudo2tau")}
        without = save_bundle(str(tmp_path / "chain.poraque"), chain)
        new = tmp_path / "bare"
        new.mkdir()
        for name in ("POSCAR", "POTCAR", "INCAR"):
            shutil.copy(os.path.join(runs, "structure_0000", name), new / name)
        with pytest.raises(SystemExit, match="holds no ext2chg"):
            poraque_inference.predict([
                str(new), "--models", without, "--output",
                str(tmp_path / "o"), "--grid", "16", "16", "16", "--add-paw",
                "--paw-source", "model", "--functional", "skip",
                "--device", "cpu"])


def accelerator():
    """The device this machine actually has, or ``None``."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


@pytest.mark.gpu
@pytest.mark.skipif(accelerator() is None, reason="requires MPS or CUDA")
class TestItEvaluatesOnWhateverAcceleratorIsHere:
    """
    A real run died on MPS at its first evaluation, ten good epochs in.

    ``evaluate_occupancies`` pooled its sums into a float64 accumulator built
    on the *operator's* device, and Metal has no float64 at all --- so the
    call raised ``TypeError`` rather than returning a number. Every earlier
    exercise of the operator had pinned ``device: cpu`` (both probes and the
    60-epoch platinum run), which is exactly why a whole-path defect survived
    26 tests. The two scalars are now summed on the CPU, which is also where
    float64 belongs: the ratio runs over every atom of every structure.
    """

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
        return FieldPairDataset(str(tmp_path / "cache"), "ext2chg")

    def test_training_with_validation_survives_its_first_evaluation(self,
                                                                    dataset):
        from poraque.ml import FieldOperator, train

        layouts = {14: (1,)}
        operator = FieldOperator(
            "ext2chg", width=6, modes=4, n_layers=1, projection_channels=8,
            device=str(accelerator()), in_channels=1, out_channels=2,
            site_layouts=layouts, readout_g_max=3.0,
            site_transform=OccupancyTransform.fit(dataset, layouts, sets=2),
            init_seed=0)
        history = train(operator, dataset, epochs=1, batch_size=2,
                        validation=dataset, eval_every=1, verbose=False)
        measured = history["val_occupancy_error"][-1]
        assert math.isfinite(measured) and measured > 0.0

    def test_the_records_come_back_from_the_accelerator(self, dataset):
        """
        The same cast, at the other end of the operator: reading predicted
        records off the device for ``--paw-source model``. ``.to("cpu",
        float64)`` asks the *source* device for the conversion, so on Metal
        both sites had to move first and cast after.
        """
        from poraque.fields import ExternalPotential
        from poraque.ml import FieldOperator

        layouts = {14: (1,)}
        operator = FieldOperator(
            "ext2chg", width=6, modes=4, n_layers=1, projection_channels=8,
            device=str(accelerator()), in_channels=1, out_channels=2,
            site_layouts=layouts, readout_g_max=3.0,
            site_transform=OccupancyTransform.fit(dataset, layouts, sets=2),
            init_seed=0)
        potential = ExternalPotential.read(
            os.path.join(dataset.root, "structure_0000", "EXTCAR"))
        records = operator.predict_occupancies(potential)
        assert len(records) == 2 and records[0][0].dtype == np.float64
        assert np.isfinite(records[0][0]).all()
