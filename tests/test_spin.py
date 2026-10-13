# -*- coding: utf-8 -*-
# file: test_spin.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Spin-polarized (``ISPIN = 2``) support, end to end.

No spin-polarized reference calculation ships with this repository — every
``INCAR`` in ``data/vasp`` sets ``ISPIN = 1`` — so the fixtures here build a
two-block ``CHGCAR`` from known arrays. That is enough to pin everything the
code path is responsible for:

* the file format round-trips through VASP's own layout;
* :math:`(\rho, m)` and :math:`(\rho_\uparrow, \rho_\downarrow)` are consistent
  views of one another;
* the dataset detects two channels from the data rather than from a flag;
* the two channels are normalized *separately*;
* an operator can be built, trained and reloaded with two channels.

What it cannot test is accuracy, which needs real ``ISPIN = 2`` data.
"""

import numpy as np
import pytest
import torch

from poraque.fields import (
    ChargeDensity,
    ExternalPotential,
    FieldGrid,
    KineticEnergyDensity,
    SpinDensity,
    is_spin_polarized,
)
from poraque.fields.vasp.poscar import Poscar
from poraque.ml import FieldOperator, FieldPairDataset
from poraque.ml.transforms import Asinh, Channelwise


@pytest.fixture
def grid():
    return FieldGrid((8, 8, 8), np.eye(3) * 4.0)


@pytest.fixture
def poscar():
    return Poscar(cell=np.eye(3) * 4.0, symbols=["Fe"], counts=[2],
                  scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture
def spin_density(grid, poscar):
    """A field with a genuinely non-zero, sign-changing magnetization."""
    rng = np.random.default_rng(0)
    up = rng.random(grid.shape) + 0.5
    down = rng.random(grid.shape) + 0.5
    return SpinDensity.from_up_down(up, down, grid, poscar)


@pytest.fixture
def spin_dataset(tmp_path, grid, poscar):
    """Three synthetic materials with spin-polarized CHGCARs."""
    rng = np.random.default_rng(1)
    for index in range(3):
        directory = tmp_path / f"structure_{index:03d}"
        directory.mkdir()
        ExternalPotential(rng.normal(size=grid.shape) * 5.0, grid,
                          poscar).write(str(directory / "EXTCAR"))
        SpinDensity.from_up_down(rng.random(grid.shape) + 0.5,
                                 rng.random(grid.shape) * 0.4,
                                 grid, poscar).write(str(directory / "CHGCAR"))
        KineticEnergyDensity(rng.random(grid.shape) * 3.0, grid,
                             poscar).write(str(directory / "TAUCAR"))
    return str(tmp_path)


# ===================================================================== #
# The field
# ===================================================================== #
class TestSpinDensity:
    def test_up_down_round_trip(self, grid, poscar):
        r"""
        :math:`(\rho, m)` and :math:`(\rho_\uparrow, \rho_\downarrow)` are the
        same information.
        """
        rng = np.random.default_rng(0)
        up, down = rng.random(grid.shape), rng.random(grid.shape)
        density = SpinDensity.from_up_down(up, down, grid, poscar)
        assert np.allclose(density.up, up)
        assert np.allclose(density.down, down)
        assert np.allclose(density.total, up + down)
        assert np.allclose(density.magnetization, up - down)

    def test_integrals(self, grid, poscar):
        """Electron count from the total, moment from the magnetization."""
        up = np.full(grid.shape, 0.5)
        down = np.full(grid.shape, 0.25)
        density = SpinDensity.from_up_down(up, down, grid, poscar)
        volume = grid.volume
        assert density.electron_count() == pytest.approx(0.75 * volume)
        assert density.magnetic_moment() == pytest.approx(0.25 * volume)

    def test_file_round_trip(self, tmp_path, spin_density):
        """Through VASP's two-block layout and back, to full precision."""
        path = str(tmp_path / "CHGCAR")
        spin_density.write(path)
        assert is_spin_polarized(path)

        back = SpinDensity.read(path)
        assert np.allclose(back.total, spin_density.total, rtol=1e-9)
        assert np.allclose(back.magnetization, spin_density.magnetization,
                           rtol=1e-9)

    def test_a_collinear_file_is_not_read_as_spin_polarized(self, tmp_path,
                                                            grid, poscar):
        """
        Refusing is the point.

        Reading a one-block CHGCAR as spin-polarized would hand back a zero
        magnetization — a physical claim ("this system is non-magnetic") that
        the file never made.
        """
        path = str(tmp_path / "CHGCAR")
        ChargeDensity(np.ones(grid.shape), grid, poscar).write(path)
        assert not is_spin_polarized(path)
        with pytest.raises(ValueError, match="not a spin-polarized"):
            SpinDensity.read(path)

    def test_normalization_preserves_local_polarization(self, spin_density):
        r"""
        Scaling both channels together is what makes it a *normalization*.

        Rescaling :math:`\rho` alone would change :math:`m/\rho` at every
        point, which is a different prediction rather than a repaired one.
        """
        before = spin_density.magnetization / spin_density.total
        after = spin_density.normalized(20.0)
        assert after.electron_count() == pytest.approx(20.0)
        assert np.allclose(after.magnetization / after.total, before)

    def test_normalization_keeps_both_spins_non_negative(self, grid, poscar):
        """Clipping acts on up/down, the quantities that cannot be negative."""
        rng = np.random.default_rng(3)
        density = SpinDensity(rng.random(grid.shape),
                              rng.normal(size=grid.shape) * 4.0,
                              grid, poscar)
        fixed = density.normalized(10.0)
        assert fixed.up.min() >= -1e-12
        assert fixed.down.min() >= -1e-12

    def test_channel_stack_shape(self, spin_density, grid):
        assert spin_density.data.shape == (2,) + tuple(grid.shape)

    def test_as_charge_density_keeps_the_electron_count(self, spin_density):
        total = spin_density.as_charge_density()
        assert isinstance(total, ChargeDensity)
        assert total.electron_count() == pytest.approx(
            spin_density.electron_count())

    def test_rejects_mismatched_shapes(self, grid, poscar):
        with pytest.raises(ValueError, match="does not match the grid"):
            SpinDensity(np.ones(grid.shape), np.ones((4, 4, 4)), grid, poscar)


# ===================================================================== #
# Per-channel normalization
# ===================================================================== #
class TestChannelwiseTransform:
    def test_applies_one_transform_per_channel(self):
        transform = Channelwise([Asinh(1.0), Asinh(10.0)])
        values = np.stack([np.full((4, 4, 4), 2.0),
                           np.full((4, 4, 4), 2.0)], axis=0)
        out = transform(values)
        assert not np.allclose(out[0], out[1]), (
            "two different scales must give two different results")

    def test_round_trips(self):
        transform = Channelwise([Asinh(0.5), Asinh(3.0)])
        rng = np.random.default_rng(0)
        values = rng.normal(size=(2, 4, 4, 4))
        assert np.allclose(transform.inverse(transform(values)), values)

    def test_works_batched_and_unbatched(self):
        """The channel axis is -4 either way, which is why it is counted so."""
        transform = Channelwise([Asinh(0.5), Asinh(3.0)])
        rng = np.random.default_rng(0)
        single = rng.normal(size=(2, 4, 4, 4))
        batched = single[None]
        assert np.allclose(transform(batched)[0], transform(single))

    def test_works_on_torch_tensors(self):
        transform = Channelwise([Asinh(0.5), Asinh(3.0)])
        values = torch.randn(1, 2, 4, 4, 4)
        assert torch.allclose(transform.inverse(transform(values)), values,
                              atol=1e-5)

    def test_state_dict_round_trip(self):
        from poraque.ml.transforms import FieldTransform

        transform = Channelwise([Asinh(0.5), Asinh(3.0)])
        rebuilt = FieldTransform.from_state_dict(transform.state_dict())
        assert isinstance(rebuilt, Channelwise)
        assert [t.scale for t in rebuilt.transforms] == [0.5, 3.0]

    def test_rejects_a_channel_count_mismatch(self):
        transform = Channelwise([Asinh(1.0), Asinh(1.0)])
        with pytest.raises(ValueError, match="channels"):
            transform(np.zeros((3, 4, 4, 4)))


# ===================================================================== #
# The dataset
# ===================================================================== #
class TestSpinDataset:
    def test_detects_spin_from_the_data(self, spin_dataset):
        dataset = FieldPairDataset(spin_dataset, task="ext2pseudo")
        assert dataset.spin is True
        assert dataset.channels == (1, 2)

    def test_pseudo2tau_is_two_in_one_out(self, spin_dataset):
        r""":math:`\tau` stays a single block even under ``ISPIN = 2``."""
        dataset = FieldPairDataset(spin_dataset, task="pseudo2tau")
        assert dataset.channels == (2, 1)

    def test_samples_carry_the_channel_axis(self, spin_dataset):
        dataset = FieldPairDataset(spin_dataset, task="ext2pseudo")
        sample = dataset[0]
        assert sample["input"].shape[0] == 1
        assert sample["target"].shape[0] == 2

    def test_transforms_are_fitted_per_channel(self, spin_dataset):
        """
        The reason :class:`Channelwise` exists.

        One scale across both channels would be set by whichever dominates the
        sample and would normalize neither.
        """
        dataset = FieldPairDataset(spin_dataset, task="ext2pseudo")
        _, target_transform = dataset.fit_transforms()
        assert isinstance(target_transform, Channelwise)
        assert len(target_transform.transforms) == 2


class TestTheMagnetizationSharesTheChargeScale:
    r"""
    The m channel of a CHGCAR takes the charge channel's Asinh scale.

    Fitted on its own median it took the noise floor: the platinum set is 64
    cells with m = 0 to round-off and 33 nanoparticles whose boxes are mostly
    vacuum, and the median |m| came out at 1.6e-6. The inverse then has a gain
    of asinh(1/s) = 14, so a 0.2 overshoot of the network's output multiplied a
    nanoparticle's moment sixteenfold. The held-out ext2chg error jumped from
    0.03 to 3.4 and 10.6 between evaluations, with the training loss smooth,
    and early stopping kept an epoch 33 % worse at occupancies than the run
    reached.
    """

    @pytest.fixture
    def mostly_unpolarized(self, tmp_path, grid, poscar):
        rng = np.random.default_rng(3)
        for index in range(4):
            directory = tmp_path / f"structure_{index:03d}"
            directory.mkdir()
            ExternalPotential(rng.normal(size=grid.shape) * 5.0, grid,
                              poscar).write(str(directory / "EXTCAR"))
            total = rng.random(grid.shape) + 0.5
            if index == 0:
                # One magnetic cell, its moment on a few voxels.
                m = np.zeros(grid.shape)
                m[:2, :2, :2] = 0.4
            else:
                m = rng.normal(size=grid.shape) * 1e-7
            SpinDensity(total, m, grid, poscar).write(
                str(directory / "CHGCAR"))
            KineticEnergyDensity(rng.random(grid.shape) * 3.0, grid,
                                 poscar).write(str(directory / "TAUCAR"))
        return str(tmp_path)

    def test_the_target_magnetization_is_not_fitted_to_its_noise(
            self, mostly_unpolarized):
        dataset = FieldPairDataset(mostly_unpolarized, task="ext2pseudo")
        _, target = dataset.fit_transforms()
        charge, magnetization = target.transforms
        assert magnetization.scale == charge.scale
        assert magnetization.scale > 1e-3

    def test_the_gain_of_the_inverse_is_the_charge_channels(
            self, mostly_unpolarized):
        """
        The quantity that failed: asinh(1/s), how many e-folds m moves per
        unit of network output in the logarithmic regime. The old rule, the
        channel's own median, is fitted alongside to show what it would be.
        """
        dataset = FieldPairDataset(mostly_unpolarized, task="ext2pseudo")
        _, target = dataset.fit_transforms()
        magnetization = np.concatenate([
            np.asarray(dataset.target_values(index))[1].ravel()
            for index in range(len(dataset))])
        own_median = Asinh.fit(magnetization)

        def gain(transform):
            return float(np.arcsinh(1.0 / transform.scale))

        assert gain(target.transforms[1]) == gain(target.transforms[0])
        assert gain(own_median) > 2 * gain(target.transforms[0])

    def test_the_pseudo2tau_input_follows_the_same_rule(self,
                                                     mostly_unpolarized):
        dataset = FieldPairDataset(mostly_unpolarized, task="pseudo2tau")
        source, _ = dataset.fit_transforms()
        assert source.transforms[1].scale == source.transforms[0].scale

    def test_a_collinear_dataset_stays_single_channel(self, tmp_path, grid,
                                                      poscar):
        """The default path must be untouched by any of this."""
        rng = np.random.default_rng(2)
        for index in range(2):
            directory = tmp_path / f"structure_{index:03d}"
            directory.mkdir()
            ExternalPotential(rng.normal(size=grid.shape), grid,
                              poscar).write(str(directory / "EXTCAR"))
            ChargeDensity(rng.random(grid.shape), grid,
                          poscar).write(str(directory / "CHGCAR"))
            KineticEnergyDensity(rng.random(grid.shape), grid,
                                 poscar).write(str(directory / "TAUCAR"))

        dataset = FieldPairDataset(str(tmp_path), task="ext2pseudo")
        assert dataset.spin is False
        assert dataset.channels == (1, 1)
        assert dataset[0]["target"].shape[0] == 1
        _, target_transform = dataset.fit_transforms()
        assert not isinstance(target_transform, Channelwise)

    def test_requesting_the_wrong_spin_is_an_error(self, spin_dataset):
        """A flag must not be able to contradict the data."""
        with pytest.raises(ValueError, match="spin-polarized"):
            FieldPairDataset(spin_dataset, task="ext2pseudo", spin=False)


# ===================================================================== #
# The operator
# ===================================================================== #
class TestSpinOperator:
    def test_predicts_a_spin_density(self, spin_dataset, grid):
        operator = FieldOperator("ext2pseudo", in_channels=1, out_channels=2,
                                 width=6, modes=3, n_layers=1,
                                 projection_channels=8, device="cpu")
        import os

        potential = ExternalPotential.read(
            os.path.join(spin_dataset, "structure_000", "EXTCAR"))
        prediction = operator.predict(potential)
        assert isinstance(prediction, SpinDensity)
        assert prediction.data.shape == (2,) + tuple(grid.shape)

    def test_consumes_a_spin_density(self, spin_dataset, grid):
        """``pseudo2tau`` takes two channels in and gives one out."""
        import os

        operator = FieldOperator("pseudo2tau", in_channels=2, out_channels=1,
                                 width=6, modes=3, n_layers=1,
                                 projection_channels=8, device="cpu")
        density = SpinDensity.read(
            os.path.join(spin_dataset, "structure_000", "CHGCAR"))
        prediction = operator.predict(density)
        assert isinstance(prediction, KineticEnergyDensity)
        assert prediction.data.shape == tuple(grid.shape)

    def test_trains(self, spin_dataset):
        from poraque.ml import train

        dataset = FieldPairDataset(spin_dataset, task="ext2pseudo")
        source_transform, target_transform = dataset.fit_transforms()
        operator = FieldOperator("ext2pseudo", in_channels=1, out_channels=2,
                                 width=6, modes=3, n_layers=1,
                                 projection_channels=8, device="cpu",
                                 input_transform=source_transform,
                                 target_transform=target_transform)
        history = train(operator, dataset, epochs=2, batch_size=1,
                        learning_rate=1e-3, verbose=False)
        assert len(history["train_loss"]) == 2
        assert np.isfinite(history["train_loss"]).all()

    def test_channel_counts_survive_a_bundle_round_trip(self, tmp_path,
                                                        spin_dataset):
        """
        The regression this guards against is silent.

        ``in_channels`` cannot be read back off the lifting layer, which sees
        ``in_channels + 3`` when coordinates are on. Inferring it would turn a
        two-channel model into a one-channel model with coordinates and load
        the weights into the wrong slots.
        """
        import os

        from poraque.ml import BUNDLE_FILENAME, load_bundle, save_bundle

        operator = FieldOperator("ext2pseudo", in_channels=1, out_channels=2,
                                 width=6, modes=3, n_layers=1,
                                 projection_channels=8, device="cpu")
        partner = FieldOperator("pseudo2tau", in_channels=2, out_channels=1,
                                width=6, modes=3, n_layers=1,
                                projection_channels=8, device="cpu")
        path = str(tmp_path / BUNDLE_FILENAME)
        save_bundle(path, {"ext2pseudo": operator, "pseudo2tau": partner})

        potential = ExternalPotential.read(
            os.path.join(spin_dataset, "structure_000", "EXTCAR"))
        expected = operator.predict(potential)

        reloaded = load_bundle(path, "ext2pseudo", device="cpu")
        assert (reloaded.in_channels, reloaded.out_channels) == (1, 2)
        assert np.allclose(reloaded.predict(potential).data, expected.data,
                           atol=1e-5)

        other = load_bundle(path, "pseudo2tau", device="cpu")
        assert (other.in_channels, other.out_channels) == (2, 1)


# ===================================================================== #
# Resolving `data.spin: auto` against the data
# ===================================================================== #
def _runs(root, grid, poscar, polarized, count=2, start=0):
    """``count`` calculation directories, spin-polarized or not."""
    rng = np.random.default_rng(7)
    for index in range(start, start + count):
        directory = root / f"structure_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        poscar.write(str(directory / "POSCAR"))
        values = rng.random(grid.shape) + 0.5
        if polarized:
            SpinDensity.from_up_down(
                values, values * 0.6, grid, poscar).write(
                    str(directory / "CHGCAR"))
        else:
            ChargeDensity(values, grid, poscar).write(
                str(directory / "CHGCAR"))
    return str(root)


class TestSpinAutoIsResolvedFromTheSourceNotFromTheCache:
    """
    Regression: ``data.spin: auto`` — the default — could never mean spin.

    ``poraque-train`` passed ``spin=data.spin is True`` to the cache builder,
    and ``"auto" is True`` is ``False``. So every ``ISPIN = 2`` magnetization
    block was discarded on the way into the cache, and the dataset's *own*
    auto-detection then reported, correctly, that the cache carried no spin.
    Two layers of detection agreeing, with the first one having thrown the
    answer away.

    The whole point of these tests is that the question is asked of the
    **sources**. A test that only inspected the cache would have passed
    throughout the bug.
    """

    def test_auto_carries_the_magnetization_of_a_polarized_source(
            self, tmp_path, grid, poscar):
        from poraque.data import build_field_cache

        root = _runs(tmp_path / "runs", grid, poscar, polarized=True)
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR",), charges={"Fe": 8.0},
                                  spin="auto")

        cached = f"{cache}/structure_000/CHGCAR"
        assert is_spin_polarized(cached), (
            "spin: auto dropped the second block of an ISPIN = 2 source")
        assert SpinDensity.read(cached).magnetic_moment() != 0.0

    def test_auto_on_an_unpolarized_source_stays_single_channel(
            self, tmp_path, grid, poscar):
        from poraque.data import build_field_cache

        root = _runs(tmp_path / "runs", grid, poscar, polarized=False)
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR",), charges={"Fe": 8.0},
                                  spin="auto")

        assert not is_spin_polarized(f"{cache}/structure_000/CHGCAR")

    def test_false_is_still_a_deliberate_opt_out(self, tmp_path, grid, poscar):
        """Discarding the channel stays possible — it just has to be asked for."""
        from poraque.data import build_field_cache

        root = _runs(tmp_path / "runs", grid, poscar, polarized=True)
        cache = build_field_cache(root, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR",), charges={"Fe": 8.0},
                                  spin=False)

        assert not is_spin_polarized(f"{cache}/structure_000/CHGCAR")

    def test_true_on_data_with_no_magnetization_raises(self, tmp_path, grid,
                                                       poscar):
        """Rather than training a channel the data has no values for."""
        from poraque.data import build_field_cache

        root = _runs(tmp_path / "runs", grid, poscar, polarized=False)
        with pytest.raises(ValueError, match="has a magnetization block"):
            build_field_cache(root, tmp_path / "cache", resolution=8,
                              fields=("CHGCAR",), charges={"Fe": 8.0},
                              spin=True)

    def test_a_mixed_set_resolves_to_spin_and_zeroes_the_rest(
            self, tmp_path, grid, poscar):
        """
        One operator has one channel count, so the decision is taken for the
        whole dataset. An unpolarized member becomes ``m = 0``, which is a true
        statement about a non-magnetic calculation rather than a padded one.
        """
        from poraque.data import build_field_cache

        runs = tmp_path / "runs"
        _runs(runs, grid, poscar, polarized=True, count=2, start=0)
        _runs(runs, grid, poscar, polarized=False, count=1, start=2)
        cache = build_field_cache(runs, tmp_path / "cache", resolution=8,
                                  fields=("CHGCAR",), charges={"Fe": 8.0},
                                  spin="auto")

        assert is_spin_polarized(f"{cache}/structure_002/CHGCAR")
        plain = SpinDensity.read(f"{cache}/structure_002/CHGCAR")
        assert np.abs(plain.magnetization).max() < 1e-9
        assert is_spin_polarized(f"{cache}/structure_000/CHGCAR")

    def test_the_two_layouts_never_share_a_cache_directory(self, tmp_path,
                                                           grid, poscar):
        """
        A one-channel cache reused for a two-channel run would be found, loaded
        and silently trained on. The resolved value names the directory, so a
        `spin: false` build and an `auto` build of the same ISPIN = 2 sources
        land in different places.
        """
        import sys

        sys.path.insert(0, "scripts")
        from poraque.ml.config import TrainingConfig
        from poraque_train import cache_tag

        root = _runs(tmp_path / "runs", grid, poscar, polarized=True)
        auto = TrainingConfig.from_dict(
            {"data": {"path": str(root), "spin": "auto"}})
        off = TrainingConfig.from_dict(
            {"data": {"path": str(root), "spin": False}})

        assert cache_tag(auto.data).endswith("_spin")
        assert not cache_tag(off.data).endswith("_spin")
        assert cache_tag(auto.data) != cache_tag(off.data)


class TestDetectingASecondBlockIsCheap:
    """
    The detector is called once per material by the cache builder and again to
    name the directory, so a full float parse of every 100 MB density to answer
    a yes/no was 14 s of pure overhead per run on a 31-material set.
    """

    def test_it_agrees_with_a_full_parse(self, tmp_path, grid, poscar,
                                         spin_density):
        from poraque.fields.vasp.volumetric import read_volumetric

        for name, field in (("spin", spin_density),
                            ("plain", ChargeDensity(
                                np.ones(grid.shape), grid, poscar))):
            path = str(tmp_path / f"CHGCAR_{name}")
            field.write(path)
            _, _, extra = read_volumetric(path, read_all=True)
            assert is_spin_polarized(path) is (len(extra) >= 1), name

    def test_augmentation_records_are_not_mistaken_for_a_second_block(
            self, tmp_path, grid, poscar):
        """
        The scan looks for the grid-dimension line repeated *exactly*. PAW
        occupancies sit between the blocks and are full of numbers; a looser
        test would read them as a magnetization channel that is not there.
        """
        path = str(tmp_path / "CHGCAR")
        ChargeDensity(np.ones(grid.shape), grid, poscar).write(
            path, augmentation=["augmentation occupancies   1  4",
                                "  8 8 8 8", "  1 2 3", "  0.1 0.2 0.3"])

        assert is_spin_polarized(path) is False

    def test_the_seek_path_answers_without_reading_the_block(
            self, tmp_path, grid, poscar, spin_density):
        """
        The probe jumps over the fixed-width grid block rather than reading
        it: it was 81 % of the preparation before the cache on the Pt set.
        A wrong jump would drop a magnetization silently, so the seek path
        must answer (not fall back) on a file Poraquê wrote, and agree.
        """
        from poraque.fields.spin import _second_block_by_seeking

        for name, field, expected in (
                ("spin", spin_density, True),
                ("plain", ChargeDensity(np.ones(grid.shape), grid, poscar),
                 False)):
            path = str(tmp_path / f"CHGCAR_{name}")
            field.write(path)
            assert _second_block_by_seeking(path) is expected, name

    def test_a_block_of_uneven_lines_falls_back_to_the_scan(
            self, tmp_path, grid, poscar, spin_density):
        """Arithmetic over line widths is only trusted where it checks out."""
        from poraque.fields.spin import _second_block_by_seeking

        path = tmp_path / "CHGCAR"
        spin_density.write(str(path))
        lines = path.read_text().splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines)
                     if line.split() == [str(n) for n in grid.shape]) + 1
        lines[start + 2] = " " + lines[start + 2]        # one line wider
        path.write_text("".join(lines))

        assert _second_block_by_seeking(str(path)) is None
        assert is_spin_polarized(str(path)) is True

    def test_the_answer_is_remembered_until_the_file_changes(
            self, tmp_path, grid, poscar, spin_density, monkeypatch):
        """
        A gzipped density cannot be seeked, so its answer costs a full
        decompression: 34.5 s on 432 CHON files, every run. The memo makes
        the second run read nothing, and a rewritten file is probed again.
        """
        import poraque.data.cache as cache

        path = str(tmp_path / "CHGCAR")
        spin_density.write(path)
        calls = []
        real = cache._spin_of
        monkeypatch.setattr(cache, "_spin_of",
                            lambda p: calls.append(p) or real(p))

        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [True]
        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [True]
        assert len(calls) == 1

        ChargeDensity(np.ones(grid.shape), grid, poscar).write(path)
        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [False]
        assert len(calls) == 2

    def test_a_store_backed_density_is_probed_not_assumed(
            self, tmp_path, spin_density):
        """
        A store's field path is ``fields.h5::CHGCAR``, which ``os.stat``
        cannot open; the memo once took that as "no file" and answered
        ``False`` without probing, so ``spin: auto`` dropped the
        magnetization of every HDF5-backed material.
        """
        import poraque.data.cache as cache
        from poraque.fields.hdf5 import join_target, write_fields

        store = str(tmp_path / "fields.h5")
        write_fields(store, {"CHGCAR": spin_density})
        path = join_target(store, "CHGCAR")

        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [True]
        assert cache.probe_spin([path], memo_root=str(tmp_path)) == [True]


# ===================================================================== #
# A spin-polarized prediction has to reach an energy
# ===================================================================== #
class TestASpinDensityCanBeIntegratedIntoAnEnergy:
    """
    Regression: it could not, and the adapter for it was never called.

    Every term in :mod:`poraque.physics.energy` is a functional of the total
    density, so a two-channel field has to be reduced before any of them can
    integrate it. ``SpinDensity.as_charge_density`` existed for exactly that
    and nothing in the package called it, so ``EnergyCalculator.compute`` —
    and therefore the ASE calculator's ``get_potential_energy`` — raised
    ``TypeError: float() argument must be ... not 'SpinDensity'`` for any model
    trained with spin resolved on. Which, since ``data.spin: auto`` now
    resolves against the data, is every model trained on ``ISPIN = 2`` runs.
    """

    def test_total_density_passes_a_plain_field_through(self, grid, poscar):
        from poraque.physics.energy import total_density

        field = ChargeDensity(np.ones(grid.shape) * 0.3, grid, poscar)
        assert total_density(field) is field

    def test_total_density_reduces_a_spin_pair_to_its_total(self,
                                                            spin_density):
        from poraque.physics.energy import total_density

        reduced = total_density(spin_density)
        assert np.allclose(np.asarray(reduced), spin_density.total)
        assert np.asarray(reduced).shape == tuple(spin_density.grid.shape)

    def test_the_energy_matches_the_one_from_the_total_alone(self,
                                                             spin_density):
        """
        The electrostatic and kinetic terms depend only on the total, so
        reducing the pair must not change them. This is what makes the
        reduction safe rather than merely convenient.
        """
        from poraque.physics import EnergyCalculator

        grid, poscar = spin_density.grid, spin_density.structure
        potential = ExternalPotential(
            np.random.default_rng(5).normal(size=grid.shape) * 5.0,
            grid, poscar)
        tau = KineticEnergyDensity(np.ones(grid.shape) * 2.0, grid, poscar)
        calculator = EnergyCalculator(grid=grid, structure=poscar,
                                      charges={"Fe": 8.0}, functional="lda")

        paired = calculator.compute(spin_density, tau, potential)
        plain = calculator.compute(spin_density.as_charge_density(), tau,
                                   potential)

        assert paired.total == pytest.approx(plain.total, rel=1e-12)
        assert paired.n_electrons == pytest.approx(
            spin_density.electron_count(), rel=1e-12)


class TestTheSpinChannelReachesThePhysicsHelpers:
    """
    `squeeze(1)` is a **no-op** on an axis of width 2, and it was used
    throughout `ml/physics.py` to mean "drop the channel axis". Every helper
    that did so was therefore correct for a one-channel field and wrong for a
    spin-polarized one -- which, since `data.spin: auto` resolves to True on
    this project's data, is every model it now trains.

    Three failure modes, all present at once:

    * a **crash**, where the surviving channel axis broadcast against the three
      Cartesian components of the gradient. This is what `loss: sobolev`
      reported;
    * a **silent wrong answer**, where a ``(B, ...)`` kernel broadcast against
      a ``(B, C, ...)`` field and, at ``C == B``, lined the channel axis up
      with the batch axis and returned finite nonsense;
    * a **meaningless second channel**, where an elementwise functional of rho
      was evaluated on the magnetization -- ``(5/3) C_TF m^(2/3)`` and
      ``v_x(m)`` -- and then broadcast into a residual built from
      single-channel potentials.
    """

    @staticmethod
    def _fields(batch=3, channels=2, n=8):
        cell = torch.eye(3).unsqueeze(0).repeat(batch, 1, 1) * 6.0
        torch.manual_seed(0)
        return torch.rand(batch, channels, n, n, n) + 0.1, cell

    def test_the_gradient_is_taken_per_channel(self):
        from poraque.ml.physics import spectral_gradient

        field, cell = self._fields()
        gradient = spectral_gradient(field, cell)

        assert gradient.shape == (3, 2, 3, 8, 8, 8)
        for channel in range(2):
            assert torch.allclose(gradient[:, channel],
                                  spectral_gradient(field[:, channel], cell),
                                  atol=1e-6)

    def test_a_single_channel_field_keeps_its_old_shape(self):
        """`(B, 1, ...)` must still give `(B, 3, ...)`: every other caller in
        the package relies on it."""
        from poraque.ml.physics import spectral_gradient

        field, cell = self._fields(channels=1)
        assert spectral_gradient(field, cell).shape == (3, 3, 8, 8, 8)

    @pytest.mark.parametrize("batch", [2, 3])
    def test_the_laplacian_does_not_align_channels_with_the_batch(self, batch):
        """
        `batch == channels == 2` is the dangerous case: it broadcast without
        error and returned a finite, wrong answer.
        """
        from poraque.ml.physics import spectral_laplacian

        field, cell = self._fields(batch=batch)
        laplacian = spectral_laplacian(field, cell)

        assert laplacian.shape == field.shape
        for channel in range(2):
            assert torch.allclose(laplacian[:, channel],
                                  spectral_laplacian(field[:, channel], cell),
                                  atol=1e-6)

    @pytest.mark.parametrize("name", ["hartree", "lda", "pbe"])
    def test_the_potentials_are_functionals_of_rho_alone(self, name):
        from poraque.ml.physics import hartree_potential, xc_potential

        field, cell = self._fields(batch=2)
        call = (hartree_potential if name == "hartree"
                else lambda f, c: xc_potential(f, name, cell=c))

        spin = call(field, cell)
        assert spin.shape[1] == 1, "the magnetization is not an argument of it"
        assert torch.allclose(spin, call(field[:, :1], cell), atol=1e-9)

    def test_the_euler_lagrange_residual_ignores_the_magnetization(self):
        from poraque.ml.physics import euler_lagrange_residual

        field, cell = self._fields(batch=2)
        external = torch.randn(2, 1, 8, 8, 8)

        residual = euler_lagrange_residual(field, external, cell, v_xc="pbe")
        assert residual.shape == (2, 1, 8, 8, 8)
        assert torch.allclose(
            residual,
            euler_lagrange_residual(field[:, :1], external, cell, v_xc="pbe"),
            atol=1e-6)

    def test_the_sobolev_loss_trains_on_two_channels(self):
        """The reported failure: `loss: sobolev` on ISPIN = 2 data."""
        from poraque.ml.losses import SobolevLoss

        target, cell = self._fields()
        prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_(True)

        value = SobolevLoss(weight=0.1)(prediction, target, cell)
        value.backward()

        assert torch.isfinite(value)
        assert prediction.grad is not None and torch.isfinite(prediction.grad).all()

    def test_the_sobolev_loss_constrains_the_magnetization_too(self):
        """
        It is a *data* term, not a physical constraint: whatever the operator
        predicts is part of the target. Taking channel 0 alone would leave
        grad(m) unconstrained and nothing would say so.
        """
        from poraque.ml.losses import SobolevLoss

        target, cell = self._fields()
        loss = SobolevLoss(weight=0.1)
        spoiled = target.clone()
        spoiled[:, 1] += 0.3 * torch.randn_like(spoiled[:, 1])

        assert float(loss(target.clone(), target, cell)) == pytest.approx(0.0, abs=1e-6)
        assert float(loss(spoiled, target, cell)) > 1e-3


class TestTheMagnetizationWeightReachesTheObjective:
    """
    ``training.magnetization_weight`` is read, not merely accepted.

    A weight that validates and then never reaches the loss is the failure
    this pins: the objective is assembled per task in ``build_loss``, far from
    the config, and the training loop reads the weight back off the criterion
    rather than off a config it does not see --- so there are two joints where
    a number can be dropped, and a run would look entirely normal with the
    magnetization term at whatever the default was.
    """

    def test_the_weight_is_inert_under_the_joint_default(self):
        """
        Under `joint` there is one term, so the weight has nothing to scale.

        Worth pinning rather than leaving implicit: someone raising
        `magnetization_weight` to chase a bad magnetization and seeing no
        change at all should find that documented, not mysterious. The knob
        needs `spin_loss: per_channel` to do anything.
        """
        import torch

        from poraque.ml.losses import PhysicsInformedLoss

        target = torch.randn(2, 2, 6, 6, 6,
                             generator=torch.Generator().manual_seed(7))
        prediction = target + 0.1

        values = set()
        for weight in (0.0, 1.0, 9.0):
            criterion = PhysicsInformedLoss(task="ext2pseudo",
                                            magnetization_weight=weight)
            assert criterion.spin_loss == "joint"
            values.add(round(float(criterion(prediction, target)["total"]), 9))
        assert len(values) == 1

    def test_build_loss_carries_it_from_the_config(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from poraque_train import build_loss
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_dict(
            {"training": {"spin": {"magnetization_weight": 2.5}}})
        criterion = build_loss(config, "ext2pseudo")
        assert criterion.magnetization_weight == 2.5
        assert criterion.data_loss.magnetization_weight == 2.5
        assert criterion.spin_loss == "joint"          # the default

    def test_the_default_is_an_equal_weighting(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from poraque_train import build_loss
        from poraque.ml.config import TrainingConfig

        criterion = build_loss(TrainingConfig.from_dict({}), "ext2pseudo")
        assert criterion.magnetization_weight == 1.0
        assert criterion.magnetization_floor == 1e-2
        assert criterion.spin_loss == "joint"

    def test_zero_removes_the_magnetization_from_the_gradient(self):
        """
        ``0.0`` is a deliberate opt-out, and it has to be exactly inert.

        Not merely small: the density term must be the whole objective, so the
        number is comparable with a single-channel run's.
        """
        import torch

        from poraque.ml.losses import PhysicsInformedLoss

        target = torch.randn(2, 2, 6, 6, 6,
                             generator=torch.Generator().manual_seed(3))
        prediction = target.clone()
        prediction[:, 1] += 5.0            # a large magnetization error
        criterion = PhysicsInformedLoss(task="ext2pseudo",
                                        spin_loss="per_channel",
                                        magnetization_weight=0.0)
        charge_only = target.clone()
        charge_only[:, 1] = target[:, 1]
        assert float(criterion(prediction, target)["total"]) == pytest.approx(
            0.0, abs=1e-12)

        # And with the density wrong instead, the term is the density's error
        # over the DENSITY's own norm -- not the joint one, which is the
        # change -- and unaffected by the magnetization weight.
        prediction = target.clone()
        prediction[:, 0] += 0.1
        value = float(criterion(prediction, target)["total"])
        difference = (prediction[:, 0] - target[:, 0]).flatten(1).norm(dim=1)
        expected = float((difference
                          / target[:, 0].flatten(1).norm(dim=1)).mean())
        assert value == pytest.approx(expected, rel=1e-6)

    def test_the_weight_changes_what_a_spin_run_optimizes(self, spin_dataset):
        """End to end: two weights, two different training losses."""
        from poraque.ml import FieldOperator, train
        from poraque.ml.losses import PhysicsInformedLoss

        dataset = FieldPairDataset(spin_dataset, task="ext2pseudo")
        source_transform, target_transform = dataset.fit_transforms()
        losses = {}
        for weight in (0.0, 5.0):
            operator = FieldOperator("ext2pseudo", in_channels=1,
                                     out_channels=2, width=6, modes=3,
                                     n_layers=1, projection_channels=8,
                                     device="cpu", init_seed=0,
                                     input_transform=source_transform,
                                     target_transform=target_transform)
            history = train(operator, dataset, epochs=1, batch_size=1,
                            verbose=False, log=lambda *_: None,
                            loss=PhysicsInformedLoss(
                                task="ext2pseudo",
                                spin_loss="per_channel",
                                magnetization_weight=weight))
            losses[weight] = history["train_loss"][-1]
        assert losses[0.0] != losses[5.0]
        # The magnetization term is additive and non-negative, so weighting it
        # up can only raise the objective.
        assert losses[5.0] > losses[0.0]
