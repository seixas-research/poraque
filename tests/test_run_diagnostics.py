# -*- coding: utf-8 -*-
# file: test_run_diagnostics.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
What a run reports about itself: its cost, its parity, its two channels.

Three things added together, because they are read together:

* the **resource profile** --- wall time per stage, peak RAM, peak VRAM --- is
  what an HPC allocation is sized from, and it must print even when the run
  fails or stops after the cache, since those are the runs whose cost matters
  most;
* the **parity figure** covers every structure of each split, through a fixed
  random sample of each one's voxels folded into a sparse histogram. Before, it
  drew the first training structure against the first validation one; and a
  split is too large to hold as voxels at all;
* a **spin-polarized** operator's objective is over (rho, m) at once, so its
  log now says how much of it is each channel.
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

from poraque.ml.profiling import (  # noqa: E402
    ResourceProfile,
    format_bytes,
    format_seconds,
    peak_rss_bytes,
)
from poraque.vis.parity import ParityAccumulator  # noqa: E402


# ===================================================================== #
# Resource profile
# ===================================================================== #
class TestTheProfileTimesAndWeighsEachStage:
    """Stages are marked imperatively and closed by the next one."""

    def test_begin_closes_the_open_stage(self):
        profile = ResourceProfile("cpu")
        profile.begin("cache")
        profile.begin("ext2pseudo: training")
        profile.end()
        profile.end()                     # a second end is a no-op
        assert [s["name"] for s in profile.stages] == ["cache",
                                                       "ext2pseudo: training"]
        assert all(s["status"] == "ok" and s["seconds"] >= 0
                   for s in profile.stages)

    def test_a_stage_left_open_is_reported_as_interrupted(self):
        """The stage a failed job died in is the row it most needs."""
        profile = ResourceProfile("cpu")
        profile.begin("ext2pseudo: training")
        text = "\n".join(profile.summary())
        assert "ext2pseudo: training  (interrupted)" in text
        assert profile.stages[-1]["status"] == "interrupted"

    def test_peak_rss_is_a_high_water_mark_in_bytes(self):
        """
        ``ru_maxrss`` is bytes on macOS and KiB on Linux; a wrong unit is a
        factor of 1024 in either direction, which an allocation of known size
        exposes.
        """
        before = peak_rss_bytes()
        block = np.ones(64 * 1024 * 1024 // 8)          # 64 MiB, touched
        block[::4096] = 2.0
        after = peak_rss_bytes()
        assert after >= before
        assert 16 * 1024 ** 2 < after < 64 * 1024 ** 4
        del block
        assert peak_rss_bytes() >= after                # never falls

    def test_the_summary_is_a_table_with_a_total(self):
        profile = ResourceProfile("cpu")
        profile.begin("cache")
        profile.begin("pseudo2tau: training")
        profile.end()
        lines = profile.summary()
        text = "\n".join(lines)
        assert "[RESOURCES]" in text
        assert "wall time" in text and "peak RSS" in text
        assert "total (wall clock)" in text
        # No VRAM column on a CPU run rather than a column of dashes.
        assert "peak VRAM" not in text
        header = next(line for line in lines if "wall time" in line)
        row = next(line for line in lines if "pseudo2tau: training" in line)
        assert len(row) == len(header)

    def test_the_record_is_json_ready(self):
        import json

        profile = ResourceProfile("cpu")
        profile.begin("cache")
        profile.end()
        record = json.loads(json.dumps(profile.as_dict()))
        assert record["stages"][0]["name"] == "cache"
        assert record["peak_rss_bytes"] > 0
        assert record["peak_vram_bytes"] is None

    def test_no_workers_means_no_worker_line(self):
        """
        An ext2chg run with ``num_workers: 0`` reported "DataLoader workers
        peaked at 3.0 GiB each". On Linux a child's ``ru_maxrss`` counts the
        pages it inherited at fork, so the LaTeX call that typesets the PDF
        read as a child the size of the parent.
        """
        import subprocess

        subprocess.run([sys.executable, "-c", "pass"], check=True)
        quiet = ResourceProfile("cpu")
        quiet.begin("cache")
        assert quiet.as_dict()["peak_worker_rss_bytes"] is None
        assert "DataLoader workers" not in "\n".join(quiet.summary())

        busy = ResourceProfile("cpu", workers=2)
        busy.begin("cache")
        assert busy.as_dict()["peak_worker_rss_bytes"] > 0

    def test_the_formats(self):
        assert format_seconds(42.34) == "42.3 s"
        assert format_seconds(782.1) == "13m 02.1s"
        assert format_seconds(7503) == "2h 05m 03s"
        assert format_bytes(None) == "—"
        assert format_bytes(3 * 1024 ** 3) == "3.0 GiB"


class TestPoraqueTrainPrintsTheSummaryLast:
    """At the end of a run, after a cache-only stop, and after a failure."""

    @staticmethod
    def _config(tmp_path):
        from test_cache_only import _config

        return _config(tmp_path)

    def test_a_training_run(self, tmp_path, capsys):
        import poraque_train

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poraque_train.run(["--config", self._config(tmp_path)])
        out = capsys.readouterr().out
        summary = out[out.rindex("[RESOURCES]"):]
        for stage in ("cache", "ext2pseudo: training",
                      "ext2pseudo: evaluation and figures", "pseudo2tau: training",
                      "checkpoint", "total (wall clock)"):
            assert stage in summary, stage
        # Nothing of the run itself comes after it.
        assert "[OVERALL]" not in summary

    def test_a_cache_only_run(self, tmp_path, capsys):
        import poraque_train

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit):
                poraque_train.run(["--config", self._config(tmp_path),
                                   "--cache-only"])
        summary = capsys.readouterr().out.split(
            "[RESOURCES]")[-1]
        assert "cache" in summary and "training" not in summary

    def test_a_run_that_fails_reports_where(self, tmp_path, capsys,
                                            monkeypatch):
        import poraque_train

        def broken(*args, profile=None, **kwargs):
            profile.begin("ext2pseudo: training")
            raise RuntimeError("out of memory, pretend")

        monkeypatch.setattr(poraque_train, "run_task", broken)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(RuntimeError, match="pretend"):
                poraque_train.run(["--config", self._config(tmp_path)])
        summary = capsys.readouterr().out.split(
            "[RESOURCES]")[-1]
        assert "ext2pseudo: training  (interrupted)" in summary


# ===================================================================== #
# Global parity
# ===================================================================== #
class TestParityIsAccumulatedNotHeld:
    """A sample per structure, a sparse histogram, running sums."""

    def test_the_statistics_are_exact_on_what_was_sampled(self):
        accumulator = ParityAccumulator(samples=10_000)
        xs, ys = [], []
        for index in range(12):
            x = np.exp(np.random.default_rng(index).normal(size=(20, 20, 20)))
            y = 1.03 * x - 0.01
            accumulator.add(x, y)       # 8000 voxels: all of them are taken
            xs.append(x.ravel())
            ys.append(y.ravel())
        x, y = np.concatenate(xs), np.concatenate(ys)
        d = x - y
        got = accumulator.metrics()
        assert got["relative_l2"] == pytest.approx(
            np.linalg.norm(d) / np.linalg.norm(x), rel=1e-12)
        assert got["r2"] == pytest.approx(
            1 - (d ** 2).sum() / ((x - x.mean()) ** 2).sum(), rel=1e-12)
        assert got["mae"] == pytest.approx(np.abs(d).mean(), rel=1e-12)
        assert got["rmse"] == pytest.approx(np.sqrt((d ** 2).mean()),
                                            rel=1e-12)
        assert (got["voxels"], got["structures"]) == (96_000, 12)

    def test_each_structure_contributes_the_same_sample(self):
        """A 64^3 cell and an 8^3 one: 10 000 voxels and all 512."""
        accumulator = ParityAccumulator(samples=10_000)
        accumulator.add(np.ones((64, 64, 64)), np.ones((64, 64, 64)))
        accumulator.add(np.ones((8, 8, 8)), np.ones((8, 8, 8)))
        assert accumulator.count == 10_512

    def test_memory_follows_occupied_bins_not_structures(self):
        """
        The OOM the design exists to prevent: a thousand structures of the same
        distribution occupy about the bins one does, and no voxel is stored.
        """
        accumulator = ParityAccumulator(samples=2000)
        rng = np.random.default_rng(0)
        sizes = []
        for index in range(300):
            x = np.exp(rng.normal(size=(16, 16, 16)))
            accumulator.add(x, x * np.exp(0.02 * rng.normal(size=x.shape)))
            if index in (29, 299):
                sizes.append(accumulator.occupied_bins)
        assert sizes[1] < 3 * sizes[0]
        held = [value for value in vars(accumulator).values()
                if isinstance(value, np.ndarray)]
        assert sum(value.size for value in held) < 16

    def test_the_draw_is_reproducible_and_the_histogram_keeps_every_sample(
            self):
        def run():
            accumulator = ParityAccumulator(samples=500, seed=7)
            rng = np.random.default_rng(3)
            for _ in range(5):
                x = np.exp(rng.normal(size=(12, 12, 12)))
                accumulator.add(x, x * 1.1)
            return accumulator

        a, b = run(), run()
        assert a.metrics() == b.metrics()
        lower, upper = a.extent(log=True)
        edges = np.logspace(np.log10(lower), np.log10(upper), 101)
        counts = a.histogram(edges, log=True)
        assert counts.sum() == a.count - a.nonpositive == 2500

    def test_a_signed_field_is_binned_linearly(self):
        accumulator = ParityAccumulator(samples=1000)
        rng = np.random.default_rng(1)
        accumulator.add(rng.normal(size=(10, 10, 10)),
                        rng.normal(size=(10, 10, 10)))
        assert accumulator.positive_share() < 0.5
        lower, upper = accumulator.extent(log=False)
        counts = accumulator.histogram(np.linspace(lower, upper, 51),
                                       log=False)
        assert counts.sum() == 1000


class TestTheParityFigureCoversEveryStructure:
    """Both splits, every structure, one figure, and its data beside it."""

    def test_the_figure_and_its_counts(self, tmp_path):
        import csv

        pytest.importorskip("matplotlib")
        from poraque.vis import TrainingReport

        rng = np.random.default_rng(0)
        train, validation = ParityAccumulator(), ParityAccumulator()
        for index in range(10):
            x = np.exp(rng.normal(-1, 1, size=(24, 24, 24)))
            (train if index < 7 else validation).add(x, x * 1.05)
        report = TrainingReport(str(tmp_path), save_data=True)
        path = report.global_parity(train, validation, label="rho", log=True)
        assert os.path.exists(path)

        totals = {}
        with open(tmp_path / "parity.csv") as handle:
            for row in csv.DictReader(handle):
                totals[row["split"]] = (totals.get(row["split"], 0)
                                        + int(row["count"]))
        assert totals == {"training set": 7 * 10_000,
                          "validation set": 3 * 10_000}

    def test_an_all_non_positive_prediction_still_draws(self, tmp_path):
        pytest.importorskip("matplotlib")
        from poraque.vis import TrainingReport

        accumulator = ParityAccumulator()
        accumulator.add(np.ones((8, 8, 8)), -np.ones((8, 8, 8)))
        path = TrainingReport(str(tmp_path)).global_parity(accumulator,
                                                           log=True)
        assert os.path.exists(path)

    def test_a_training_run_draws_every_structure(self, tmp_path,
                                                 monkeypatch):
        """
        Before, the figure drew the first training structure against the first
        validation one. The accumulators it is handed now hold every structure
        of each split, all 512 voxels of each 8^3 cell (under the per-structure
        sample), for both tasks.
        """
        pytest.importorskip("matplotlib")
        import yaml

        import poraque_train
        from poraque.vis import TrainingReport
        from test_cache_only import _config
        from test_data_sources import write_calculation

        path = _config(tmp_path)
        settings = yaml.safe_load(open(path))
        settings["report"]["figures"] = True
        settings["training"]["validation"]["fraction"] = 0.5
        with open(path, "w") as handle:
            yaml.safe_dump(settings, handle)
        for index in range(2, 6):
            write_calculation(tmp_path / "runs" / f"structure_{index:04d}",
                              shape=(8, 8, 8), seed=index)

        drawn = {}
        original = TrainingReport.global_parity

        def spy(report, training, validation=None, name="parity", **kwargs):
            drawn[(report.prefix, name)] = (training, validation)
            return original(report, training, validation, name=name, **kwargs)

        monkeypatch.setattr(TrainingReport, "global_parity", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poraque_train.run(["--config", path])

        assert set(drawn) == {("ext2pseudo", "parity"), ("pseudo2tau", "parity")}
        for training, validation in drawn.values():
            assert training.structures + validation.structures == 6
            assert validation.structures >= 1
            assert training.count == 512 * training.structures
            assert validation.count == 512 * validation.structures
        plots = tmp_path / "models" / "cache_only_smoke" / "plots"
        assert (plots / "ext2pseudo_parity.png").exists()


# ===================================================================== #
# The two channels of a spin-polarized objective
# ===================================================================== #
class TestTheChargeAndMagnetizationPartsOfTheLoss:
    """
    The two channels, reported separately and combined by `training.spin_loss`.

    `joint` (the default) is one norm over both channels -- each channel's
    error over the whole (rho, m) target's norm, added in quadrature, which is
    identically ||d(rho, m)|| / ||(rho, m)||. `per_channel` divides each
    channel by its own norm (floored) and sums them with
    magnetization_weight.

    The default is `joint` **because it was measured to be better**, against
    the arithmetic. Under `joint` the magnetization's share of the objective
    is 3e-8 in the median on the 432-cell CHON set, which makes `per_channel`
    look obviously right; measured head to head on that cache it is not.
    With 12 magnetic and 100 noise cells -- the real 11 % imbalance -- `joint`
    reaches predicted/reference ||m|| of 0.96 and `per_channel` 0.001, because
    the floored term gives each of the many noise cells an order-one "drive m
    to zero" gradient that buries the few magnetic ones. What trains the
    magnetization is the mean share, not the median.

    These tests pin the arithmetic of both modes and that `joint` reproduces
    the single-norm loss it replaced bit for bit.
    """

    def test_per_channel_sums_the_parts_and_joint_takes_quadrature(self):
        """The two modes' combination rules, each against the other's."""
        from poraque.ml.losses import channel_errors, data_error

        rng = torch.Generator().manual_seed(0)
        target = torch.randn(3, 2, 8, 8, 8, generator=rng)
        prediction = target + 0.1 * torch.randn(3, 2, 8, 8, 8, generator=rng)

        parts = channel_errors(prediction, target, spin_loss="per_channel")
        assert parts.shape == (3, 2)
        assert torch.allclose(
            parts.sum(dim=1),
            data_error(prediction, target, spin_loss="per_channel"),
            rtol=1e-6)

        joint_parts = channel_errors(prediction, target, spin_loss="joint")
        assert torch.allclose(
            joint_parts.pow(2).sum(dim=1).sqrt(),
            data_error(prediction, target, spin_loss="joint"), rtol=1e-6)

    def test_the_weight_scales_the_magnetization_term_alone(self):
        from poraque.ml.losses import channel_errors, data_error

        rng = torch.Generator().manual_seed(1)
        target = torch.randn(2, 2, 6, 6, 6, generator=rng)
        prediction = target + 0.1 * torch.randn(2, 2, 6, 6, 6, generator=rng)
        parts = channel_errors(prediction, target, spin_loss="per_channel")
        for weight in (0.0, 0.5, 3.0):
            assert torch.allclose(
                data_error(prediction, target, spin_loss="per_channel",
                           magnetization_weight=weight),
                parts[:, 0] + weight * parts[:, 1], rtol=1e-6)

    def test_a_single_channel_target_is_untouched_by_the_split(self):
        """
        An unpolarized run must be bit-identical to before the change.

        Every platinum config sets `spin: false`, so the separation may not
        move a number on the runs this project has already measured.
        """
        from poraque.ml.losses import data_error, relative_error

        rng = torch.Generator().manual_seed(2)
        target = torch.randn(3, 1, 8, 8, 8, generator=rng)
        prediction = target + 0.1 * torch.randn(3, 1, 8, 8, 8, generator=rng)
        for weight in (0.0, 1.0, 7.0):
            assert torch.equal(
                data_error(prediction, target, magnetization_weight=weight),
                relative_error(prediction, target))

    @pytest.mark.parametrize("name", ["absolute_l2", "relative_l2",
                                      "absolute_h1", "relative_h1"])
    def test_one_channel_reproduces_the_plain_loss_it_replaced(self, name):
        """
        Each of the four objectives, against the class it used to be.

        `spectral_gradient` does **not** keep a width-1 channel axis: it
        returns (B, 3, Nx, Ny, Nz) for one channel and (B, C, 3, ...) only for
        C > 1. Reducing the first of those on axis 1 reads the three spatial
        components as three channels, which tripled the value term of a
        single-channel H1 objective and handed the magnetization weight to two
        of the three derivatives. Nothing caught it: the objective and the
        per-sample error shared the bug, so they still agreed with each other,
        and the single-channel identity above only exercised relative_l2.
        """
        from poraque.ml.losses import (DATA_LOSSES, AbsoluteL2Loss,
                                       RelativeL2Loss, SobolevLoss,
                                       build_data_loss, channel_errors)

        # float64, so the assertion can be tight: the two paths reduce in a
        # different order and agree only to float32 epsilon otherwise, which
        # is far too loose to catch the factor of three this pins.
        rng = torch.Generator().manual_seed(4)
        target = torch.randn(2, 1, 8, 8, 8, generator=rng,
                             dtype=torch.float64)
        prediction = target + 0.1 * torch.randn(2, 1, 8, 8, 8, generator=rng,
                                                dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64).expand(2, 3, 3) * 5.0
        spec = DATA_LOSSES[name]

        # One channel means one column, whatever the objective.
        assert channel_errors(prediction, target, cell, loss=name,
                              sobolev_weight=0.3).shape == (2, 1)

        if spec["gradient"]:
            previous = SobolevLoss(0.3, relative=spec["relative"])(
                prediction, target, cell)
        elif spec["relative"]:
            previous = RelativeL2Loss()(prediction, target)
        else:
            previous = AbsoluteL2Loss()(prediction, target)
        current = build_data_loss(name, sobolev_weight=0.3)(
            prediction, target, cell)
        assert float(current) == pytest.approx(float(previous), rel=1e-9)

    def test_two_channels_keep_one_column_each(self):
        """The other half of the shape contract the H1 bug broke."""
        from poraque.ml.losses import DATA_LOSSES, channel_errors

        rng = torch.Generator().manual_seed(5)
        target = torch.randn(2, 2, 8, 8, 8, generator=rng)
        prediction = target + 0.1 * torch.randn(2, 2, 8, 8, 8, generator=rng)
        cell = torch.eye(3).expand(2, 3, 3) * 5.0
        for name in DATA_LOSSES:
            assert channel_errors(prediction, target, cell, loss=name,
                                  sobolev_weight=0.3).shape == (2, 2), name

    def test_the_parts_stay_finite_when_the_magnetization_is_zero(self):
        """
        The reason the magnetization's denominator carries a floor.

        m == 0 identically on 126 of the 432 CHON cells, so an unfloored
        per-channel relative error would divide by zero on a quarter of the
        set. The floored term stays finite and still penalizes a spurious
        moment, which is the behavior that matters: the old operator invented
        one on precisely these cells.
        """
        from poraque.ml.losses import channel_errors, data_error

        target = torch.zeros(1, 2, 6, 6, 6)
        target[:, 0] = 1.0
        prediction = target.clone()
        prediction[:, 1] = 0.01
        parts = channel_errors(prediction, target, spin_loss="per_channel")
        assert math.isfinite(float(parts[0, 1]))
        assert float(parts[0, 1]) > 0.0
        assert math.isfinite(float(data_error(prediction, target,
                                              spin_loss="per_channel")))

    def test_each_channel_is_measured_against_its_own_norm(self):
        """
        The change the shared denominator did not make.

        A cell whose magnetization is 5 % of its density: predicting m = 0
        must read as ~100 % wrong in the mag column, not as 5 % of the joint
        norm. Under the old shared denominator this term was
        ||m|| / ||(rho, m)|| ~ 0.05, and on the median CHON cell ~3e-8.
        """
        from poraque.ml.losses import channel_errors

        target = torch.zeros(2, 2, 8, 8, 8)
        target[:, 0] = 1.0
        target[:, 1] = 0.05
        prediction = target.clone()
        prediction[:, 1] = 0.0                  # no magnetization at all

        parts = channel_errors(prediction, target, spin_loss="per_channel")
        assert float(parts[0, 0]) == pytest.approx(0.0, abs=1e-12)
        # Its own norm, floored: ||m|| = 0.05||rho|| is well above the 1e-2
        # floor, so this is the true relative error and close to 1.
        assert float(parts[0, 1]) > 0.9

        # The old arrangement, for contrast: the same error shared against the
        # whole target's norm is the number that made this unlearnable.
        joint = float((prediction - target).flatten(1).norm(dim=1)[0]
                      / target.flatten(1).norm(dim=1)[0])
        assert joint < 0.06
        assert float(parts[0, 1]) > 15.0 * joint

    def test_the_floor_bounds_the_term_on_a_dead_cell(self):
        """A dead cell cannot contribute an unbounded magnetization term."""
        from poraque.ml.losses import channel_errors

        target = torch.zeros(1, 2, 6, 6, 6)
        target[:, 0] = 1.0                      # m identically zero
        prediction = target.clone()
        prediction[:, 1] = 1e-4                 # a small spurious moment

        tight = channel_errors(prediction, target, spin_loss="per_channel",
                               magnetization_floor=1e-3)
        loose = channel_errors(prediction, target, spin_loss="per_channel",
                               magnetization_floor=1e-1)
        # Same error, read against two floors: the tighter floor penalizes it
        # more, and both are finite.
        assert math.isfinite(float(tight[0, 1]))
        assert float(tight[0, 1]) > float(loose[0, 1])
        assert float(loose[0, 1]) == pytest.approx(1e-3, rel=1e-3)

    def test_the_magnetization_term_has_a_gradient_of_its_own(self):
        """
        The mechanism, stated as a measurement.

        With a density error a thousand times the magnetization error, a norm
        over both channels delivers a gradient to m suppressed by that ratio;
        a sum delivers one that does not depend on the density's error at all.
        """
        from poraque.ml.losses import (AbsoluteL2Loss, build_data_loss)

        target = torch.zeros(1, 2, 6, 6, 6)
        target[:, 0] = 1.0

        def magnetization_gradient(objective):
            prediction = target.clone()
            prediction[:, 0] += 1.0          # a large density error
            prediction[:, 1] += 1e-3         # a small magnetization error
            prediction.requires_grad_(True)
            objective(prediction, target).backward()
            return float(prediction.grad[:, 1].abs().max())

        separated = magnetization_gradient(
            lambda p, t: build_data_loss("absolute_l2",
                                         spin_loss="per_channel")(p, t))
        joined = magnetization_gradient(AbsoluteL2Loss())
        assert separated > 100.0 * joined

    @pytest.fixture
    def spin_cache(self, tmp_path):
        from poraque.data import build_field_cache
        from test_data_sources import write_calculation
        from test_ext2chg import _with_records

        for index in range(4):
            _with_records(write_calculation(
                tmp_path / "runs" / f"structure_{index:04d}",
                shape=(12, 12, 12), seed=index), spin=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            build_field_cache([str(tmp_path / "runs")], str(tmp_path / "c"),
                              resolution=12, spin=True, charges={"Si": 4.0},
                              log=lambda *_: None)
        return str(tmp_path / "c")

    def test_a_spin_run_logs_both_and_records_them(self, spin_cache):
        from poraque.ml import FieldOperator, train
        from poraque.ml.data import FieldPairDataset

        dataset = FieldPairDataset(spin_cache, "ext2pseudo")
        assert dataset.channels == (1, 2)
        operator = FieldOperator("ext2pseudo", width=4, modes=2, n_layers=1,
                                 projection_channels=4, device="cpu",
                                 in_channels=1, out_channels=2)
        lines = []
        history = train(operator, dataset, epochs=2, batch_size=2,
                        eval_every=1, log=lines.append, verbose=True)
        header = next(line for line in lines if "train loss" in line
                      and "epoch" in line)
        assert "charge" in header and "mag" in header
        assert len(history["train_loss_charge"]) == 2
        assert len(history["train_loss_magnetization"]) == 2
        assert any("sqrt(charge^2 + mag^2)" in line for line in lines)

    def test_a_one_channel_run_has_no_such_columns(self, spin_cache):
        from poraque.ml import FieldOperator, train
        from poraque.ml.data import FieldPairDataset

        dataset = FieldPairDataset(spin_cache, "pseudo2tau")
        operator = FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=4, device="cpu",
                                 in_channels=2, out_channels=1)
        lines = []
        history = train(operator, dataset, epochs=1, batch_size=2,
                        eval_every=1, log=lines.append, verbose=True)
        assert "train_loss_charge" not in history
        assert not any("mag" in line for line in lines if "epoch" in line)


class TestAKilledRunKeepsItsBestWeights:
    """
    Before, a fit that died left nothing: ``train()`` has always taken a
    ``checkpoint`` path and written to it on every improvement, but
    ``poraque-train`` never passed one, so the best weights sat in memory until
    the run finished. A 3-hour ``ext2chg`` fit was killed at epoch 240 with a
    better model than its predecessor's already measured and no file to show
    for it.
    """

    def test_the_best_weights_reach_disk_while_the_fit_runs(self, tmp_path):
        import poraque_train
        from poraque.ml import FieldOperator
        from test_cache_only import _config

        import yaml

        from poraque.ml.config import TrainingConfig
        from poraque.ml.tasks import TASKS

        config = TrainingConfig.from_dict(
            yaml.safe_load(open(_config(tmp_path))))
        progress = poraque_train.progress_checkpoint_path(config,
                                                          TASKS["ext2pseudo"])
        assert progress.endswith("_ext2pseudo_best_so_far.pt")

        operator = FieldOperator("ext2pseudo", width=4, modes=2, n_layers=1,
                                 projection_channels=4, device="cpu")
        operator.save(progress)                      # what train() does on '*'
        assert os.path.exists(progress)
        # A bare operator state, not a bundle: it reloads on its own.
        back = FieldOperator.load(progress, device="cpu")
        assert back.task.name == "ext2pseudo"

    def test_the_finished_bundle_supersedes_it(self, tmp_path, capsys):
        import poraque_train

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            poraque_train.run(["--config", _cache_only_config(tmp_path)])
        run = tmp_path / "models" / "cache_only_smoke"
        assert not list(run.glob("*_best_so_far.pt")), \
            "the in-progress copy outlived the bundle that supersedes it"
        assert list(run.glob("*.poraque"))


def _cache_only_config(tmp_path):
    from test_cache_only import _config

    return _config(tmp_path)


class TestTheDataTermHasAGradientAtZeroError:
    """
    A prediction that is exactly right must not produce NaN gradients.

    `combine_channel_parts` reconstitutes the joint norm over the channels,
    and writing that as `pow(2).sum().sqrt()` rather than `norm()` agrees in
    value and differs in gradient at the origin: the explicit form is 0/0
    there and back-propagates NaN. `tests/test_physics_informed.py` scores an
    exactly-correct prediction on purpose -- to check a physics term's
    gradient survives a zero data term -- and that is what caught it.
    """

    @pytest.mark.parametrize("channels", [1, 2])
    @pytest.mark.parametrize("name", ["absolute_l2", "relative_l2",
                                      "absolute_h1", "relative_h1"])
    @pytest.mark.parametrize("mode", ["joint", "per_channel"])
    def test_a_perfect_prediction_back_propagates_zero_not_nan(
            self, channels, name, mode):
        from poraque.ml.losses import build_data_loss

        prediction = torch.randn(2, channels, 6, 6, 6, requires_grad=True)
        cell = torch.eye(3).expand(2, 3, 3) * 4.0
        objective = build_data_loss(name, sobolev_weight=0.3, spin_loss=mode)
        objective(prediction, prediction.detach(), cell).backward()
        assert torch.isfinite(prediction.grad).all()
        assert float(prediction.grad.abs().max()) == pytest.approx(0.0,
                                                                   abs=1e-12)
