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
* a **spin-polarised** operator's objective is over (rho, m) at once, so its
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
        profile.begin("ext2chg: training")
        profile.end()
        profile.end()                     # a second end is a no-op
        assert [s["name"] for s in profile.stages] == ["cache",
                                                       "ext2chg: training"]
        assert all(s["status"] == "ok" and s["seconds"] >= 0
                   for s in profile.stages)

    def test_a_stage_left_open_is_reported_as_interrupted(self):
        """The stage a failed job died in is the row it most needs."""
        profile = ResourceProfile("cpu")
        profile.begin("ext2chg: training")
        text = "\n".join(profile.summary())
        assert "ext2chg: training  (interrupted)" in text
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
        profile.begin("chg2tau: training")
        profile.end()
        lines = profile.summary()
        text = "\n".join(lines)
        assert "RESOURCE PROFILING SUMMARY" in text
        assert "wall time" in text and "peak RSS" in text
        assert "total (wall clock)" in text
        # No VRAM column on a CPU run rather than a column of dashes.
        assert "peak VRAM" not in text
        header = next(line for line in lines if "wall time" in line)
        row = next(line for line in lines if "chg2tau: training" in line)
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
        summary = out[out.rindex("RESOURCE PROFILING SUMMARY"):]
        for stage in ("cache", "ext2chg: training",
                      "ext2chg: evaluation and figures", "chg2tau: training",
                      "checkpoint", "total (wall clock)"):
            assert stage in summary, stage
        # Nothing of the run itself comes after it.
        assert "OVERALL" not in summary

    def test_a_cache_only_run(self, tmp_path, capsys):
        import poraque_train

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit):
                poraque_train.run(["--config", self._config(tmp_path),
                                   "--cache-only"])
        summary = capsys.readouterr().out.split(
            "RESOURCE PROFILING SUMMARY")[-1]
        assert "cache" in summary and "training" not in summary

    def test_a_run_that_fails_reports_where(self, tmp_path, capsys,
                                            monkeypatch):
        import poraque_train

        def broken(*args, profile=None, **kwargs):
            profile.begin("ext2chg: training")
            raise RuntimeError("out of memory, pretend")

        monkeypatch.setattr(poraque_train, "run_task", broken)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(RuntimeError, match="pretend"):
                poraque_train.run(["--config", self._config(tmp_path)])
        summary = capsys.readouterr().out.split(
            "RESOURCE PROFILING SUMMARY")[-1]
        assert "ext2chg: training  (interrupted)" in summary


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
        settings["output"]["plot_figures"] = True
        settings["training"]["valid_fraction"] = 0.5
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

        assert set(drawn) == {("ext2chg", "parity"), ("chg2tau", "parity")}
        for training, validation in drawn.values():
            assert training.structures + validation.structures == 6
            assert validation.structures >= 1
            assert training.count == 512 * training.structures
            assert validation.count == 512 * validation.structures
        plots = tmp_path / "models" / "cache_only_smoke" / "plots"
        assert (plots / "ext2chg_parity.png").exists()


# ===================================================================== #
# The two channels of a spin-polarised objective
# ===================================================================== #
class TestTheChargeAndMagnetisationPartsOfTheLoss:
    """
    Reported beside the objective, never instead of it: the objective is still
    over (rho, m) at once, and a relative error of m alone would divide by
    m = 0 on every non-magnetic cell of the platinum set.
    """

    def test_the_parts_combine_in_quadrature_to_the_data_term(self):
        from poraque.ml.losses import data_error

        rng = torch.Generator().manual_seed(0)
        target = torch.randn(3, 2, 8, 8, 8, generator=rng)
        prediction = target + 0.1 * torch.randn(3, 2, 8, 8, 8, generator=rng)
        whole = data_error(prediction, target)
        parts = []
        for channel in range(2):
            alone = target.clone()
            alone[:, channel] = prediction[:, channel]
            parts.append(data_error(alone, target))
        assert torch.allclose(torch.sqrt(parts[0] ** 2 + parts[1] ** 2),
                              whole, rtol=1e-6)

    def test_the_parts_stay_finite_when_the_magnetisation_is_zero(self):
        from poraque.ml.losses import data_error

        target = torch.zeros(1, 2, 6, 6, 6)
        target[:, 0] = 1.0
        prediction = target.clone()
        prediction[:, 1] = 0.01
        alone = target.clone()
        alone[:, 1] = prediction[:, 1]
        assert math.isfinite(float(data_error(alone, target)))

    @pytest.fixture
    def spin_cache(self, tmp_path):
        from poraque.data import build_field_cache
        from test_data_sources import write_calculation
        from test_ext2paw import _with_records

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

        dataset = FieldPairDataset(spin_cache, "ext2chg")
        assert dataset.channels == (1, 2)
        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=4, device="cpu",
                                 in_channels=1, out_channels=2)
        lines = []
        history = train(operator, dataset, epochs=2, batch_size=2,
                        eval_every=1, log=lines.append, verbose=True)
        header = next(line for line in lines if "train loss" in line
                      and "epoch" in line)
        assert "charge" in header and "mag" in header
        assert len(history["train_loss_charge"]) == 2
        assert len(history["train_loss_magnetisation"]) == 2
        assert any("sqrt(charge^2 + mag^2)" in line for line in lines)

    def test_a_one_channel_run_has_no_such_columns(self, spin_cache):
        from poraque.ml import FieldOperator, train
        from poraque.ml.data import FieldPairDataset

        dataset = FieldPairDataset(spin_cache, "chg2tau")
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
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
    the run finished. A 3-hour ``ext2paw`` fit was killed at epoch 240 with a
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
                                                          TASKS["ext2chg"])
        assert progress.endswith("_ext2chg_best_so_far.pt")

        operator = FieldOperator("ext2chg", width=4, modes=2, n_layers=1,
                                 projection_channels=4, device="cpu")
        operator.save(progress)                      # what train() does on '*'
        assert os.path.exists(progress)
        # A bare operator state, not a bundle: it reloads on its own.
        back = FieldOperator.load(progress, device="cpu")
        assert back.task.name == "ext2chg"

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
