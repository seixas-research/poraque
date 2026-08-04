# -*- coding: utf-8 -*-
# file: test_split.py

"""
Tests for the train/validation split, the evaluation interval and early
stopping.

The split is defined by one number, ``valid_fraction``; K-fold cross-validation
is the only variation on the protocol.

The split resolver lives in ``scripts/poraque_train.py`` rather than the package,
so it is imported by path here — the alternative, duplicating the logic in a
library module, would let the two drift apart and only the script's copy is
what actually runs.
"""

import importlib.util
import os
import sys

import pytest
import torch

from poraque.ml import FieldOperator, train
from poraque.ml.config import TrainingConfig

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_poraque_train():
    """Import ``scripts/poraque_train.py`` as a module."""
    path = os.path.join(_ROOT, "scripts", "poraque_train.py")
    spec = importlib.util.spec_from_file_location("_poraque_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_poraque_train"] = module
    spec.loader.exec_module(module)
    return module


poraque_train = _load_poraque_train()


class _Material:
    """Stand-in for a dataset record; only the identifier is consulted."""

    def __init__(self, identifier):
        self.identifier = identifier


class _Dataset:
    def __init__(self, count):
        self.materials = [_Material(f"struct_{i:03d}") for i in range(count)]


# ===================================================================== #
# valid_fraction
# ===================================================================== #
class TestValidFraction:
    def test_default_is_a_fifth(self):
        assert TrainingConfig().training.valid_fraction == pytest.approx(0.2)

    def test_zero_falls_back_to_the_universal_fit(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.0
        validation, origin = poraque_train.resolve_validation_split(_Dataset(5), config)
        assert validation == set()
        assert "TRAINING FIT" in origin

    @pytest.mark.parametrize("fraction, expected", [
        (0.2, 2), (0.3, 3), (0.5, 5), (0.1, 1),
    ])
    def test_reserves_the_requested_share(self, fraction, expected):
        config = TrainingConfig()
        config.training.valid_fraction = fraction
        validation, _ = poraque_train.resolve_validation_split(_Dataset(10), config)
        assert len(validation) == expected

    def test_split_is_at_the_structure_level(self):
        """Whole materials move together; identifiers come out intact."""
        config = TrainingConfig()
        config.training.valid_fraction = 0.4
        validation, _ = poraque_train.resolve_validation_split(_Dataset(5), config)
        assert validation <= {f"struct_{i:03d}" for i in range(5)}

    def test_is_reproducible_for_a_given_seed(self):
        def draw(seed):
            config = TrainingConfig()
            config.training.valid_fraction = 0.3
            config.training.seed = seed
            return poraque_train.resolve_validation_split(_Dataset(10), config)[0]

        assert draw(0) == draw(0)

    def test_seed_changes_the_draw(self):
        """Otherwise 'randomized' would be a claim the code does not make."""
        draws = set()
        for seed in range(8):
            config = TrainingConfig()
            config.training.valid_fraction = 0.3
            config.training.seed = seed
            draws.add(frozenset(
                poraque_train.resolve_validation_split(_Dataset(10), config)[0]))
        assert len(draws) > 1

    def test_never_empties_either_side(self):
        """
        A fraction that rounds to zero would silently degrade into a universal
        fit while the config still claims a validation split.
        """
        config = TrainingConfig()
        config.training.valid_fraction = 0.01
        validation, _ = poraque_train.resolve_validation_split(_Dataset(3), config)
        assert len(validation) == 1

        config.training.valid_fraction = 0.99
        validation, _ = poraque_train.resolve_validation_split(_Dataset(3), config)
        assert len(validation) == 2      # one structure is always kept to train on

    def test_rejects_a_fraction_outside_the_range(self):
        config = TrainingConfig()
        for bad in (1.0, 1.5, -0.2):
            config.training.valid_fraction = bad
            with pytest.raises(SystemExit, match="valid_fraction"):
                poraque_train.resolve_validation_split(_Dataset(5), config)

    def test_rejects_a_single_structure_dataset(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.5
        with pytest.raises(SystemExit, match="two structures"):
            poraque_train.resolve_validation_split(_Dataset(1), config)

    def test_origin_records_the_protocol(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.25
        config.training.seed = 7
        _, origin = poraque_train.resolve_validation_split(_Dataset(8), config)
        assert "valid_fraction=0.25" in origin and "seed=7" in origin


# ===================================================================== #
# eval_epoch
# ===================================================================== #
class TestEvalInterval:
    def test_config_default_is_ten(self):
        assert TrainingConfig().training.eval_epoch == 10

    def test_reports_only_on_the_interval(self, toy):
        lines = []
        _train_toy(toy, epochs=10, eval_every=3, log=lines.append)
        # epochs 3, 6, 9 and the final 10
        assert len(_rows(lines)) == 4
        assert _rows(lines)[-1].split()[0] == "10/10"

    def test_final_epoch_always_reports(self, toy):
        """A run must not end without a current number."""
        lines = []
        _train_toy(toy, epochs=7, eval_every=5, log=lines.append)
        assert _rows(lines)[-1].split()[0] == "7/7"

    def test_emits_a_column_header(self, toy):
        """
        A bare column of numbers is not self-describing, and the two values are
        measured differently — reading the validation error as if it were the
        objective is an easy and expensive mistake.
        """
        lines = []
        _train_toy(toy, epochs=4, eval_every=2, log=lines.append, validate=True)
        header = next(line for line in lines if "epoch" in line)
        assert "train loss" in header and "val rel L2" in header
        # The legend says what each column actually is.
        legend = next(line for line in lines if "mean" in line)
        assert "per batch" in legend and "physical units" in legend

    def test_header_omits_the_validation_column_when_absent(self, toy):
        lines = []
        _train_toy(toy, epochs=4, eval_every=2, log=lines.append)
        header = next(line for line in lines if "epoch" in line)
        assert "train loss" in header and "val rel L2" not in header
        assert any("TRAINING FIT" in line for line in lines)

    def test_rows_line_up_under_the_header(self, toy):
        """Header and rows share their widths; drift would be silent."""
        lines = []
        _train_toy(toy, epochs=4, eval_every=2, log=lines.append, validate=True)
        header = next(line for line in lines if "epoch" in line)
        for row in _rows(lines):
            assert len(row) == len(header) or row.endswith("*")

    def test_train_loss_is_recorded_every_epoch(self, toy):
        """The interval throttles reporting, not the loss history."""
        history = _train_toy(toy, epochs=9, eval_every=4)
        assert len(history["train_loss"]) == 9

    def test_validation_series_carries_its_epochs(self, toy):
        """
        With eval_every > 1 the validation list is shorter than the training
        one, so anything plotting it needs the matching epoch numbers.
        """
        history = _train_toy(toy, epochs=9, eval_every=4, validate=True)
        assert len(history["val_error"]) == len(history["val_epoch"])
        assert len(history["val_error"]) < len(history["train_loss"])
        assert history["val_epoch"] == [4, 8, 9]

    def test_zero_and_negative_are_clamped_to_every_epoch(self, toy):
        for interval in (0, -3):
            history = _train_toy(toy, epochs=4, eval_every=interval,
                                 validate=True)
            assert history["val_epoch"] == [1, 2, 3, 4]

    def test_silent_when_not_verbose(self, toy):
        lines = []
        _train_toy(toy, epochs=6, eval_every=2, log=lines.append,
                   verbose=False)
        assert lines == []

    def test_loss_curves_plot_against_the_recorded_epochs(self, toy, tmp_path):
        """
        A sparse validation series plotted against 1..N would land on the
        wrong x-axis and silently misreport when the model improved.
        """
        pytest.importorskip("matplotlib")
        from poraque.vis import TrainingReport

        history = _train_toy(toy, epochs=12, eval_every=4, validate=True)
        report = TrainingReport(str(tmp_path), prefix="t")
        assert os.path.exists(report.loss_curves(history))


# ===================================================================== #
# early_stopping
# ===================================================================== #
class TestEarlyStopping:
    def test_config_default_is_fifty(self):
        assert TrainingConfig().training.early_stopping == 50

    def test_triggers_once_patience_runs_out(self, toy, monkeypatch):
        """
        Driven by a scripted validation curve rather than a real one: a toy
        model may or may not plateau, and a test that only sometimes exercises
        the branch is not a test of it.
        """
        errors = iter([1.0, 0.5, 0.6, 0.7, 0.8, 0.9] + [1.0] * 50)
        monkeypatch.setattr("poraque.ml.training.evaluate",
                            lambda *a, **k: next(errors))
        lines = []
        history = _train_toy(toy, epochs=50, eval_every=1, validate=True,
                             early_stopping=3, log=lines.append)

        assert history["stopped_early"] is True
        assert history["best_epoch"] == 2          # the 0.5
        assert history["best_error"] == pytest.approx(0.5)
        # epoch 5 is the third without improvement since epoch 2
        assert len(history["val_error"]) == 5
        assert any("stopped early at epoch 5" in line for line in lines)

    def test_does_not_trigger_while_improving(self, toy, monkeypatch):
        errors = iter([1.0 / (i + 1) for i in range(40)])
        monkeypatch.setattr("poraque.ml.training.evaluate",
                            lambda *a, **k: next(errors))
        history = _train_toy(toy, epochs=8, eval_every=1, validate=True,
                             early_stopping=3)
        assert history["stopped_early"] is False
        assert len(history["train_loss"]) == 8

    def test_restores_the_best_weights(self, toy, monkeypatch):
        """
        Stopping partway down a degrading curve must not hand back the
        degraded model — the point is to keep the best one *measured*.

        Verified by snapshotting the weights at every evaluation, then checking
        the returned model against the snapshot from the best-scoring epoch.
        """
        errors = iter([1.0, 0.1] + [9.0] * 50)
        snapshots = []

        def fake_evaluate(operator, loader):
            snapshots.append({k: v.detach().clone()
                              for k, v in operator.model.state_dict().items()})
            return next(errors)

        monkeypatch.setattr("poraque.ml.training.evaluate", fake_evaluate)

        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        history = train(operator, toy, epochs=30, batch_size=1,
                        learning_rate=1e-2, eval_every=1, validation=toy,
                        early_stopping=2, verbose=False)

        assert history["stopped_early"] is True
        assert history["best_epoch"] == 2

        best = snapshots[1]                      # epoch 2, the 0.1
        last = snapshots[-1]
        final = operator.model.state_dict()
        assert all(torch.allclose(final[k], best[k]) for k in best)
        # And genuinely different from where training actually stopped.
        assert any(not torch.allclose(last[k], best[k]) for k in best)

    def test_zero_disables_it(self, toy, monkeypatch):
        errors = iter([1.0] * 60)
        monkeypatch.setattr("poraque.ml.training.evaluate",
                            lambda *a, **k: next(errors))
        history = _train_toy(toy, epochs=10, eval_every=1, validate=True,
                             early_stopping=0)
        assert history["stopped_early"] is False
        assert len(history["train_loss"]) == 10

    def test_warns_without_a_validation_split(self, toy):
        """
        The training loss falls monotonically by construction, so it can never
        signal that training should stop. Silently doing nothing would leave
        the user believing the run was protected.
        """
        with pytest.warns(RuntimeWarning, match="without a validation split"):
            history = _train_toy(toy, epochs=4, eval_every=2,
                                 early_stopping=2)
        assert "stopped_early" not in history

    def test_patience_is_counted_in_epochs_not_evaluations(self, toy,
                                                           monkeypatch):
        """
        With eval_every=5 and patience=10, two consecutive non-improving
        evaluations exhaust it — not ten.
        """
        errors = iter([0.1] + [9.0] * 50)
        monkeypatch.setattr("poraque.ml.training.evaluate",
                            lambda *a, **k: next(errors))
        history = _train_toy(toy, epochs=60, eval_every=5, validate=True,
                             early_stopping=10)
        assert history["stopped_early"] is True
        assert history["best_epoch"] == 5
        assert history["val_epoch"] == [5, 10, 15]


@pytest.fixture
def toy(tmp_path):
    """A two-material dataset on one grid shape, enough to drive `train`."""
    import numpy as np

    from poraque.fields import ChargeDensity, KineticEnergyDensity
    from poraque.fields import ExternalPotential, FieldGrid
    from poraque.fields.vasp.poscar import Poscar
    from poraque.ml import FieldPairDataset

    rng = np.random.default_rng(0)
    for index in range(2):
        directory = tmp_path / f"mat_{index}"
        directory.mkdir()
        grid = FieldGrid((8, 8, 8), np.eye(3) * 5.0)
        structure = Poscar(np.eye(3) * 5.0, ["Si"], [2], rng.random((2, 3)))
        ExternalPotential.compute(structure, grid, {"Si": 4.0},
                                  widths={"Si": 0.5}).write(directory / "EXTCAR")
        density = rng.random(grid.shape) * 0.1 + 0.01
        ChargeDensity(density, grid, structure).write(directory / "CHGCAR")
        KineticEnergyDensity(density * 50.0, grid,
                             structure).write(directory / "TAUCAR")
    return FieldPairDataset(str(tmp_path), task="chg2tau")


def _rows(lines):
    """The per-epoch data rows, excluding the legend, header and rule."""
    return [line for line in lines
            if line.strip() and line.strip()[0].isdigit()]


def _train_toy(dataset, epochs, eval_every, log=None, validate=False,
               verbose=True, early_stopping=0):
    """Run a tiny training loop and return its history."""
    torch.manual_seed(0)
    operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu")
    return train(operator, dataset, epochs=epochs, batch_size=1,
                 learning_rate=1e-3, eval_every=eval_every,
                 validation=dataset if validate else None,
                 early_stopping=early_stopping, log=log, verbose=verbose)
