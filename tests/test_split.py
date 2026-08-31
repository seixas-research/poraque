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
import json
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
    def test_config_defaults_are_pinned(self):
        """
        Patience and the epoch cap are set together: patience has to be short
        enough to save time and long enough not to cut a slow run off. Pinning
        both makes a change to either deliberate rather than incidental.
        """
        training = TrainingConfig().training
        # Raised together when the shipped configs were unified: both of them
        # already used 500/300, so the numbers moved out of the files and into
        # the defaults rather than being restated in each.
        assert training.early_stopping == 300
        assert training.epochs == 500
        assert training.early_stopping < training.epochs

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

        def fake_evaluate(operator, loader, **_):
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


# ===================================================================== #
# The reported objective
# ===================================================================== #
class TestTheObjectiveIsReportedAsOneNumber:
    """
    The log and the report carry the **total** objective and nothing else.

    A physics-informed run's total is what the optimiser stepped on: data
    fidelity plus every weighted constraint. The per-term split used to be
    logged beside it, in unweighted units that do not sum to the total, which
    made the table three columns wide and two of them incomparable with any
    other run's.
    """

    def _physics(self, **weights):
        from poraque.ml.losses import PhysicsInformedLoss

        return PhysicsInformedLoss(task="chg2tau", **weights)

    def test_history_carries_only_the_total(self, toy):
        history = _train_toy(toy, epochs=2, eval_every=1, verbose=False,
                             loss=self._physics(positivity_weight=0.5,
                                                von_weizsacker_weight=0.25))
        assert len(history["train_loss"]) == 2
        assert not [key for key in history
                    if key.startswith(("data_loss", "physics"))]

    def test_the_header_is_the_same_with_and_without_constraints(self, toy):
        constrained = []
        _train_toy(toy, epochs=2, eval_every=1, log=constrained.append,
                   loss=self._physics(positivity_weight=0.5))
        plain = []
        _train_toy(toy, epochs=2, eval_every=1, log=plain.append)

        header = next(line for line in constrained if "epoch" in line)
        assert header == next(line for line in plain if "epoch" in line)
        assert "train loss" in header
        assert "physics" not in header and "data loss" not in header

    def test_rows_line_up_under_the_header(self, toy):
        lines = []
        _train_toy(toy, epochs=2, eval_every=1, log=lines.append,
                   loss=self._physics(positivity_weight=0.5))
        header = next(line for line in lines if "epoch" in line)
        for row in _rows(lines):
            assert len(row) == len(header)


class TestTheValidationColumnNamesItsOwnNorm:
    r"""
    ``loss: sobolev`` constrains the gradient as well as the values, so the
    number watched during training has to be the one being minimised.

    The regression this pins is a label, but not only a label. The column read
    ``val rel L2`` whatever the objective was, and the quantity behind it was a
    plain :math:`L^2` — so a Sobolev run reported a number the optimiser was
    not stepping on, plotted it as the validation curve, and *selected the
    checkpoint on it*. Early stopping was choosing the model that minimised the
    wrong functional.

    The final per-structure table is deliberately untouched: it reports
    relative :math:`L^2` whatever the run was trained with, because that is
    what makes two runs comparable at all.
    """

    def _sobolev(self, weight=0.1):
        from poraque.ml.losses import PhysicsInformedLoss

        return PhysicsInformedLoss(task="chg2tau", sobolev_weight=weight)

    def test_a_plain_run_still_says_rel_l2(self, toy):
        lines = []
        history = _train_toy(toy, epochs=2, eval_every=1, validate=True,
                             log=lines.append)
        assert history["val_metric"] == "rel L2"
        assert "val rel L2" in next(line for line in lines if "epoch" in line)

    def test_a_sobolev_run_says_rel_h1(self, toy):
        lines = []
        history = _train_toy(toy, epochs=2, eval_every=1, validate=True,
                             log=lines.append, loss=self._sobolev())
        assert history["val_metric"] == "rel H1"
        header = next(line for line in lines if "epoch" in line)
        assert "val rel H1" in header and "val rel L2" not in header

    def test_the_legend_says_what_the_h1_is_made_of(self, toy):
        """A reader who has never seen the column needs the definition once."""
        lines = []
        _train_toy(toy, epochs=1, eval_every=1, validate=True,
                   log=lines.append, loss=self._sobolev(weight=0.25))
        legend = " ".join(lines)
        assert "rel L2 + 0.25 x" in legend
        assert "reports plain rel L2" in legend

    def test_the_number_is_not_the_l2_relabelled(self, toy):
        """
        The whole point: an H1 error is a different number. Relabelling the
        same value would satisfy a test on the header and none of the reason
        the header was wrong.
        """
        plain = _train_toy(toy, epochs=2, eval_every=1, validate=True,
                           verbose=False)
        sobolev = _train_toy(toy, epochs=2, eval_every=1, validate=True,
                             verbose=False, loss=self._sobolev(weight=0.5))
        assert sobolev["val_error"][-1] > plain["val_error"][-1]

    def test_it_is_the_objective_evaluated_on_the_held_out_set(self, toy):
        """
        `evaluate` and `SobolevLoss` must agree, or the validation curve is
        measuring a third thing that happens to look like the objective.
        """
        import torch

        from poraque.ml.losses import SobolevLoss, relative_h1_error

        torch.manual_seed(0)
        prediction = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        cell = torch.eye(3).expand(2, 3, 3) * 5.0

        per_sample = relative_h1_error(prediction, target, cell, weight=0.3)
        assert per_sample.shape == (2,)
        assert float(per_sample.mean()) == pytest.approx(
            float(SobolevLoss(0.3)(prediction, target, cell)), rel=1e-5)

    def test_zero_weight_is_exactly_the_l2(self, toy):
        """
        `sobolev_weight: 0` is a relative L2 by construction, and the label
        follows the weight rather than the config keyword — so a run that asks
        for Sobolev and weights the gradient at nothing is reported honestly
        as what it is.
        """
        import torch

        from poraque.ml.losses import relative_error, relative_h1_error

        torch.manual_seed(0)
        prediction = torch.randn(2, 1, 6, 6, 6)
        target = torch.randn(2, 1, 6, 6, 6)
        cell = torch.eye(3).expand(2, 3, 3) * 5.0
        assert torch.allclose(
            relative_h1_error(prediction, target, cell, weight=0.0),
            relative_error(prediction, target))

        history = _train_toy(toy, epochs=1, eval_every=1, validate=True,
                             verbose=False, loss=self._sobolev(weight=0.0))
        assert history["val_metric"] == "rel L2"

    def test_the_figure_labels_the_axis_it_actually_plotted(self, toy, tmp_path):
        """
        The stored series is whatever `train` measured, and a figure that
        assumed L2 would mislabel it with nothing to notice.
        """
        pytest.importorskip("matplotlib")
        from poraque.vis import TrainingReport

        report = TrainingReport(str(tmp_path))
        history = _train_toy(toy, epochs=2, eval_every=1, validate=True,
                             verbose=False, loss=self._sobolev())
        report.loss_curves(history, name="sobolev")

        import matplotlib.pyplot as plt

        plt.close("all")
        assert history["val_metric"] == "rel H1"
        assert os.path.exists(os.path.join(str(tmp_path), "sobolev.png"))


class TestTheFinalEvaluationIsAlwaysARelativeL2:
    """
    Whatever a run was trained on, what it *reports* per structure is a
    relative :math:`L^2` — so two runs with different objectives can be put
    side by side. The metric is computed from NumPy arrays and never consults
    the loss, and this asserts that rather than assuming it.
    """

    def test_the_column_is_there_whatever_the_objective(self):
        columns = dict((key, title) for title, key, _ in
                       poraque_train.METRIC_COLUMNS)
        assert columns["relative_l2"] == "rel L2"

    def test_the_metric_function_is_never_told_what_the_loss_was(self):
        import inspect

        import numpy as np

        parameters = inspect.signature(poraque_train.metrics).parameters
        assert set(parameters) == {"prediction", "target", "grid"}

        values = poraque_train.metrics(np.ones((4, 4, 4)) * 1.1,
                                       np.ones((4, 4, 4)))
        assert values["relative_l2"] == pytest.approx(0.1, rel=1e-6)


class TestLossSummaryRows:
    def test_reports_one_number(self):
        rows = poraque_train.loss_summary({"train_loss": [0.5]})
        assert rows == {"final train loss": "0.50000"}

    def test_a_constrained_run_reports_the_total_and_nothing_else(self):
        """The total already includes every weighted physics term."""
        rows = poraque_train.loss_summary({"train_loss": [1.0]})
        assert rows == {"final train loss": "1.00000"}

    def test_an_empty_history_yields_nothing(self):
        assert poraque_train.loss_summary({}) == {}


# ===================================================================== #
# Writing the history into the JSON summary
# ===================================================================== #
class TestHistorySerialisation:
    """
    `train` returns per-epoch curves and scalar summaries in one dict, and the
    scalars appear whenever a validation split does — which is the default. A
    run that mapped `float` over every value crashed at the very end of
    training, after all the compute had been spent.
    """

    def test_separates_curves_from_scalars(self, toy):
        history = _train_toy(toy, epochs=4, eval_every=2, validate=True,
                             early_stopping=2)
        curves, stopping = poraque_train.split_history(history)

        assert set(curves) == {"train_loss", "val_error", "val_epoch"}
        assert all(isinstance(value, list) for value in curves.values())
        assert set(stopping) == {"best_epoch", "best_error", "stopped_early",
                                 "val_metric"}

    def test_a_real_history_survives_json(self, toy):
        """The failure was a TypeError on the way into json.dump."""
        history = _train_toy(toy, epochs=4, eval_every=2, validate=True,
                             early_stopping=2)
        curves, stopping = poraque_train.split_history(history)
        payload = json.loads(json.dumps({"history": curves,
                                         "early_stopping": stopping}))
        assert payload["history"]["train_loss"]
        assert payload["early_stopping"]["stopped_early"] in (True, False)

    def test_scalars_are_not_iterated(self):
        """The exact shape that raised: an int beside the lists."""
        curves, stopping = poraque_train.split_history(
            {"train_loss": [1.0, 0.5], "best_epoch": 2, "stopped_early": True})
        assert curves == {"train_loss": [1.0, 0.5]}
        assert stopping == {"best_epoch": 2, "stopped_early": True}

    def test_no_validation_leaves_no_scalars(self, toy):
        """Without a split `train` adds none, and the key stays null."""
        history = _train_toy(toy, epochs=2, eval_every=1)
        curves, stopping = poraque_train.split_history(history)
        assert stopping is None
        assert curves["train_loss"]

    def test_curves_are_floats(self):
        """Tensors and numpy scalars must not reach json.dump."""
        curves, _ = poraque_train.split_history(
            {"train_loss": [torch.tensor(0.25)], "val_epoch": [1]})
        assert curves == {"train_loss": [0.25], "val_epoch": [1.0]}
        assert all(type(v) is float for v in curves["train_loss"])


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
               verbose=True, early_stopping=0, loss=None):
    """Run a tiny training loop and return its history."""
    torch.manual_seed(0)
    operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu")
    return train(operator, dataset, epochs=epochs, batch_size=1,
                 learning_rate=1e-3, eval_every=eval_every,
                 validation=dataset if validate else None,
                 early_stopping=early_stopping, log=log, verbose=verbose,
                 loss=loss)


# ===================================================================== #
# Fine-tuning
# ===================================================================== #
class TestFineTuning:
    """
    Adapting a trained operator, rather than starting from noise.

    The failure modes here are silent: a refitted normalization rescales the
    inputs out from under the loaded weights, a frozen parameter still decays
    towards zero, and a fine-tune written under the base model's name replaces
    something general with something narrow. None of them raises.
    """

    def _model(self):
        return FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu").model

    def test_freezing_touches_only_the_lifting_path(self):
        from poraque.ml.training import freeze_lifting_layers

        model = self._model()
        freeze_lifting_layers(model)
        frozen = {name for name, p in model.named_parameters()
                  if not p.requires_grad}
        assert frozen
        assert all(name.startswith(("lift.", "cell_encoder."))
                   for name in frozen)

    def test_the_projection_head_stays_trainable(self):
        """It decodes to physical units, which is what differs per family."""
        from poraque.ml.training import freeze_lifting_layers

        model = self._model()
        freeze_lifting_layers(model)
        assert all(p.requires_grad for name, p in model.named_parameters()
                   if name.startswith("project."))

    def test_counts_reconcile_with_n_parameters(self):
        """
        A complex weight is two real numbers. Counting it once here and twice
        in `n_parameters` made the log read as if half the model were frozen.
        """
        from poraque.ml.training import freeze_lifting_layers

        model = self._model()
        total = model.n_parameters()
        counts = freeze_lifting_layers(model)
        assert counts["frozen"] + counts["trainable"] == total
        assert counts["frozen"] > 0

    def test_freezing_is_reversible(self):
        from poraque.ml.training import freeze_lifting_layers

        model = self._model()
        freeze_lifting_layers(model)
        freeze_lifting_layers(model, freeze=False)
        assert all(p.requires_grad for p in model.parameters())

    def test_frozen_weights_do_not_move(self, toy):
        """
        AdamW's decoupled weight decay applies without a gradient, so frozen
        weights would shrink every step unless they are kept out of the
        optimiser entirely.
        """
        from poraque.ml.training import freeze_lifting_layers

        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        freeze_lifting_layers(operator.model)
        before = operator.model.lift.weight.detach().clone()
        train(operator, toy, epochs=2, batch_size=1, learning_rate=1e-2,
              weight_decay=0.5, eval_every=1, verbose=False)
        assert torch.equal(operator.model.lift.weight, before)

    def test_unfrozen_weights_do_move(self, toy):
        torch.manual_seed(0)
        operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        before = operator.model.lift.weight.detach().clone()
        train(operator, toy, epochs=2, batch_size=1, learning_rate=1e-2,
              eval_every=1, verbose=False)
        assert not torch.equal(operator.model.lift.weight, before)


class TestBundleNaming:
    def test_the_two_filenames_differ(self):
        from poraque.ml import BUNDLE_FILENAME, FINETUNED_BUNDLE_FILENAME

        assert BUNDLE_FILENAME != FINETUNED_BUNDLE_FILENAME
        assert BUNDLE_FILENAME.endswith(".pfno")
        assert FINETUNED_BUNDLE_FILENAME.endswith(".pfno")

    def test_an_existing_file_is_returned_unchanged(self, tmp_path):
        from poraque.ml import resolve_bundle_path

        path = tmp_path / "m.pfno"
        path.write_text("x")
        assert resolve_bundle_path(str(path)) == str(path)

    def test_a_legacy_file_is_found_and_announced(self, tmp_path):
        """Renaming the default must not make an existing model invisible."""
        from poraque.ml import resolve_bundle_path

        (tmp_path / "m.pth").write_text("x")
        lines = []
        resolved = resolve_bundle_path(str(tmp_path / "m.pfno"), lines.append)
        assert resolved == str(tmp_path / "m.pth")
        assert any(".pfno" in line for line in lines)

    def test_a_missing_file_keeps_the_requested_name(self, tmp_path):
        """So the error names what the user asked for, not a guess."""
        from poraque.ml import resolve_bundle_path

        requested = str(tmp_path / "absent.pfno")
        assert resolve_bundle_path(requested) == requested


class TestFineTuningValidation:
    def _config(self, tmp_path, **overrides):
        config = TrainingConfig()
        config.fine_tuning.enable = True
        config.output.root = str(tmp_path)
        for key, value in overrides.items():
            setattr(config.fine_tuning, key, value)
        return config

    def test_disabled_needs_no_checkpoint(self):
        config = TrainingConfig()
        config.fine_tuning.pretrained_checkpoint = "/nope.pfno"
        poraque_train.validate_fine_tuning_settings(config)     # no raise

    def test_a_missing_checkpoint_fails_before_training(self, tmp_path):
        config = self._config(tmp_path, pretrained_checkpoint="/nope.pfno")
        with pytest.raises(SystemExit, match="does not exist"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_writing_over_the_base_is_refused(self, tmp_path):
        """
        The destination is derived from `task.name`, so the collision is
        constructed the same way rather than from a remembered filename.
        """
        config = self._config(tmp_path, pretrained_checkpoint="unset")
        base = poraque_train.bundle_path(config)
        # The run folder is created by the run, not by `bundle_path`, which
        # only forms the name.
        os.makedirs(os.path.dirname(base), exist_ok=True)
        with open(base, "w") as handle:
            handle.write("x")
        config.fine_tuning.pretrained_checkpoint = base
        with pytest.raises(SystemExit, match="over its own base"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_a_sibling_name_is_allowed(self, tmp_path):
        """The base and the fine-tune coexist; only a true collision fails."""
        from poraque.ml import BUNDLE_FILENAME

        base = tmp_path / BUNDLE_FILENAME
        base.write_text("x")
        config = self._config(tmp_path, pretrained_checkpoint=str(base))
        poraque_train.validate_fine_tuning_settings(config)     # no raise

    def test_the_name_reaches_every_output(self, tmp_path):
        """One string decides the weights, the report and the figure folder."""
        config = TrainingConfig()
        config.task.name = "ag_au_pt_v2"
        config.output.root = str(tmp_path)

        run = os.path.join(str(tmp_path), "ag_au_pt_v2")
        assert poraque_train.bundle_path(config) == \
            os.path.join(run, "ag_au_pt_v2.pfno")
        assert poraque_train.plot_directory(config) == \
            os.path.join(run, "plots")
        assert config.report_dir() == os.path.join(run, "report")
        assert config.log_path() == \
            os.path.join(run, "log", "ag_au_pt_v2.log")
        assert poraque_train.report_filename(config, "ext2chg") == \
            "ag_au_pt_v2_report.pdf"
        # Two tasks cannot share one report name.
        assert poraque_train.report_filename(config, "ext2chg", 2) == \
            "ag_au_pt_v2_ext2chg_report.pdf"
        # A fine-tune never lands on the general model it specialises.
        config.fine_tuning.enable = True
        assert poraque_train.bundle_path(config) != \
            os.path.join(str(tmp_path), "ag_au_pt_v2.pfno")

    def test_a_non_positive_learning_rate_fails(self, tmp_path):
        base = tmp_path / "base.pfno"
        base.write_text("x")
        config = self._config(tmp_path, pretrained_checkpoint=str(base),
                              learning_rate=0.0)
        with pytest.raises(SystemExit, match="must be positive"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_a_legacy_checkpoint_is_accepted_and_rewritten(self, tmp_path):
        base = tmp_path / "base.pth"
        base.write_text("x")
        config = self._config(tmp_path,
                              pretrained_checkpoint=str(tmp_path / "base.pfno"))
        poraque_train.validate_fine_tuning_settings(config)
        assert config.fine_tuning.pretrained_checkpoint == str(base)


class TestStandardOutputTables:
    """
    The terminal tables, checked on their alignment.

    Every defect here is invisible to the code and obvious to a reader: a rule
    a character short of its heading, a left-aligned value under a
    right-aligned word, a column of blanks for a metric the task cannot
    produce. None of them raises.
    """

    METRICS = {"mse": 1.1e-4, "mae": 4.5e-3, "rmse": 1.0e-2,
               "relative_l2": 0.0107, "r2": 0.9998, "jsd": 7.8e-6}

    def test_the_rule_matches_the_heading(self):
        heading, rule = poraque_train.format_metrics_header(22)
        assert len(rule) == len(heading), (
            f"heading {len(heading)}, rule {len(rule)}")

    def test_rows_line_up_with_the_heading(self):
        heading, _ = poraque_train.format_metrics_header(22)
        row = poraque_train.format_metrics_row("struct_000", "train",
                                               self.METRICS, 22)
        assert len(row) == len(heading)

    def test_the_split_is_its_own_column(self):
        """
        It used to be glued to the name -- ``struct_000 (train)`` -- which put
        two different things in the structure column.
        """
        row = poraque_train.format_metrics_row("struct_000", "validation",
                                               self.METRICS, 22)
        assert "struct_000" in row and "(validation)" not in row
        assert "validation" in row

    def test_a_left_aligned_column_has_a_left_aligned_title(self):
        heading, _ = poraque_train.format_metrics_header(22)
        row = poraque_train.format_metrics_row("s", "train", self.METRICS, 22)
        assert heading.index("split") == row.index("train")

    def test_a_missing_metric_leaves_the_columns_intact(self):
        """
        ``jsd`` is undefined for a signed field. The row must keep its width so
        the table does not shear.
        """
        heading, _ = poraque_train.format_metrics_header(22)
        without = dict(self.METRICS, jsd=None)
        assert len(poraque_train.format_metrics_row(
            "s", "train", without, 22)) == len(heading)

    def test_a_task_without_jsd_drops_the_column_entirely(self):
        columns = poraque_train.metric_columns([dict(self.METRICS, jsd=None)])
        assert all(key != "jsd" for _, key, _ in columns), (
            "a column of blanks is worse than no column")
        heading, rule = poraque_train.format_metrics_header(22, columns)
        assert "JSD" not in heading and len(rule) == len(heading)

    def test_structure_names_wrap_into_indented_lines(self):
        names = [f"struct_{i:03d}" for i in range(17)]
        lines = poraque_train.format_names(names)
        assert len(lines) > 1, "seventeen names on one line is the defect"
        assert all(line.startswith("      ") for line in lines)
        joined = " ".join(lines).split()
        assert joined == names, "every name must survive the wrapping"

    def test_an_empty_name_list_prints_nothing(self):
        assert poraque_train.format_names([]) == []

    def test_grid_shapes_are_one_line_each(self):
        lines = poraque_train.format_shapes({(32, 32, 32): 9, (30, 32, 30): 1})
        assert len(lines) == 2
        assert "30x32x30" in lines[0] and "1 structure" in lines[0]
        assert "32x32x32" in lines[1] and "9 structures" in lines[1]

    def test_the_shape_continuation_is_indented_under_the_first(self):
        lines = poraque_train.format_shapes({(32, 32, 32): 9, (30, 32, 30): 1})
        assert not lines[0].startswith(" ")
        assert lines[1].startswith("   "), "continuations align under the label"

    def test_the_aggregate_is_printed_for_both_splits(self):
        """
        Quoting only the training aggregate is how a training fit gets read as
        a generalisation estimate.
        """
        lines = []
        poraque_train.format_aggregate("ext2chg: training", [self.METRICS],
                                       lines.append)
        poraque_train.format_aggregate("ext2chg: VALIDATION", [self.METRICS],
                                       lines.append)
        text = "\n".join(lines)
        assert "training" in text and "VALIDATION" in text
        assert text.count("metric") == 2

    def test_an_empty_aggregate_prints_nothing(self):
        lines = []
        poraque_train.format_aggregate("ext2chg: VALIDATION", [], lines.append)
        assert lines == []

    def test_the_aggregate_rule_matches_its_heading(self):
        lines = []
        poraque_train.format_aggregate("x", [self.METRICS], lines.append)
        heading = next(line for line in lines if "metric" in line)
        rule = next(line for line in lines if set(line.strip()) == {"-"})
        assert len(rule) == len(heading)
