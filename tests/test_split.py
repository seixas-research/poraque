# -*- coding: utf-8 -*-
# file: test_split.py

"""
Tests for the train/validation split and the evaluation interval.

The split resolver lives in ``scripts/run_train.py`` rather than the package,
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


def _load_run_train():
    """Import ``scripts/run_train.py`` as a module."""
    path = os.path.join(_ROOT, "scripts", "run_train.py")
    spec = importlib.util.spec_from_file_location("_run_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_run_train"] = module
    spec.loader.exec_module(module)
    return module


run_train = _load_run_train()


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
    def test_zero_falls_back_to_the_universal_fit(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.0
        holdout, origin = run_train.resolve_validation_split(_Dataset(5), config)
        assert holdout == set()
        assert "TRAINING FIT" in origin

    @pytest.mark.parametrize("fraction, expected", [
        (0.2, 2), (0.3, 3), (0.5, 5), (0.1, 1),
    ])
    def test_reserves_the_requested_share(self, fraction, expected):
        config = TrainingConfig()
        config.training.valid_fraction = fraction
        holdout, _ = run_train.resolve_validation_split(_Dataset(10), config)
        assert len(holdout) == expected

    def test_split_is_at_the_structure_level(self):
        """Whole materials move together; identifiers come out intact."""
        config = TrainingConfig()
        config.training.valid_fraction = 0.4
        holdout, _ = run_train.resolve_validation_split(_Dataset(5), config)
        assert holdout <= {f"struct_{i:03d}" for i in range(5)}

    def test_is_reproducible_for_a_given_seed(self):
        def draw(seed):
            config = TrainingConfig()
            config.training.valid_fraction = 0.3
            config.training.seed = seed
            return run_train.resolve_validation_split(_Dataset(10), config)[0]

        assert draw(0) == draw(0)

    def test_seed_changes_the_draw(self):
        """Otherwise 'randomized' would be a claim the code does not make."""
        draws = set()
        for seed in range(8):
            config = TrainingConfig()
            config.training.valid_fraction = 0.3
            config.training.seed = seed
            draws.add(frozenset(
                run_train.resolve_validation_split(_Dataset(10), config)[0]))
        assert len(draws) > 1

    def test_never_empties_either_side(self):
        """
        A fraction that rounds to zero would silently degrade into a universal
        fit while the config still claims a validation split.
        """
        config = TrainingConfig()
        config.training.valid_fraction = 0.01
        holdout, _ = run_train.resolve_validation_split(_Dataset(3), config)
        assert len(holdout) == 1

        config.training.valid_fraction = 0.99
        holdout, _ = run_train.resolve_validation_split(_Dataset(3), config)
        assert len(holdout) == 2      # one structure is always kept to train on

    def test_rejects_a_fraction_outside_the_range(self):
        config = TrainingConfig()
        for bad in (1.0, 1.5, -0.2):
            config.training.valid_fraction = bad
            with pytest.raises(SystemExit, match="valid_fraction"):
                run_train.resolve_validation_split(_Dataset(5), config)

    def test_rejects_a_single_structure_dataset(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.5
        with pytest.raises(SystemExit, match="two structures"):
            run_train.resolve_validation_split(_Dataset(1), config)

    def test_origin_records_the_protocol(self):
        config = TrainingConfig()
        config.training.valid_fraction = 0.25
        config.training.seed = 7
        _, origin = run_train.resolve_validation_split(_Dataset(8), config)
        assert "valid_fraction=0.25" in origin and "seed=7" in origin


class TestHoldout:
    def test_named_structures_are_held_out(self):
        config = TrainingConfig()
        config.training.holdout = ["struct_001", "struct_003"]
        holdout, origin = run_train.resolve_validation_split(_Dataset(5), config)
        assert holdout == {"struct_001", "struct_003"}
        assert "explicit" in origin

    def test_rejects_an_unknown_name(self):
        config = TrainingConfig()
        config.training.holdout = ["struct_099"]
        with pytest.raises(SystemExit, match="not present"):
            run_train.resolve_validation_split(_Dataset(5), config)

    def test_conflicts_with_valid_fraction(self):
        """
        Both define a validation split. Letting one win silently would make
        the protocol depend on which key the reader looked at.
        """
        config = TrainingConfig()
        config.training.holdout = ["struct_001"]
        config.training.valid_fraction = 0.2
        with pytest.raises(SystemExit, match="both define"):
            run_train.resolve_validation_split(_Dataset(5), config)


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
        assert len(lines) == 4
        assert "epoch    10/10" in lines[-1]

    def test_final_epoch_always_reports(self, toy):
        """A run must not end without a current number."""
        lines = []
        _train_toy(toy, epochs=7, eval_every=5, log=lines.append)
        assert "epoch     7/7" in lines[-1]

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


def _train_toy(dataset, epochs, eval_every, log=None, validate=False,
               verbose=True):
    """Run a tiny training loop and return its history."""
    torch.manual_seed(0)
    operator = FieldOperator("chg2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu")
    return train(operator, dataset, epochs=epochs, batch_size=1,
                 learning_rate=1e-3, eval_every=eval_every,
                 validation=dataset if validate else None,
                 log=log, verbose=verbose)
