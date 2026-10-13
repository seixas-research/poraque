# -*- coding: utf-8 -*-
# file: test_split.py

"""
Tests for the train/validation split, the evaluation interval and early
stopping.

The split is defined by one number, ``validation.fraction``; K-fold cross-validation
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
# training.validation.fraction
# ===================================================================== #
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
    return FieldPairDataset(str(tmp_path), task="pseudo2tau")


def _rows(lines):
    """The per-epoch data rows, excluding the legend, header and rule."""
    return [line for line in lines
            if line.strip() and line.strip()[0].isdigit()]


def _train_toy(dataset, epochs, eval_every, log=None, validate=False,
               verbose=True, early_stopping=0, loss=None):
    """Run a tiny training loop and return its history."""
    torch.manual_seed(0)
    operator = FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
                             projection_channels=8, device="cpu")
    return train(operator, dataset, epochs=epochs, batch_size=1,
                 learning_rate=1e-3, eval_every=eval_every,
                 validation=dataset if validate else None,
                 early_stopping=early_stopping, log=log, verbose=verbose,
                 loss=loss)


# ===================================================================== #
# Fine-tuning
# ===================================================================== #


class TestValidFraction:
    def test_default_is_a_fifth(self):
        assert TrainingConfig().training.validation.fraction == pytest.approx(0.2)

    def test_zero_falls_back_to_the_universal_fit(self):
        config = TrainingConfig()
        config.training.validation.fraction = 0.0
        validation, origin = poraque_train.resolve_validation_split(_Dataset(5), config)
        assert validation == set()
        assert "TRAINING FIT" in origin

    @pytest.mark.parametrize("fraction, expected", [
        (0.2, 2), (0.3, 3), (0.5, 5), (0.1, 1),
    ])
    def test_reserves_the_requested_share(self, fraction, expected):
        config = TrainingConfig()
        config.training.validation.fraction = fraction
        validation, _ = poraque_train.resolve_validation_split(_Dataset(10), config)
        assert len(validation) == expected

    def test_split_is_at_the_structure_level(self):
        """Whole materials move together; identifiers come out intact."""
        config = TrainingConfig()
        config.training.validation.fraction = 0.4
        validation, _ = poraque_train.resolve_validation_split(_Dataset(5), config)
        assert validation <= {f"struct_{i:03d}" for i in range(5)}

    def test_is_reproducible_for_a_given_seed(self):
        def draw(seed):
            config = TrainingConfig()
            config.training.validation.fraction = 0.3
            config.run.seed = seed
            return poraque_train.resolve_validation_split(_Dataset(10), config)[0]

        assert draw(0) == draw(0)

    def test_seed_changes_the_draw(self):
        """Otherwise 'randomized' would be a claim the code does not make."""
        draws = set()
        for seed in range(8):
            config = TrainingConfig()
            config.training.validation.fraction = 0.3
            config.run.seed = seed
            draws.add(frozenset(
                poraque_train.resolve_validation_split(_Dataset(10), config)[0]))
        assert len(draws) > 1

    def test_never_empties_either_side(self):
        """
        A fraction that rounds to zero would silently degrade into a universal
        fit while the config still claims a validation split.
        """
        config = TrainingConfig()
        config.training.validation.fraction = 0.01
        validation, _ = poraque_train.resolve_validation_split(_Dataset(3), config)
        assert len(validation) == 1

        config.training.validation.fraction = 0.99
        validation, _ = poraque_train.resolve_validation_split(_Dataset(3), config)
        assert len(validation) == 2      # one structure is always kept to train on

    def test_rejects_a_fraction_outside_the_range(self):
        config = TrainingConfig()
        for bad in (1.0, 1.5, -0.2):
            config.training.validation.fraction = bad
            with pytest.raises(SystemExit, match="fraction"):
                poraque_train.resolve_validation_split(_Dataset(5), config)

    def test_rejects_a_single_structure_dataset(self):
        config = TrainingConfig()
        config.training.validation.fraction = 0.5
        with pytest.raises(SystemExit, match="two structures"):
            poraque_train.resolve_validation_split(_Dataset(1), config)

    def test_origin_records_the_protocol(self):
        config = TrainingConfig()
        config.training.validation.fraction = 0.25
        config.run.seed = 7
        _, origin = poraque_train.resolve_validation_split(_Dataset(8), config)
        assert "fraction=0.25" in origin and "seed=7" in origin


# ===================================================================== #
# training.validation.every
# ===================================================================== #
class TestEvalInterval:
    def test_config_default_is_ten(self):
        assert TrainingConfig().training.validation.every == 10

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
# training.validation.patience
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
        assert training.validation.patience == 300
        assert training.epochs == 500
        assert training.validation.patience < training.epochs

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

        operator = FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
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


class TestFineTuning:
    """
    Adapting a trained operator, rather than starting from noise.

    The failure modes here are silent: a refitted normalization rescales the
    inputs out from under the loaded weights, a frozen parameter still decays
    towards zero, and a fine-tune written under the base model's name replaces
    something general with something narrow. None of them raises.
    """

    def _model(self):
        return FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
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
        optimizer entirely.
        """
        from poraque.ml.training import freeze_lifting_layers

        torch.manual_seed(0)
        operator = FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        freeze_lifting_layers(operator.model)
        before = operator.model.lift.weight.detach().clone()
        train(operator, toy, epochs=2, batch_size=1, learning_rate=1e-2,
              weight_decay=0.5, eval_every=1, verbose=False)
        assert torch.equal(operator.model.lift.weight, before)

    def test_unfrozen_weights_do_move(self, toy):
        torch.manual_seed(0)
        operator = FieldOperator("pseudo2tau", width=4, modes=2, n_layers=1,
                                 projection_channels=8, device="cpu")
        before = operator.model.lift.weight.detach().clone()
        train(operator, toy, epochs=2, batch_size=1, learning_rate=1e-2,
              eval_every=1, verbose=False)
        assert not torch.equal(operator.model.lift.weight, before)


class TestBundleNaming:
    def test_the_two_filenames_differ(self):
        from poraque.ml import BUNDLE_FILENAME, FINETUNED_BUNDLE_FILENAME

        assert BUNDLE_FILENAME != FINETUNED_BUNDLE_FILENAME
        assert BUNDLE_FILENAME.endswith(".poraque")
        assert FINETUNED_BUNDLE_FILENAME.endswith(".poraque")

    def test_an_existing_file_is_returned_unchanged(self, tmp_path):
        from poraque.ml import resolve_bundle_path

        path = tmp_path / "m.poraque"
        path.write_text("x")
        assert resolve_bundle_path(str(path)) == str(path)

    def test_a_legacy_file_is_found_and_announced(self, tmp_path):
        """Renaming the default must not make an existing model invisible."""
        from poraque.ml import resolve_bundle_path

        (tmp_path / "m.pth").write_text("x")
        lines = []
        resolved = resolve_bundle_path(str(tmp_path / "m.poraque"), lines.append)
        assert resolved == str(tmp_path / "m.pth")
        assert any(".poraque" in line for line in lines)

    def test_a_missing_file_keeps_the_requested_name(self, tmp_path):
        """So the error names what the user asked for, not a guess."""
        from poraque.ml import resolve_bundle_path

        requested = str(tmp_path / "absent.poraque")
        assert resolve_bundle_path(requested) == requested


class TestTheExtensionIsPoraqueAndPfnoStillLoads:
    """
    The bundle extension became ``.poraque`` on 2026-09-02, replacing
    ``.pfno``.

    Every model trained before that date is called ``.pfno`` — including the
    one this repository ships — and the *format* did not change, only the
    name. A rename that could not find those files would report a trained
    model as "no model", which is the failure the ``.pth`` rename already
    caused once and which :func:`resolve_bundle_path` exists to prevent.
    """

    def test_the_suffix_has_one_definition(self):
        """
        Two copies of a filename extension is how the last rename left a
        default path behind, so the string is defined once and imported.
        """
        from poraque.ml import BUNDLE_FILENAME, FINETUNED_BUNDLE_FILENAME
        from poraque.ml.config import BUNDLE_SUFFIX

        assert BUNDLE_SUFFIX == ".poraque"
        assert BUNDLE_FILENAME.endswith(BUNDLE_SUFFIX)
        assert FINETUNED_BUNDLE_FILENAME.endswith(BUNDLE_SUFFIX)

    def test_the_old_extension_is_a_legacy_name_not_the_current_one(self):
        from poraque.ml import LEGACY_BUNDLE_SUFFIXES
        from poraque.ml.config import BUNDLE_SUFFIX

        assert ".pfno" in LEGACY_BUNDLE_SUFFIXES
        assert BUNDLE_SUFFIX not in LEGACY_BUNDLE_SUFFIXES

    def test_a_pfno_beside_the_requested_name_is_found(self, tmp_path):
        from poraque.ml import resolve_bundle_path

        (tmp_path / "m.pfno").write_text("x")
        lines = []
        resolved = resolve_bundle_path(str(tmp_path / "m.poraque"),
                                       lines.append)
        assert resolved == str(tmp_path / "m.pfno")
        assert any(".poraque" in line for line in lines)

    def test_a_renamed_base_is_found_from_a_recorded_pfno_path(self, tmp_path):
        """
        The other direction, which the single-direction search missed: a LoRA
        checkpoint stores the path of the base it adapts. Written before the
        rename it names a ``.pfno``; read after the base was renamed, that
        path no longer exists and the adapter holds no weights of its own.
        """
        from poraque.ml import resolve_bundle_path

        (tmp_path / "base.poraque").write_text("x")
        resolved = resolve_bundle_path(str(tmp_path / "base.pfno"))
        assert resolved == str(tmp_path / "base.poraque")

    def test_a_different_stem_is_never_substituted(self, tmp_path):
        """Only the extension is guessed at; another stem is another model."""
        from poraque.ml import resolve_bundle_path

        (tmp_path / "other.pfno").write_text("x")
        requested = str(tmp_path / "wanted.poraque")
        assert resolve_bundle_path(requested) == requested

    def test_the_checkpoint_path_carries_the_new_extension(self, tmp_path):
        config = TrainingConfig()
        config.run.name = "pt_res32"
        config.run.output = str(tmp_path)
        assert config.checkpoint_path().endswith("pt_res32.poraque")


class TestFineTuningValidation:
    def _config(self, tmp_path, **overrides):
        config = TrainingConfig.from_dict({"fine_tuning": True})
        config.run.output = str(tmp_path)
        for key, value in overrides.items():
            setattr(config.fine_tuning, key, value)
        return config

    def test_disabled_needs_no_checkpoint(self):
        config = TrainingConfig()
        assert config.fine_tuning is None
        poraque_train.validate_fine_tuning_settings(config)     # no raise

    def test_a_missing_checkpoint_fails_before_training(self, tmp_path):
        config = self._config(tmp_path, checkpoint="/nope.poraque")
        with pytest.raises(SystemExit, match="does not exist"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_writing_over_the_base_is_refused(self, tmp_path):
        """
        The destination is derived from `run.name`, so the collision is
        constructed the same way rather than from a remembered filename.
        """
        config = self._config(tmp_path, checkpoint="unset")
        base = poraque_train.bundle_path(config)
        # The run folder is created by the run, not by `bundle_path`, which
        # only forms the name.
        os.makedirs(os.path.dirname(base), exist_ok=True)
        with open(base, "w") as handle:
            handle.write("x")
        config.fine_tuning.checkpoint = base
        with pytest.raises(SystemExit, match="over its own base"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_a_sibling_name_is_allowed(self, tmp_path):
        """The base and the fine-tune coexist; only a true collision fails."""
        from poraque.ml import BUNDLE_FILENAME

        base = tmp_path / BUNDLE_FILENAME
        base.write_text("x")
        config = self._config(tmp_path, checkpoint=str(base))
        poraque_train.validate_fine_tuning_settings(config)     # no raise

    def test_the_name_reaches_every_output(self, tmp_path):
        """One string decides the weights, the report and the figure folder."""
        config = TrainingConfig()
        config.run.name = "ag_au_pt_v2"
        config.run.output = str(tmp_path)

        run = os.path.join(str(tmp_path), "ag_au_pt_v2")
        assert poraque_train.bundle_path(config) == \
            os.path.join(run, "ag_au_pt_v2.poraque")
        assert poraque_train.plot_directory(config) == \
            os.path.join(run, "plots")
        assert config.report_dir() == os.path.join(run, "report")
        assert config.log_path() == \
            os.path.join(run, "log", "ag_au_pt_v2.log")
        assert poraque_train.report_filename(config, "ext2pseudo") == \
            "ag_au_pt_v2_report.pdf"
        # Two tasks cannot share one report name.
        assert poraque_train.report_filename(config, "ext2pseudo", 2) == \
            "ag_au_pt_v2_ext2pseudo_report.pdf"
        # A fine-tune never lands on the general model it specializes.
        config.apply_overrides({"fine_tuning": True})
        assert poraque_train.bundle_path(config) != \
            os.path.join(str(tmp_path), "ag_au_pt_v2.poraque")

    def test_a_non_positive_learning_rate_fails(self, tmp_path):
        base = tmp_path / "base.poraque"
        base.write_text("x")
        config = self._config(tmp_path, checkpoint=str(base),
                              learning_rate=0.0)
        with pytest.raises(SystemExit, match="must be positive"):
            poraque_train.validate_fine_tuning_settings(config)

    def test_a_legacy_checkpoint_is_accepted_and_rewritten(self, tmp_path):
        base = tmp_path / "base.pth"
        base.write_text("x")
        config = self._config(tmp_path,
                              checkpoint=str(tmp_path / "base.poraque"))
        poraque_train.validate_fine_tuning_settings(config)
        assert config.fine_tuning.checkpoint == str(base)


class TestTheReportedMetricsAreTheDensityChannel:
    """
    A spin-polarized prediction is a ``(2, ...)`` stack, and the table is not.

    ``operator.predict`` returns a :class:`~poraque.fields.SpinDensity`
    whenever ``data.spin`` resolved on -- every model trained on the platinum
    data -- and the per-structure metrics were taken over that stack, in a
    table whose unit column says ``e/Ang^3``. A prediction *exact* in rho and
    wrong only in the magnetization was measured reporting ``relative_l2``
    1.3e-3, ``mae`` 7.9e-4 and ``max_abs`` 6.1e-3: a density error that does
    not exist. The two new integral metrics fared differently and no better --
    a ``(2, ...)`` shape is not the grid's, so both came back ``None`` and
    their columns would have been silently absent from every spin run.

    This is the defect the six inference sites carried until 2026-09-02, found
    in a seventh place.
    """

    @staticmethod
    def _pair(magnetization_factor=3.0):
        import numpy as np

        from poraque.fields import SpinDensity
        from poraque.fields.grid import FieldGrid
        from poraque.fields.structure import Structure

        cell = np.eye(3) * 5.0
        grid = FieldGrid((8, 8, 8), cell)
        structure = Structure(cell, ["Pt"], [1], np.zeros((1, 3)))
        rng = np.random.RandomState(0)
        density = rng.rand(8, 8, 8) + 1.0
        moment = 1e-3 * rng.randn(8, 8, 8)
        return (SpinDensity(density, moment * magnetization_factor, grid,
                            structure),
                SpinDensity(density, moment, grid, structure))

    def test_a_density_predicted_exactly_scores_zero(self):
        prediction, target = self._pair()
        values = poraque_train.compare_fields(prediction, target)
        for key in ("relative_l2", "mae", "max_abs", "relative_h1",
                    "integral_error"):
            assert values[key] == pytest.approx(0.0, abs=1e-12), key

    def test_the_stack_would_have_reported_an_error_that_is_not_there(self):
        """The counterfactual, so the fix is not mistaken for a tolerance."""
        prediction, target = self._pair()
        stacked = poraque_train.metrics(prediction.data, target.data,
                                        grid=target.grid)
        assert stacked["relative_l2"] > 1e-4
        assert stacked["relative_h1"] is None
        assert stacked["integral_error"] is None

    def test_the_magnetization_is_measured_rather_than_dropped(self):
        """
        Reducing to rho must not leave half a two-channel model unmeasured ---
        that is the mirror of the mistake being fixed.
        """
        prediction, target = self._pair(magnetization_factor=3.0)
        values = poraque_train.compare_fields(prediction, target)
        # predicted 3m against m: ||2m|| / ||m|| = 2.
        assert values["magnetization_relative_l2"] == pytest.approx(2.0)

    def test_a_one_channel_field_reports_no_magnetization_at_all(self):
        import numpy as np

        from poraque.fields import ChargeDensity
        from poraque.fields.grid import FieldGrid
        from poraque.fields.structure import Structure

        cell = np.eye(3) * 5.0
        grid = FieldGrid((8, 8, 8), cell)
        structure = Structure(cell, ["Pt"], [1], np.zeros((1, 3)))
        field = ChargeDensity(np.ones((8, 8, 8)), grid, structure)
        values = poraque_train.compare_fields(field, field)
        assert "magnetization_relative_l2" not in values
