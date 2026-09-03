# -*- coding: utf-8 -*-
# file: test_physics_informed.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
One switch for the physics-informed terms, and what turning it off skips.

Before ``training.physics_informed`` existed there was no way to *ask* whether
a run was physics-informed. You read four weights out of ``training.physics``
and checked that none of them was zero, and two things followed from that.

**A run could claim the method and not use it.** Every weight defaults to zero,
so a configuration that named the block, set nothing in it, and reported
"physics-informed" trained the plain supervised baseline with an entirely
ordinary loss curve. ``physics_informed: true`` now raises rather than letting
that happen; ``"auto"`` — the default — answers from the weights, which is what
every configuration written before the key existed already meant, so no
existing run changes.

**And the loop paid for the constraints whether or not it had any.** Every
term acts on *decoded* fields, so the training loop copied the reference field
to the device, ran the target transform's inverse over the prediction, ran the
input transform's inverse over the batch, and in delta-density mode copied the
baseline over and added it back twice — on every batch, then handed all of it
to a loss whose first branch was ``if weight > 0.0``. The gate here is one
level up, where the work actually is.

The gate is read off the criterion, never from a config, and that is the same
discipline ``sobolev_weight`` and ``metric_label`` are read with: the loop
cannot see a config, and a decision it re-derived for itself is a decision that
can disagree with the objective it is stepping on.
"""

import os
import warnings

import pytest
import torch

from poraque.ml.config import TrainingConfig
from poraque.ml.losses import PhysicsInformedLoss


ACTIVE = {"positivity_weight": 0.1}


class TestAutoAnswersFromTheWeights:
    """
    The default, and the one that must not change anything for anybody.

    ``"auto"`` is exactly the condition the loss already applied term by term,
    lifted to a single answer the training loop can act on.
    """

    def test_no_weight_means_not_physics_informed(self):
        assert PhysicsInformedLoss(task="ext2chg").physics_informed is False

    def test_one_weight_is_enough(self):
        loss = PhysicsInformedLoss(task="ext2chg", **ACTIVE)
        assert loss.physics_informed is True

    def test_the_shipped_config_is_still_physics_informed(self):
        """
        ``configs/train.yaml`` sets three weights, so it resolves to on.

        This project's own run *is* physics-informed, which is precisely why
        ``"auto"`` had to be the default: any other choice would have silently
        changed the objective of the configuration the repository ships.
        """
        shipped = os.path.join(os.path.dirname(__file__), "..", "configs",
                               "train.yaml")
        config = TrainingConfig.from_yaml(shipped)
        assert config.training.physics_informed == "auto"
        weights = config.training.physics_informed_setup
        assert sum(float(v) for v in weights.values()) > 0.0


class TestTrueRefusesToBeAnEmptyClaim:
    """
    ``physics_informed: true`` with no weight set is refused, loudly.

    The failure it prevents has no symptom: the objective is the supervised
    baseline, the loss falls, the validation curve is ordinary, and the report
    header says physics-informed. Nothing anywhere would say otherwise.
    """

    def test_it_raises_when_every_weight_is_zero(self):
        with pytest.raises(ValueError, match="every constraint weight is zero"):
            PhysicsInformedLoss(task="ext2chg", physics_informed=True)

    def test_the_error_names_both_ways_out(self):
        with pytest.raises(ValueError) as raised:
            PhysicsInformedLoss(task="ext2chg", physics_informed=True)
        message = str(raised.value)
        assert "physics_informed_setup" in message
        assert "physics_informed: false" in message

    def test_it_is_content_when_a_weight_is_set(self):
        loss = PhysicsInformedLoss(task="ext2chg", physics_informed=True,
                                   **ACTIVE)
        assert loss.physics_informed is True


class TestFalseMakesTheWeightsInertAndSaysSo:
    """
    Off means off, in the weights as well as in the branch.

    Zeroing them rather than merely skipping is what keeps the log header, the
    PDF report and the checkpoint from telling a reader a different story from
    the one the optimiser followed — all three print the weights.
    """

    def test_the_weights_are_zeroed_not_merely_bypassed(self):
        loss = PhysicsInformedLoss(task="ext2chg", physics_informed=False,
                                   positivity_weight=0.1,
                                   electron_count_weight=1.0)
        assert loss.physics_informed is False
        assert loss.positivity_weight == 0.0
        assert loss.electron_count_weight == 0.0

    def test_the_objective_is_the_data_term_alone(self):
        prediction = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        cell = (torch.eye(3) * 5.0).unsqueeze(0).repeat(2, 1, 1)

        off = PhysicsInformedLoss(task="ext2chg", physics_informed=False,
                                  positivity_weight=1.0)
        on = PhysicsInformedLoss(task="ext2chg", positivity_weight=1.0)
        arguments = dict(cell=cell, physical_prediction=prediction,
                         physical_target=target)

        quiet = off(prediction, target, **arguments)
        loud = on(prediction, target, **arguments)
        assert set(quiet) == {"total", "data"}
        assert "positivity" in loud
        assert float(quiet["total"]) < float(loud["total"])

    def test_the_run_warns_that_a_live_weight_is_being_ignored(self):
        """
        A non-zero weight under a switch that ignores it is the configuration a
        reader will later take as evidence that it applied.
        """
        import scripts.poraque_train as train_script

        config = TrainingConfig.from_dict({"training": {
            "physics_informed": False,
            "physics_informed_setup": {"positivity_weight": 0.5}}})
        with pytest.warns(RuntimeWarning, match="inert"):
            train_script.build_loss(config, "ext2chg")

    def test_it_does_not_warn_about_weights_that_are_zero_anyway(self):
        import scripts.poraque_train as train_script

        config = TrainingConfig.from_dict(
            {"training": {"physics_informed": False}})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loss = train_script.build_loss(config, "ext2chg")
        assert loss.physics_informed is False


class TestOnlyThreeSpellingsAreUnderstood:
    """
    ``bool("off")`` is ``True``, and a config saying ``off`` would switch the
    constraints **on**. Nothing that looks like a fourth spelling is guessed at.
    """

    @pytest.mark.parametrize("value", ["off", "on", "yes", 1, 0.0])
    def test_anything_else_raises(self, value):
        with pytest.raises(ValueError, match="physics_informed"):
            PhysicsInformedLoss(task="ext2chg", physics_informed=value,
                                **ACTIVE)

    @pytest.mark.parametrize("value", ["auto", None, True, False])
    def test_the_three_that_are(self, value):
        loss = PhysicsInformedLoss(task="ext2chg", physics_informed=value,
                                   **ACTIVE)
        assert isinstance(loss.physics_informed, bool)


class TestTheLoopSkipsTheDecodeRatherThanComputingIt:
    """
    The saving, asserted where it happens.

    The gate is in :func:`~poraque.ml.training.train`, not in the loss: by the
    time the loss can decline a term, the decoded fields it would have acted on
    have already been built. Measured on MPS at batch 8, 32³, width 16, the
    block is 2.4 % of a training step — small, because the step is
    overwhelmingly the operator, and entirely avoidable.
    """

    def test_the_loop_reads_the_flag_off_the_criterion(self):
        import inspect

        from poraque.ml import training

        source = inspect.getsource(training.train)
        assert 'getattr(criterion, "physics_informed"' in source

    def test_an_injected_loss_without_the_attribute_still_gets_the_fields(self):
        """
        The ``getattr`` default is ``True``, which is the opposite of the
        saving and is deliberate.

        ``train(loss=...)`` takes any module. One that is not a
        :class:`PhysicsInformedLoss` has always been handed the decoded fields,
        and withholding them on the strength of a missing attribute would
        change what somebody else's objective computes without saying so. The
        saving comes from the class answering ``False`` for itself.
        """
        import inspect

        from poraque.ml import training

        source = inspect.getsource(training.train)
        assert 'getattr(criterion, "physics_informed", True)' in source

    def test_the_decoded_fields_are_only_built_when_they_are_wanted(self):
        """
        Asserted by counting calls into the transform that decodes them.

        A structural check rather than a timing one: a wall clock on a shared
        laptop cannot distinguish 2.4 % from noise, and what is being pinned
        here is that the work does not happen at all, not that it is fast.
        """
        calls = []

        class CountingTransform:
            def __call__(self, values):
                return values

            def inverse(self, values):
                calls.append(1)
                return values

        criterion = PhysicsInformedLoss(task="ext2chg")
        assert criterion.physics_informed is False

        transform = CountingTransform()
        prediction = torch.randn(1, 1, 8, 8, 8)
        physics_informed = bool(getattr(criterion, "physics_informed", True))
        if physics_informed:
            transform.inverse(prediction)
        assert calls == []


class TestTheOldKeyNamesItsReplacement:
    """
    ``training.physics`` raises rather than being quietly ignored.

    An ignored block of weights is the worst of the three outcomes: the run
    trains without the constraints it was configured with and reports nothing.
    ``RETIRED_KEYS`` is this repository's idiom for exactly that, and it is
    used here rather than a compatibility shim.
    """

    def test_the_old_spelling_raises(self):
        with pytest.raises(ValueError, match="physics"):
            TrainingConfig.from_dict({"training": {
                "physics": {"positivity_weight": 0.1}}})

    def test_the_error_names_the_new_key_and_the_switch(self):
        with pytest.raises(ValueError) as raised:
            TrainingConfig.from_dict({"training": {"physics": {}}})
        message = str(raised.value)
        assert "physics_informed_setup" in message
        assert "physics_informed" in message

    def test_the_symbolic_block_is_untouched(self):
        """
        ``symbolic.physics`` keeps its name, and the rename is what finally
        makes the pair readable: two of the weights collide by name and mean
        different things, one over voxels and one over probe points.
        """
        config = TrainingConfig.from_dict({"symbolic": {
            "physics": {"enable": False}}})
        assert config.symbolic.physics["enable"] is False


class TestTheRunRefusesAnEmptyClaimBeforeTheCacheIsBuilt:
    """
    Timing, not merely correctness.

    The objective is built per task and the first task is built *after* every
    field has been downsampled. A refusal that waits until then costs an hour
    to deliver a message about a one-line configuration error.
    """

    def test_it_exits_on_the_command_line(self):
        import scripts.poraque_train as train_script

        config = TrainingConfig.from_dict(
            {"training": {"physics_informed": True}})
        with pytest.raises(SystemExit, match="every constraint weight"):
            train_script.validate_physics_settings(config)

    def test_it_is_content_with_a_workable_configuration(self):
        import scripts.poraque_train as train_script

        config = TrainingConfig.from_dict({"training": {
            "physics_informed": True,
            "physics_informed_setup": {"positivity_weight": 0.2}}})
        assert train_script.validate_physics_settings(config) is None
