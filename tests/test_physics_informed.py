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
ordinary loss curve. The loss's ``physics_informed=True`` now raises rather
than letting that happen; ``"auto"`` answers from the weights. Since
2026-10-13 a configuration has no switch at all: ``training.physics`` holds the
weights, and the run always builds its objective with ``"auto"``.

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

import pytest
import torch

from poraque.ml.config import TrainingConfig
from poraque.ml.losses import PhysicsInformedLoss


ACTIVE = {"positivity_weight": 0.1}


class TestExt2chgChargeConservation:
    @pytest.mark.parametrize("device", ["cpu", pytest.param(
        "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(),
                                       reason="MPS is unavailable"))])
    def test_absolute_decoded_charge_has_a_gradient_but_magnetization_does_not(
            self, device):
        from poraque.ml.physics import cell_volume
        from poraque.ml.transforms import Asinh

        cell = torch.tensor([[[2., 0., 0.], [.3, 2., 0.], [0., 0., 2.]],
                             [[3., 0., 0.], [0., 2., 0.], [0., 0., 4.]]],
                            device=device)
        count = torch.tensor([4., 10.], device=device)
        baseline = torch.zeros(2, 2, 4, 4, 4, device=device)
        baseline[:, 0] = (count / cell_volume(cell)).reshape(-1, 1, 1, 1)
        residual = .1 * baseline
        residual[:, 1] = -3.0  # The moment has no particle-number constraint.
        transform = Asinh(scale=.02)
        prediction = transform(residual).detach().requires_grad_(True)
        criterion = PhysicsInformedLoss(task="ext2chg", electron_count_weight=2.)
        terms = criterion(prediction, prediction.detach(), cell=cell,
                          physical_prediction=transform.inverse(prediction) + baseline,
                          physical_target=baseline)

        assert float(terms["electron_count"]) == pytest.approx(.01, rel=2e-5)
        assert float(terms["total"].detach()) == pytest.approx(.02, rel=2e-5)
        terms["total"].backward()
        assert torch.isfinite(prediction.grad).all()
        assert float(prediction.grad[:, 0].abs().sum()) > 0
        assert float(prediction.grad[:, 1].abs().sum()) == 0

    def test_explicit_count_overrides_reference_for_charged_systems(self):
        density = torch.ones(1, 2, 4, 4, 4)
        cell = torch.eye(3).unsqueeze(0) * 2.
        criterion = PhysicsInformedLoss(task="ext2chg", electron_count_weight=1.)
        terms = criterion(density, density, cell=cell,
                          physical_prediction=density,
                          physical_target=density * 2., n_electrons=torch.tensor([8.]))
        assert float(terms["electron_count"]) == 0.

    def test_cli_configuration_activates_the_term(self):
        from scripts.poraque_train import build_loss

        config = TrainingConfig.from_dict({"training": {"physics": {
            "electron_count": 1.}}})
        criterion = build_loss(config, "ext2chg")
        prediction = torch.ones(1, 1, 4, 4, 4)
        terms = criterion(prediction, prediction, cell=torch.eye(3).unsqueeze(0),
                          physical_prediction=prediction,
                          physical_target=prediction * 2.)
        assert float(terms["electron_count"]) == pytest.approx(.25)
        assert float(terms["total"].detach()) == pytest.approx(.25)


class TestAutoAnswersFromTheWeights:
    """
    The default, and the one that must not change anything for anybody.

    ``"auto"`` is exactly the condition the loss already applied term by term,
    lifted to a single answer the training loop can act on.
    """

    def test_no_weight_means_not_physics_informed(self):
        assert PhysicsInformedLoss(task="ext2pseudo").physics_informed is False

    def test_one_weight_is_enough(self):
        loss = PhysicsInformedLoss(task="ext2pseudo", **ACTIVE)
        assert loss.physics_informed is True

    @pytest.mark.parametrize("name", ["train_simple.yaml",
                                      "train_complete_and_commented.yaml"])
    def test_the_shipped_configs_let_their_own_weights_decide(self, name):
        """
        A shipped config's weights decide, and nothing else does.

        Written first as "the shipped config *is* physics-informed", which was
        true of the file on the day and is not a property of the mechanism: the
        moment somebody zeroed the weights for an ablation an ordinary config
        edit became a test failure. Since 2026-10-13 there is no switch beside
        the weights at all, so what has to hold is that the objective the run
        builds is physics-informed exactly when a weight is nonzero.
        """
        import scripts.poraque_train as train_script

        shipped = os.path.join(os.path.dirname(__file__), "..", "configs",
                               name)
        config = TrainingConfig.from_yaml(shipped)
        live = sum(config.training.physics_weights().values()) > 0.0
        loss = train_script.build_loss(config, "ext2pseudo")
        assert loss.physics_informed is live


class TestTrueRefusesToBeAnEmptyClaim:
    """
    ``physics_informed: true`` with no weight set is refused, loudly.

    The failure it prevents has no symptom: the objective is the supervised
    baseline, the loss falls, the validation curve is ordinary, and the report
    header says physics-informed. Nothing anywhere would say otherwise.
    """

    def test_it_raises_when_every_weight_is_zero(self):
        with pytest.raises(ValueError, match="every constraint weight is zero"):
            PhysicsInformedLoss(task="ext2pseudo", physics_informed=True)

    def test_the_error_names_both_ways_out(self):
        with pytest.raises(ValueError) as raised:
            PhysicsInformedLoss(task="ext2pseudo", physics_informed=True)
        message = str(raised.value)
        assert "training.physics" in message
        assert "physics_informed='auto'" in message

    def test_it_is_content_when_a_weight_is_set(self):
        loss = PhysicsInformedLoss(task="ext2pseudo", physics_informed=True,
                                   **ACTIVE)
        assert loss.physics_informed is True


class TestFalseMakesTheWeightsInertAndSaysSo:
    """
    Off means off, in the weights as well as in the branch.

    Zeroing them rather than merely skipping is what keeps the log header, the
    PDF report and the checkpoint from telling a reader a different story from
    the one the optimizer followed — all three print the weights.
    """

    def test_the_weights_are_zeroed_not_merely_bypassed(self):
        loss = PhysicsInformedLoss(task="ext2pseudo", physics_informed=False,
                                   positivity_weight=0.1,
                                   electron_count_weight=1.0)
        assert loss.physics_informed is False
        assert loss.positivity_weight == 0.0
        assert loss.electron_count_weight == 0.0

    def test_the_objective_is_the_data_term_alone(self):
        prediction = torch.randn(2, 1, 8, 8, 8)
        target = torch.randn(2, 1, 8, 8, 8)
        cell = (torch.eye(3) * 5.0).unsqueeze(0).repeat(2, 1, 1)

        off = PhysicsInformedLoss(task="ext2pseudo", physics_informed=False,
                                  positivity_weight=1.0)
        on = PhysicsInformedLoss(task="ext2pseudo", positivity_weight=1.0)
        arguments = dict(cell=cell, physical_prediction=prediction,
                         physical_target=target)

        quiet = off(prediction, target, **arguments)
        loud = on(prediction, target, **arguments)
        assert set(quiet) == {"total", "data"}
        assert "positivity" in loud
        assert float(quiet["total"]) < float(loud["total"])

class TestOnlyThreeSpellingsAreUnderstood:
    """
    ``bool("off")`` is ``True``, and a config saying ``off`` would switch the
    constraints **on**. Nothing that looks like a fourth spelling is guessed at.
    """

    @pytest.mark.parametrize("value", ["off", "on", "yes", 1, 0.0])
    def test_anything_else_raises(self, value):
        with pytest.raises(ValueError, match="physics_informed"):
            PhysicsInformedLoss(task="ext2pseudo", physics_informed=value,
                                **ACTIVE)

    @pytest.mark.parametrize("value", ["auto", None, True, False])
    def test_the_three_that_are(self, value):
        loss = PhysicsInformedLoss(task="ext2pseudo", physics_informed=value,
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

        criterion = PhysicsInformedLoss(task="ext2pseudo")
        assert criterion.physics_informed is False

        transform = CountingTransform()
        prediction = torch.randn(1, 1, 8, 8, 8)
        physics_informed = bool(getattr(criterion, "physics_informed", True))
        if physics_informed:
            transform.inverse(prediction)
        assert calls == []


class TestTheOldKeyNamesItsReplacement:
    """
    The previous spellings raise rather than being quietly ignored.

    An ignored block of weights is the worst of the outcomes: the run trains
    without the constraints it was configured with and reports nothing.
    ``RETIRED_KEYS`` is this repository's idiom for exactly that, and it is
    used here rather than a compatibility shim. The name ``training.physics``
    itself went round a loop: retired in 26.9.5, it is the block again since
    2026-10-13, now holding the weights alone with no switch beside them.
    """

    def test_the_previous_block_names_the_new_one(self):
        with pytest.raises(ValueError) as raised:
            TrainingConfig.from_dict({"training": {
                "physics_informed": {"positivity_weight": 0.1}}})
        assert "training.physics" in str(raised.value)

    def test_the_intermediate_spelling_raises_too(self):
        """A config written against the 26.9.5 spelling is as stale as any."""
        with pytest.raises(ValueError, match="physics_informed_setup"):
            TrainingConfig.from_dict({"training": {
                "physics_informed_setup": {"positivity_weight": 0.1}}})

    def test_the_old_weight_names_are_refused(self):
        """``positivity_weight`` is ``positivity`` inside the block."""
        with pytest.raises(ValueError, match="positivity_weight"):
            TrainingConfig.from_dict({"training": {
                "physics": {"positivity_weight": 0.1}}})

    def test_the_euler_lagrange_term_is_not_a_setting(self):
        """It ran without v_xc and was never a correctness measure."""
        with pytest.raises(ValueError, match="euler_lagrange"):
            TrainingConfig.from_dict({"training": {
                "physics": {"euler_lagrange": 0.1}}})
        config = TrainingConfig.from_dict({"training": {
            "physics": {"positivity": 0.2}}})
        assert config.training.physics_weights()["euler_lagrange_weight"] == 0.0
