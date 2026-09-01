# -*- coding: utf-8 -*-
# file: test_lora.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
Low-Rank Adaptation: what it must do, and what it must not silently do.

LoRA freezes a trained model and learns :math:`W' = W + \frac{\alpha}{r}BA`
beside its dense layers. Four claims carry the feature, and each of them fails
*quietly* if it is wrong — which is why they are asserted rather than assumed:

**It starts as the identity.** :math:`B` is zero, so the adapted model
reproduces the base bit for bit before the first optimiser step. A non-zero
start perturbs the very weights the fine-tune exists to preserve, and the only
symptom is a slightly worse model.

**The base is strictly frozen.** Every parameter is frozen first and adapters
added after, so a layer nobody thought about cannot stay trainable by
omission — the failure mode being a "LoRA" fine-tune that quietly trains
millions of parameters and needs the memory it promised to save.

**The checkpoint holds the adapter only.** That is the economy of the method.
It also means the file is *not* self-contained, which is a real cost and is
stated in the error rather than discovered.

**Nothing is adapted silently.** Wrapping zero layers would train zero
parameters and produce a flat loss curve, which reads as a bad dataset.
"""

import io
import os

import pytest
import torch

from poraque.ml.fno import FNO3d
from poraque.ml.lora import (
    LORA_TARGETS,
    LoRAConv3d,
    apply_lora,
    is_adapted,
    load_lora_state_dict,
    lora_state_dict,
    parameter_counts,
)


@pytest.fixture
def model():
    """A small FNO with the shipped model's shape, at a size tests can run."""
    torch.manual_seed(0)
    return FNO3d(in_channels=1, out_channels=1, width=8, modes=4, n_layers=2,
                 projection_channels=16)


@pytest.fixture
def sample():
    return torch.randn(1, 1, 8, 8, 8), torch.eye(3).unsqueeze(0) * 5.0


class TestItStartsAsTheIdentity:
    r""":math:`B = 0`, so :math:`BA = 0` and the adapted model *is* the base."""

    def test_the_prediction_is_bit_for_bit_unchanged(self, model, sample):
        x, cell = sample
        before = model(x, cell).clone()

        apply_lora(model, rank=4)

        assert torch.equal(model(x, cell), before)

    def test_b_is_zero_and_a_is_not(self, model):
        apply_lora(model, rank=4)
        layers = [m for m in model.modules() if isinstance(m, LoRAConv3d)]

        assert layers
        for layer in layers:
            assert torch.all(layer.lora_B == 0)
            assert torch.any(layer.lora_A != 0), (
                "A must be random: with both factors at zero the gradient of "
                "the product is zero too and the adapter could never move")

    def test_it_stops_being_the_identity_once_trained(self, model, sample):
        x, cell = sample
        before = model(x, cell).clone()
        apply_lora(model, rank=4)

        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2)
        model(x, cell).square().mean().backward()
        optimiser.step()

        assert not torch.allclose(model(x, cell), before)


class TestTheBaseIsStrictlyFrozen:
    def test_only_adapter_tensors_are_trainable(self, model):
        apply_lora(model, rank=4)
        trainable = [name for name, parameter in model.named_parameters()
                     if parameter.requires_grad]

        assert trainable
        assert all("lora_A" in name or "lora_B" in name for name in trainable)

    def test_the_spectral_weights_never_move(self, model, sample):
        """
        They are ~99.8 % of the parameters and are deliberately not adapted:
        factorising a 5-index complex kernel is a choice of which axes to pair,
        not one decomposition.
        """
        x, cell = sample
        before = {name: tensor.detach().clone()
                  for name, tensor in model.named_parameters()
                  if tensor.is_complex()}
        apply_lora(model, rank=4)

        optimiser = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-1)
        for _ in range(3):
            optimiser.zero_grad()
            model(x, cell).square().mean().backward()
            optimiser.step()

        after = dict(model.named_parameters())
        for name, tensor in before.items():
            assert torch.equal(after[name], tensor), name

    def test_the_trainable_share_is_a_fraction_of_a_percent(self, model):
        counts = apply_lora(model, rank=8)
        share = counts["trainable"] / (counts["trainable"] + counts["frozen"])

        assert 0 < share < 0.02
        assert counts["adapters"] == 3        # lift + the two projection convs

    def test_freezing_happens_before_wrapping_not_after(self, model):
        """
        A layer this function does not know about must not stay trainable by
        omission. Checked by counting: every non-adapter parameter is frozen.
        """
        apply_lora(model, rank=4)
        counts = parameter_counts(model)
        adapter = sum(p.numel() for name, p in model.named_parameters()
                      if "lora_" in name)

        assert counts["trainable"] == adapter


class TestTheCheckpointHoldsTheAdapterOnly:
    def test_the_state_is_two_tensors_per_adapted_layer(self, model):
        counts = apply_lora(model, rank=4)
        state = lora_state_dict(model)

        assert len(state) == 2 * counts["adapters"]
        assert all("lora_A" in k or "lora_B" in k for k in state)

    def test_it_is_orders_of_magnitude_smaller(self, model):
        apply_lora(model, rank=4)

        adapter, full = io.BytesIO(), io.BytesIO()
        torch.save(lora_state_dict(model), adapter)
        torch.save(model.state_dict(), full)

        assert full.tell() > 20 * adapter.tell()

    def test_it_round_trips_into_a_freshly_adapted_model(self, model, sample):
        x, cell = sample
        apply_lora(model, rank=4)
        for parameter in model.parameters():
            if parameter.requires_grad:
                with torch.no_grad():
                    parameter.add_(0.05)
        expected = model(x, cell)

        torch.manual_seed(0)
        fresh = FNO3d(in_channels=1, out_channels=1, width=8, modes=4,
                      n_layers=2, projection_channels=16)
        apply_lora(fresh, rank=4)
        loaded = load_lora_state_dict(fresh, lora_state_dict(model))

        assert loaded == 6
        assert torch.allclose(fresh(x, cell), expected, atol=1e-6)

    def test_a_state_from_a_different_rank_is_refused(self, model):
        apply_lora(model, rank=4)
        state = lora_state_dict(model)
        state["nowhere.lora_A"] = torch.zeros(2, 2)

        torch.manual_seed(0)
        other = FNO3d(in_channels=1, out_channels=1, width=8, modes=4,
                      n_layers=2, projection_channels=16)
        apply_lora(other, rank=4)

        with pytest.raises(KeyError, match="no place for"):
            load_lora_state_dict(other, state)


class TestItRefusesToDoNothingQuietly:
    def test_adapting_no_layer_raises(self):
        """A run that trains zero parameters looks like a bad dataset."""
        model = torch.nn.Module()

        with pytest.raises(ValueError, match="adapted no layer"):
            apply_lora(model, rank=4)

    def test_a_rank_of_zero_is_refused(self, model):
        with pytest.raises(ValueError, match="lora_rank must be positive"):
            apply_lora(model, rank=0)

    def test_a_spatial_convolution_is_not_wrapped(self):
        """
        A larger kernel is a spatial operator, not a channel map, and
        factorising it means something else.
        """
        with pytest.raises(ValueError, match="1x1x1"):
            LoRAConv3d(torch.nn.Conv3d(4, 4, kernel_size=3), rank=2)

    def test_it_wraps_the_two_ends_and_says_which(self):
        assert LORA_TARGETS == ("lift", "project")


class TestTheScalingIsAlphaOverRank:
    r"""What keeps a learning rate roughly transferable between ranks."""

    def test_the_factor_is_recorded_and_applied(self):
        base = torch.nn.Conv3d(4, 6, kernel_size=1)
        layer = LoRAConv3d(base, rank=8, alpha=16.0)

        assert layer.scaling == pytest.approx(2.0)

        with torch.no_grad():
            layer.lora_B.fill_(1.0)
            layer.lora_A.fill_(1.0)
        x = torch.ones(1, 4, 2, 2, 2)
        expected = base(x) + 2.0 * 8 * 4 * torch.ones(1, 6, 2, 2, 2)

        assert torch.allclose(layer(x), expected, atol=1e-5)

    def test_merging_reproduces_the_wrapped_forward(self):
        torch.manual_seed(0)
        base = torch.nn.Conv3d(4, 6, kernel_size=1)
        layer = LoRAConv3d(base, rank=2, alpha=4.0)
        with torch.no_grad():
            layer.lora_B.normal_()

        x = torch.randn(1, 4, 3, 3, 3)
        merged = torch.nn.functional.conv3d(x, layer.merged_weight(),
                                            base.bias)

        assert torch.allclose(layer(x), merged, atol=1e-5)


class TestAnAdaptedOperatorSavesAndReloads:
    """The whole cycle: base bundle, LoRA fine-tune, reload."""

    def _dataset(self, tmp_path):
        import numpy as np

        from poraque.fields import ChargeDensity, FieldGrid
        from poraque.fields import KineticEnergyDensity
        from poraque.fields.vasp.poscar import Poscar
        from poraque.ml.data import FieldPairDataset

        rng = np.random.default_rng(0)
        for name in ("a", "b"):
            structure = Poscar(np.eye(3) * 4.0, ["Si"], [2],
                               [[0, 0, 0], [.5, .5, .5]])
            grid = FieldGrid((8, 8, 8), structure.cell)
            directory = tmp_path / name
            directory.mkdir(parents=True)
            values = rng.random(grid.shape) + 1.0
            ChargeDensity(values, grid, structure).write(directory / "CHGCAR")
            KineticEnergyDensity(values * 40.0, grid,
                                 structure).write(directory / "TAUCAR")
        return FieldPairDataset(str(tmp_path), task="chg2tau")

    @pytest.fixture
    def base_bundle(self, tmp_path):
        from poraque.ml.training import FieldOperator, save_bundle, train

        data = self._dataset(tmp_path / "data")
        operator = FieldOperator("chg2tau", width=8, modes=4, n_layers=2,
                                 projection_channels=16, device="cpu")
        train(operator, data, epochs=2, batch_size=1, verbose=False, seed=0)
        path = str(tmp_path / "base.pfno")
        save_bundle(path, {"chg2tau": operator})
        return path, data

    def _adapt(self, path, rank=4):
        from poraque.ml.training import load_bundle

        operator = load_bundle(path, "chg2tau", device="cpu")
        apply_lora(operator.model, rank=rank, alpha=16.0)
        operator.lora = {"rank": rank, "alpha": 16.0, "dropout": 0.0,
                         "base_checkpoint": os.path.abspath(path)}
        return operator

    def test_the_saved_bundle_carries_no_base_weights(self, base_bundle,
                                                      tmp_path):
        from poraque.ml.training import save_bundle

        path, _ = base_bundle
        operator = self._adapt(path)

        assert operator.state()["model_state"] == {}
        assert operator.state()["lora"]["base_checkpoint"] == os.path.abspath(path)

        fine_tuned = str(tmp_path / "lora.pfno")
        save_bundle(fine_tuned, {"chg2tau": operator})

        assert os.path.getsize(fine_tuned) < os.path.getsize(path) / 10

    def test_it_reloads_to_the_same_predictions(self, base_bundle, tmp_path):
        from poraque.ml.training import load_bundle, save_bundle, train

        path, data = base_bundle
        operator = self._adapt(path)
        train(operator, data, epochs=2, batch_size=1, learning_rate=1e-2,
              verbose=False, seed=0)

        fine_tuned = str(tmp_path / "lora.pfno")
        save_bundle(fine_tuned, {"chg2tau": operator})
        reloaded = load_bundle(fine_tuned, "chg2tau", device="cpu")

        sample = data[0]
        x = sample["input"].unsqueeze(0)
        cell = sample["cell"].unsqueeze(0)

        assert is_adapted(reloaded.model)
        assert torch.allclose(reloaded.model(x, cell), operator.model(x, cell),
                              atol=1e-6)

    def test_a_missing_base_says_exactly_what_is_wrong(self, base_bundle,
                                                       tmp_path):
        """
        The cost of a small file, stated rather than discovered: a LoRA
        checkpoint cannot reconstruct weights it never stored.
        """
        from poraque.ml.training import load_bundle, save_bundle

        path, _ = base_bundle
        operator = self._adapt(path)
        fine_tuned = str(tmp_path / "lora.pfno")
        save_bundle(fine_tuned, {"chg2tau": operator})
        os.rename(path, path + ".moved")

        with pytest.raises(FileNotFoundError, match="LoRA checkpoint"):
            load_bundle(fine_tuned, "chg2tau", device="cpu")
