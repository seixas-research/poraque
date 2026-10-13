# -*- coding: utf-8 -*-
# file: test_config_schema.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The shape of the configuration schema, and every committed file's fidelity to it.

The schema was rewritten on 2026-10-13. The previous one had a single
``model`` block shared by every task, so the larger ``ext2chg`` network lived
in a corner of it (``model.paw.width`` beside ``model.width``), a ``task.type``
that had to be cross-checked against ``model.paw.enable``, and a dozen keys
that were not choices. Now:

* an operator is trained when its section (``ext2chg``, ``chg2tau``) is
  present, and each carries its own ``model``;
* a feature that can be switched off is a block whose presence is the switch
  (``false`` | ``true`` | a mapping), with no ``enable`` key;
* the previous schema's sections and keys raise, naming where they went.

Every YAML under ``configs/`` is parsed here, because a config nobody loads is
a config that rots: a rename once broke eight committed files and none of the
four the change was made against would have noticed.
"""

import glob
import os
import re
from dataclasses import fields

import pytest
import yaml

from poraque.ml.config import (RETIRED_KEYS, RETIRED_SECTIONS, ModelConfig,
                               TrainingConfig)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "**", "*.yaml"),
                           recursive=True))
REFERENCE = os.path.join(ROOT, "configs", "train_complete_and_commented.yaml")


def label(path):
    """``configs/train_simple.yaml``, for a readable test id."""
    return os.path.relpath(path, ROOT)


def both_operators(**sections):
    """A mapping that trains both operators, plus ``sections``."""
    return {"ext2chg": None, "chg2tau": None, **sections}


class TestEveryCommittedConfigLoads:
    """The two committed configs parse against the current schema."""

    def test_there_are_exactly_two(self):
        assert sorted(os.path.basename(path) for path in CONFIGS) == [
            "train_complete_and_commented.yaml", "train_simple.yaml"]

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_it_parses_and_trains_something(self, path):
        assert TrainingConfig.from_yaml(path).task_names()

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_it_is_written_in_block_style(self, path):
        """No flow-style mappings or lists: ``{a: 1}`` reads as one setting."""
        for number, line in enumerate(open(path), start=1):
            code = line.split("#", 1)[0]
            assert not re.search(r"[{}\[\]]", code), (
                f"{label(path)}:{number} uses flow style: {line.strip()}")


class TestTheSectionsPresentAreTheTasksTrained:
    """``task.type`` and ``model.paw.enable`` were two switches for one choice."""

    def test_nothing_present_is_nothing_to_train(self):
        with pytest.raises(ValueError, match="Nothing to train"):
            TrainingConfig.from_dict({}).task_names()

    def test_both_sections_train_both_operators(self):
        config = TrainingConfig.from_dict(both_operators())
        assert config.task_names() == ["ext2chg", "chg2tau"]

    def test_a_bare_section_takes_its_defaults(self):
        """``chg2tau:`` with nothing under it is YAML ``null``: present."""
        config = TrainingConfig.from_dict(yaml.safe_load("chg2tau:\n"))
        assert config.task_names() == ["chg2tau"]

    def test_false_is_absent(self):
        config = TrainingConfig.from_dict({"ext2chg": None, "chg2tau": False})
        assert config.task_names() == ["ext2chg"]

    def test_occupancies_off_is_the_grid_task(self):
        config = TrainingConfig.from_dict({
            "ext2chg": {"occupancies": False},
            "chg2tau": {"occupancies": False}})
        assert config.task_names() == ["ext2pseudo", "pseudo2tau"]

    def test_the_operator_section_is_found_by_internal_name(self):
        config = TrainingConfig.from_dict({"ext2chg": {"occupancies": False}})
        assert config.operator("ext2pseudo") is config.ext2chg
        with pytest.raises(ValueError, match="does not train"):
            config.operator("ext2chg")


class TestEachOperatorHasItsOwnModel:
    """The reason the schema was rewritten: two networks, two blocks."""

    def test_the_defaults_differ_per_operator(self):
        config = TrainingConfig.from_dict(both_operators())
        assert (config.ext2chg.model.width, config.ext2chg.model.layers,
                config.ext2chg.model.modes) == (64, 4, 16)
        assert (config.chg2tau.model.width, config.chg2tau.model.layers,
                config.chg2tau.model.modes) == (16, 3, 8)

    def test_a_partial_model_keeps_its_operator_defaults(self):
        config = TrainingConfig.from_dict({"ext2chg": {"model": {"width": 8}}})
        assert config.ext2chg.model.modes == 16

    def test_the_backbone_keywords(self):
        model = ModelConfig(band=16.0, equivariant=None)
        kwargs = model.fno_kwargs()
        assert kwargs["mode_selection"] == "physical"
        assert kwargs["g_max"] == 16.0
        assert kwargs["use_coordinates"] is True

    def test_equivariance_drops_the_coordinates(self):
        config = TrainingConfig.from_dict(
            {"ext2chg": {"model": {"equivariant": True}}})
        kwargs = config.model_kwargs("ext2chg")
        assert kwargs["equivariant"] is True
        assert kwargs["use_coordinates"] is False
        assert kwargs["spherical_cutoff"] is True

    def test_cell_conditioning_false_drops_film(self):
        config = TrainingConfig.from_dict(
            {"chg2tau": {"model": {"cell_conditioning": False}}})
        assert config.model_kwargs("chg2tau")["cell_conditioning"] is False


class TestABlockIsSwitchedByItsPresence:
    """There is no ``enable`` key, so no block can be off and look on."""

    def test_true_is_on_with_defaults(self):
        config = TrainingConfig.from_dict(
            {"ext2chg": {"structure_input": True}})
        assert config.ext2chg.structure_input.elements == "auto"

    def test_a_mapping_is_on_with_settings(self):
        config = TrainingConfig.from_dict(
            {"ext2chg": {"model": {"equivariant": {"n_radial": 4}}}})
        assert config.ext2chg.model.equivariant.n_radial == 4

    def test_a_scalar_is_refused(self):
        with pytest.raises(ValueError, match="false, true or a block"):
            TrainingConfig.from_dict({"fine_tuning": "yes"})

    def test_enable_is_not_a_key(self):
        with pytest.raises(ValueError, match="Unknown key"):
            TrainingConfig.from_dict(
                {"ext2chg": {"model": {"equivariant": {"enable": True}}}})


class TestTheRoundTrip:
    """``to_dict`` is what a checkpoint stores, and must read back."""

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_to_dict_reads_back_equal(self, path):
        config = TrainingConfig.from_yaml(path)
        assert TrainingConfig.from_dict(config.to_dict()).to_dict() \
            == config.to_dict()

    def test_an_absent_operator_is_written_as_false(self):
        config = TrainingConfig.from_dict({"ext2chg": None})
        assert config.to_dict()["chg2tau"] is False

    def test_the_minimal_form_keeps_the_operators(self):
        trimmed = TrainingConfig.from_dict(both_operators()).non_default_dict()
        assert "ext2chg" in trimmed and "chg2tau" in trimmed


class TestValuesAreCheckedAtLoad:
    """Before the cache is built, not an hour into it."""

    @pytest.mark.parametrize("mapping,fragment", [
        ({"ext2chg": {"model": {"activation": "kan"}}}, "activation"),
        ({"ext2chg": {"model": {"precision": "float16"}}}, "precision"),
        ({"ext2chg": {"model": {"band": -1}}}, "band"),
        ({"ext2chg": {"target": "half"}}, "target"),
        ({"chg2tau": {"head": "pauli"}}, "pauli"),
        ({"data": {"cache": {"compression": "gzip"}}}, "hdf5"),
        ({"training": {"validation": {"kfold": 1}}}, "kfold"),
        ({"training": {"loss": {"norm": "sobolev"}}}, "norm"),
        ({"ext2chg": {"occupancies": {"neighbor_min": 6.0}}}, "neighbor_min"),
        ({"ext2chg": {"target": "full", "structure_input": True}},
         "structure_input"),
    ])
    def test_a_bad_value_raises_naming_the_key(self, mapping, fragment):
        with pytest.raises(ValueError, match=fragment):
            TrainingConfig.from_dict(mapping)

    def test_kfold_and_fine_tuning_are_refused_together(self):
        with pytest.raises(ValueError, match="kfold"):
            TrainingConfig.from_dict({"training": {"validation": {"kfold": 3}},
                                      "fine_tuning": True})


class TestThePreviousSchemaIsRefusedByName:
    """A config written before 2026-10-13 is told where each setting went."""

    @pytest.mark.parametrize("name", sorted(RETIRED_SECTIONS))
    def test_a_retired_section_names_its_replacement(self, name):
        with pytest.raises(ValueError, match="Removed configuration section"):
            TrainingConfig.from_dict({name: {}})

    @pytest.mark.parametrize("section,key", [
        ("data", "data_paths"), ("data", "resolution"),
        ("training", "learning_rate"), ("training", "early_stopping"),
        ("training", "seed"), ("training", "device"),
    ])
    def test_a_retired_key_names_its_replacement(self, section, key):
        with pytest.raises(ValueError, match="Replaced:") as raised:
            TrainingConfig.from_dict({section: {key: 1}})
        assert RETIRED_KEYS[(section, key)].split()[0] in str(raised.value)

    def test_a_retired_model_key_names_its_replacement(self):
        with pytest.raises(ValueError, match="ext2chg.model.layers"):
            TrainingConfig.from_dict({"ext2chg": {"model": {"n_layers": 2}}})

    def test_the_archived_features_name_the_archive(self):
        with pytest.raises(ValueError, match="experiments/kan_symbolic"):
            TrainingConfig.from_dict({"symbolic": {}})
        with pytest.raises(ValueError, match="experiments/kan_symbolic"):
            TrainingConfig.from_dict(
                {"chg2tau": {"model": {"kan_setup": None}}})

    def test_an_ordinary_typo_gets_no_misleading_hint(self):
        with pytest.raises(ValueError) as raised:
            TrainingConfig.from_dict({"run": {"nmae": "x"}})
        assert "Replaced:" not in str(raised.value)


class TestOverridesWalkThePath:
    """The command line names a setting by its dotted path."""

    def test_a_nested_override(self):
        config = TrainingConfig.from_dict(both_operators()).apply_overrides({
            "training.optimizer.learning_rate": 1e-3,
            "ext2chg.model.width": 8})
        assert config.training.optimizer.learning_rate == 1e-3
        assert config.ext2chg.model.width == 8

    def test_none_is_skipped(self):
        config = TrainingConfig().apply_overrides({"run.name": None})
        assert config.run.name == "poraque_models"

    def test_a_path_into_an_absent_block_is_refused(self):
        with pytest.raises(ValueError, match="which is off"):
            TrainingConfig().apply_overrides({"fine_tuning.learning_rate": 1})

    def test_an_unknown_path_is_refused(self):
        with pytest.raises(ValueError, match="Unknown override"):
            TrainingConfig().apply_overrides({"training.nonesuch": 1})

    def test_a_bare_key_is_refused(self):
        with pytest.raises(ValueError, match="Unknown override"):
            TrainingConfig().apply_overrides({"epochs": 1})

    def test_an_override_is_validated(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainingConfig().apply_overrides({"training.epochs": 0})


class TestTheAnnotatedReferenceCoversTheSchema:
    """
    ``train_complete_and_commented.yaml`` is the documentation of last resort.

    A key that exists and is not in it is a key whose only description is a
    docstring nobody reading a config file will open.
    """

    def test_it_states_the_defaults(self):
        stated = TrainingConfig.from_yaml(REFERENCE)
        default = TrainingConfig.from_dict(both_operators())
        assert stated.to_dict() == default.to_dict()

    def test_every_key_appears_somewhere_in_the_file(self):
        """Stated or commented: a block that is off lists its settings."""
        text = open(REFERENCE).read()

        def names(cls):
            for entry in fields(cls):
                yield entry.name
                nested = entry.metadata.get("section") \
                    or entry.metadata.get("block")
                if nested is not None:
                    yield from names(nested)

        missing = sorted({name for name in names(TrainingConfig)
                          if not re.search(rf"^\s*#?\s*{re.escape(name)}\s*:",
                                           text, re.MULTILINE)})
        assert missing == []
