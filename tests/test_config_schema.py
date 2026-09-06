# -*- coding: utf-8 -*-
# file: test_config_schema.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The shape of the configuration schema, and every committed file's fidelity to it.

**One convention, stated once.** Every optional feature is a block::

    <group>:
      <feature>:
        enable: <switch>
        <setting>: <value>

with the switch *inside* the settings it governs. Before 26.9.8 two of them
were not: ``model.equivariant`` was a bare boolean beside an
``equivariant_setup`` group, and ``training.physics_informed`` a bare
``auto``/``true``/``false`` beside a ``physics_informed_setup`` group. The
two-key spelling drifts in a way the one-key spelling cannot --
``equivariant: false`` above a populated ``equivariant_setup`` reads at a
glance as an equivariant run and is not one -- and nothing brings the switch
and the settings into the same field of view. ``symbolic.physics`` and
``fine_tuning`` were already blocks; those two joined them, and
``symbolic.enable_symbolic_distillation``, which restated its own section in
its own name, became ``symbolic.enable``.

**The audit is here rather than in a script**, because it is the kind of check
that is run once by hand and then quietly stops being true. Eight configs under
``configs/kan/`` were broken by the 26.9.8 rename and none of the four the
change was made against would have noticed: nothing loaded them. The count
asserted below is a floor, not the current number: it exists so an empty glob
cannot pass as a clean audit.
"""

import glob
import os
import re

import pytest
import yaml

from poraque.ml.config import (RETIRED_KEYS, TrainingConfig,
                               retired_replacement, split_enable_block)

CONFIGS = sorted(glob.glob(
    os.path.join(os.path.dirname(__file__), "..", "configs", "**", "*.yaml"),
    recursive=True))


def label(path):
    """``configs/kan/compare_kan_rbf_ext2chg.yaml``, for a readable test id."""
    return os.path.relpath(path, os.path.join(os.path.dirname(__file__), ".."))


#: Every ``{enable: ...}`` block, as ``(section, key, settings, switch values)``.
BLOCKS = TrainingConfig.ENABLE_BLOCKS


class TestEveryCommittedConfigLoads:
    """
    A config nobody loads is a config that rots.

    The 26.9.8 rename broke the eight ``configs/kan/`` comparison files and left
    the four named in the change request working, which is exactly the split a
    hand audit produces. This parametrisation is what makes the number stop
    mattering.
    """

    def test_there_are_configs_to_check(self):
        assert len(CONFIGS) >= 10, (
            f"only {len(CONFIGS)} configs found; the glob is probably wrong")

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_it_parses_against_the_current_schema(self, path):
        TrainingConfig.from_yaml(path)

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_it_survives_a_round_trip(self, path, tmp_path):
        """
        Written back out and read again, it is the same configuration.

        This is what catches a field whose default is a mutable the dataclass
        shares, and a block that ``to_yaml`` flattens on the way out.
        """
        config = TrainingConfig.from_yaml(path)
        written = tmp_path / "round_trip.yaml"
        config.to_yaml(str(written))
        assert TrainingConfig.from_yaml(str(written)).to_dict() \
            == config.to_dict()


class TestTheKeysAreSpelledOneWay:
    """
    snake_case, no hyphens, no camelCase, and no key defined twice.

    None of these is a matter of taste in a schema that refuses unknown keys:
    ``batch-size`` and ``batchSize`` do not fail as *style*, they fail as
    ``Unknown key``, and the reader has to work out which of the two dozen
    valid spellings was meant. A duplicate is worse, because YAML does not
    fail at all -- ``safe_load`` keeps the last and discards the first
    silently.
    """

    #: A mapping key at the start of a line, with its indentation.
    KEY = re.compile(r"^(\s*)([A-Za-z_][\w\-]*)\s*:")

    def keys(self, path):
        for number, line in enumerate(open(path).read().splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            match = self.KEY.match(line)
            if match:
                yield number, len(match.group(1)), match.group(2)

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_no_hyphens_and_no_camel_case(self, path):
        offenders = [(number, key) for number, _, key in self.keys(path)
                     if "-" in key or re.search(r"[a-z][A-Z]", key)
                     or key != key.lower()]
        assert offenders == [], f"{label(path)}: {offenders}"

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_no_key_is_defined_twice_in_one_mapping(self, path):
        seen, duplicates = {}, []
        for number, indent, key in self.keys(path):
            for deeper in [level for level in seen if level > indent]:
                seen.pop(deeper)
            lines = seen.setdefault(indent, {}).setdefault(key, [])
            if lines:
                duplicates.append((key, lines + [number]))
            lines.append(number)
        assert duplicates == [], f"{label(path)}: {duplicates}"

    @pytest.mark.parametrize("path", CONFIGS, ids=label)
    def test_no_retired_key_survives(self, path):
        """
        Per section, because a word retired in one is live in another.

        Written as a bare-word scan this test failed on ``configs/train.yaml``
        for ``output.root``, which is not retired at all --- the same mistake
        the ``RETIRED_KEYS`` table itself made until it was keyed by
        ``(section, key)``. ``from_yaml`` above would catch these too; the
        value here is that the failure names the file and the key rather than
        arriving as one line of a stack trace.
        """
        stated = yaml.safe_load(open(path)) or {}
        stale = sorted(f"{section}.{key}"
                       for section, values in stated.items()
                       if isinstance(values, dict)
                       for key in values
                       if (section, key) in RETIRED_KEYS)
        assert stale == [], f"{label(path)}: {stale}"


class TestEveryOptionalFeatureIsAnEnableBlock:
    """
    The convention, asserted against the schema rather than against a list.

    ``ENABLE_BLOCKS`` is the registry :meth:`TrainingConfig.validate_blocks`
    walks, so a new block is registered rather than hand-validated --- and the
    tests below iterate the same table, so a block that is added and not
    registered is invisible to *both*, which is the one failure mode a registry
    has. `test_no_setup_group_is_left_anywhere` is the counterweight: it looks
    at the dataclasses, not at the registry.
    """

    @pytest.mark.parametrize("section,key,settings,allowed", BLOCKS,
                             ids=[f"{s}.{k}" for s, k, _, _ in BLOCKS])
    def test_the_default_is_a_mapping_carrying_enable(self, section, key,
                                                      settings, allowed):
        block = getattr(getattr(TrainingConfig(), section), key)
        assert isinstance(block, dict)
        assert block.get("enable", ...) in allowed, (
            f"{section}.{key} has no default switch")

    @pytest.mark.parametrize("section,key,settings,allowed", BLOCKS,
                             ids=[f"{s}.{k}" for s, k, _, _ in BLOCKS])
    def test_the_scalar_spelling_says_how_to_translate_itself(
            self, section, key, settings, allowed):
        """
        The old spelling is a *type* error, not an unknown key.

        ``RETIRED_KEYS`` cannot catch it --- the key is still there and still
        valid --- so the block parser owns the message, and the message has to
        show the replacement rather than only reject the value.
        """
        with pytest.raises(ValueError) as raised:
            TrainingConfig.from_dict({section: {key: allowed[0]}})
        message = str(raised.value)
        assert "must be a block" in message
        assert f"{key}:" in message and "enable:" in message

    @pytest.mark.parametrize("section,key,settings,allowed", BLOCKS,
                             ids=[f"{s}.{k}" for s, k, _, _ in BLOCKS])
    def test_an_unknown_setting_names_its_own_block(self, section, key,
                                                    settings, allowed):
        with pytest.raises(ValueError, match=rf"{section}\.{key}"):
            TrainingConfig.from_dict(
                {section: {key: {"enable": allowed[0], "nonesuch": 1}}})

    @pytest.mark.parametrize("section,key,settings,allowed", BLOCKS,
                             ids=[f"{s}.{k}" for s, k, _, _ in BLOCKS])
    def test_an_unknown_switch_value_is_refused(self, section, key,
                                                settings, allowed):
        with pytest.raises(ValueError, match="enable is"):
            TrainingConfig.from_dict(
                {section: {key: {"enable": "perhaps"}}})

    def test_no_setup_group_is_left_anywhere(self):
        """
        The `_setup` suffix is the pattern this convention replaced.

        ``kan_setup`` is the one that stays, and deliberately: it is selected
        by ``model.activation``, which is a *choice among five* rather than a
        switch, so there is no `enable` for it to hold and no boolean it could
        contradict.
        """
        from dataclasses import fields as dataclass_fields

        leftovers = [f"{section}.{f.name}"
                     for section, cls in TrainingConfig._SECTIONS.items()
                     for f in dataclass_fields(cls)
                     if f.name.endswith("_setup") and f.name != "kan_setup"]
        assert leftovers == []

    def test_every_switch_is_spelled_enable(self):
        """
        And not ``enable_<the thing it is in>``.

        ``symbolic.enable_symbolic_distillation`` was 37 characters, the
        longest key in the schema, and named its own section twice.
        """
        from dataclasses import fields as dataclass_fields

        wrong = [f"{section}.{f.name}"
                 for section, cls in TrainingConfig._SECTIONS.items()
                 for f in dataclass_fields(cls)
                 if f.name.startswith("enable") and f.name != "enable"
                 and f.name != "enable_kfold"]
        assert wrong == [], (
            "a feature's switch is `enable`, inside the feature's own block")


class TestTheBlockParser:
    """
    :func:`split_enable_block` is the one place the hierarchy is read.

    Four blocks parsed by four hand-written readers is four chances to accept
    a key one of them rejects, and to phrase the same error four ways.
    """

    def test_none_is_an_empty_block(self):
        assert split_enable_block(None, "x.y", ("a",)) == (None, {})

    def test_an_unstated_switch_comes_back_as_none(self):
        """
        Not as the default: ``to_yaml`` has to tell "unstated" from "stated as
        the default", and the caller is what knows which default applies.
        """
        enable, stated = split_enable_block({"a": 1}, "x.y", ("a",))
        assert enable is None and stated == {"a": 1}

    def test_the_switch_is_removed_from_the_settings(self):
        enable, stated = split_enable_block(
            {"enable": True, "a": 1}, "x.y", ("a",))
        assert enable is True and stated == {"a": 1}

    def test_it_does_not_mutate_what_it_was_given(self):
        block = {"enable": True, "a": 1}
        split_enable_block(block, "x.y", ("a",))
        assert block == {"enable": True, "a": 1}

    def test_the_error_names_the_block_and_the_alternatives(self):
        with pytest.raises(ValueError) as raised:
            split_enable_block({"b": 1}, "training.physics_informed", ("a",))
        message = str(raised.value)
        assert "training.physics_informed" in message
        assert "'b'" in message and "'a'" in message and "'enable'" in message

    def test_the_scalar_message_quotes_yaml_not_python(self):
        """``true``, not ``True``: the reader is looking at a YAML file."""
        with pytest.raises(ValueError, match="enable: true"):
            split_enable_block(True, "model.equivariant", ("n_radial",))


class TestRetiredKeysAreScopedToTheirSection:
    """
    The same word is retired in one section and live in another.

    ``root`` is gone from ``data`` and is still how ``output.root`` names the
    run folder; ``physics`` is gone from ``training`` and is still the symbolic
    search's own constraint block. Keyed by the bare word --- which is how the
    table was written until 26.9.8 --- a config saying ``model: {root: ...}``
    was answered with "write ``data_paths``", which is advice about a section
    the reader was not editing.
    """

    def test_a_retired_key_names_its_replacement(self):
        assert retired_replacement("data", "root").startswith("data_paths")

    def test_the_same_word_is_silent_in_another_section(self):
        assert retired_replacement("model", "root") is None

    def test_a_live_key_elsewhere_is_untouched(self):
        assert TrainingConfig.from_dict(
            {"output": {"root": "models"}}).output.root == "models"
        assert TrainingConfig.from_dict(
            {"symbolic": {"physics": {"enable": False}}}
        ).symbolic.physics["enable"] is False

    def test_the_hint_reaches_the_message(self):
        with pytest.raises(ValueError, match="Replaced:"):
            TrainingConfig.from_dict({"data": {"root": "x"}})

    def test_an_ordinary_typo_gets_no_misleading_hint(self):
        with pytest.raises(ValueError) as raised:
            TrainingConfig.from_dict({"model": {"root": "x"}})
        assert "Replaced:" not in str(raised.value)


class TestABareOverrideMustBeUnambiguous:
    """
    Four field names live in two sections each, and "first match wins" guessed.

    ``precision`` is how the *fields are stored* in ``data`` and what the
    *operator computes in* in ``model``; ``seed`` is the validation draw in
    ``training`` and the search seed in ``symbolic``. Resolving either to
    whichever section is listed first is a decision nobody made, taken silently
    -- and ``enable`` joined the list when ``symbolic`` gained it in 26.9.8.
    """

    def test_a_dotted_override_still_works(self):
        config = TrainingConfig().apply_overrides({"symbolic.enable": True})
        assert config.symbolic.enable is True
        assert config.fine_tuning.enable is False

    def test_an_unambiguous_bare_override_still_works(self):
        assert TrainingConfig().apply_overrides({"width": 64}).model.width == 64

    @pytest.mark.parametrize(
        "key", ["enable", "learning_rate", "precision", "seed"])
    def test_an_ambiguous_bare_override_is_refused(self, key):
        with pytest.raises(ValueError, match="Ambiguous"):
            TrainingConfig().apply_overrides({key: 1})

    def test_the_message_names_every_section_that_carries_it(self):
        with pytest.raises(ValueError) as raised:
            TrainingConfig().apply_overrides({"precision": "float64"})
        message = str(raised.value)
        assert "data.precision" in message and "model.precision" in message

    def test_an_unknown_override_is_still_unknown(self):
        with pytest.raises(ValueError, match="Unknown override"):
            TrainingConfig().apply_overrides({"nonesuch": 1})


class TestTheAnnotatedReferenceCoversTheSchema:
    """
    ``train_complete_and_commented.yaml`` is the documentation of last resort.

    A key that exists and is not in it is a key whose only description is a
    docstring nobody reading a config file will open.
    """

    REFERENCE = os.path.join(os.path.dirname(__file__), "..", "configs",
                             "train_complete_and_commented.yaml")

    def test_every_section_appears(self):
        stated = yaml.safe_load(open(self.REFERENCE)) or {}
        assert set(stated) == set(TrainingConfig._SECTIONS)

    def test_every_key_appears_somewhere_in_the_file(self):
        """
        Stated *or* commented: the file deliberately comments out the settings
        of a block that is off, so that turning it on is uncommenting rather
        than remembering the names.
        """
        from dataclasses import fields as dataclass_fields

        text = open(self.REFERENCE).read()
        missing = [f"{section}.{f.name}"
                   for section, cls in TrainingConfig._SECTIONS.items()
                   for f in dataclass_fields(cls)
                   if not re.search(rf"^\s*#?\s*{re.escape(f.name)}\s*:",
                                    text, re.MULTILINE)]
        assert missing == []

    @pytest.mark.parametrize("section,key,settings,allowed", BLOCKS,
                             ids=[f"{s}.{k}" for s, k, _, _ in BLOCKS])
    def test_every_block_setting_appears_beneath_its_block(
            self, section, key, settings, allowed):
        text = open(self.REFERENCE).read()
        for setting in settings:
            assert re.search(rf"^\s*#?\s*{re.escape(setting)}\s*:", text,
                             re.MULTILINE), (
                f"{section}.{key}.{setting} is undocumented")
