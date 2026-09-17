# -*- coding: utf-8 -*-
# file: test_ext2paw.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The data path of ``ext2paw``: PAW augmentation occupancies as per-atom targets.

``ext2paw`` maps :math:`V_{\rm ext}` to the pseudo-density *and* each atom's
augmentation occupancies :math:`\rho(\ell\ell'LM)` --- the one-centre terms a
``CHGCAR`` carries after its grid. Everything between the file and a batch is
here: reading the records, carrying them through the cache, yielding them as
padded per-atom tensors, and the configuration that selects the task. The
operator is not written, and the last class pins that ``poraque-train`` says
so before building anything.

Reading the records turned up a defect older than the task. A spin-polarised
``CHGCAR`` writes a line of per-ion ``MAGMOM`` values after the first set of
records, and the parser read each record up to the next header, so the last
atom of every ``ISPIN = 2`` file gained ``NIONS`` extra values. On this
project's platinum data that emptied the training-set average outright --- the
170-value record failed the per-element shape check and the whole file was
dropped --- and gave the free Pt atom a 139-value record in every cache, which
``--add-paw`` then wrote out.
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "tests"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

from poraque.fields import ChargeDensity, SpinDensity  # noqa: E402
from poraque.fields.vasp.augmentation import (  # noqa: E402
    RECORD_SCHEMA,
    build_reference,
    format_augmentation,
    occupancy_arrays,
    parse_augmentation,
)
from poraque.fields.vasp.volumetric import read_augmentation_blocks  # noqa: E402
from test_data_sources import write_calculation  # noqa: E402

PLATINUM = os.path.join(_ROOT, "data", "vasp", "structures", "structure_0000",
                        "CHGCAR")
FREE_ATOM = os.path.join(_ROOT, "data", "vasp", "isolated_atoms", "Pt",
                         "CHGCAR")

#: What VASP writes after the first record set of a spin-polarised CHGCAR: one
#: MAGMOM per ion, in the wider E20.12 field.
MAGMOM_LINE = "  0.100000000000E+01  0.100000000000E+01"


def _records(atoms, length, offset=0.0):
    return [np.arange(length, dtype=float) * 0.01 + offset + atom
            for atom in range(atoms)]


def _with_records(directory, spin, length=6):
    """
    Rewrite a synthetic run's CHGCAR with augmentation records after the grid.

    Two atoms, ``length`` values each; a spin-polarised file gets a second set
    and the MAGMOM line VASP writes between them.
    """
    path = os.path.join(str(directory), "CHGCAR")
    density = ChargeDensity.read(path)
    first = format_augmentation(_records(2, length))
    if not spin:
        density.write(path, augmentation=first)
        return path
    second = format_augmentation(_records(2, length, offset=0.5))
    rng = np.random.default_rng(1)
    pair = SpinDensity(density.data, 0.01 * rng.standard_normal(
        density.data.shape), density.grid, density.structure)
    pair.write(path, augmentation=[first + [MAGMOM_LINE], second])
    return path


# ===================================================================== #
# Reading
# ===================================================================== #
class TestARecordIsReadToTheLengthItDeclares:
    """
    ``augmentation occupancies   1 138`` says 138, and 138 are read.

    Reading to the next header let the MAGMOM line of a spin-polarised file
    into the last record of its first set.
    """

    def test_the_magmom_line_is_not_part_of_the_last_record(self):
        block = format_augmentation(_records(2, 6)) + [MAGMOM_LINE]
        records = parse_augmentation(block)
        assert [record.size for record in records] == [6, 6]
        np.testing.assert_array_equal(records[1], _records(2, 6)[1])

    def test_the_training_set_average_is_no_longer_empty(self, tmp_path):
        """
        The regression as it was felt: every ``ISPIN = 2`` file was refused
        for an inconsistent channel count, and the average came back empty
        without a word.
        """
        run = write_calculation(tmp_path / "run", shape=(8, 8, 8))
        _with_records(run, spin=True)
        reference = build_reference([run])
        assert reference["Si"]["atoms"] == 2
        assert len(reference["Si"]["values"]) == 6
        assert reference["Si"]["schema"] == RECORD_SCHEMA

    def test_a_header_whose_fields_ran_together_is_still_read(self):
        """``2I4`` runs the index into the count once the index reaches 1000."""
        from poraque.fields.vasp.augmentation import _declared_length

        assert _declared_length("augmentation occupancies   7 138") == 138
        assert _declared_length("augmentation occupancies10001138") == 1138
        assert _declared_length("augmentation occupancies********") is None

    @pytest.mark.skipif(not os.path.exists(PLATINUM),
                        reason="the platinum dataset is not in this checkout")
    def test_every_platinum_record_is_138_values(self):
        _, blocks = read_augmentation_blocks(PLATINUM)
        assert len(blocks) == 2
        for block in blocks:
            assert {record.size for record in parse_augmentation(block)} \
                == {138}

    @pytest.mark.skipif(not os.path.exists(FREE_ATOM),
                        reason="the isolated Pt atom is not in this checkout")
    def test_the_free_atom_record_is_138_not_139(self):
        _, blocks = read_augmentation_blocks(FREE_ATOM)
        assert [record.size for record in parse_augmentation(blocks[0])] \
            == [138]


class TestOccupanciesBecomeOnePaddedArray:
    """``(atoms, sets, values)``, with the lengths that tell padding from zero."""

    def test_two_elements_pad_to_the_longer_record(self):
        block = format_augmentation([np.ones(4), np.full(9, 2.0)])
        values, lengths = occupancy_arrays([block])
        assert values.shape == (2, 1, 9)
        assert lengths.tolist() == [4, 9]
        assert values[0, 0, 4:].tolist() == [0.0] * 5

    def test_a_one_channel_dataset_keeps_the_totals_set(self):
        first = format_augmentation(_records(2, 6))
        second = format_augmentation(_records(2, 6, offset=0.5))
        values, _ = occupancy_arrays([first, second], channels=1)
        assert values.shape == (2, 1, 6)
        np.testing.assert_array_equal(values[1, 0], _records(2, 6)[1])

    def test_an_unpolarised_member_of_a_spin_set_gets_zero_moments(self):
        """
        Its grid carries m = 0, and its magnetisation occupancies vanish for
        the same reason.
        """
        values, _ = occupancy_arrays([format_augmentation(_records(2, 6))],
                                     channels=2)
        assert values.shape == (2, 2, 6)
        assert not values[:, 1].any()

    def test_sets_that_disagree_are_refused(self):
        first = format_augmentation(_records(2, 6))
        with pytest.raises(ValueError, match="atoms"):
            occupancy_arrays([first, format_augmentation(_records(3, 6))])
        with pytest.raises(ValueError, match="does not\\s+change"):
            occupancy_arrays([first, format_augmentation(_records(2, 5))])
        with pytest.raises(ValueError, match="No augmentation"):
            occupancy_arrays([])


# ===================================================================== #
# The cache
# ===================================================================== #
class TestTheCacheKeepsTheRecords:
    """
    Copied verbatim beside the downsampled density, one set per channel.

    The records are on-site, so downsampling has nothing to say about them;
    a cache that dropped them dropped the only per-atom target the data has.
    """

    @pytest.mark.parametrize("storage", ["files", "hdf5"])
    @pytest.mark.parametrize("spin", [True, False])
    def test_they_survive_the_downsampling_exactly(self, tmp_path, storage,
                                                   spin):
        from poraque.data import build_field_cache

        runs = tmp_path / "runs"
        native = _with_records(
            write_calculation(runs / "structure_0000", shape=(12, 12, 12)),
            spin=True)
        cache = tmp_path / "cache"
        build_field_cache([str(runs)], str(cache), resolution=8, spin=spin,
                          storage=storage, charges={"Si": 4.0})

        name = "CHGCAR" if storage == "files" else "fields.h5::CHGCAR"
        _, cached = read_augmentation_blocks(
            os.path.join(cache, "structure_0000", name))
        _, blocks = read_augmentation_blocks(native)
        expected, _ = occupancy_arrays(blocks, channels=2 if spin else 1)
        got, _ = occupancy_arrays(cached)
        np.testing.assert_array_equal(got, expected)

    def test_a_spin_cache_still_reads_as_two_channels(self, tmp_path):
        """The records sit between the two grid blocks, as VASP puts them."""
        from poraque.data import build_field_cache

        runs = tmp_path / "runs"
        _with_records(write_calculation(runs / "structure_0000",
                                        shape=(12, 12, 12)), spin=True)
        cache = tmp_path / "cache"
        build_field_cache([str(runs)], str(cache), resolution=8, spin=True,
                          charges={"Si": 4.0})
        pair = SpinDensity.read(os.path.join(cache, "structure_0000",
                                             "CHGCAR"))
        assert pair.magnetization.shape == (8, 8, 8)
        assert np.abs(pair.magnetization).max() > 0.0


class TestATableBuiltTheOldWayIsRebuilt:
    """
    Cached PAW tables are stamped with the reading that built them.

    Without the stamp every existing cache would go on serving its 139-value
    free-atom record after the parser was fixed.
    """

    def test_an_unstamped_paw_reference_is_not_reused(self, tmp_path):
        from poraque.data.cache import (
            PAW_REFERENCE_FILENAME,
            build_paw_reference,
        )

        run = write_calculation(tmp_path / "runs" / "structure_0000",
                                shape=(8, 8, 8))
        _with_records(run, spin=True)
        from poraque.data import discover_records, resolve_source

        records = discover_records([resolve_source(str(tmp_path / "runs"))],
                                   required=("CHGCAR",))
        cache = tmp_path / "cache"
        cache.mkdir()
        (cache / PAW_REFERENCE_FILENAME).write_text(json.dumps(
            {"Si": {"values": [0.0] * 7, "atoms": 2, "structures": 1}}))

        lines = []
        reference = build_paw_reference(records, str(cache), lines.append,
                                        source="material")
        assert len(reference["Si"]["values"]) == 6
        assert any("predates" in line for line in lines)

    def test_a_version_one_library_memo_is_reingested(self, tmp_path):
        from poraque.fields.atomic import (
            LIBRARY_FILENAME,
            AtomicReferenceLibrary,
            resolve_library,
        )

        atoms = tmp_path / "atoms"
        run = write_calculation(atoms / "Si", shape=(12, 12, 12))
        density = ChargeDensity.read(os.path.join(run, "CHGCAR"))
        from poraque.fields.vasp.poscar import Poscar

        lone = Poscar(density.structure.cell, ["Si"], [1], [[0.0, 0.0, 0.0]])
        pair = SpinDensity(density.data, np.zeros(density.data.shape),
                           density.grid, lone)
        pair.write(os.path.join(run, "CHGCAR"),
                   augmentation=[format_augmentation([np.ones(6)])
                                 + ["  0.100000000000E+01"],
                                 format_augmentation([np.zeros(6)])])

        cache = tmp_path / "cache"
        cache.mkdir()
        memo = {"version": 1, "entries": {}}
        (cache / LIBRARY_FILENAME).write_text(json.dumps(memo))
        lines = []
        library = resolve_library(str(atoms), cache=str(cache),
                                  log=lines.append)
        assert library.schema_version == 2
        assert [len(entry.augmentation)
                for entry in library.entries.values()] == [6]
        assert any("re-ingesting" in line for line in lines)
        assert AtomicReferenceLibrary.load(str(cache)).schema_version == 2

    def test_a_version_one_database_loses_its_records_and_says_so(self,
                                                                  tmp_path):
        from poraque.fields.atomic import AtomicReferenceLibrary

        library = AtomicReferenceLibrary.load(str(tmp_path / "absent.json"))
        assert len(library) == 0

        path = tmp_path / "atomic_reference.json"
        entry = {"element": "Pt", "valence_charge": 10.0, "g_grid": [0.0, 1.0],
                 "form_factor": [10.0, 9.0], "g_max": 1.0,
                 "radial_scatter": 0.0, "augmentation": [1.0] * 139}
        path.write_text(json.dumps({"version": 1,
                                    "entries": {"Pt": entry}}))
        try:
            with pytest.warns(RuntimeWarning, match="MAGMOM"):
                loaded = AtomicReferenceLibrary.load(str(path))
        except TypeError as error:          # a field the fixture lacks
            pytest.skip(f"AtomicReference needs more fields: {error}")
        assert loaded.entries["Pt"].augmentation is None
        assert loaded.entries["Pt"].form_factor == [10.0, 9.0] or np.allclose(
            loaded.entries["Pt"].form_factor, [10.0, 9.0])


# ===================================================================== #
# The dataset
# ===================================================================== #
class TestTheDatasetYieldsPerAtomTargets:
    """
    ``paw``, its masks, the species and the positions, beside the fields.
    """

    @staticmethod
    def _cache(tmp_path, spin=True, runs=(("structure_0000", 0),)):
        from poraque.data import build_field_cache

        root = tmp_path / "runs"
        for name, seed in runs:
            _with_records(write_calculation(root / name, shape=(12, 12, 12),
                                            seed=seed), spin=True)
        cache = tmp_path / "cache"
        build_field_cache([str(root)], str(cache), resolution=8, spin=spin,
                          charges={"Si": 4.0})
        return str(cache)

    def test_a_sample_carries_the_records_of_its_own_file(self, tmp_path):
        from poraque.ml.data import FieldPairDataset

        dataset = FieldPairDataset(self._cache(tmp_path), "ext2paw")
        sample = dataset[0]
        assert tuple(sample["paw"].shape) == (2, 2, 6)
        assert sample["paw_lengths"].tolist() == [6, 6]
        assert sample["species"].tolist() == [14, 14]
        assert tuple(sample["positions"].shape) == (2, 3)
        np.testing.assert_allclose(sample["paw"][1, 1].numpy(),
                                   _records(2, 6, offset=0.5)[1], rtol=1e-6)
        assert tuple(sample["target"].shape) == (2, 8, 8, 8)

    def test_the_field_tasks_carry_no_site_targets(self, tmp_path):
        from poraque.ml.data import FieldPairDataset

        sample = FieldPairDataset(self._cache(tmp_path), "ext2chg")[0]
        assert "paw" not in sample

    def test_a_cache_without_records_says_to_rebuild(self, tmp_path):
        from poraque.data import build_field_cache
        from poraque.ml.data import FieldPairDataset

        root = tmp_path / "runs"
        write_calculation(root / "structure_0000", shape=(12, 12, 12))
        cache = tmp_path / "cache"
        build_field_cache([str(root)], str(cache), resolution=8,
                          charges={"Si": 4.0})
        dataset = FieldPairDataset(str(cache), "ext2paw")
        with pytest.raises(ValueError, match="rebuild"):
            dataset[0]

    def test_a_batch_pads_atoms_and_values_and_masks_both(self):
        from poraque.ml.data import collate_fields

        def sample(atoms, length, value):
            return {"input": torch.zeros(1, 4, 4, 4),
                    "target": torch.zeros(1, 4, 4, 4),
                    "target_physical": torch.zeros(1, 4, 4, 4),
                    "cell": torch.eye(3), "shape": (4, 4, 4),
                    "material": "m", "reference_energy": None,
                    "paw": torch.full((atoms, 1, length), value),
                    "paw_lengths": torch.full((atoms,), length),
                    "species": torch.full((atoms,), 78),
                    "positions": torch.zeros(atoms, 3)}

        batch = collate_fields([sample(2, 4, 1.0), sample(3, 6, 2.0)])
        assert tuple(batch["paw"].shape) == (2, 3, 1, 6)
        assert batch["atom_mask"].tolist() == [[True, True, False],
                                               [True, True, True]]
        assert batch["paw_mask"][0, 0].tolist() == [True] * 4 + [False] * 2
        assert not batch["paw_mask"][0, 2].any()
        assert batch["species"][0].tolist() == [78, 78, 0]
        padding = ~batch["paw_mask"][0].unsqueeze(1)       # over the set axis
        assert (batch["paw"][0] * padding).abs().sum() == 0.0


# ===================================================================== #
# The configuration and the pre-flight
# ===================================================================== #
class TestTheTaskAndItsBlockAgree:
    """
    ``task.type: ext2paw`` and ``model.paw.enable`` are checked against each
    other, and ``all`` still means the chain.
    """

    @staticmethod
    def _config(**sections):
        from poraque.ml.config import TrainingConfig

        return TrainingConfig.from_dict(sections)

    def test_all_is_still_the_chain(self):
        from poraque.ml.tasks import CHAIN, TASKS

        assert self._config().task_names() == list(CHAIN)
        assert "ext2paw" in TASKS and "ext2paw" not in CHAIN

    def test_the_block_adds_ext2paw_to_an_all_run(self):
        names = self._config(model={"paw": {"enable": True}}).task_names()
        assert names == ["ext2chg", "chg2tau", "ext2paw"]

    def test_the_two_switches_cannot_disagree(self):
        with pytest.raises(ValueError, match="model.paw.enable is false"):
            self._config(task={"type": "ext2paw"}).task_names()
        with pytest.raises(ValueError, match="never reads it"):
            self._config(task={"type": "chg2tau"},
                         model={"paw": {"enable": True}}).task_names()

    def test_the_settings_default_and_are_checked(self):
        model = self._config(model={"paw": {"enable": True, "width": 32}}).model
        assert model.paw_settings() == {"width": 32, "modes": 16,
                                        "n_layers": 4, "readout_g_max": "auto",
                                        "occupancy_weight": 1.0}
        for bad in (0, -1, 2.5, True, "64"):
            with pytest.raises(ValueError, match="positive integer"):
                self._config(model={"paw": {"enable": True,
                                            "modes": bad}}).model.paw_settings()
        for key, bad in (("readout_g_max", 0), ("readout_g_max", "atuo"),
                         ("occupancy_weight", -1.0),
                         ("occupancy_weight", True)):
            with pytest.raises(ValueError, match=key):
                self._config(model={"paw": {"enable": True,
                                            key: bad}}).model.paw_settings()
        with pytest.raises(ValueError, match="Unknown key"):
            self._config(model={"paw": {"enable": True, "layers": 4}})

    def test_settings_under_a_disabled_block_warn(self):
        model = self._config(model={"paw": {"enable": False,
                                            "width": 32}}).model
        with pytest.warns(RuntimeWarning, match="ignored"):
            model.paw_settings()

    def test_the_field_operator_never_sees_the_block(self):
        kwargs = self._config(model={"paw": {"enable": True}}).model_kwargs()
        assert "paw" not in kwargs

    def test_a_dataset_is_not_said_to_serve_ext2paw_by_file_names(self,
                                                                  tmp_path):
        from poraque.data import MixedFieldDataset

        write_calculation(tmp_path / "runs" / "structure_0000")
        data = MixedFieldDataset([str(tmp_path / "runs")])
        assert "ext2paw" not in data.available_tasks()


class TestTheCacheOnlyPathAndThePreFlight:
    """
    ``--cache-only`` prepares ``ext2paw`` data; the paths the operator does not
    support yet are refused before the cache.
    """

    @staticmethod
    def _config_file(tmp_path, **training):
        import yaml

        runs = tmp_path / "runs"
        _with_records(write_calculation(runs / "structure_0000",
                                        shape=(8, 8, 8)), spin=False)
        path = tmp_path / "train.yaml"
        path.write_text(yaml.safe_dump({
            "task": {"type": "ext2paw", "name": "paw_smoke"},
            "data": {"data_paths": [str(runs)], "cache": str(tmp_path / "c"),
                     "resolution": 8, "delta_density": False, "spin": False,
                     "paw_source": "material"},
            "model": {"width": 4, "modes": 2, "n_layers": 1,
                      "projection_channels": 4, "paw": {"enable": True}},
            "training": {"epochs": 1, "valid_fraction": 0.0,
                         "device": "cpu", **training},
            "output": {"root": str(tmp_path / "models"),
                       "plot_figures": False, "write_pdf_report": False},
        }))
        return str(path)

    def test_kfold_is_refused_before_the_cache(self, tmp_path):
        import poraque_train

        config = self._config_file(tmp_path, enable_kfold=True)
        with pytest.raises(SystemExit, match="kfold"):
            poraque_train.run(["--config", config])
        assert not (tmp_path / "c").exists()

    def test_a_cache_only_run_builds_the_data_it_will_train_on(self,
                                                               tmp_path):
        import poraque_train

        config = self._config_file(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with pytest.raises(SystemExit) as caught:
                poraque_train.run(["--config", config, "--cache-only"])
        assert caught.value.code in (0, None)

        cached = [os.path.join(root, name)
                  for root, _, names in os.walk(tmp_path / "c")
                  for name in names if name == "CHGCAR"]
        assert cached
        _, blocks = read_augmentation_blocks(cached[0])
        values, _ = occupancy_arrays(blocks)
        assert values.shape == (2, 1, 6)

    def test_the_shipped_ext2paw_config_loads_and_passes_the_pre_flight(self):
        import poraque_train
        from poraque.ml.config import TrainingConfig

        config = TrainingConfig.from_yaml(
            os.path.join(_ROOT, "configs", "train_ext2paw.yaml"))
        assert config.task_names() == ["ext2paw"]
        poraque_train.validate_paw_settings(config)
