# -*- coding: utf-8 -*-
# file: test_cache_only.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
``poraque-train --cache-only``: build the dataset cache on a CPU and stop.

The mode exists for HPC job separation. Building the cache is NumPy work --
parsing, spectral downsampling, writing -- and a GPU allocation spent on it is
a GPU allocation spent idle, so a CPU job builds it first and a GPU job then
trains from it. Three things make that split trustworthy, and each has a test
here:

* the CPU job forms **no** process group and builds **no** model, whatever the
  config asked for -- ``training.device`` is forced to the CPU and
  ``training.distributed`` to ``off`` before either is consulted;
* it exits **0** after a summary that names the cache, counts the materials
  per task and sizes the fields on disk and in RAM, so the GPU job can be
  sized against it;
* the GPU job, run with the same config, **finds the cache and rebuilds
  nothing** -- every row of its cache table reads ``cached``.
"""

import os
import sys

import pytest
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import poraque_train  # noqa: E402
from test_data_sources import write_calculation  # noqa: E402


def _config(tmp_path, **overrides):
    """A config for two tiny synthetic runs, written where the test can read it."""
    runs = tmp_path / "runs"
    write_calculation(runs / "structure_0000", shape=(8, 8, 8), seed=0)
    write_calculation(runs / "structure_0001", shape=(8, 8, 8), seed=1)
    settings = {
        "task": {"type": "all", "name": "cache_only_smoke"},
        "data": {
            "data_paths": [str(runs)],
            "cache": str(tmp_path / "cache"),
            "resolution": 8,
            # No atomic reference and no augmentation records in a synthetic
            # run: neither default is under test here.
            "delta_density": False,
            "paw_source": "material",
            "spin": False,
        },
        "model": {"width": 4, "modes": 2, "n_layers": 1,
                  "projection_channels": 4},
        "training": {"epochs": 1, "valid_fraction": 0.0, "eval_epoch": 1,
                     "early_stopping": 0, "device": "cpu"},
        "output": {"root": str(tmp_path / "models"),
                   "plot_figures": False, "write_pdf_report": False},
    }
    for section, values in overrides.items():
        settings[section].update(values)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(settings))
    return str(path)


def _log(tmp_path):
    return (tmp_path / "models" / "cache_only_smoke" / "log"
            / "cache_only_smoke.log").read_text()


@pytest.fixture
def no_training(monkeypatch):
    """Turn every route into a model or a group into a loud failure."""
    def refuse(*_, **__):
        raise AssertionError("--cache-only reached the training path")

    monkeypatch.setattr(poraque_train, "run_task", refuse)
    monkeypatch.setattr(poraque_train, "run_task_kfold", refuse)
    monkeypatch.setattr(poraque_train, "initialize_distributed", refuse)
    monkeypatch.setattr(poraque_train, "build_operator", refuse)


class TestTheFlagIsExposed:
    def test_it_is_off_by_default_and_is_not_a_config_override(self):
        args = poraque_train.build_parser().parse_args([])
        assert args.cache_only is False

    def test_the_help_says_what_it_is_for(self):
        parser = poraque_train.build_parser()
        text = parser.format_help()
        assert "--cache-only" in text
        assert "HPC" in text
        assert "exits" in text


class TestACacheOnlyRunBuildsTheCacheAndStops:
    def test_it_exits_zero_before_any_model_exists(self, tmp_path, no_training):
        config = _config(tmp_path)

        with pytest.raises(SystemExit) as stop:
            poraque_train.run(["--config", config, "--cache-only"])

        assert stop.value.code == 0

    def test_the_cache_is_on_disk_and_no_weights_are(self, tmp_path,
                                                    no_training):
        config = _config(tmp_path)
        with pytest.raises(SystemExit):
            poraque_train.run(["--config", config, "--cache-only"])

        cache = tmp_path / "cache"
        (tag,) = [entry for entry in os.listdir(cache)]
        for material in ("structure_0000", "structure_0001"):
            for field in ("EXTCAR", "CHGCAR", "TAUCAR"):
                assert (cache / tag / material / field).exists()
        assert (cache / tag / "cache_fingerprint.json").exists()

        weights = [name for _, _, files in os.walk(tmp_path / "models")
                   for name in files if name.endswith(".poraque")]
        assert weights == []

    def test_the_summary_names_the_cache_and_counts_the_materials(
            self, tmp_path, no_training, capsys):
        config = _config(tmp_path)
        with pytest.raises(SystemExit):
            poraque_train.run(["--config", config, "--cache-only"])

        out = capsys.readouterr().out
        assert "[CACHE SUMMARY]" in out
        assert "built; no training was run (--cache-only)" in out
        assert "materials   : 2 cached  (ext2pseudo: 2, pseudo2tau: 2)" in out
        assert "grid shapes : 8x8x8" in out
        assert "on disk     :" in out
        assert "decoded     :" in out
        assert "elements    : Si" in out
        # The same summary is in the run's own log, which is what a batch job
        # leaves behind.
        assert "[CACHE SUMMARY]" in _log(tmp_path)

    def test_the_device_is_the_cpu_whatever_was_asked_for(self, tmp_path,
                                                          no_training):
        """A CPU node has no GPU to honor ``--device cuda`` on, and the mode
        must not fail there for want of one -- nor form a group."""
        config = _config(tmp_path, training={"device": "cuda",
                                             "strict_device": True,
                                             "distributed": "auto"})
        with pytest.raises(SystemExit) as stop:
            poraque_train.run(["--config", config, "--cache-only",
                               "--device", "cuda", "--strict-device"])

        assert stop.value.code == 0
        assert "device : cpu" in _log(tmp_path)

    def test_a_second_cache_only_run_reuses_every_material(self, tmp_path,
                                                           no_training, capsys):
        config = _config(tmp_path)
        with pytest.raises(SystemExit):
            poraque_train.run(["--config", config, "--cache-only"])
        capsys.readouterr()

        with pytest.raises(SystemExit) as stop:
            poraque_train.run(["--config", config, "--cache-only"])

        assert stop.value.code == 0
        rows = [line for line in capsys.readouterr().out.splitlines()
                if _is_material_row(line) and "8x8x8" in line]
        assert len(rows) == 2 and all(row.endswith("cached") for row in rows)


def _is_material_row(line):
    """A cache-table row: ``i/N`` in the first column, the material next."""
    tokens = line.split()
    return (len(tokens) > 1 and "/" in tokens[0]
            and tokens[1].startswith("structure_"))


class TestTheTrainingJobFindsTheCache:
    """The half of the contract that makes the split worth making."""

    def test_the_gpu_job_rebuilds_nothing(self, tmp_path, capsys):
        config = _config(tmp_path)
        with pytest.raises(SystemExit):
            poraque_train.run(["--config", config, "--cache-only"])
        capsys.readouterr()

        results = poraque_train.run(["--config", config, "--no-plots"])

        out = capsys.readouterr().out
        # The cache table's rows carry the native and cached grid shapes; the
        # per-material metrics rows further down do not.
        rows = [line for line in out.splitlines()
                if _is_material_row(line) and "8x8x8" in line]
        assert len(rows) == 2 and all(row.endswith("cached") for row in rows)
        assert {result["task"] for result in results} == {"ext2pseudo",
                                                            "pseudo2tau"}
        assert (tmp_path / "models" / "cache_only_smoke"
                / "cache_only_smoke.poraque").exists()
