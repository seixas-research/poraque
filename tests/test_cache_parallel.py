# -*- coding: utf-8 -*-
# file: test_cache_parallel.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
The cache build over several processes, one material each.

The build is a Python text codec, not the FFT work its Slurm script claimed:
measured per material on the platinum set, the spectral downsampling is 2.2 %
of the clock and the Fortran ``E17.11`` reader and writer are 86 %. Threads
cannot touch that --- there is no BLAS call in the path and ``np.fft`` is
single-threaded --- so the axis that scales is materials, and
``data.cache_workers`` splits them over processes.

What has to stay true, and is what these tests pin:

* the cache a pool writes is **byte-identical** to the one the serial build
  writes, down to the text format's eleventh digit;
* nothing is left half-written, because ``_build_one`` already writes through
  ``.partial-<pid>`` and ``os.replace``;
* the bookkeeping stays in the parent: ``cache_summary.json`` and
  ``cache_fingerprint.json`` are plain writes and would lose updates from
  several processes;
* a record can actually be **shipped** to a worker. It cannot be pickled as it
  stands --- its source's options dict holds the build's log callable --- and
  ``spawn``, which is macOS's default, has to pickle it;
* ``auto`` sizes itself on the largest grid in the set and on the cores the
  process may use, not on the node's core count.
"""

import json
import multiprocessing
import os
import pickle

import numpy as np
import pytest

from poraque.data.cache import (CACHE_FINGERPRINT_FILENAME,
                                CACHE_SUMMARY_FILENAME, WORKER_BASE_BYTES,
                                WORKER_BYTES_PER_VOXEL, WORKER_MEMORY_BUDGET,
                                WORKER_MINIMUM_VOXELS, build_field_cache,
                                resolve_cache_workers, shippable_options)
from poraque.fields import ChargeDensity, FieldGrid
from poraque.fields.vasp.poscar import Poscar


def write_material(directory, shape=(12, 12, 12), seed=0, tau=True):
    """One material's directory: a CHGCAR and, optionally, a TAUCAR."""
    os.makedirs(directory, exist_ok=True)
    cell = np.diag([6.0, 6.0, 6.0])
    structure = Poscar(cell, ["H"], [2],
                       np.array([[0.1, 0.1, 0.1], [0.6, 0.6, 0.6]]))
    grid = FieldGrid(shape, cell)
    rng = np.random.default_rng(seed)
    values = 1.0 + 0.1 * rng.standard_normal(shape)
    ChargeDensity(values, grid, structure=structure).write(
        os.path.join(directory, "CHGCAR"))
    if tau:
        from poraque.fields import KineticEnergyDensity

        KineticEnergyDensity(np.abs(values), grid,
                             structure=structure).write(
            os.path.join(directory, "TAUCAR"))
    return directory


def write_dataset(root, count=6, shape=(12, 12, 12)):
    """A directory of ``count`` per-material subdirectories."""
    os.makedirs(root, exist_ok=True)
    for index in range(count):
        write_material(os.path.join(root, f"material_{index:03d}"),
                       shape=shape, seed=index)
    return root


def read_tree(root):
    """``{relative path: bytes}`` for every file under ``root``."""
    contents = {}
    for directory, _, names in os.walk(root):
        for name in names:
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                contents[os.path.relpath(path, root)] = handle.read()
    return contents


class TestAPooledCacheIsTheSameCache:
    """
    Regression guard for the whole feature: parallelism must not be visible in
    the result. The build writes a text format with eleven significant digits,
    and a reordering, a re-rounding or a dropped augmentation record would all
    train a different model from the same config.
    """

    def test_a_text_cache_is_byte_identical_to_the_serial_build(self,
                                                                tmp_path):
        root = write_dataset(str(tmp_path / "data"), count=6)
        serial = str(tmp_path / "serial")
        pooled = str(tmp_path / "pooled")

        build_field_cache(root, serial, resolution=8, charges={"H": 1.0},
                          storage="files", workers=1)
        build_field_cache(root, pooled, resolution=8, charges={"H": 1.0},
                          storage="files", workers=3)

        skip = {CACHE_SUMMARY_FILENAME, CACHE_FINGERPRINT_FILENAME}
        left = {name: data for name, data in read_tree(serial).items()
                if os.path.basename(name) not in skip}
        right = {name: data for name, data in read_tree(pooled).items()
                 if os.path.basename(name) not in skip}
        assert sorted(left) == sorted(right)
        for name in left:
            assert left[name] == right[name], f"{name} differs"

    def test_an_hdf5_cache_holds_the_same_values_as_the_serial_build(
            self, tmp_path):
        """
        Values rather than bytes, and the reason is not parallelism: an HDF5
        store stamps its root with ``created``, an ISO timestamp to the second
        (``hdf5.py``), so two builds a second apart differ in two bytes and
        nowhere else. Comparing the bytes here would pin the clock, not the
        cache.
        """
        from poraque.fields.hdf5 import field_names, read_volumetric

        root = write_dataset(str(tmp_path / "data"), count=6)
        serial = str(tmp_path / "serial")
        pooled = str(tmp_path / "pooled")

        build_field_cache(root, serial, resolution=8, charges={"H": 1.0},
                          storage="hdf5", workers=1)
        build_field_cache(root, pooled, resolution=8, charges={"H": 1.0},
                          storage="hdf5", workers=3)

        materials = sorted(name for name in os.listdir(serial)
                           if os.path.isdir(os.path.join(serial, name)))
        assert materials == [f"material_{i:03d}" for i in range(6)]
        for material in materials:
            left = os.path.join(serial, material, "fields.h5")
            right = os.path.join(pooled, material, "fields.h5")
            assert sorted(field_names(left)) == sorted(field_names(right))
            for name in field_names(left):
                _, one, _ = read_volumetric(f"{left}::{name}")
                _, two, _ = read_volumetric(f"{right}::{name}")
                assert np.array_equal(one, two), f"{material}/{name} differs"

    def test_it_leaves_no_partial_files(self, tmp_path):
        """
        `.partial-<pid>` is what makes the pool safe; a survivor means a writer
        raised and the cleanup missed it, and the next build would move a torn
        field into place.
        """
        root = write_dataset(str(tmp_path / "data"), count=6)
        cache = str(tmp_path / "cache")
        build_field_cache(root, cache, resolution=8, charges={"H": 1.0},
                          workers=3)
        leftovers = [name for name in read_tree(cache)
                     if ".partial" in os.path.basename(name)]
        assert leftovers == []

    def test_the_summary_records_every_material_once(self, tmp_path):
        """
        `_write_summary` is a plain `open(..., "w")`. It stays in the parent
        precisely so several workers cannot lose each other's updates.
        """
        root = write_dataset(str(tmp_path / "data"), count=6)
        cache = str(tmp_path / "cache")
        build_field_cache(root, cache, resolution=8, charges={"H": 1.0},
                          workers=3)
        with open(os.path.join(cache, CACHE_SUMMARY_FILENAME)) as handle:
            summary = json.load(handle)
        assert sorted(summary) == [f"material_{i:03d}" for i in range(6)]
        assert all("ranges" in entry for entry in summary.values())

    def test_a_pooled_build_resumes_like_a_serial_one(self, tmp_path):
        """
        Resume is `_all_present`, which the pool inherits: a second run must
        rebuild nothing and must still fill the table from the summary.
        """
        root = write_dataset(str(tmp_path / "data"), count=6)
        cache = str(tmp_path / "cache")
        build_field_cache(root, cache, resolution=8, charges={"H": 1.0},
                          workers=3)
        lines = []
        build_field_cache(root, cache, resolution=8, charges={"H": 1.0},
                          workers=3, log=lines.append)
        rows = [line for line in lines if "material_" in line]
        assert rows and all("cached" in row for row in rows)


class TestOneBadMaterialDoesNotLoseTheArchive:
    """
    A worker returns its exception instead of dying with it, so an archive of
    thousands is not lost to one truncated density. The parent still writes the
    summary for everything that succeeded -- which is what makes a rerun resume
    -- and only then raises, naming what failed.
    """

    def test_the_good_materials_are_kept_and_the_failure_is_named(
            self, tmp_path):
        root = write_dataset(str(tmp_path / "data"), count=5)
        broken = os.path.join(root, "material_002", "CHGCAR")
        with open(broken, "r") as handle:
            text = handle.read()
        with open(broken, "w") as handle:
            handle.write(text[:len(text) // 3])     # truncated mid-block

        cache = str(tmp_path / "cache")
        with pytest.raises(RuntimeError, match="material_002"):
            build_field_cache(root, cache, resolution=8,
                              charges={"H": 1.0}, workers=3)

        with open(os.path.join(cache, CACHE_SUMMARY_FILENAME)) as handle:
            summary = json.load(handle)
        assert "material_002" not in summary
        assert sorted(summary) == ["material_000", "material_001",
                                   "material_003", "material_004"]

    def test_the_message_says_how_many_of_how_many(self, tmp_path):
        root = write_dataset(str(tmp_path / "data"), count=4)
        broken = os.path.join(root, "material_001", "CHGCAR")
        with open(broken, "w") as handle:
            handle.write("not a density\n")
        with pytest.raises(RuntimeError, match="1 of 4 material"):
            build_field_cache(root, str(tmp_path / "cache"), resolution=8,
                              charges={"H": 1.0}, workers=2)


class TestARecordHasToBeShippableToAWorker:
    """
    The blocker that made the first pool fail, and why `shippable_options`
    exists. A `MaterialRecord` carries its source, the source carries its
    options, and `build_field_cache` puts its log callable in those options --
    so the record holds a live closure over an open file. `fork` hides this;
    macOS defaults to `spawn` and a Slurm `srun` launch has no fork either.
    """

    def test_the_log_is_what_makes_a_record_unpicklable(self, tmp_path):
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=2)
        source = resolve_source(root, format="auto", log=lambda *_: None)
        record = discover_records([source], required=("CHGCAR",))[0]
        with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
            pickle.dumps(record)

    def test_stripping_the_log_makes_the_options_shippable(self, tmp_path):
        root = write_dataset(str(tmp_path / "data"), count=2)
        options = {"log": lambda *_: None, "potcar_dir": None,
                   "format": "auto", "charges": {"H": 1.0}}
        shipped = shippable_options(options)
        assert "log" not in shipped
        assert shipped["charges"] == {"H": 1.0}
        pickle.loads(pickle.dumps(shipped))         # does not raise
        assert str(root)                            # the fixture is real

    def test_the_build_ships_records_under_spawn(self, tmp_path):
        """
        End to end with the start method that actually pickles. `fork` would
        pass this test without proving anything.
        """
        assert "spawn" in multiprocessing.get_all_start_methods()
        root = write_dataset(str(tmp_path / "data"), count=4)
        cache = str(tmp_path / "cache")
        build_field_cache(root, cache, resolution=8, charges={"H": 1.0},
                          workers=2, log=lambda *_: None)
        assert os.path.exists(os.path.join(cache, "material_000", "CHGCAR"))


class TestTheWorkerCountIsResolvedBeforeTheBuild:
    """
    A worker count is a setting, so a bad one raises by name before any file is
    written -- and `auto` is sized on memory, not only on cores, because one
    dataset's materials are not one size.
    """

    def test_zero_and_one_are_the_single_process_build(self):
        assert resolve_cache_workers(0, []) == 1
        assert resolve_cache_workers(1, []) == 1
        assert resolve_cache_workers("0", []) == 1

    def test_a_count_is_taken_as_given(self):
        assert resolve_cache_workers(5, []) == 5
        assert resolve_cache_workers("5", []) == 5

    def test_a_misspelling_raises_and_names_the_key(self):
        with pytest.raises(ValueError, match="data.cache_workers"):
            resolve_cache_workers("yes please", [])

    def test_a_negative_count_raises(self):
        with pytest.raises(ValueError, match="negative"):
            resolve_cache_workers(-2, [])

    def test_auto_stays_serial_when_the_whole_set_is_tiny(self, tmp_path):
        """
        A pool has to pay for its own start-up. Measured: a `spawn`ed pool
        costs 0.49-0.60 s once each worker has imported the package, and the
        build runs at 1.03 s per million native voxels -- so below about 1 M
        voxels in total the pool is pure loss. Without this, every synthetic
        8^3 run in the end-to-end tests spawned processes to do microseconds
        of work.
        """
        from poraque.data.cache import _native_voxels
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=4, shape=(8, 8, 8))
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert sum(_native_voxels(records)) < WORKER_MINIMUM_VOXELS

        lines = []
        assert resolve_cache_workers("auto", records, lines.append) == 1
        assert any("pays for its own start-up" in line for line in lines)

    def test_an_explicit_count_ignores_the_floor(self, tmp_path):
        """
        The floor bounds `auto`, not the feature: someone who asks for four
        workers on a toy set is testing the pool and should get it.
        """
        root = write_dataset(str(tmp_path / "data"), count=4, shape=(8, 8, 8))
        from poraque.data.sources import discover_records, resolve_source

        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert resolve_cache_workers(4, records) == 4

    def test_auto_never_exceeds_the_material_count(self, tmp_path):
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=2)
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert resolve_cache_workers("auto", records) <= 2

    def test_auto_is_sized_on_the_largest_grid_not_the_average(self, tmp_path):
        """
        A set of small cells and one huge cell has to be sized on the huge one:
        every worker may be handed it, and sizing on the mean is how a build
        goes into swap on the last eleven materials.
        """
        from poraque.data.cache import _native_voxels
        from poraque.data.sources import discover_records, resolve_source

        root = str(tmp_path / "data")
        write_dataset(root, count=3, shape=(8, 8, 8))
        write_material(os.path.join(root, "material_big"), shape=(24, 24, 24),
                       seed=9)
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert max(_native_voxels(records)) == 24 ** 3

        budget = WORKER_BASE_BYTES + 24 ** 3 * WORKER_BYTES_PER_VOXEL
        assert resolve_cache_workers("auto", records) <= max(
            1, WORKER_MEMORY_BUDGET // budget)

    def test_the_budget_is_the_slurm_allocation_when_there_is_one(
            self, monkeypatch):
        """
        A constant is the wrong answer on a cluster. `--cpus-per-task=8
        --mem=32G` against an 8 GiB default picks four workers for a 288^3 set
        and idles half the allocation -- the exact job this feature is for.
        """
        from poraque.data.cache import _memory_budget

        for name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU",
                     "SLURM_CPUS_PER_TASK"):
            monkeypatch.delenv(name, raising=False)
        assert _memory_budget() == (WORKER_MEMORY_BUDGET, "default budget")

        monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
        budget, source = _memory_budget()
        assert source == "Slurm allocation"
        assert budget == int(0.8 * 32768 * 1024 ** 2)

        monkeypatch.delenv("SLURM_MEM_PER_NODE")
        monkeypatch.setenv("SLURM_MEM_PER_CPU", "4096")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
        budget, source = _memory_budget()
        assert source == "Slurm allocation"
        assert budget == int(0.8 * 4096 * 8 * 1024 ** 2)

    def test_the_core_count_is_the_slurm_allocation_not_the_node(
            self, monkeypatch, tmp_path):
        """
        `--cpus-per-task=4` on a 96-core node must mean four.

        `sched_getaffinity` *usually* says so, because Slurm narrows the mask
        through `task/affinity` or `task/cgroup` -- but that is a site
        configuration, not a guarantee, and the call cannot be simulated off a
        cluster. Found exactly that way: simulating the sbatch's allocations on
        a laptop, `--cpus-per-task=4` picked six workers, because the fallback
        was `os.cpu_count()`.
        """
        from poraque.data import cache as cache_module
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=8)
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        # A real set, so the start-up floor does not answer first, and plenty
        # of memory, so the core count is the bound under test.
        monkeypatch.setattr(cache_module, "_native_voxels",
                            lambda _: [1_000_000] * 8)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(256 * 1024))
        monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
        assert cache_module._usable_cores() == 4
        assert resolve_cache_workers("auto", records) == 4

        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
        assert resolve_cache_workers("auto", records) == 2

    def test_without_slurm_the_core_count_falls_back_to_the_machine(
            self, monkeypatch):
        """Off a cluster there is no allocation to read, and that is fine."""
        from poraque.data import cache as cache_module

        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        assert cache_module._usable_cores() >= 1

    def test_a_malformed_slurm_variable_falls_back_rather_than_raising(
            self, monkeypatch):
        """
        Slurm writes `32G` in some versions and `32768` in others, and an empty
        value is normal. None of those is worth failing a build over.
        """
        from poraque.data.cache import _memory_budget

        monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        for value in ("", "unlimited", "???"):
            monkeypatch.setenv("SLURM_MEM_PER_NODE", value)
            assert _memory_budget() == (WORKER_MEMORY_BUDGET, "default budget")
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "32G")
        budget, source = _memory_budget()
        assert source == "Slurm allocation"
        assert budget == int(0.8 * 32 * 1024 ** 2)

    def test_the_log_line_names_which_bound_was_hit(self, tmp_path,
                                                    monkeypatch):
        """
        "4 within the 8 GiB default budget" and "4 within the 25.6 GiB Slurm
        allocation" are different facts, and only one of them is worth editing
        a submission script over.
        """
        from poraque.data import cache as cache_module
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=2)
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        # A real set, not these two 12^3 toys, which the start-up floor would
        # (correctly) answer before any of the three bounds is consulted. The
        # grids are reported rather than written: this is about the line, and
        # writing 1 M voxels of text to assert on a log message is not a trade.
        monkeypatch.setattr(cache_module, "_native_voxels",
                            lambda _: [1_260_000, 23_890_000])

        monkeypatch.delenv("SLURM_MEM_PER_CPU", raising=False)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", "32768")
        lines = []
        resolve_cache_workers("auto", records, lines.append)
        assert any("Slurm allocation" in line for line in lines)
        assert any("usable core(s)" in line for line in lines)
        assert any("23.89 M voxels" in line for line in lines), \
            "the line must name the largest grid, which sets the estimate"

    def test_the_largest_grid_comes_from_the_header_not_the_file_size(
            self, tmp_path):
        """
        A spin-polarized CHGCAR holds two grid blocks, so bytes/18 reads as
        twice the voxel count and would halve the worker count on exactly the
        datasets that need them most. The header carries the shape.
        """
        from poraque.data.cache import _native_voxels
        from poraque.data.sources import discover_records, resolve_source

        root = str(tmp_path / "data")
        write_dataset(root, count=2, shape=(10, 10, 10))
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert max(_native_voxels(records)) == 1000


class TestTheSpinProbeIsNotASerialFullArchivePass:
    """
    `_resolve_spin` asks every density in the set whether it carries a
    magnetization block, before the first field is written. On the 97-cell
    platinum set that cost 29.2 s -- 11 % of the serial build, and under Amdahl
    the term that would have capped a pool at 3.2x however many cores were
    added. It is per-record independent, so it takes the same pool, and it must
    reach the same answer either way.
    """

    @pytest.mark.parametrize("workers", [1, 3])
    def test_it_agrees_with_the_serial_answer(self, tmp_path, workers):
        from poraque.data.cache import _polarized_records
        from poraque.data.sources import discover_records, resolve_source

        root = write_dataset(str(tmp_path / "data"), count=4)
        source = resolve_source(root, format="auto")
        records = discover_records([source], required=("CHGCAR",))
        assert _polarized_records(records, workers) == []

    def test_an_unreadable_density_is_not_polarized_rather_than_fatal(
            self, tmp_path):
        from poraque.data.cache import _spin_of

        missing = str(tmp_path / "nothing" / "CHGCAR")
        assert _spin_of(missing) is False
        assert _spin_of(None) is False
