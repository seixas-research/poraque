# -*- coding: utf-8 -*-
# file: profile_cache.py

# This code is part of Poraquê.
# MIT License
#
# Copyright (c) 2026 Leandro Seixas Rocha <leandro.rocha@ilum.cnpem.br>

r"""
What is the cache build actually spending its time on, and what parallelizes?

``scripts/slurm/poraque_cache.sbatch`` asks for eight cores, exports
``OMP_NUM_THREADS=8``, runs one task, and its own comment says the work is
"NumPy work -- reading every CHGCAR/TAUCAR, computing V_ext, spectrally
downsampling, writing". Two of those four are text codecs written in Python and
one is an FFT, so before choosing between OpenMP and MPI the question is which
of them the clock is in.

Three measurements, each independent of the others:

``--profile``
    Wall-clock time per stage of :func:`poraque.data.cache._build_one`, for
    both storage layouts. No ``cProfile``: the hot paths are a few million
    one-line calls and the profiler's per-call overhead triples them, which
    inverts the ranking. ``time.perf_counter`` around the real calls instead.
    Note that this measures the build **as it is now**, with the two fixes in:
    the ``grid header`` and text-read rows are smaller than the table in this
    directory's README, which is the pre-fix profile and is labeled as such.

``--fixes``
    Two places where the serial build did work it did not need, and one that
    is still there. **Both fixes have landed**, so this mode carries the
    pre-fix implementations itself (:func:`legacy_read_block`,
    :func:`legacy_grid_from_file`) and measures the shipped code against them
    --- otherwise it would benchmark the new reader against itself and print
    1.0x. Each is checked to return the same values, which is the half of the
    claim that matters.

``--scaling``
    The parallel measurement. One process per material, over a pool, and the
    resulting cache compared byte for byte against the serial one. This is the
    proposal, not a sketch of it: the worker is the real ``_build_one``, and
    the pool uses ``spawn``, which is what macOS gives by default and what
    exposes everything the fork path hides.

Usage
-----
``python experiments/cache_parallelism/profile_cache.py --profile``
``python experiments/cache_parallelism/profile_cache.py --fixes``
``python experiments/cache_parallelism/profile_cache.py --scaling --materials 16``

``--data`` points at a directory of per-material subdirectories (default
``data/vasp/structures``); ``--potcar-dir`` at a ``POTCAR`` library, without
which V_ext is the Gaussian pseudo-ion model and the EXTCAR stage measures a
different computation.
"""

import argparse
import filecmp
import json
import multiprocessing
import os
import resource
import shutil
import sys
import time

import numpy as np

#: Fields the build writes, in file order.
FIELDS = ("EXTCAR", "CHGCAR", "TAUCAR")

#: The source, rebuilt inside each worker because it cannot be shipped --- see
#: :func:`_initialize`. One per process, module level, which is the only state
#: a ``spawn``ed worker can keep between tasks.
_SOURCE = None


# ---------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------- #
def peak_rss_bytes(children=False):
    """
    Peak resident set size, in bytes.

    ``getrusage`` reports bytes on macOS and KiB on Linux, the same correction
    :mod:`poraque.ml.profiling` makes; and it is a running maximum, so it
    never falls.
    """
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    peak = resource.getrusage(who).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def source_options(arguments):
    """
    The source options, **without** a ``log``.

    The omission is the finding: ``CalculationSource`` keeps its options dict,
    ``MaterialRecord`` keeps the source, and the cache builder puts its logging
    callable in that dict. A record therefore holds a live closure over an open
    log file, and pickling one raises --- which is exactly what a ``spawn``ed
    pool must do to send it work.
    """
    return {"potcar_dir": arguments.potcar_dir, "format": "auto"}


def discover(arguments, limit=None):
    """Records under ``--data``, smallest native grid first when limited."""
    from poraque.data.sources import discover_records, resolve_source

    source = resolve_source(arguments.data, **source_options(arguments))
    records = discover_records([source], required=("CHGCAR",))
    if limit:
        records = sorted(
            records, key=lambda record: os.path.getsize(
                record.files["CHGCAR"]))[:limit]
    return source, records


def fresh(path):
    """An empty directory at ``path``, whatever was there before."""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------- #
# --profile: where the time goes
# ---------------------------------------------------------------------- #
def measure_profile(arguments):
    """
    Per-stage wall clock of one material, averaged over ``--materials``.

    The stages are the ones :func:`~poraque.data.cache._build_one` runs, in
    its order, called directly rather than through it so each can be timed
    without the others.
    """
    from poraque.data import cache as cache_module
    from poraque.fields.resample import downsampled_grid, resample_field

    _, records = discover(arguments, limit=arguments.materials)
    report = {}

    for storage in ("files", "hdf5"):
        destination = fresh(os.path.join(arguments.scratch,
                                         f"profile_{storage}"))
        stages = {}
        started = time.perf_counter()
        for record in records:
            clock = time.perf_counter()
            native = record.source.grid(record)
            stages["grid header"] = stages.get("grid header", 0.0) + \
                time.perf_counter() - clock
            reduced = downsampled_grid(native, arguments.resolution, None)
            os.makedirs(os.path.join(destination, record.identifier),
                        exist_ok=True)

            for name in FIELDS:
                clock = time.perf_counter()
                field = record.source.read(record, name, native,
                                           spin=arguments.spin)
                stages[f"read {name}"] = stages.get(f"read {name}", 0.0) + \
                    time.perf_counter() - clock

                clock = time.perf_counter()
                small = resample_field(field, reduced.shape, grid=reduced)
                stages["resample"] = stages.get("resample", 0.0) + \
                    time.perf_counter() - clock

                augmentation = None
                if name == "CHGCAR":
                    clock = time.perf_counter()
                    augmentation = cache_module._augmentation_for(record,
                                                                  small)
                    stages["augmentation rescan"] = \
                        stages.get("augmentation rescan", 0.0) + \
                        time.perf_counter() - clock

                clock = time.perf_counter()
                _write(destination, record.identifier, name, small,
                       augmentation, storage)
                stages[f"write {name}"] = stages.get(f"write {name}", 0.0) + \
                    time.perf_counter() - clock
        wall = time.perf_counter() - started

        count = len(records)
        print(f"\nstorage={storage}  materials={count}  "
              f"spin={arguments.spin}  resolution={arguments.resolution}")
        for name, total in sorted(stages.items(), key=lambda item: -item[1]):
            print(f"  {name:22s} {total / count:7.3f} s/material  "
                  f"{100 * total / wall:5.1f} %")
        print(f"  {'TOTAL':22s} {wall / count:7.3f} s/material")
        report[storage] = {"seconds_per_material": wall / count,
                           "stages": {k: v / count for k, v in stages.items()}}

    text, binary = (report["files"]["seconds_per_material"],
                    report["hdf5"]["seconds_per_material"])
    print(f"\nhdf5 storage is {text / binary:.2f}x faster per material, "
          f"entirely on the write side.")
    return report


def _write(destination, identifier, name, field, augmentation, storage):
    """One field into one material's cache directory, in either layout."""
    directory = os.path.join(destination, identifier)
    if storage == "hdf5":
        from poraque.fields.hdf5 import write_fields

        write_fields(os.path.join(directory, "fields.h5"), {name: field},
                     augmentation=({name: augmentation} if augmentation
                                   else None))
    else:
        field.write(os.path.join(directory, name),
                    augmentation=augmentation or None)


# ---------------------------------------------------------------------- #
# --fixes: serial work the build does not need
# ---------------------------------------------------------------------- #
def measure_fixes(arguments):
    """
    The two landed fixes against the code they replaced, plus the one left.

    Reported as a speedup *and* as an equality assertion, because a faster
    reader that reads something else is not a fix. The "before" arm is
    :func:`legacy_grid_from_file` / :func:`legacy_read_block` in this file,
    not the package --- the package is the "after".
    """
    from poraque.fields import FieldGrid
    from poraque.fields.spin import is_spin_polarized
    from poraque.fields.vasp import volumetric

    _, records = discover(arguments, limit=arguments.materials)
    path = records[0].files["CHGCAR"]
    report = {"density": path}

    print(f"\nfix 1 -- FieldGrid.from_file parsed the whole density for a "
          f"shape\n  {path}")
    before = _best(lambda: legacy_grid_from_file(path))
    after = _best(lambda: FieldGrid.from_file(path))
    reference, candidate = legacy_grid_from_file(path), FieldGrid.from_file(path)
    assert tuple(reference.shape) == tuple(candidate.shape), "shape differs"
    assert np.allclose(reference.cell, candidate.cell), "cell differs"
    print(f"  before (parse all)  {before:7.3f} s\n"
          f"  after  (header)     {after:7.4f} s"
          f"\n  -> same grid, {before / max(after, 1e-9):.0f}x, "
          f"saves {before - after:.3f} s per material")
    report["header_only"] = {"before": before, "after": after}

    print("\nfix 2 -- _read_block called np.array once per line")
    shipped = volumetric._read_block
    try:
        after = _best(lambda: volumetric.read_volumetric(path,
                                                         read_all=True))
        candidate = volumetric.read_volumetric(path, read_all=True)
        volumetric._read_block = legacy_read_block
        before = _best(lambda: volumetric.read_volumetric(path,
                                                          read_all=True))
        reference = volumetric.read_volumetric(path, read_all=True)
    finally:
        volumetric._read_block = shipped
    assert np.array_equal(reference[1], candidate[1]), "first block differs"
    assert len(reference[2]) == len(candidate[2]), "block count differs"
    assert all(np.array_equal(x, y)
               for x, y in zip(reference[2], candidate[2])), "spin differs"
    print(f"  before (per line)   {before:7.3f} s\n"
          f"  after  (chunked)    {after:7.3f} s"
          f"\n  -> bit-identical values, {before / after:.2f}x, "
          f"saves {before - after:.3f} s per density")
    report["chunked_parse"] = {"before": before, "after": after}

    print("\nfix 3 -- the dataset-wide spin probe, before the build starts")
    probe = _best(lambda: is_spin_polarized(path))
    total = 0.0
    for record in records:
        clock = time.perf_counter()
        record.source.is_spin_polarized(record)
        total += time.perf_counter() - clock
    print(f"  one material {probe:7.3f} s (a full line scan, no float "
          f"conversion)\n  {len(records)} materials {total:7.2f} s serial, "
          f"before the first field is written")
    report["spin_probe"] = {"per_material": probe, "total": total,
                            "materials": len(records)}
    return report


def _best(call, repeat=3):
    """Fastest of ``repeat`` runs, which is the one least polluted by noise."""
    best = float("inf")
    for _ in range(repeat):
        clock = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - clock)
    return best


def legacy_grid_from_file(path):
    """
    ``FieldGrid.from_file`` **as it was before the fix**, for the "before" arm.

    It went through :func:`read_volumetric`, which converts every value in the
    file to float and reshapes them, and then returned ``data.shape`` --- a
    number the ``NGXF NGYF NGZF`` line carries a few hundred bytes in. The
    shipped version is :func:`poraque.fields.vasp.volumetric.read_grid_header`.
    """
    from poraque.fields import FieldGrid
    from poraque.fields.vasp.volumetric import read_volumetric

    structure, data, _ = read_volumetric(path)
    return FieldGrid(data.shape, structure.cell)


def legacy_read_block(lines, n_points, shape, path=""):
    """
    ``_read_block`` **as it was before the fix**, for the "before" arm.

    One ``np.array(line.split())`` per line, so NumPy's array-construction
    overhead was paid once per five values. Kept here verbatim rather than in
    the package, so the measurement in this directory's README stays
    reproducible after the shipped reader replaced it.
    """
    values = np.empty(n_points, dtype=float)
    filled = 0
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
        chunk = np.array(tokens, dtype=float)
        take = min(chunk.size, n_points - filled)
        values[filled:filled + take] = chunk[:take]
        filled += take
        if filled >= n_points:
            break

    if filled < n_points:
        raise ValueError(
            f"{path}: volumetric file truncated - expected {n_points} values, "
            f"got {filled}.")
    return values.reshape(shape, order="F")


# ---------------------------------------------------------------------- #
# --scaling: one process per material
# ---------------------------------------------------------------------- #
def _initialize(root, options):
    """
    Rebuild the source inside a worker.

    A ``MaterialRecord`` cannot be pickled: it carries its source, the source
    carries its options, and the options carry the build's log callable. So
    the parent sends the three strings a record is made of and the worker
    rebuilds the rest --- which it would have to do anyway under ``spawn``,
    and which costs one directory probe per process rather than per material.
    """
    global _SOURCE
    from poraque.data.sources import resolve_source

    _SOURCE = resolve_source(root, **options)


def _build(payload):
    """One material, in a worker. Returns what ``_build_one`` returns."""
    from poraque.data import cache as cache_module
    from poraque.ml.data import MaterialRecord

    identifier, directory, files, settings = payload
    record = MaterialRecord(identifier, directory, files=dict(files),
                            source=_SOURCE)
    clock = time.perf_counter()
    entry = cache_module._build_one(
        record, settings["cache"], settings["resolution"], FIELDS,
        settings["spin"], storage=settings["storage"])
    return identifier, entry, time.perf_counter() - clock, peak_rss_bytes()


def measure_scaling(arguments):
    """
    Wall clock against worker count, and the caches compared byte for byte.

    Records are handed out **largest first** with ``chunksize=1``: the native
    grids of one dataset span a factor of 19 in size here, and longest-job-
    first is the standard remedy for the tail a static split would leave.
    """
    source, records = discover(arguments, limit=arguments.materials)
    records = sorted(records,
                     key=lambda record: -os.path.getsize(
                         record.files["CHGCAR"]))
    settings = {"resolution": arguments.resolution, "spin": arguments.spin,
                "storage": arguments.storage}
    print(f"\nstorage={arguments.storage}  materials={len(records)}  "
          f"spin={arguments.spin}  cores={os.cpu_count()}  "
          f"start method=spawn")

    report, baseline, reference = {}, None, None
    for workers in arguments.workers:
        destination = fresh(os.path.join(
            arguments.scratch, f"scaling_{arguments.storage}_w{workers}"))
        payloads = [(record.identifier, record.directory, record.files,
                     dict(settings, cache=destination))
                    for record in records]

        started = time.perf_counter()
        if workers == 1:
            _initialize(arguments.data, source_options(arguments))
            results = [_build(payload) for payload in payloads]
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(workers, initializer=_initialize,
                              initargs=(arguments.data,
                                        source_options(arguments))) as pool:
                results = list(pool.imap(_build, payloads, chunksize=1))
        wall = time.perf_counter() - started

        busy = sum(result[2] for result in results)
        worker_peak = max(result[3] for result in results)
        baseline = baseline or wall
        reference = reference or destination
        identical = (destination == reference
                     or _trees_match(reference, destination))
        print(f"  workers={workers:2d}  wall={wall:8.2f} s  "
              f"{wall / len(records):6.3f} s/material  "
              f"speedup={baseline / wall:5.2f}x  "
              f"efficiency={busy / (wall * workers):5.2f}  "
              f"peak RSS/worker={worker_peak / 2 ** 30:5.2f} GiB  "
              f"cache {'identical' if identical else 'DIFFERS'}")
        report[workers] = {"wall": wall, "speedup": baseline / wall,
                           "efficiency": busy / (wall * workers),
                           "peak_rss_bytes_per_worker": worker_peak,
                           "identical_to_serial": identical}
    return report


def _trees_match(left, right):
    """Whether two cache directories hold the same files with equal bytes."""
    for directory, _, names in os.walk(left):
        relative = os.path.relpath(directory, left)
        other = os.path.join(right, relative)
        if not os.path.isdir(other):
            return False
        for name in names:
            if not filecmp.cmp(os.path.join(directory, name),
                               os.path.join(other, name), shallow=False):
                return False
    return True


# ---------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data", default="data/vasp/structures")
    parser.add_argument("--potcar-dir",
                        default="/Users/leseixas/Simulations/vasp/POTCARs")
    parser.add_argument("--scratch", default="/tmp/poraque_cache_parallelism")
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--materials", type=int, default=8)
    parser.add_argument("--storage", default="files", choices=("files",
                                                               "hdf5"))
    parser.add_argument("--workers", type=int, nargs="+",
                        default=[1, 2, 4, 8])
    parser.add_argument("--spin", action="store_true",
                        help="cache the magnetization channel too")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fixes", action="store_true")
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args(argv)

    if not (arguments.profile or arguments.fixes or arguments.scaling):
        parser.error("choose at least one of --profile, --fixes, --scaling")
    os.makedirs(arguments.scratch, exist_ok=True)

    report = {}
    if arguments.profile:
        report["profile"] = measure_profile(arguments)
    if arguments.fixes:
        report["fixes"] = measure_fixes(arguments)
    if arguments.scaling:
        report["scaling"] = measure_scaling(arguments)
    print(f"\npeak RSS of this process: "
          f"{peak_rss_bytes() / 2 ** 30:.2f} GiB")

    if arguments.json:
        with open(arguments.json, "w") as handle:
            json.dump(report, handle, indent=1, sort_keys=True, default=str)
        print(f"wrote {arguments.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
